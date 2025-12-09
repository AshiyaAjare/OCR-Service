# app/services/rag/retriever.py
from typing import List, Dict, Any, Tuple, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging
import re
from app.services.index_pdf_to_pgvector import pg_conn, VECTOR_TABLE, EMBED_DIM
from app.config import settings

logger = logging.getLogger(__name__)

# Config (can also live in app/config.py)
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# Table name for documents (use prefix from settings)
DOCUMENT_TABLE = os.getenv("DOCUMENT_TABLE", f"{settings.DATABASE_TABLE_PREFIX}extracted_document")

def extract_company_name_from_question(question: str) -> Optional[str]:
    """
    Try to extract company name from question using simple heuristics.
    This is a basic implementation - you might want to use NER or more sophisticated parsing.
    """
    # Common words to exclude
    exclude_words = {'the', 'a', 'an', 'what', 'is', 'are', 'was', 'were', 'broker', 
                     'estimate', 'target', 'data', 'information', 'about', 'for', 'on'}
    
    # Look for patterns like "for X", "about X", "data for X", "X's", etc.
    # Use more specific patterns that capture full company names
    patterns = [
        # Pattern for "data for X" or "information for X" - capture everything until question mark or end
        r"(?:data|information|details)\s+(?:for|about|on)\s+([A-Z][A-Za-z\s&]+?)(?:\s*\?|$)",
        # Pattern for "for X" - capture everything until question mark, space before common words, or end
        r"(?:for|about|on)\s+([A-Z][A-Za-z\s&]+?)(?:\s*\?|$|(?=\s+(?:broker|estimate|target|is|are)))",
        # Pattern for company name before "broker", "estimate", etc.
        r"([A-Z][A-Za-z\s&]+?)(?:\s+(?:broker|estimate|target))",
        # Pattern for company name at the end before question mark
        r"([A-Z][A-Za-z\s&]{2,})(?:\s*\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            company = match.group(1).strip()
            # Remove trailing question marks or spaces
            company = company.rstrip('?').strip()
            
            # Filter out common words and short matches
            company_lower = company.lower()
            words = company_lower.split()
            
            # If all words are in exclude list, skip
            if all(word in exclude_words for word in words):
                continue
            
            # If any word is in exclude list but not all, try to extract the company name part
            if any(word in exclude_words for word in words):
                # Try to find the company name part (usually the last significant word or phrase)
                significant_words = [w for w in words if w not in exclude_words]
                if significant_words:
                    # Reconstruct from original to preserve capitalization
                    original_words = company.split()
                    # Find the position of first significant word
                    first_sig_idx = next((i for i, w in enumerate(original_words) 
                                         if w.lower() not in exclude_words), None)
                    if first_sig_idx is not None:
                        company = ' '.join(original_words[first_sig_idx:])
            
            if len(company) > 2 and company_lower not in exclude_words:
                logger.info(f"Extracted company name from question: '{company}'")
                return company
    logger.debug(f"Could not extract company name from question: '{question}'")
    return None

def knn_query_with_metadata_filter(
    conn, 
    query_vector: List[float], 
    top_k: int = TOP_K,
    company_name: Optional[str] = None,
    broker_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Run a KNN search with optional metadata filtering by joining with document table.
    Only searches documents that are vector_indexed = true.
    """
    # Build WHERE clause for metadata filtering
    where_clauses = []
    where_params = []
    
    # Always filter to only indexed documents
    where_clauses.append("d.vector_indexed = true")
    
    if company_name:
        # Try exact match first (case-insensitive), then LIKE for partial matches
        # This handles variations like "Asian Paints" vs "ASIAN PAINTS"
        company_clean = company_name.strip()
        # Build a flexible matching pattern that handles:
        # 1. Exact match (case-insensitive, trimmed)
        # 2. Partial match with LIKE
        # 3. Word-based matching (for multi-word company names)
        words = company_clean.split()
        if len(words) > 1:
            # For multi-word names, match if all words are present (handles word order variations)
            # Use a pattern that matches all words: "%word1%word2%"
            word_pattern = "%".join(words)
            where_clauses.append("(LOWER(TRIM(d.company_name)) = LOWER(TRIM(%s)) OR LOWER(d.company_name) LIKE LOWER(%s) OR LOWER(d.company_name) LIKE LOWER(%s))")
            where_params.append(company_clean)
            where_params.append(f"%{company_clean}%")
            where_params.append(f"%{word_pattern}%")
        else:
            # Single word - simpler matching
            where_clauses.append("(LOWER(TRIM(d.company_name)) = LOWER(TRIM(%s)) OR LOWER(d.company_name) LIKE LOWER(%s))")
            where_params.append(company_clean)
            where_params.append(f"%{company_clean}%")
        logger.info(f"Filtering by company_name: '{company_name}' (clean: '{company_clean}', words: {words})")
    
    if broker_name:
        where_clauses.append("(LOWER(TRIM(d.broker_name)) = LOWER(TRIM(%s)) OR LOWER(d.broker_name) LIKE LOWER(%s))")
        where_params.append(broker_name.strip())
        where_params.append(f"%{broker_name.strip()}%")
        logger.info(f"Filtering by broker_name: '{broker_name}'")
    
    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)
    
    # Use a subquery approach: first get top K*3 candidates, then filter and take top K
    # This ensures we still get good vector similarity while applying metadata filters
    # The INNER JOIN ensures we only get chunks from documents that exist in the document table
    sql = f"""
    WITH ranked_chunks AS (
        SELECT 
            v.id, 
            v.document_id, 
            v.page, 
            v.chunk_index, 
            v.chunk_text, 
            v.metadata, 
            v.embedding,
            v.embedding <-> %s::vector AS distance
        FROM {VECTOR_TABLE} v
        INNER JOIN {DOCUMENT_TABLE} d ON CAST(v.document_id AS INTEGER) = d.id
        {where_sql}
        ORDER BY v.embedding <-> %s::vector
        LIMIT %s
    )
    SELECT id, document_id, page, chunk_index, chunk_text, metadata, embedding, distance
    FROM ranked_chunks
    ORDER BY distance
    LIMIT %s;
    """
    
    # Build parameter list in correct order:
    # 1. query_vector (for distance calculation in SELECT)
    # 2. WHERE clause params (vector_indexed check, company_name, broker_name if present)
    # 3. query_vector (for ORDER BY)
    # 4. search_k (for first LIMIT)
    # 5. top_k (for second LIMIT)
    # Always have at least vector_indexed filter, so where_clauses is never empty
    # If filtering by company/broker name, get more candidates first to ensure we have enough after filtering
    if company_name or broker_name:
        # Increase multiplier when filtering by company/broker name to get better results
        search_k = top_k * 10  # Increased from 3 to 10 to get more candidates
        logger.debug(f"Using metadata filter with company/broker name, search_k={search_k}, top_k={top_k}")
    else:
        # Only filtering by vector_indexed, use smaller multiplier
        search_k = top_k * 2
        logger.debug(f"Using metadata filter (vector_indexed only), search_k={search_k}, top_k={top_k}")
    
    final_params = [query_vector] + where_params + [query_vector, search_k, top_k]
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                # Add diagnostic query before the main query
                if company_name:
                    # Check how many documents match the company name
                    diag_sql = f"""
                    SELECT COUNT(*) as total_count,
                           SUM(CASE WHEN vector_indexed = true THEN 1 ELSE 0 END) as indexed_count
                    FROM {DOCUMENT_TABLE}
                    WHERE LOWER(company_name) LIKE LOWER(%s) OR LOWER(company_name) LIKE LOWER(%s)
                    """
                    company_clean = company_name.strip()
                    words = company_clean.split()
                    if len(words) > 1:
                        word_pattern = "%".join(words)
                        cur.execute(diag_sql, (f"%{company_clean}%", f"%{word_pattern}%"))
                    else:
                        cur.execute(diag_sql, (f"%{company_clean}%", f"%{company_clean}%"))
                    diag_result = cur.fetchone()
                    logger.info(f"Diagnostic: Found {diag_result['total_count']} total documents matching '{company_name}', {diag_result['indexed_count']} are indexed")
                    
                    # Check how many chunks exist for indexed documents with this company name
                    chunks_sql = f"""
                    SELECT COUNT(*) as chunk_count
                    FROM {VECTOR_TABLE} v
                    INNER JOIN {DOCUMENT_TABLE} d ON CAST(v.document_id AS INTEGER) = d.id
                    WHERE d.vector_indexed = true
                      AND (LOWER(d.company_name) LIKE LOWER(%s) OR LOWER(d.company_name) LIKE LOWER(%s))
                    """
                    if len(words) > 1:
                        cur.execute(chunks_sql, (f"%{company_clean}%", f"%{word_pattern}%"))
                    else:
                        cur.execute(chunks_sql, (f"%{company_clean}%", f"%{company_clean}%"))
                    chunks_result = cur.fetchone()
                    logger.info(f"Diagnostic: Found {chunks_result['chunk_count']} chunks for indexed documents matching '{company_name}'")
                
                # Execute the main query
                logger.debug(f"Executing metadata filter query with {len(final_params)} parameters")
                cur.execute(sql, final_params)
                rows = cur.fetchall()
                logger.info(f"Retrieved {len(rows)} chunks with metadata filter (company_name={company_name}, broker_name={broker_name})")
                if rows:
                    doc_ids = set([r["document_id"] for r in rows])
                    logger.info(f"Retrieved chunks from documents: {doc_ids}")
                else:
                    logger.warning(f"No chunks retrieved. SQL: {sql[:200]}...")
                    logger.warning(f"Parameters: {[str(p)[:50] if isinstance(p, str) else type(p).__name__ for p in final_params[:5]]}")
            except Exception as e:
                logger.error(f"Error executing KNN query with metadata filter: {e}")
                logger.error(f"SQL: {sql}")
                logger.error(f"Params count: {len(final_params)}, types: {[type(p).__name__ for p in final_params[:5]]}...")
                raise
    return rows

def knn_query(conn, query_vector: List[float], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Run a KNN search on pgvector table and return rows with distance.
    Only searches documents that are vector_indexed = true.
    NOTE: this uses the pgvector nearest-neighbour operator `<->` for distance ordering.
    If your install supports cosine ops you can adapt accordingly.
    """
    sql = f"""
    SELECT v.id, v.document_id, v.page, v.chunk_index, v.chunk_text, v.metadata, v.embedding
    FROM {VECTOR_TABLE} v
    INNER JOIN {DOCUMENT_TABLE} d ON CAST(v.document_id AS INTEGER) = d.id
    WHERE d.vector_indexed = true
    ORDER BY v.embedding <-> %s::vector
    LIMIT %s;
    """
    # psycopg2 accepts Python lists for vector parameters with pgvector installed
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (query_vector, top_k))
        rows = cur.fetchall()
    return rows

def check_document_indexed(conn, document_id: int) -> bool:
    """
    Check if a document is indexed in the vector table.
    """
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT COUNT(*) as count FROM {VECTOR_TABLE} WHERE document_id = %s",
                (str(document_id),)
            )
            result = cur.fetchone()
            count = result["count"] if result else 0
            logger.info(f"Document {document_id} has {count} chunks in vector table")
            return count > 0
    except Exception as e:
        logger.error(f"Error checking if document {document_id} is indexed: {e}")
        return False

def cleanup_orphaned_chunks(conn=None) -> int:
    """
    Remove chunks from vector table that belong to documents that no longer exist.
    Returns the number of orphaned chunks deleted.
    """
    close_conn = False
    if conn is None:
        conn = pg_conn()
        close_conn = True
    try:
        with conn.cursor() as cur:
            # Delete chunks where the document_id doesn't exist in the document table
            delete_sql = f"""
            DELETE FROM {VECTOR_TABLE} v
            WHERE NOT EXISTS (
                SELECT 1 FROM {DOCUMENT_TABLE} d 
                WHERE CAST(v.document_id AS INTEGER) = d.id
            )
            """
            cur.execute(delete_sql)
            deleted_count = cur.rowcount
            conn.commit()
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} orphaned chunks from vector table")
            return deleted_count
    except Exception as e:
        logger.error(f"Error cleaning up orphaned chunks: {e}")
        if conn:
            conn.rollback()
        return 0
    finally:
        if close_conn:
            conn.close()

def retrieve_similar_chunks(
    query_vector: List[float], 
    top_k: int = TOP_K, 
    conn=None,
    company_name: Optional[str] = None,
    broker_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    close_conn = False
    if conn is None:
        conn = pg_conn()
        close_conn = True
    try:
        # Use metadata filtering if company_name or broker_name is provided
        if company_name or broker_name:
            logger.info(f"Using metadata filter: company_name={company_name}, broker_name={broker_name}")
            try:
                rows = knn_query_with_metadata_filter(
                    conn, query_vector, top_k=top_k, 
                    company_name=company_name, broker_name=broker_name
                )
                # If metadata filtering returns no results, fall back to regular search
                if not rows:
                    logger.warning(f"Metadata filter returned no results for company_name='{company_name}', broker_name='{broker_name}'. Falling back to regular vector search.")
                    # Check if any documents with this company name are indexed
                    if company_name:
                        with conn.cursor(cursor_factory=RealDictCursor) as cur:
                            cur.execute(
                                f"SELECT id, company_name, vector_indexed FROM {DOCUMENT_TABLE} WHERE LOWER(company_name) LIKE LOWER(%s) LIMIT 5",
                                (f"%{company_name}%",)
                            )
                            matching_docs = cur.fetchall()
                            if matching_docs:
                                logger.info(f"Found {len(matching_docs)} documents matching company_name '{company_name}':")
                                for doc in matching_docs:
                                    indexed = check_document_indexed(conn, doc["id"])
                                    logger.info(f"  - Document {doc['id']}: company_name='{doc['company_name']}', vector_indexed={doc.get('vector_indexed')}, chunks_in_vector_table={indexed}")
                    rows = knn_query(conn, query_vector, top_k)
            except Exception as e:
                logger.warning(f"Metadata filter query failed: {e}, falling back to regular vector search")
                rows = knn_query(conn, query_vector, top_k)
        else:
            rows = knn_query(conn, query_vector, top_k)
        
        # map fields into a consistent structure and validate documents exist
        results = []
        doc_ids_to_check = set()
        for r in rows:
            doc_id_str = str(r["document_id"])
            doc_ids_to_check.add(doc_id_str)
            results.append({
                "id": r["id"],
                "document_id": doc_id_str,
                "page": r["page"],
                "chunk_index": r["chunk_index"],
                "chunk_text": r["chunk_text"],
                "metadata": r["metadata"],
                # optionally compute similarity from returned embedding (not necessary)
            })
        
        # Validate that all documents still exist in the database (safety check)
        if doc_ids_to_check:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Check which document IDs actually exist
                    doc_ids_list = [int(did) for did in doc_ids_to_check if did.isdigit()]
                    if doc_ids_list:
                        placeholders = ','.join(['%s'] * len(doc_ids_list))
                        
                        # First, let's verify what table we're querying
                        logger.info(f"Validation: Querying table '{DOCUMENT_TABLE}' for document IDs: {doc_ids_list}")
                        
                        # Execute the query and log the actual SQL
                        query_sql = f"SELECT id FROM {DOCUMENT_TABLE} WHERE id IN ({placeholders})"
                        logger.debug(f"Validation SQL: {query_sql} with params: {doc_ids_list}")
                        
                        cur.execute(query_sql, doc_ids_list)
                        existing_docs = {str(row["id"]) for row in cur.fetchall()}
                        
                        # Also check if there are any chunks for these document IDs in the vector table
                        cur.execute(
                            f"SELECT DISTINCT document_id FROM {VECTOR_TABLE} WHERE document_id IN ({placeholders})",
                            [str(did) for did in doc_ids_list]
                        )
                        chunks_doc_ids = {str(row["document_id"]) for row in cur.fetchall()}
                        
                        # Log what we found
                        logger.info(f"Validation: Checking {len(doc_ids_list)} document IDs: {doc_ids_list}")
                        logger.info(f"Validation: Found {len(existing_docs)} existing documents in {DOCUMENT_TABLE}: {existing_docs}")
                        logger.info(f"Validation: Found {len(chunks_doc_ids)} document IDs with chunks in {VECTOR_TABLE}: {chunks_doc_ids}")
                        
                        # Check for mismatches
                        missing_docs = set(doc_ids_list) - {int(did) for did in existing_docs}
                        if missing_docs:
                            logger.warning(f"Validation: Documents {missing_docs} are NOT in {DOCUMENT_TABLE} but have chunks in {VECTOR_TABLE}")
                        
                        # Filter out chunks from non-existent documents
                        filtered_results = [r for r in results if r["document_id"] in existing_docs]
                        
                        if len(filtered_results) < len(results):
                            removed_count = len(results) - len(filtered_results)
                            removed_docs = doc_ids_to_check - existing_docs
                            logger.warning(f"Filtered out {removed_count} chunks from {len(removed_docs)} non-existent documents: {removed_docs}")
                            results = filtered_results
                        else:
                            if missing_docs:
                                logger.error(f"Validation ERROR: Documents {missing_docs} don't exist but were returned by INNER JOIN query! This should not happen.")
                            else:
                                logger.info(f"Validation: All {len(results)} chunks are from existing documents")
            except Exception as e:
                logger.warning(f"Error validating document existence: {e}. Proceeding with all results.")
                import traceback
                logger.error(traceback.format_exc())
        
        # Log which documents were retrieved
        if results:
            doc_ids = set([r["document_id"] for r in results])
            logger.info(f"Retrieved chunks from {len(results)} chunks across {len(doc_ids)} documents: {doc_ids}")
        
        return results
    finally:
        if close_conn:
            conn.close()
