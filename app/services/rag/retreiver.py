# app/services/rag/retriever.py
from typing import List, Dict, Any, Tuple, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging
from app.services.index_pdf_to_pgvector import pg_conn, VECTOR_TABLE, EMBED_DIM

logger = logging.getLogger(__name__)

# Config (can also live in app/config.py)
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

def knn_query(conn, query_vector: List[float], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Run a KNN search on pgvector table and return rows with distance.
    NOTE: this uses the pgvector nearest-neighbour operator `<->` for distance ordering.
    If your install supports cosine ops you can adapt accordingly.
    """
    sql = f"""
    SELECT id, document_id, page, chunk_index, chunk_text, metadata, embedding
    FROM {VECTOR_TABLE}
    ORDER BY embedding <-> %s::vector
    LIMIT %s;
    """
    # psycopg2 accepts Python lists for vector parameters with pgvector installed
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (query_vector, top_k))
        rows = cur.fetchall()
    return rows

def retrieve_similar_chunks(query_vector: List[float], top_k: int = TOP_K, conn=None) -> List[Dict[str, Any]]:
    close_conn = False
    if conn is None:
        conn = pg_conn()
        close_conn = True
    try:
        rows = knn_query(conn, query_vector, top_k)
        # map fields into a consistent structure
        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "document_id": r["document_id"],
                "page": r["page"],
                "chunk_index": r["chunk_index"],
                "chunk_text": r["chunk_text"],
                "metadata": r["metadata"],
                # optionally compute similarity from returned embedding (not necessary)
            })
        return results
    finally:
        if close_conn:
            conn.close()
