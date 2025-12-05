#!/usr/bin/env python3
"""
index_pdf_to_pgvector.py

- Chunk `merged_text` (page-aware)
- Normalize simple dates found in text (uses dateutil)
- Call local Ollama embedding model (nomic-embed-text:latest)
- Store embeddings + metadata in Postgres pgvector table
"""

import os
import re
import json
import math
import time
import logging
from typing import List, Dict, Any, Tuple, Optional

import requests
import psycopg2
from psycopg2.extras import Json
from dateutil import parser as dateparser

# ---------- CONFIG ----------
DB_NAME = os.getenv("DB_NAME", "fortress_db")
DB_USER = os.getenv("DB_USER", "ashiya")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123456")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", None)  # optional
OLLAMA_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

# Embedding dim: set to actual model dimension if known (default 768).
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# chunk parameters (characters)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))   # desired chunk size in characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# DB table name
VECTOR_TABLE = os.getenv("VECTOR_TABLE", "document_vectors")

# Batch size for embeddings API
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- Utilities ----------
def pg_conn():
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    return conn


def ensure_vector_table(conn, dim: int = EMBED_DIM, table_name: str = VECTOR_TABLE):
    """
    Create vector extension and vectors table if not exists.
    Table schema:
      id serial primary key
      document_id text -- source doc identifier (file path or URL)
      page int
      chunk_index int
      chunk_text text
      metadata jsonb
      embedding vector(dim)
      created_at timestamptz default now()
    """
    with conn.cursor() as cur:
        # Create vector extension if allowed
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Create table with vector column dimension
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            document_id TEXT,
            page INT,
            chunk_index INT,
            chunk_text TEXT,
            metadata JSONB,
            embedding VECTOR({dim}),
            created_at TIMESTAMPTZ DEFAULT now()
        );
        -- add index for nearest neighbor search (Optional L2 / cosine)
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes WHERE tablename = '{table_name}' AND indexname = '{table_name}_embedding_idx'
            ) THEN
                CREATE INDEX {table_name}_embedding_idx ON {table_name} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            END IF;
        EXCEPTION WHEN others THEN
            -- index creation might fail on unsupported Postgres versions; ignore
            RAISE NOTICE 'Could not create ivfflat index (maybe unsupported) - continuing';
        END;
        $$;
        """
        cur.execute(create_sql)
    conn.commit()
    logger.info(f"Ensured table `{table_name}` with embedding dim={dim}")


# ---------- Text normalization & chunking ----------
PAGE_MARKER_RE = re.compile(r"\[PAGE\s+(\d+)\s+(PDF|OCR)\]", flags=re.IGNORECASE)


def split_by_pages(merged_text: str) -> Dict[int, str]:
    """
    Split merged_text into pages using the markers produced earlier:
    [PAGE {n} PDF] and [PAGE {n} OCR]
    We prefer the PDF text marker; if both exist we can merge or prefer PDF.
    Return mapping page_number -> page_text (combined PDF + OCR optionally).
    """
    parts = PAGE_MARKER_RE.split(merged_text)
    # parts will be like ['', page_num, marker, content, page_num, marker, content, ...]
    pages: Dict[int, List[str]] = {}
    i = 0
    while i < len(parts):
        if parts[i] == "":
            i += 1
            continue
        # Expect sequence: page_num, marker, content
        try:
            page_num = int(parts[i])
            marker = parts[i + 1]
            content = parts[i + 2]
            pages.setdefault(page_num, []).append((marker.upper(), content.strip()))
            i += 3
        except Exception:
            # fallback: break
            break
    # Consolidate: prefer PDF content over OCR if both exist; otherwise join.
    consolidated: Dict[int, str] = {}
    for pnum, contents in pages.items():
        pdfs = [c for m, c in contents if "PDF" in m]
        ocrs = [c for m, c in contents if "OCR" in m]
        if pdfs:
            # join pdf segments (usually single)
            consolidated[pnum] = "\n\n".join(pdfs)
        elif ocrs:
            consolidated[pnum] = "\n\n".join(ocrs)
        else:
            consolidated[pnum] = "\n\n".join([c for m, c in contents])
    return consolidated


def paragraph_split(text: str) -> List[str]:
    """
    Split text into paragraphs (blank line or long newline separation).
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return paras


def chunk_paragraph(paragraph: str, max_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Chunk a single paragraph into chunks of max_size with overlap (character-based).
    """
    if len(paragraph) <= max_size:
        return [paragraph]
    chunks = []
    start = 0
    while start < len(paragraph):
        end = start + max_size
        chunk = paragraph[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(paragraph):
            break
    return chunks


def page_to_chunks(page_num: int, page_text: str, doc_id: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Turn a page text into list of chunks each with metadata.
    Returns list of dicts:
      {
        'document_id': doc_id,
        'page': page_num,
        'chunk_index': i,
        'chunk_text': text,
        'metadata': {...}
      }
    """
    paras = paragraph_split(page_text)
    offset_cursor = 0
    chunks = []
    chunk_index = 0
    for para in paras:
        para_chunks = chunk_paragraph(para, max_size=chunk_size, overlap=overlap)
        for c in para_chunks:
            # compute approximate offsets relative to page (best-effort)
            # find c occurrence in page_text starting at offset_cursor (best-effort)
            idx = page_text.find(c, offset_cursor)
            if idx == -1:
                idx = offset_cursor
            start_offset = idx
            end_offset = idx + len(c)
            meta = {"page": page_num, "char_start": start_offset, "char_end": end_offset}
            chunks.append({
                "document_id": doc_id,
                "page": page_num,
                "chunk_index": chunk_index,
                "chunk_text": c,
                "metadata": meta,
            })
            chunk_index += 1
            offset_cursor = end_offset
    return chunks


# ---------- Simple date normalization helper ----------
def attempt_normalize_date(raw: str) -> Optional[str]:
    """
    Try to parse raw date-like strings into ISO YYYY-MM-DD using dateutil.
    Returns string or None.
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    # quick guard: ignore obviously wrong short tokens
    if re.match(r"^\d{1,2}-\d{1,2}-\d{1,4}$", raw) or re.search(r"[A-Za-z]", raw):
        try:
            dt = dateparser.parse(raw, dayfirst=False)  # defaults; user timezone not needed
            if dt:
                return dt.date().isoformat()
        except Exception:
            return None
    else:
        # try generic parse
        try:
            dt = dateparser.parse(raw)
            if dt:
                return dt.date().isoformat()
        except Exception:
            return None
    return None


# ---------- Ollama embedding call ----------
def call_ollama_embeddings(texts: List[str], model: str = OLLAMA_MODEL, base_url: str = OLLAMA_URL, api_key: Optional[str] = OLLAMA_API_KEY) -> List[List[float]]:
    """
    Call Ollama's embedding API in batches.
    Expect endpoint: POST {base_url}/api/embed  (or /api/embeddings)
    Body example: { "model": "nomic-embed-text:latest", "input": ["one text", "another"] }
    Response: depends on version - try to handle common shapes.
    Returns list of embedding vectors (list of floats) in same order as texts.
    """
    embeddings = []
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    endpoint = f"{base_url.rstrip('/')}/api/embed"  # common Ollama embedding path
    logger.info(f"Calling Ollama embeddings endpoint: {endpoint} (model={model})")

    # batch
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        payload = {"model": model, "input": batch}
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=6000)
        if resp.status_code != 200:
            # try fallback path /api/embeddings
            alt_endpoint = f"{base_url.rstrip('/')}/api/embeddings"
            logger.warning(f"Got status {resp.status_code} from {endpoint}, trying {alt_endpoint}")
            resp = requests.post(alt_endpoint, headers=headers, json=payload, timeout=6000)

        resp.raise_for_status()
        j = resp.json()
        # Handle multiple response shapes:
        # Shape A: {"data": [{"embedding": [...]}, ...]}
        # Shape B: {"embedding": [...]} for single
        # Shape C: {"embeddings": [[...], [...]]}
        if isinstance(j, dict) and "data" in j and isinstance(j["data"], list):
            for item in j["data"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
                elif isinstance(item, list):
                    embeddings.append(item)
                else:
                    raise ValueError("Unexpected item in j['data'] from Ollama: " + str(item)[:200])
        elif isinstance(j, dict) and "embeddings" in j:
            for e in j["embeddings"]:
                embeddings.append(e)
        elif isinstance(j, dict) and "embedding" in j:
            # single vector - but we requested batch; handle by wrapping
            e = j["embedding"]
            if isinstance(e[0], list):  # maybe nested
                embeddings.extend(e)
            else:
                embeddings.append(e)
        elif isinstance(j, list) and all(isinstance(x, (list, float)) for x in j):
            # direct list of vectors
            if isinstance(j[0], list):
                embeddings.extend(j)
            else:
                # list of floats (single vector)
                embeddings.append(j)
        else:
            # as a last resort, try extract any arrays
            # attempt keys check
            found = False
            for k, v in (j.items() if isinstance(j, dict) else []):
                if isinstance(v, list) and v and isinstance(v[0], (list, float)):
                    embeddings.extend(v)
                    found = True
                    break
            if not found:
                raise ValueError(f"Unable to parse embedding response shape: {j}")
        time.sleep(0.01)  # tiny throttle
    if len(embeddings) != len(texts):
        logger.warning(f"Embeddings count {len(embeddings)} != texts {len(texts)} (some mismatch)")
    return embeddings


# ---------- DB upsert ----------
def upsert_chunks_with_embeddings(
    conn,
    table_name: str,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    upsert_on_conflict: bool = False,
):
    """
    Insert rows into Postgres table. One row per chunk.
    columns: document_id, page, chunk_index, chunk_text, metadata (jsonb), embedding (vector)
    """
    assert len(chunks) == len(embeddings), "chunks/embeddings length mismatch"

    with conn.cursor() as cur:
        for chunk, emb in zip(chunks, embeddings):
            doc_id = chunk["document_id"]
            page = chunk["page"]
            idx = chunk["chunk_index"]
            text = chunk["chunk_text"]
            meta = chunk.get("metadata", {})
            # Insert - simple insert (no unique constraint). If you want dedupe/upsert, add unique constraints & ON CONFLICT.
            insert_sql = f"""
            INSERT INTO {table_name} (document_id, page, chunk_index, chunk_text, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            # psycopg2 will accept Python list for vector column with pgvector extension.
            cur.execute(insert_sql, (doc_id, page, idx, text, Json(meta), emb))
    conn.commit()
    logger.info(f"Inserted {len(chunks)} vectors into {table_name}")


# ---------- Orchestrator ----------
def index_merged_text(merged_text: str, document_id: str = "local_doc", conn=None):
    """
    High-level function:
      - split by pages
      - chunk
      - embed in batches
      - write to DB
    """
    if conn is None:
        conn = pg_conn()

    ensure_vector_table(conn, dim=EMBED_DIM, table_name=VECTOR_TABLE)

    page_map = split_by_pages(merged_text)
    all_chunks: List[Dict[str, Any]] = []
    for pnum in sorted(page_map.keys()):
        ptext = page_map[pnum]
        chunks = page_to_chunks(pnum, ptext, document_id)
        # annotate with a few heuristics: try to extract date mentions from chunk and normalize
        for c in chunks:
            # find 1-2 date-like substrings and normalize
            found_dates = []
            for m in re.findall(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b[^\n,]{0,30}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b", c["chunk_text"], flags=re.IGNORECASE):
                norm = attempt_normalize_date(m)
                if norm:
                    found_dates.append({"raw": m, "normalized": norm})
            if found_dates:
                c["metadata"]["dates"] = found_dates
            all_chunks.append(c)

    logger.info(f"Prepared {len(all_chunks)} chunks from {len(page_map)} pages")

    # call embeddings in batches
    texts = [c["chunk_text"] for c in all_chunks]
    if not texts:
        logger.info("No text to embed.")
        return

    embeddings = call_ollama_embeddings(texts, model=OLLAMA_MODEL, base_url=OLLAMA_URL, api_key=OLLAMA_API_KEY)
    if len(embeddings) < len(texts):
        # if mismatch, try to handle partial result or pad with zeros
        logger.warning("Embeddings fewer than texts; padding with zero vectors")
        while len(embeddings) < len(texts):
            embeddings.append([0.0] * EMBED_DIM)

    upsert_chunks_with_embeddings(conn, VECTOR_TABLE, all_chunks, embeddings)
    logger.info("Indexing complete.")


