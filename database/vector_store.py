import os
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

load_dotenv(Path(__file__).parent.parent / ".env")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "ragdb"),
    "user":     os.getenv("DB_USER", "vc"),
    "password": os.getenv("DB_PASSWORD", "")
}


def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


def save_chunks(chunks: list[dict]) -> None:
    """
    Takes a list of chunk dicts (with embeddings already attached).
    Inserts all of them into the document_chunks table.
    """
    conn = get_connection()
    cur = conn.cursor()

    rows = [
        (
            chunk["text"],
            chunk["source"],
            chunk["page"],
            chunk["chunk_index"],
            chunk["embedding"]
        )
        for chunk in chunks
    ]

    insert_query = """
        INSERT INTO document_chunks (text, source, page, chunk_index, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
    """

    execute_batch(cur, insert_query, rows)
    conn.commit()
    print(f"Saved {len(chunks)} chunks to the database.")
    cur.close()
    conn.close()


def search_similar_chunks(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Takes a pre-computed query embedding (list of floats).
    Returns the top_k most similar chunks from the database.
    """
    conn = get_connection()
    cur = conn.cursor()

    search_query = """
        SELECT
            text,
            source,
            page,
            chunk_index,
            1 - (embedding <=> %s::vector) AS similarity
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    cur.execute(search_query, (query_embedding, query_embedding, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "text":        row[0],
            "source":      row[1],
            "page":        row[2],
            "chunk_index": row[3],
            "similarity":  round(row[4], 4)
        }
        for row in rows
    ]


def clear_all_chunks() -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks;")
    conn.commit()
    print("Cleared all chunks from the database.")
    cur.close()
    conn.close()


def get_chunk_count() -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM document_chunks;")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from ingestion.loader import load_all_pdfs
    from ingestion.chunker import chunk_pages
    from ingestion.embedder import embed_chunks

    print("=== Week 3 Test ===\n")

    print("--- Step 1: Loading PDFs ---")
    pages = load_all_pdfs("data")

    print("\n--- Step 2: Chunking ---")
    chunks = chunk_pages(pages)

    print("\n--- Step 3: Embedding ---")
    chunks = embed_chunks(chunks)

    print("\n--- Step 4: Clearing old data and saving to database ---")
    clear_all_chunks()
    save_chunks(chunks)
    print(f"Total chunks in database: {get_chunk_count()}")

    print("\n--- Step 5: Testing search ---")
    query = "What are the technical skills?"
    print(f"Query: '{query}'\n")

    query_chunk = embed_chunks([{"text": query, "source": "query", "page": 0, "chunk_index": -1}])
    query_embedding = query_chunk[0]["embedding"]
    results = search_similar_chunks(query_embedding, top_k=3)

    for i, result in enumerate(results):
        print(f"Result {i+1} | similarity: {result['similarity']} | {result['source']} p.{result['page']}")
        print(f"{result['text'][:200]}...")
        print()
