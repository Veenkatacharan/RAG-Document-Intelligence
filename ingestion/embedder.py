import torch
from sentence_transformers import SentenceTransformer

# detects your GPU automatically
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "all-mpnet-base-v2"

print(f"Loading embedding model on {DEVICE}...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)


def embed_chunks(chunks: list[dict], batch_size: int = 64) -> list[dict]:
    """
    Takes a list of chunk dicts.
    Adds an "embedding" key to each dict containing a list of floats.
    Returns the updated list.
    """
    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks in batches of {batch_size}...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # important for cosine similarity search
    )

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    print(f"Done. Each embedding has {len(chunks[0]['embedding'])} dimensions.")
    return chunks


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from ingestion.loader import load_all_pdfs
    from ingestion.chunker import chunk_pages

    pages = load_all_pdfs("data")
    chunks = chunk_pages(pages)
    chunks = embed_chunks(chunks)

    # inspect first chunk
    c = chunks[0]
    print(f"\nChunk: {c['text'][:100]}...")
    print(f"Embedding preview: {c['embedding'][:5]}...")
    print(f"Embedding length: {len(c['embedding'])}")
