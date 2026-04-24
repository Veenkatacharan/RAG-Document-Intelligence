import os
from pathlib import Path
import requests
from dotenv import load_dotenv
from database.vector_store import search_similar_chunks
from ingestion.embedder import embed_chunks

load_dotenv(Path(__file__).parent / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = "llama3.1"
TOP_K = 5  # how many chunks to retrieve


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Assembles the context chunks and question into a single prompt string.
    The quality of this prompt directly affects answer quality.
    """
    # build the context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"--- Source {i+1}: {chunk['source']} | Page {chunk['page']} | Similarity: {chunk['similarity']} ---\n"
            f"{chunk['text']}"
        )

    context_block = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided context documents.

Rules you must follow:
- Only use information from the provided context to answer
- If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer that"
- Always mention which source/page your answer comes from
- Be concise and clear
- Do not make up or infer information beyond what is explicitly stated

CONTEXT:
{context_block}

QUESTION:
{question}

ANSWER:"""

    return prompt


def ask(question: str, verbose: bool = False) -> dict:
    """
    The main function. Takes a plain English question.
    Retrieves relevant chunks, builds a prompt, calls Claude,
    and returns the answer with sources.

    Args:
        question: the user's question
        verbose: if True, prints the retrieved chunks before the answer
    """
    print(f"\nQuestion: {question}")
    print("-" * 60)

    # Step 1 — embed the question then retrieve relevant chunks from pgvector
    print("Retrieving relevant chunks...")
    query_chunk = embed_chunks([{"text": question, "source": "query", "page": 0, "chunk_index": -1}])
    query_embedding = query_chunk[0]["embedding"]
    chunks = search_similar_chunks(query_embedding, top_k=TOP_K)

    if not chunks:
        return {
            "question": question,
            "answer": "No relevant documents found in the database.",
            "sources": []
        }

    # Step 2 — optionally show what was retrieved
    if verbose:
        print(f"\nTop {len(chunks)} retrieved chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\n[{i+1}] {chunk['source']} p.{chunk['page']} | similarity: {chunk['similarity']}")
            print(f"    {chunk['text'][:150]}...")

    # Step 3 — build the prompt
    prompt = build_prompt(question, chunks)

    # Step 4 — call Ollama API
    print("\nGenerating answer...")
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
    answer = response.json()["response"]

    # Step 5 — collect unique sources for citation
    sources = []
    seen = set()
    for chunk in chunks:
        key = f"{chunk['source']} p.{chunk['page']}"
        if key not in seen:
            sources.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "similarity": chunk["similarity"]
            })
            seen.add(key)

    result = {
        "question": question,
        "answer": answer,
        "sources": sources
    }

    # Step 6 — print the result
    print(f"\nAnswer:\n{answer}")
    print(f"\nSources:")
    for s in sources:
        print(f"  - {s['source']}, page {s['page']}")

    return result


if __name__ == "__main__":
    # test with a few questions about your CV
    test_questions = [
        "What is the candidate's educational background?",
        "What technical skills does the candidate have?",
        "What work experience does the candidate have?"
    ]

    for question in test_questions:
        result = ask(question, verbose=False)
        print("\n" + "=" * 60 + "\n")
