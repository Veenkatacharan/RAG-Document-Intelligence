import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# import your existing pipeline components
from rag_pipeline import ask
from database.vector_store import (
    save_chunks,
    get_chunk_count,
    clear_all_chunks
)
from ingestion.loader import load_all_pdfs
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_chunks

# initialise FastAPI app
app = FastAPI(
    title="RAG Document Intelligence API",
    description="Upload documents and ask questions about them using AI",
    version="1.0.0"
)

# allow requests from any origin
# this is needed so your Streamlit UI can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FOLDER = "data"


# ── Request / Response models ──────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the technical skills?",
                "top_k": 5
            }
        }


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]


class IngestResponse(BaseModel):
    message: str
    chunks_created: int


class HealthResponse(BaseModel):
    status: str
    chunks_in_database: int


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Simple health check endpoint.
    Returns server status and how many chunks are in the database.
    Every real API has a health check — interviewers notice this.
    """
    return HealthResponse(
        status="ok",
        chunks_in_database=get_chunk_count()
    )


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Main endpoint. Takes a question, retrieves relevant chunks,
    passes them to the LLM, and returns a cited answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if get_chunk_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please ingest documents first via POST /ingest"
        )

    result = ask(request.question)

    return AnswerResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"]
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents():
    """
    Triggers the full ingestion pipeline on everything in the data/ folder.
    Clears existing chunks first so you don't get duplicates on re-ingestion.
    """
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        raise HTTPException(
            status_code=400,
            detail=f"No PDF files found in {DATA_FOLDER}/ folder"
        )

    # clear existing chunks to avoid duplicates
    clear_all_chunks()

    # run the full pipeline
    pages = load_all_pdfs(DATA_FOLDER)
    chunks = chunk_pages(pages)
    chunks = embed_chunks(chunks)
    save_chunks(chunks)

    return IngestResponse(
        message=f"Successfully ingested {len(pdf_files)} PDF(s)",
        chunks_created=len(chunks)
    )


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload and saves it to the data/ folder.
    After uploading, call POST /ingest to process it.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    file_path = os.path.join(DATA_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": f"Successfully uploaded {file.filename}",
        "filename": file.filename,
        "next_step": "Call POST /ingest to process this document"
    }


@app.get("/chunks/count")
def chunk_count():
    """
    Returns how many chunks are currently in the database.
    Useful for verifying ingestion worked correctly.
    """
    count = get_chunk_count()
    return {"chunks_in_database": count}
