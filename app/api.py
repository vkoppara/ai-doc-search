from pydantic import BaseModel
from fastapi import FastAPI
from .langgraph_agent import  agent_infer_langgraph_stream
from .util.embedding_client import get_embedding, get_embeddings, get_message
from fastapi import Query
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from .utils import (
    extract_text_from_docx, 
    extract_text_from_pdf,
    extract_rich_pdf_segments,    
    chunk_text,
    chunk_segments,
)
from .db import get_connection
import tempfile, os, asyncio
from app.util.chunk_summary import extractive_summary

router = APIRouter()

@router.post("/upload")
async def upload_file(
    text_extract_only: bool = Query(False),
    file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.txt', '.docx', ".pdf")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .txt, .docx and .pdf are allowed.")
    
    suffix = os.path.splitext(file.filename)[1] or ".tmp"
    tmp_path = None
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await file.read())
        tmp_path = tmp.name
        tmp.close()  # Ensure file is closed before further use

        lowered = file.filename.lower()
        if lowered.endswith('.txt'):
            with open(tmp_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif lowered.endswith('.docx'):
            text = extract_text_from_docx(tmp_path)
        elif lowered.endswith(".pdf"):
            segments = extract_rich_pdf_segments(tmp_path, text_extract_only)
            if segments:
                chunks = chunk_segments(segments)
            else:
                text = extract_text_from_pdf(tmp_path)
                chunks = chunk_text(text)
        else:
            chunks = []
        if lowered.endswith(('.txt','.docx')):                            
            chunks = chunk_text(text) 
        vectors = await asyncio.gather(*[get_embedding(chunk) for chunk in chunks])
        summary_text = extractive_summary(chunks, vectors, num_summary_chunks=5)
        summary_embedding =await get_embedding(summary_text)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT into documents (filename, summary_text, summary_embedding) VALUES (%s, %s, %s) RETURNING id", (file.filename, summary_text, summary_embedding ))
                doc_id = cur.fetchone()[0]
                
                for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
                    cur.execute("INSERT INTO embeddings (document_id, chunk, vector_embedding, chunk_index) VALUES (%s, %s, %s, %s)", (doc_id, chunk, vector, idx))
            conn.commit()        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except PermissionError:
                pass  # File is still in use, skip deletion

    return {"status": "success", "message": f"Uploaded and processed {file.filename}", "document_id": doc_id, "num_chunks": len(chunks)}


@router.post("/agent_infer")
async def agent_infer(document_id: int, query: str, top_k: int = Query(3, ge=1, le=10)):
    conn = get_connection()
    return StreamingResponse(agent_infer_langgraph_stream(conn, document_id, query, top_k), media_type="application/json")

@router.get("/documents")
async def list_documents():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT id, filename FROM documents ORDER BY id DESC")
        docs = cur.fetchall()
    conn.close()
    return [{"id": doc[0], "filename": doc[1]} for doc in docs]

class ChatHistoryRequest(BaseModel):
    message: str
    code: str
    timestamp: str
    email_id: str

@router.post("/chat_history")
async def chat_history(request: ChatHistoryRequest):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_history (code, message, timestamp, email_id) VALUES (%s, %s, %s, %s)
                on conflict (code) do update set message = EXCLUDED.message, timestamp = EXCLUDED.timestamp, email_id = EXCLUDED.email_id
                """,
            (request.code, request.message, request.timestamp, request.email_id)
        )
        conn.commit()
    return {"status": "success", "message": "Chat history saved."}

@router.get("/chat_history")
async def get_chat_history(email_id: str = None, top_k: int = Query(10, ge=1, le=100)):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT code, message, timestamp, email_id FROM chat_history WHERE email_id = %s ORDER BY timestamp DESC LIMIT %s", (email_id, top_k))
            rows = cur.fetchall()
            return [
                {"code": row[0], "message": row[1], "timestamp": row[2], "email_id": row[3]}
                for row in rows
            ]