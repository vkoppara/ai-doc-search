from app.util.openai_client import get_chat_content
from app.util.openai_client import openai_client
import asyncio
from openai import OpenAI
import httpx
import os

EMBEDDING_MODEL = "text-embedding-3-small"

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_OPENAI_API_KEY = os.getenv("EMBEDDING_OPENAI_API_KEY","")
EMBEDDING_OPENAI_API_BASE = os.getenv("EMBEDDING_OPENAI_API_BASE", "https://api.openai.com/v1")

_http_client = httpx.Client(timeout=60.0)
openai_client = OpenAI(api_key=EMBEDDING_OPENAI_API_KEY, base_url=EMBEDDING_OPENAI_API_BASE, http_client=_http_client)

async def get_embedding(text: str) -> list[float]:
    loop = asyncio.get_event_loop()
    def sync_embed():
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,  
            input=[text]
        )
        return response.data[0].embedding
    return await loop.run_in_executor(None, sync_embed)

def get_embeddings(texts: list[str]) -> list[list[float]]:
    if not isinstance(texts, list):
        raise ValueError("Input must be a list of strings.")

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [data.embedding for data in response.data]

def get_message(query: str, context: str) -> list[dict]:
    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question, response is to be understandable by a general audience and avoid technical jargon."
    )
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User question: {query}\nContext: {context}"}
    ]
    return get_chat_content(
        messages=message, model="gpt-4", max_retries=3, temperature=0.7, max_tokens=65355
    )