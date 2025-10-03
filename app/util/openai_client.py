import os
import httpx
from openai import OpenAI
from typing import Optional
import base64
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

_http_client = httpx.Client(timeout=60.0)
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE, http_client=_http_client)

def get_chat_content(messages: list[dict], model: str = "gpt-4", max_retries: int = 3, temperature: float = 0.7, max_tokens: int = 65355) -> str:
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    raise RuntimeError("Failed to get chat content after retries.")

    #return response.choices[0].message.content.strip()

def get_image_caption(image_bytes: bytes, prompt: Optional[str]= None, model: str = "sonar", max_retries: int=3) -> str:
    if prompt is None:
        prompt = (
            """
            You are an AI assistant that extracts documents for Retrieval-Augmented Generation (RAG) storage.
            Your responsibilities:
            1. Text Extraction
            Extract all visible text verbatim from the document or page.
            Preserve structure (headings, subheadings, bullet points, numbering, paragraphs).
            Do not paraphrase or summarize.
            2. Tables
            Convert tables into a textual table-like structure (rows and columns in plain text, CSV-style or Markdown table if possible).
            3. Graphs/Charts/Images
            For each non-textual element, insert a clear description inline where it occurs.
            Example for graphs:
            “[Graph: Line chart showing India’s inflation rate (2018–2023). Trend indicates steady decline from 6% in 2018 to 4% in 2020, followed by sharp increase to 7% in 2022, then slight dip to 6% in 2023].”
            Example for logos/figures:
            “[Figure: RBI emblem/logo displayed at top of the page].”
            4. Ordering
            Maintain the original flow and order of the document.
            Clearly separate sections with headings if present.
            5. Output Format
            Output should be plain text (or Markdown if tables/structure is useful).
            No interpretation beyond describing visual elements; the goal is to preserve raw content for later retrieval.
        """            
        )
    
    b64 = base64.b64encode(image_bytes).decode("utf-8");
    messages = [
        {
            "role": "user",
            "content": [
                {"type":"text", "text": prompt},
                {"type":"image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        }
    ]
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = openai_client.chat.completions.create(
                model = model,
                messages = messages,
                temperature = 0.0                
            )
            return (resp.choices[0].message.content or "").strip()
            #return "sample";
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Vision caption failed after retries: {last_err}")


