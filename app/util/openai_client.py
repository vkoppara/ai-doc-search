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
            " You are an expert visual captioner. Describe the key content of the image "
            "then provide 5 short, comma-seperated tags. Keep it factual, no specualtion. "            
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
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Vision caption failed after retries: {last_err}")


