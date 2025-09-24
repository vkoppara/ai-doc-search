import os
import httpx
from openai import OpenAI

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

    return response.choices[0].message.content.strip()

