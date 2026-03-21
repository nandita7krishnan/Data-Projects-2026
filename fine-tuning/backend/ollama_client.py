"""
LLM API client. All model calls go through this file.

Uses Groq API for fast Llama inference.
Set GROQ_API_KEY environment variable before running.
"""

import asyncio
import os
from typing import List, Dict

from dotenv import load_dotenv
import httpx

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# --- Configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.1-8b-instant"


async def generate_response(
    system_prompt: str,
    messages: List[Dict],
    character_name: str,
) -> str:
    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages.extend(messages)

    payload = {
        "model": MODEL_NAME,
        "messages": full_messages,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 256,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
            if response.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"Rate limited on {character_name}, waiting {wait}s...")
                await asyncio.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            await asyncio.sleep(3)
            return data["choices"][0]["message"]["content"].strip()

    raise Exception(f"Rate limited after 3 retries for {character_name}")
