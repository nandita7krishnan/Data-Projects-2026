"""
LLM API client. All model calls go through this file.

Uses the fine-tuned boardroom model served via Modal.
Set MODAL_ENDPOINT in backend/.env, e.g.:
    MODAL_ENDPOINT=https://<your-username>--boardroom-inference-serve.modal.run
"""

import os
from typing import List, Dict

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

MODAL_ENDPOINT = os.environ.get("MODAL_ENDPOINT", "").rstrip("/")
MODEL_NAME = "boardroom"

# Backwards compat — main.py imports this to gate requests.
GROQ_API_KEY = MODAL_ENDPOINT or ""


async def generate_response(
    system_prompt: str,
    messages: List[Dict],
    character_name: str,
) -> str:
    if not MODAL_ENDPOINT:
        raise RuntimeError("MODAL_ENDPOINT is not set in backend/.env")

    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages.extend(messages)

    payload = {
        "model": MODEL_NAME,
        "messages": full_messages,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 256,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{MODAL_ENDPOINT}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
