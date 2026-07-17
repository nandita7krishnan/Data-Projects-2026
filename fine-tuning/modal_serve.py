"""
Modal deployment for the Boardroom fine-tuned model.

Setup (one-time):
    pip install modal
    modal setup          # login
    modal volume create boardroom-model
    modal volume put boardroom-model Llama-3.1-8B-Instruct.Q4_K_M.gguf /model.gguf

Deploy:
    modal deploy modal_serve.py

This exposes an OpenAI-compatible endpoint at:
    https://<your-modal-username>--boardroom-inference-serve.modal.run
"""

import modal

app = modal.App("boardroom-inference")

volume = modal.Volume.from_name("boardroom-model")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "llama-cpp-python==0.3.4",
        "fastapi==0.115.0",
        "uvicorn==0.30.6",
    )
)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/model": volume},
    timeout=600,
    scaledown_window=120,
)
@modal.asgi_app()
def serve():
    import os
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Dict, Optional
    from llama_cpp import Llama

    MODEL_PATH = "/model/model.gguf"

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Did you run: modal volume put boardroom-model <gguf> /model.gguf")

    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,   # offload all layers to GPU
        n_ctx=2048,
        verbose=False,
    )

    web_app = FastAPI()

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: Optional[str] = "boardroom"
        messages: List[Message]
        temperature: Optional[float] = 0.8
        top_p: Optional[float] = 0.9
        max_tokens: Optional[int] = 256

    @web_app.get("/health")
    def health():
        return {"status": "ok"}

    @web_app.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        result = llm.create_chat_completion(
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        return result

    return web_app
