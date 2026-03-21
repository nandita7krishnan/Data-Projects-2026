"""
FastAPI backend for the Boardroom debate app.

Usage:
    cd fine-tuning
    uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .characters import CHARACTERS, CHARACTER_ORDER
from .debate import run_debate

app = FastAPI(title="Boardroom API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PitchRequest(BaseModel):
    pitch: str


class CharacterInfo(BaseModel):
    key: str
    name: str
    initials: str
    color: str


@app.post("/pitch")
async def submit_pitch(request: PitchRequest):
    """Submit a pitch and get a full debate response."""
    if not request.pitch.strip():
        raise HTTPException(status_code=400, detail="Pitch cannot be empty")

    result = await run_debate(request.pitch.strip())
    return result


@app.get("/characters")
async def get_characters():
    """Return the list of characters in debate order."""
    return [
        CharacterInfo(
            key=key,
            name=CHARACTERS[key]["name"],
            initials=CHARACTERS[key]["initials"],
            color=CHARACTERS[key]["color"],
        )
        for key in CHARACTER_ORDER
    ]


@app.get("/health")
async def health():
    return {"status": "ok"}
