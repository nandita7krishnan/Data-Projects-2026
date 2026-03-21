# Boardroom: Fine-Tuned Character Debate App

Pitch a business idea and watch 6 fictional characters debate it. Built as a learning project to understand the full LLM fine-tuning pipeline — from data collection to QLoRA training to deployment.

## Characters

| Character | Source | Personality |
|-----------|--------|-------------|
| Harvey Specter | Suits | Sharp, deal-obsessed, dismissive |
| Bob the Builder | Bob the Builder | Optimistic, construction metaphors |
| Logan Roy | Succession | Brutal, terse, power-focused |
| Ross Geller | Friends | Over-analytical, academic tangents |
| Miranda Priestly | The Devil Wears Prada | Withering contempt, fashion lens |
| Noddy | Toyland | Confidently nonsensical |

## How It Works

1. **Data pipeline** (`scripts/`) — Parsed TV scripts, quote collections, and character bios. Generated synthetic dialogue for characters with limited data (Bob, Noddy, Logan) using a base LLM.
2. **Fine-tuning** (`notebooks/finetune_unsloth.ipynb`) — QLoRA fine-tuning of Llama 3.1 8B on Google Colab (free T4). Single model conditioned via system prompts to switch between characters.
3. **Backend** (`backend/`) — FastAPI app that orchestrates a 2-phase debate: opening reactions, then a final vote with yes/no/abstain tally.
4. **Frontend** (`frontend/`) — React + Vite UI with dark theme, character avatars, and vote visualization.

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- A [Groq API key](https://console.groq.com/) (free tier works) for fast inference

### Backend
```bash
cd fine-tuning
pip install -r requirements.txt
cp backend/.env.example backend/.env
# Edit backend/.env and add your GROQ_API_KEY
uvicorn backend.main:app --reload --port 8000
```

### Frontend
```bash
cd fine-tuning/frontend
npm install
npm run dev
```

The app runs at `http://localhost:5173` with the API at `http://localhost:8000`.

### Fine-Tuning (Optional)
To train your own model, open `notebooks/finetune_unsloth.ipynb` in Google Colab with a T4 GPU. Upload `data/training/boardroom_train.jsonl` and follow the notebook. It covers:
- 4-bit quantization and why it works
- LoRA adapter configuration (rank, alpha, target modules)
- SFTTrainer hyperparameters
- Base vs fine-tuned evaluation
- GGUF export for local deployment via Ollama

## Project Structure
```
fine-tuning/
├── data/
│   ├── raw/                    # TV scripts, quote collections, character bios
│   ├── processed/              # Parsed dialogue + character profiles
│   └── training/               # Chat-format JSONL for fine-tuning
├── notebooks/
│   └── finetune_unsloth.ipynb  # Colab notebook — the core learning artifact
├── scripts/
│   ├── parse_transcripts.py    # Extract dialogue from scripts/quotes/bios
│   ├── generate_synthetic.py   # Generate in-character dialogue for low-data characters
│   ├── build_dataset.py        # Convert dialogue → Llama 3.1 chat JSONL
│   └── validate_dataset.py     # Dataset stats, balance checks, sample preview
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── characters.py           # Character definitions & system prompts
│   ├── ollama_client.py        # Groq API client (originally Ollama, swapped for speed)
│   └── debate.py               # Debate orchestration
├── frontend/                   # React + Vite UI
├── Modelfile                   # Ollama Modelfile for local GGUF deployment
└── requirements.txt
```

## Notes
- The inference backend uses **Groq** (cloud) instead of local Ollama because running an 8B model on 8GB RAM was too slow for interactive use. The Groq API is free and runs Llama 3.1 8B at ~500 tokens/sec.
- The file `ollama_client.py` retains its original name since it was the initial swap point — it now calls Groq's OpenAI-compatible API.
- The GGUF model file is not included in the repo (4.6GB). To use the fine-tuned model locally via Ollama, train it yourself using the Colab notebook and import with the provided `Modelfile`.
