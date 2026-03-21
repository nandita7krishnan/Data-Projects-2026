# Boardroom: Fine-Tuned Character Debate App

## Context
A learning project focused on LLM fine-tuning. Users pitch ideas and 6 fictional characters (Harvey Specter, Bob the Builder, Logan Roy, Ross Geller, Miranda Priestly, Noddy) debate them. The core learning goal is the fine-tuning pipeline — the app is the vehicle to exercise and showcase the fine-tuned model.

## Project Structure
```
fine-tuning/
├── data/
│   ├── raw/                    # User drops raw transcripts here
│   ├── processed/              # Cleaned, attributed dialogue
│   └── training/               # Final JSONL for fine-tuning
├── notebooks/
│   └── finetune_unsloth.ipynb  # Colab notebook for QLoRA training
├── scripts/
│   ├── parse_transcripts.py    # Flexible parser for scripts, quotes, and profiles
│   ├── generate_synthetic.py   # Generate dialogue for low-data characters via Ollama
│   ├── build_dataset.py        # Converts parsed dialogue → chat JSONL
│   └── validate_dataset.py     # Stats, balance checks, sample preview
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── characters.py           # Character definitions & system prompts
│   ├── ollama_client.py        # All Ollama calls (swap point for fine-tuned model)
│   └── debate.py               # Debate orchestration (3 phases)
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── PitchInput.jsx
│   │   │   ├── DebateThread.jsx
│   │   │   ├── CharacterMessage.jsx
│   │   │   └── VoteTally.jsx
│   │   └── styles/
│   │       └── app.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── Modelfile                   # Ollama Modelfile for importing fine-tuned GGUF
└── README.md
```

---

## Phase 1: Data Pipeline (the most important part)

### Raw Data Reality
The raw data in `data/raw/` is a mix of formats:

| File | Type | Character | Dialogue Lines |
|------|------|-----------|---------------|
| suits_101_harvey.rtf | Full TV script | Multi-character | ~199 (mostly Harvey) |
| Suits_02.pdf | Full TV script | Multi-character | ~217 (mostly Harvey) |
| Suits_1x07.pdf | Episode script | Multi-character | ~104 (mostly Harvey) |
| miranada_quotes.rtf | Scene-by-scene dialogue | Miranda Priestly | ~75 |
| harvey spectar quotes.rtf | Quote collection | Harvey Specter | ~68 |
| ross_quotes.rtfd | Numbered quotes + conversations | Ross Geller | ~56 |
| ross_wiki.rtf | Character bio + speech style | Ross Geller | ~1 |
| logan_roy.rtf | Character bio + a few lines | Logan Roy | ~2 |
| about_miranada.rtf | Character bio | Miranda Priestly | ~1 |
| bob_the_buider.rtf | Character bio (no dialogue) | Bob the Builder | 0 |
| noddy_wiki.rtf | Wikipedia article (no dialogue) | Noddy | 0 |

**Key takeaway**: Harvey has plenty of data, Miranda is okay, Ross has enough from quotes. Logan Roy, Bob, and Noddy need synthetic data generation.

### Step 1.1 — Flexible Parser (`scripts/parse_transcripts.py`)
The parser auto-classifies each file and handles three content types:

- **Scripts** (Suits RTFs/PDFs, Miranda quotes): Extract `CHARACTER: dialogue` lines with regex. Handles wiki-formatted scripts (with `[[links]]`) and plain screenplays.
- **Quote collections** (Harvey quotes): Extract standalone quoted strings and attribute them to the character via filename hints.
- **Character profiles** (Bob, Noddy, Logan, Miranda bios): Extract personality traits, speech style, and iconic lines → saved to `data/processed/profiles/` for synthetic data generation and system prompt enrichment.

Also handles RTF (via macOS `textutil` or `striprtf`) and PDF (via `pdfplumber`) extraction.

Output:
```
data/processed/*.json           — extracted dialogue lines
data/processed/profiles/*.json  — character profiles
```

### Step 1.1b — Synthetic Data Generation (`scripts/generate_synthetic.py`)
For characters with insufficient real dialogue (Bob, Noddy, Ross, Logan), generates in-character responses using a base LLM via Ollama:

- Reads character profiles from `data/processed/profiles/`
- Falls back to system prompts from `characters.py` for characters without profiles
- Generates responses to ~50 diverse business/pitch scenarios
- Outputs to `data/processed/synthetic_*.json` (same format as real dialogue)
- Target: at least 50 lines per character

```bash
python scripts/generate_synthetic.py --dry-run          # preview what needs generating
python scripts/generate_synthetic.py --model llama3.1:8b # generate with Ollama
python scripts/generate_synthetic.py --character "Ross Geller" --count 60  # specific character
```

### Step 1.2 — Dataset Builder (`scripts/build_dataset.py`)
Converts processed dialogue (both real and synthetic) into Llama 3.1 chat-format JSONL for fine-tuning.

**Key design decisions for learning:**
- **Single model, character-conditioned**: Each training example includes a system prompt identifying the character, so the model learns to switch voices
- **Chat template format**: Uses Llama 3.1's native `<|begin_of_text|><|start_header_id|>system<|end_header_id|>` format
- **Context-aware examples**: Where possible, group dialogue into multi-turn exchanges (not just isolated lines) so the model learns conversational patterns

Each JSONL row:
```json
{
  "messages": [
    {"role": "system", "content": "You are Harvey Specter from Suits. You see everything through a legal/deal lens..."},
    {"role": "user", "content": "What do you think of this idea: [pitch or previous dialogue]"},
    {"role": "assistant", "content": "actual Harvey dialogue from transcript"}
  ]
}
```

### Step 1.3 — Dataset Validator (`scripts/validate_dataset.py`)
- Print per-character example counts (flag if any character < 30)
- Show average/min/max token lengths per character
- Print 3 random samples for manual inspection
- Check for duplicate entries
- Warn about class imbalance (some characters will have more data than others — that's fine, just be aware)

---

## Phase 2: Fine-Tuning on Colab (`notebooks/finetune_unsloth.ipynb`)

This is the core learning artifact. The notebook is structured as a tutorial with clear sections:

### 2.1 — Setup & Installation
```python
!pip install unsloth
# Unsloth handles: bitsandbytes, peft, trl, transformers
```

### 2.2 — Load Base Model with Unsloth
- Load `unsloth/llama-3.1-8b-instruct-bnb-4bit` (pre-quantized for Colab)
- Explain what 4-bit quantization means and why it lets 8B params fit in ~5GB VRAM

### 2.3 — Configure LoRA
- `r=16, lora_alpha=32, lora_dropout=0.05`
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Explain each parameter: what rank means, why alpha = 2*r is a good starting point, which layers we're adapting and why

### 2.4 — Load & Inspect Dataset
- Upload the JSONL from Phase 1
- Load with HuggingFace `datasets`
- Apply Llama 3.1 chat template tokenization
- Show token length distribution, verify formatting looks correct

### 2.5 — Training with SFTTrainer
```python
SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=2048,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # effective batch size = 8
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=10,
    logging_steps=5,
    save_steps=50,
)
```
- Explain each hyperparameter and what tuning it does
- ~800 examples, 3 epochs ≈ 20-40 min on Colab T4

### 2.6 — Evaluate Before Export
- Run inference on test prompts for each character (same prompt, different system prompts)
- Compare base model vs fine-tuned outputs side by side
- Qualitative check: does Harvey sound like Harvey? Does Noddy sound like Noddy?

### 2.7 — Export to GGUF for Ollama
```python
model.save_pretrained_gguf("boardroom-model", tokenizer, quantization_method="q4_k_m")
```
- Explain GGUF format and why q4_k_m is a good quality/size tradeoff
- Download the `.gguf` file

### 2.8 — Experimentation Section (stretch goals documented in notebook)
- Hyperparameter sweep: try r=8 vs r=32, different learning rates
- Ablation: train on 3 characters only, see if quality improves with more data per character
- Compare 1 epoch vs 3 vs 5

---

## Phase 3: Local Deployment via Ollama

### 3.1 — Modelfile
```
FROM ./boardroom-model-q4_k_m.gguf
TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"
```

### 3.2 — Import & Test
```bash
ollama create boardroom -f Modelfile
ollama run boardroom
```

---

## Phase 4: Backend (FastAPI)

### `characters.py`
- Dict of 6 characters, each with: name, system_prompt, color, initials
- System prompts are the same ones used in training data so the model recognizes them

### `ollama_client.py`
- Single `generate_response(system_prompt, messages, character_name)` function
- Calls Ollama HTTP API (`POST http://localhost:11434/api/chat`)
- Model name is a config variable at the top (easy swap between `llama3.1:8b` and `boardroom`)
- Returns parsed response text

### `debate.py`
- Orchestrates the 3 phases, builds conversation history incrementally
- Phase 1 (Opening): 6 calls, each character reacts to pitch
- Phase 2 (Cross-debate): 2 rounds x 6 characters, full history as context
- Phase 3 (Vote): 6 calls, each returns verdict + vote, parsed into tally

### `main.py`
- `POST /pitch` — accepts `{"pitch": "..."}`, returns full debate JSON
- CORS middleware for React dev server

---

## Phase 5: Frontend (React + Vite)

- Dark theme, no component libraries
- `PitchInput`: textarea + submit button
- `DebateThread`: renders phases with headers ("Opening Reactions", "Cross-Debate Round 1", etc.)
- `CharacterMessage`: avatar circle (initials + character color), name, message text
- `VoteTally`: bar chart showing yes/no/abstain counts
- Loading state while debate generates (this will take ~60-90s for all calls)

---

## Build Order
1. **Data pipeline scripts** (parse, generate synthetic, build, validate) — `scripts/`
2. **Colab notebook** — the main learning deliverable
3. **Ollama import + Modelfile** — get model running locally
4. **Backend** — FastAPI wired to Ollama
5. **Frontend** — React UI
6. **Integration test** — end-to-end pitch → debate flow

## Verification
- Run `parse_transcripts.py` and confirm all files are classified correctly
- Run `generate_synthetic.py` for characters with < 50 lines
- Run `validate_dataset.py` and confirm balanced, well-formatted training data
- In Colab, compare base vs fine-tuned outputs for each character
- After Ollama import: `ollama run boardroom "You are Harvey Specter..." "What do you think of selling ice to penguins?"`
- Start backend + frontend, submit a pitch, verify all 3 phases render with correct character attribution
- Check that swapping `MODEL_NAME` in `ollama_client.py` between base and fine-tuned changes response quality
