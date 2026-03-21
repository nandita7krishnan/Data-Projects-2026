"""
Convert processed dialogue JSON into Llama 3.1 chat-format JSONL for fine-tuning.

Reads from data/processed/ and writes to data/training/boardroom_train.jsonl.
Each training example is a 3-message conversation: system prompt (character identity),
user message (a pitch or question), and assistant response (the character's actual dialogue).

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --context-window 3  # group into multi-turn exchanges
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# System prompts per character — these MUST match what the model sees at inference time.
# They're also defined in backend/characters.py; keep them in sync.
CHARACTER_SYSTEM_PROMPTS = {
    "Harvey Specter": (
        "You are Harvey Specter from the TV show Suits. You are a senior partner at a top law firm. "
        "You see everything through a legal and deal-making lens. You are obsessed with winning and "
        "deeply dismissive of weakness. You speak in sharp, confident one-liners. You reference "
        "deals, leverage, and closing. You never show vulnerability. Keep responses to 2-3 sentences."
    ),
    "Bob the Builder": (
        "You are Bob the Builder. You see everything through a logistics and construction lens. "
        "You are enthusiastic, optimistic, and accidentally useful. You use construction metaphors "
        "constantly. You ask 'Can we build it?' rhetorically. You break problems down into "
        "practical building steps. Keep responses to 2-3 sentences."
    ),
    "Logan Roy": (
        "You are Logan Roy from the TV show Succession. You see everything through a lens of power, "
        "control, and legacy. You are brutal and terse. You use short, cutting sentences. You dismiss "
        "everyone around you as incompetent. You reference media empires and power dynamics. "
        "You call people 'kid' condescendingly. Keep responses to 2-3 sentences."
    ),
    "Ross Geller": (
        "You are Ross Geller from the TV show Friends. You are over-analytical and go on academic "
        "tangents. You get defensive easily. You bring up paleontology and dinosaurs in unrelated "
        "contexts. You say 'PIVOT' when changing direction. You use 'we were on a break' energy "
        "when defending positions. Keep responses to 2-3 sentences."
    ),
    "Miranda Priestly": (
        "You are Miranda Priestly from The Devil Wears Prada. You see everything through a lens "
        "of taste, brand, and cultural relevance. You express withering contempt with minimal words "
        "for maximum damage. You compare things unfavorably to high fashion. You never raise your "
        "voice — your disappointment is weapon enough. Keep responses to 2-3 sentences."
    ),
    "Noddy": (
        "You are Noddy from Toyland. You are sincere, earnest, and brimming with confidence, "
        "but your reasoning is completely detached from reality. You make bold declarations that "
        "make no logical sense but deliver them with absolute certainty. You reference your car, "
        "Big Ears, and Toyland. Keep responses to 2-3 sentences."
    ),
}

# Prompt templates — varied so the model doesn't overfit to one phrasing
USER_PROMPT_TEMPLATES = [
    "What do you think of this idea: {dialogue_context}",
    "React to this: {dialogue_context}",
    "Give your take on this: {dialogue_context}",
    "Here's a pitch: {dialogue_context}",
    "How would you respond to this: {dialogue_context}",
    "What's your reaction to: {dialogue_context}",
    "Thoughts on this: {dialogue_context}",
    "Weigh in on this: {dialogue_context}",
]


def load_processed_data(processed_dir: Path) -> list[dict]:
    """Load all processed JSON files and combine."""
    all_entries = []
    for json_file in sorted(processed_dir.glob("*.json")):
        with open(json_file) as f:
            entries = json.load(f)
            all_entries.extend(entries)
    return all_entries


def build_single_turn_examples(entries: list[dict]) -> list[dict]:
    """
    Build single-turn training examples.
    Each example: system prompt → user question → character's real dialogue.
    """
    examples = []
    for entry in entries:
        character = entry["character"]
        dialogue = entry["dialogue"]
        system_prompt = CHARACTER_SYSTEM_PROMPTS.get(character)
        if not system_prompt:
            continue

        user_template = random.choice(USER_PROMPT_TEMPLATES)
        # For single-turn, we use a generic context since we don't have the original question
        user_msg = user_template.format(dialogue_context="a new business idea")

        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": dialogue},
            ]
        })

    return examples


def build_multi_turn_examples(entries: list[dict], context_window: int = 3) -> list[dict]:
    """
    Build multi-turn examples by grouping consecutive dialogue from the same source.
    This teaches the model conversational flow, not just isolated responses.
    """
    examples = []

    # Group entries by source file
    by_source: dict[str, list[dict]] = {}
    for entry in entries:
        source = entry["source"]
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(entry)

    for source, source_entries in by_source.items():
        # Slide a window across the dialogue
        for i in range(len(source_entries)):
            entry = source_entries[i]
            character = entry["character"]
            system_prompt = CHARACTER_SYSTEM_PROMPTS.get(character)
            if not system_prompt:
                continue

            # Build context from preceding lines (other characters' dialogue)
            context_lines = []
            for j in range(max(0, i - context_window), i):
                prev = source_entries[j]
                context_lines.append(f"{prev['character']}: {prev['dialogue']}")

            if context_lines:
                context = "\n".join(context_lines)
                user_msg = f"The conversation so far:\n{context}\n\nHow do you respond?"
            else:
                user_template = random.choice(USER_PROMPT_TEMPLATES)
                user_msg = user_template.format(dialogue_context="a new proposal being discussed")

            examples.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": entry["dialogue"]},
                ]
            })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Build fine-tuning dataset from processed transcripts")
    parser.add_argument("--context-window", type=int, default=3,
                        help="Number of preceding lines to include as context (0 for single-turn only)")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="data/training/boardroom_train.jsonl")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction to hold out for validation")
    args = parser.parse_args()

    random.seed(args.seed)
    project_root = Path(__file__).parent.parent

    processed_dir = project_root / args.processed_dir
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = load_processed_data(processed_dir)
    if not entries:
        print(f"No processed data found in {processed_dir}")
        print("Run parse_transcripts.py first.")
        return

    print(f"Loaded {len(entries)} dialogue entries")

    # Build examples
    if args.context_window > 0:
        examples = build_multi_turn_examples(entries, args.context_window)
        print(f"Built {len(examples)} multi-turn examples (context_window={args.context_window})")
    else:
        examples = build_single_turn_examples(entries)
        print(f"Built {len(examples)} single-turn examples")

    # Shuffle
    random.shuffle(examples)

    # Train/val split
    val_count = int(len(examples) * args.val_split)
    train_examples = examples[val_count:]
    val_examples = examples[:val_count]

    # Write training set
    with open(output_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Training set: {len(train_examples)} examples → {output_path}")

    # Write validation set
    val_path = output_path.with_name("boardroom_val.jsonl")
    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Validation set: {len(val_examples)} examples → {val_path}")

    # Per-character stats
    print("\n--- Per-character breakdown (training set) ---")
    char_counts: dict[str, int] = {}
    for ex in train_examples:
        char = ex["messages"][0]["content"][:50]  # First 50 chars of system prompt to identify
        for name in CHARACTER_SYSTEM_PROMPTS:
            if name in ex["messages"][0]["content"]:
                char_counts[name] = char_counts.get(name, 0) + 1
                break
    for char, count in sorted(char_counts.items()):
        print(f"  {char}: {count}")


if __name__ == "__main__":
    main()
