from __future__ import annotations

"""
Validate and inspect the fine-tuning dataset.

Checks for balance, quality, duplicates, and shows sample entries.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --dataset data/training/boardroom_train.jsonl
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def get_character(example: dict) -> str:
    """Extract character name from system prompt."""
    system = example["messages"][0]["content"]
    # Look for "You are X from" or "You are X."
    if "You are " in system:
        after = system.split("You are ")[1]
        # Take up to " from " or ". "
        for delimiter in [" from ", ". "]:
            if delimiter in after:
                return after.split(delimiter)[0]
    return "Unknown"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def main():
    parser = argparse.ArgumentParser(description="Validate fine-tuning dataset")
    parser.add_argument("--dataset", type=str, default="data/training/boardroom_train.jsonl")
    parser.add_argument("--samples", type=int, default=3, help="Number of random samples to show per character")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    dataset_path = project_root / args.dataset

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Run build_dataset.py first.")
        return

    examples = load_jsonl(dataset_path)
    print(f"Dataset: {dataset_path}")
    print(f"Total examples: {len(examples)}\n")

    # --- Per-character stats ---
    char_examples: dict[str, list[dict]] = {}
    for ex in examples:
        char = get_character(ex)
        if char not in char_examples:
            char_examples[char] = []
        char_examples[char].append(ex)

    print("=" * 60)
    print("PER-CHARACTER BREAKDOWN")
    print("=" * 60)
    for char in sorted(char_examples.keys()):
        exs = char_examples[char]
        assistant_texts = [ex["messages"][-1]["content"] for ex in exs]
        token_counts = [estimate_tokens(t) for t in assistant_texts]
        char_lengths = [len(t) for t in assistant_texts]

        count = len(exs)
        avg_tokens = sum(token_counts) / count if count else 0
        min_tokens = min(token_counts) if count else 0
        max_tokens = max(token_counts) if count else 0

        flag = " ⚠️  LOW COUNT" if count < 30 else ""
        print(f"\n{char}: {count} examples{flag}")
        print(f"  Tokens (est.): avg={avg_tokens:.0f}, min={min_tokens}, max={max_tokens}")
        print(f"  Chars: avg={sum(char_lengths)/count:.0f}, min={min(char_lengths)}, max={max(char_lengths)}")

    # --- Duplicate check ---
    print("\n" + "=" * 60)
    print("DUPLICATE CHECK")
    print("=" * 60)
    assistant_texts = [ex["messages"][-1]["content"] for ex in examples]
    text_counts = Counter(assistant_texts)
    duplicates = {text: count for text, count in text_counts.items() if count > 1}
    if duplicates:
        print(f"Found {len(duplicates)} duplicate assistant responses:")
        for text, count in sorted(duplicates.items(), key=lambda x: -x[1])[:5]:
            print(f"  [{count}x] {text[:80]}...")
    else:
        print("No duplicate assistant responses found.")

    # --- Total sequence length ---
    print("\n" + "=" * 60)
    print("SEQUENCE LENGTH DISTRIBUTION")
    print("=" * 60)
    total_tokens = []
    for ex in examples:
        full_text = " ".join(m["content"] for m in ex["messages"])
        total_tokens.append(estimate_tokens(full_text))
    print(f"Total tokens per example (est.): avg={sum(total_tokens)/len(total_tokens):.0f}, "
          f"min={min(total_tokens)}, max={max(total_tokens)}")
    over_2048 = sum(1 for t in total_tokens if t > 2048)
    if over_2048:
        print(f"  ⚠️  {over_2048} examples may exceed 2048 token context window")
    else:
        print("  All examples fit within 2048 token context window")

    # --- Random samples ---
    print("\n" + "=" * 60)
    print("RANDOM SAMPLES")
    print("=" * 60)
    random.seed(42)
    for char in sorted(char_examples.keys()):
        exs = char_examples[char]
        sample_count = min(args.samples, len(exs))
        samples = random.sample(exs, sample_count)
        print(f"\n--- {char} ---")
        for i, ex in enumerate(samples):
            user_msg = ex["messages"][1]["content"]
            assistant_msg = ex["messages"][-1]["content"]
            print(f"  Sample {i+1}:")
            print(f"    User: {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}")
            print(f"    Assistant: {assistant_msg[:120]}{'...' if len(assistant_msg) > 120 else ''}")

    # --- Class balance warning ---
    print("\n" + "=" * 60)
    print("BALANCE ASSESSMENT")
    print("=" * 60)
    counts = {char: len(exs) for char, exs in char_examples.items()}
    if counts:
        max_count = max(counts.values())
        min_count = min(counts.values())
        ratio = max_count / min_count if min_count > 0 else float("inf")
        print(f"Max/min ratio: {ratio:.1f}x")
        if ratio > 5:
            print("⚠️  Significant class imbalance. Consider upsampling underrepresented characters "
                  "or using weighted loss during training.")
        elif ratio > 2:
            print("Moderate imbalance — acceptable but keep an eye on underrepresented characters' quality.")
        else:
            print("Good balance across characters.")


if __name__ == "__main__":
    main()
