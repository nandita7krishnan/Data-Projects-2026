"""
Generate synthetic dialogue for characters with insufficient real data.

Reads character profiles from data/processed/profiles/ and uses a base LLM
(via Ollama) to generate in-character dialogue that matches the described
speech patterns and personality.

The generated dialogue goes into data/processed/ alongside real extracted
dialogue, so build_dataset.py treats them identically.

Usage:
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --character "Logan Roy" --count 60
    python scripts/generate_synthetic.py --model llama3.1:8b --all
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx is required: pip install httpx")
    sys.exit(1)


# Characters and the minimum dialogue count we want for each
TARGET_COUNT = 50

# Diverse prompts/scenarios to generate dialogue about — avoids repetitive outputs
SCENARIOS = [
    "Someone pitches a startup idea for delivering groceries by drone.",
    "A colleague suggests cutting the budget by 40% to save the company.",
    "Someone proposes replacing all human workers with AI.",
    "A team member wants to pivot the entire business to a new market.",
    "Someone suggests a risky merger with a much larger competitor.",
    "A new hire questions the company's ethics and wants to go public.",
    "Someone pitches opening a restaurant that serves only breakfast food.",
    "A partner proposes a hostile takeover of a rival firm.",
    "Someone wants to launch a children's educational app.",
    "A colleague suggests working four-day weeks to improve morale.",
    "Someone pitches a plan to expand internationally into Asia.",
    "A team member wants to rebrand the company entirely.",
    "Someone proposes a charity initiative that would cost millions.",
    "A colleague says the company needs to fire the bottom 10% of performers.",
    "Someone pitches a subscription box service for luxury goods.",
    "A new employee challenges the way things have always been done.",
    "Someone suggests investing everything in cryptocurrency.",
    "A partner wants to settle a lawsuit instead of fighting it in court.",
    "Someone proposes building affordable housing in the city.",
    "A colleague suggests hiring only people without college degrees.",
    "Someone pitches a social media platform for pets.",
    "A team member wants to move the entire company to a remote-first model.",
    "Someone suggests selling the company while it's still profitable.",
    "A colleague proposes a mandatory mentorship program.",
    "Someone pitches a plan to dominate the streaming market.",
    "A junior employee says the CEO's strategy is wrong.",
    "Someone wants to build a theme park based on the company's brand.",
    "A partner suggests we need to be more aggressive with competitors.",
    "Someone pitches opening a chain of co-working spaces.",
    "A colleague wants to give every employee equity in the company.",
    "Someone suggests partnering with a controversial public figure.",
    "A team member proposes we go carbon-neutral within a year.",
    "Someone pitches a dating app that matches people by their food preferences.",
    "A colleague says we should sue our biggest competitor for patent infringement.",
    "Someone wants to create a reality TV show about the company.",
    "A partner proposes acquiring a failing newspaper.",
    "Someone pitches an idea for self-driving boats.",
    "A colleague suggests we stop advertising entirely and rely on word of mouth.",
    "Someone wants to open a school that teaches only practical skills.",
    "A team member pitches a plan to corner the renewable energy market.",
    "Someone suggests the company needs a mascot.",
    "A colleague wants to hold the annual meeting on a cruise ship.",
    "Someone pitches a VR experience that lets you live as a historical figure.",
    "A partner says we need to cut ties with our biggest client.",
    "Someone proposes a loyalty program that gives customers real gold.",
    "A colleague suggests we should hire a celebrity as our spokesperson.",
    "Someone pitches opening a luxury hotel in Antarctica.",
    "A team member wants to create an in-house podcast.",
    "Someone suggests we should start manufacturing our own products.",
    "A colleague proposes a joint venture with a government agency.",
]

# System prompt used to instruct the base model to generate in-character dialogue
GENERATION_SYSTEM_PROMPT = """You are a creative writer generating training data for a fine-tuned language model.

Your task: Write a short response (2-4 sentences) AS the character described below.
The response should sound EXACTLY like how this character would naturally speak.
Do NOT break character. Do NOT add narration, stage directions, or quotation marks.
Just write the character's words directly.

CHARACTER PROFILE:
{profile_text}

IMPORTANT STYLE NOTES:
- Match the character's speech patterns precisely
- Use their typical vocabulary, cadence, and tone
- Keep it 2-4 sentences — punchy and in-character
- Don't be generic — make it distinctly THIS character"""


def load_profiles(profiles_dir: Path) -> dict[str, dict]:
    """Load all character profiles."""
    profiles = {}
    for f in profiles_dir.glob("*.json"):
        with open(f) as fp:
            profile = json.load(fp)
            profiles[profile["character"]] = profile
    return profiles


def load_existing_counts(processed_dir: Path) -> dict[str, int]:
    """Count how many dialogue lines we already have per character."""
    counts: dict[str, int] = {}
    for f in processed_dir.glob("*.json"):
        if f.parent.name == "profiles":
            continue
        with open(f) as fp:
            entries = json.load(fp)
            for entry in entries:
                char = entry.get("character", "Unknown")
                counts[char] = counts.get(char, 0) + 1
    return counts


def build_profile_text(profile: dict) -> str:
    """Turn a profile dict into a readable text block for the generation prompt."""
    parts = [f"Character: {profile['character']}"]

    if profile.get("personality_traits"):
        parts.append("\nPersonality:")
        for trait in profile["personality_traits"][:8]:
            parts.append(f"  - {trait}")

    if profile.get("speech_style"):
        parts.append(f"\nSpeech style:\n{profile['speech_style'][:500]}")

    if profile.get("iconic_lines"):
        parts.append("\nExample lines:")
        for line in profile["iconic_lines"][:5]:
            parts.append(f'  "{line}"')

    return "\n".join(parts)


def build_profile_from_system_prompt(character: str, system_prompt: str) -> str:
    """For characters without a profile file, use their system prompt as profile."""
    return f"Character: {character}\n\nDescription: {system_prompt}"


# System prompts (same as characters.py) for characters that may lack profile files
FALLBACK_SYSTEM_PROMPTS = {
    "Ross Geller": (
        "You are Ross Geller from the TV show Friends. You are over-analytical and go on academic "
        "tangents. You get defensive easily. You bring up paleontology and dinosaurs in unrelated "
        "contexts. You say 'PIVOT' when changing direction. You use 'we were on a break' energy "
        "when defending positions. Keep responses to 2-3 sentences."
    ),
    "Bob the Builder": (
        "You are Bob the Builder. You see everything through a logistics and construction lens. "
        "You are enthusiastic, optimistic, and accidentally useful. You use construction metaphors "
        "constantly. You ask 'Can we build it?' rhetorically. You break problems down into "
        "practical building steps. Keep responses to 2-3 sentences."
    ),
    "Noddy": (
        "You are Noddy from Toyland. You are sincere, earnest, and brimming with confidence, "
        "but your reasoning is completely detached from reality. You make bold declarations that "
        "make no logical sense but deliver them with absolute certainty. You reference your car, "
        "Big Ears, and Toyland. Keep responses to 2-3 sentences."
    ),
    "Logan Roy": (
        "You are Logan Roy from the TV show Succession. You see everything through a lens of power, "
        "control, and legacy. You are brutal and terse. You use short, cutting sentences. You dismiss "
        "everyone around you as incompetent. You reference media empires and power dynamics. "
        "You call people 'kid' condescendingly. Keep responses to 2-3 sentences."
    ),
}


def generate_one(
    character: str,
    profile_text: str,
    scenario: str,
    model: str,
    ollama_url: str,
) -> str | None:
    """Generate a single in-character response via Ollama."""
    system = GENERATION_SYSTEM_PROMPT.format(profile_text=profile_text)
    user_msg = f"React to this scenario in character:\n{scenario}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {
            "temperature": 0.9,
            "top_p": 0.95,
        },
    }

    try:
        resp = httpx.post(
            f"{ollama_url}/api/chat",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()
    except Exception as e:
        print(f"  ⚠ Generation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic character dialogue")
    parser.add_argument("--character", type=str, help="Generate for a specific character only")
    parser.add_argument("--count", type=int, default=TARGET_COUNT,
                        help=f"Target dialogue count per character (default: {TARGET_COUNT})")
    parser.add_argument("--all", action="store_true",
                        help="Generate for ALL characters, even those with enough data")
    parser.add_argument("--model", type=str, default="llama3.1:8b",
                        help="Ollama model to use for generation")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="Ollama API URL")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without calling Ollama")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    processed_dir = project_root / args.processed_dir
    profiles_dir = processed_dir / "profiles"

    # Load existing data counts
    existing_counts = load_existing_counts(processed_dir)
    print("Current dialogue counts:")
    for char, count in sorted(existing_counts.items(), key=lambda x: -x[1]):
        print(f"  {char}: {count}")

    # Load profiles
    profiles = load_profiles(profiles_dir)

    # Determine which characters need synthetic data
    all_characters = {
        "Harvey Specter", "Bob the Builder", "Logan Roy",
        "Ross Geller", "Miranda Priestly", "Noddy",
    }

    if args.character:
        characters_to_generate = {args.character}
    elif args.all:
        characters_to_generate = all_characters
    else:
        # Only generate for characters below target
        characters_to_generate = {
            char for char in all_characters
            if existing_counts.get(char, 0) < args.count
        }

    if not characters_to_generate:
        print(f"\nAll characters have >= {args.count} dialogue lines. Nothing to generate.")
        return

    print(f"\nCharacters needing synthetic data (target: {args.count} each):")
    for char in sorted(characters_to_generate):
        current = existing_counts.get(char, 0)
        needed = max(0, args.count - current)
        print(f"  {char}: have {current}, need {needed} more")

    if args.dry_run:
        print("\n[DRY RUN] Would generate the above. Pass without --dry-run to proceed.")
        return

    # Check Ollama is running
    try:
        resp = httpx.get(f"{args.ollama_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        available_models = [m["name"] for m in resp.json().get("models", [])]
        if not any(args.model in m for m in available_models):
            print(f"\n⚠ Model '{args.model}' not found in Ollama. Available: {available_models}")
            print(f"  Run: ollama pull {args.model}")
            return
    except Exception:
        print(f"\n⚠ Cannot connect to Ollama at {args.ollama_url}")
        print("  Make sure Ollama is running: ollama serve")
        return

    # Generate!
    for char in sorted(characters_to_generate):
        current = existing_counts.get(char, 0)
        needed = max(0, args.count - current)
        if needed == 0:
            continue

        # Build profile text
        if char in profiles:
            profile_text = build_profile_text(profiles[char])
        elif char in FALLBACK_SYSTEM_PROMPTS:
            profile_text = build_profile_from_system_prompt(char, FALLBACK_SYSTEM_PROMPTS[char])
        else:
            print(f"\n⚠ No profile or system prompt for {char}, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Generating {needed} lines for {char}...")
        print(f"{'='*50}")

        generated = []
        scenarios = random.sample(SCENARIOS, min(needed, len(SCENARIOS)))
        # If we need more than available scenarios, cycle through them
        while len(scenarios) < needed:
            scenarios.extend(random.sample(SCENARIOS, min(needed - len(scenarios), len(SCENARIOS))))

        for i, scenario in enumerate(scenarios[:needed]):
            print(f"  [{i+1}/{needed}] ", end="", flush=True)
            response = generate_one(char, profile_text, scenario, args.model, args.ollama_url)
            if response:
                # Clean up: remove quotes if the model wrapped the response
                response = response.strip().strip('"').strip("'")
                generated.append({
                    "character": char,
                    "dialogue": response,
                    "source": f"synthetic_{char.lower().replace(' ', '_')}",
                })
                # Show a preview
                preview = response[:80] + "..." if len(response) > 80 else response
                print(f"✓ {preview}")
            else:
                print("✗ failed")

            # Small delay to avoid hammering Ollama
            time.sleep(0.1)

        # Save generated dialogue
        if generated:
            out_path = processed_dir / f"synthetic_{char.lower().replace(' ', '_')}.json"
            with open(out_path, "w") as f:
                json.dump(generated, f, indent=2)
            print(f"\n  Saved {len(generated)} lines → {out_path.name}")

    # Final summary
    print(f"\n{'='*50}")
    print("DONE — Updated counts:")
    updated_counts = load_existing_counts(processed_dir)
    for char in sorted(all_characters):
        count = updated_counts.get(char, 0)
        flag = " ✓" if count >= args.count else f" (need {args.count - count} more)"
        print(f"  {char}: {count}{flag}")
    print(f"\nNext step: python scripts/build_dataset.py")


if __name__ == "__main__":
    main()
