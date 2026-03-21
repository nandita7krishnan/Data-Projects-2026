"""
Flexible parser for mixed raw data formats.

Handles the reality that raw data comes in different shapes:
- Full TV scripts (Suits RTFs, Suits PDFs) → extract CHARACTER: dialogue lines
- Quote collections (harvey spectar quotes.rtf) → extract standalone quotes
- Wiki/bio/profile files (bob, noddy, about_miranda, logan) → extract character
  profiles (personality, speech style, iconic lines) for system prompt enrichment
  and synthetic data generation

Reads from data/raw/, outputs to:
- data/processed/*.json           — extracted dialogue lines
- data/processed/profiles/*.json  — character profiles from wiki/bio files

Usage:
    python scripts/parse_transcripts.py
    python scripts/parse_transcripts.py --input data/raw/suits_101_harvey.rtf
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Character aliases → canonical names
# ---------------------------------------------------------------------------
CHARACTER_ALIASES = {
    # Harvey Specter
    "harvey": "Harvey Specter",
    "harvey specter": "Harvey Specter",
    "specter": "Harvey Specter",
    # Bob the Builder
    "bob": "Bob the Builder",
    "bob the builder": "Bob the Builder",
    # Logan Roy
    "logan": "Logan Roy",
    "logan roy": "Logan Roy",
    "roy": "Logan Roy",
    # Ross Geller
    "ross": "Ross Geller",
    "ross geller": "Ross Geller",
    "geller": "Ross Geller",
    # Miranda Priestly
    "miranda": "Miranda Priestly",
    "miranda priestly": "Miranda Priestly",
    "priestly": "Miranda Priestly",
    # Noddy
    "noddy": "Noddy",
}

# Which canonical character does each raw file map to?
# Used for quote files and profile files where speaker isn't in the text.
FILE_CHARACTER_HINTS = {
    "harvey spectar quotes": "Harvey Specter",
    "suits_101_harvey": None,  # multi-character script, don't force
    "suits_1x07": None,
    "suits_02": None,
    "miranada_quotes": "Miranda Priestly",
    "about_miranada": "Miranda Priestly",
    "bob_the_buider": "Bob the Builder",
    "logan_roy": "Logan Roy",
    "noddy_wiki": "Noddy",
    "ross_quotes": "Ross Geller",
    "ross_wiki": "Ross Geller",
}

# Keywords that signal a file is a wiki/bio rather than a transcript
PROFILE_SIGNALS = [
    "character overview",
    "personality traits",
    "speech style",
    "cultural legacy",
    "key relationships",
    "iconic moments",
]


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------
def read_file_text(filepath: Path) -> str:
    """Extract plain text from any supported file format (.txt, .rtf, .rtfd, .pdf)."""
    suffix = filepath.suffix.lower()

    if suffix == ".rtfd":
        # .rtfd is a macOS bundle directory containing TXT.rtf
        inner = filepath / "TXT.rtf"
        if inner.exists():
            return _read_rtf(inner)
        # Fallback: try textutil on the bundle itself
        return _read_rtf(filepath)
    elif suffix == ".rtf":
        return _read_rtf(filepath)
    elif suffix == ".pdf":
        return _read_pdf(filepath)
    else:
        return filepath.read_text(encoding="utf-8", errors="replace")


def _read_rtf(filepath: Path) -> str:
    """Read RTF → plain text. Uses macOS textutil if available, else striprtf."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["textutil", "-convert", "txt", "-stdout", str(filepath)],
                capture_output=True, text=True, check=True,
            )
            return result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Fallback: striprtf library
    try:
        from striprtf.striprtf import rtf_to_text
        raw = filepath.read_text(encoding="utf-8", errors="replace")
        return rtf_to_text(raw)
    except ImportError:
        print(f"  ⚠ Cannot read RTF (install striprtf or use macOS): {filepath.name}")
        return ""


def _read_pdf(filepath: Path) -> str:
    """Read PDF → plain text using pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    except ImportError:
        print(f"  ⚠ Cannot read PDF (install pdfplumber): {filepath.name}")
        return ""


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------
def classify_file(filepath: Path, content: str) -> str:
    """
    Classify a file as one of:
      'script'  — full TV script with CHARACTER: dialogue lines
      'quotes'  — standalone quote collection for one character
      'profile' — wiki/bio/character analysis (minimal dialogue)
    """
    stem = filepath.stem.lower()
    content_lower = content[:3000].lower()

    # Check for profile signals
    profile_hits = sum(1 for sig in PROFILE_SIGNALS if sig in content_lower)
    if profile_hits >= 2:
        return "profile"

    # Check for quote-list pattern: count quoted strings anywhere in content
    # (quotes may be on same line, not one-per-line)
    quote_count = len(re.findall(r'["\u201c].{15,}?["\u201d]', content[:5000]))
    if quote_count >= 5 and "quote" in stem:
        return "quotes"

    # Check for script-style dialogue patterns (CHARACTER: line)
    dialogue_pattern = re.compile(r"^[A-Z][A-Z\s\.]+:\s*.+", re.MULTILINE)
    dialogue_matches = dialogue_pattern.findall(content[:5000])
    if len(dialogue_matches) >= 5:
        return "script"

    # Script-style with [[ wiki links ]] (like suits_101_harvey which has both)
    wiki_pattern = re.compile(r"\[\[.+?\]\]|\[\d+\]")
    wiki_matches = wiki_pattern.findall(content[:5000])
    if len(dialogue_matches) >= 2 and len(wiki_matches) >= 3:
        return "script"

    # Lots of quotes without the filename hint
    if quote_count >= 5:
        return "quotes"

    # Heavy wiki references with no dialogue → profile (catches noddy_wiki)
    if len(wiki_matches) >= 10 and len(dialogue_matches) < 3:
        return "profile"

    # Default: if any profile signal, treat as profile
    if profile_hits >= 1:
        return "profile"

    return "script"  # optimistic default


def normalize_character(name: str) -> str | None:
    """Map a raw character name to its canonical form, or None if not a target."""
    key = name.strip().lower()
    return CHARACTER_ALIASES.get(key)


def clean_dialogue(text: str) -> str:
    """Remove stage directions, wiki markup, normalize whitespace."""
    # Remove [[File:...]] wiki image tags
    text = re.sub(r"\[\[File:.*?\]\]", "", text)
    # Convert [[Display|Link]] to just Display, and [[Link]] to Link
    text = re.sub(r"\[\[([^|\]]*?\|)?(.+?)\]\]", r"\2", text)
    # Remove [stage directions] and (stage directions)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    # Remove wiki refs like [1], [23]
    text = re.sub(r"\[\d+\]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Parsers by file type
# ---------------------------------------------------------------------------
def parse_script(content: str, source_name: str) -> list[dict]:
    """Parse a TV script with CHARACTER: dialogue lines."""
    results = []

    # Pattern: CHARACTER_NAME: dialogue (handles wiki-formatted scripts too)
    # e.g. "HARVEY: I don't play the odds"
    pattern_colon = re.compile(
        r"^([A-Z][A-Za-z\s\.]+?):\s*(.+?)$", re.MULTILINE
    )
    for match in pattern_colon.finditer(content):
        raw_name = match.group(1).strip()
        dialogue = clean_dialogue(match.group(2))
        canonical = normalize_character(raw_name)
        if canonical and len(dialogue) > 10:
            results.append({
                "character": canonical,
                "dialogue": dialogue,
                "source": source_name,
            })

    # If colon pattern found nothing, try NAME on own line + dialogue below
    if not results:
        lines = content.split("\n")
        i = 0
        while i < len(lines) - 1:
            line = lines[i].strip()
            canonical = normalize_character(line)
            if canonical:
                dialogue_lines = []
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line or normalize_character(next_line):
                        break
                    dialogue_lines.append(next_line)
                    i += 1
                dialogue = clean_dialogue(" ".join(dialogue_lines))
                if len(dialogue) > 10:
                    results.append({
                        "character": canonical,
                        "dialogue": dialogue,
                        "source": source_name,
                    })
            else:
                i += 1

    return results


def parse_quotes(content: str, source_name: str, character: str) -> list[dict]:
    """Parse a quote collection — handles both plain quotes and numbered conversation format."""
    results = []

    # First, try to extract dialogue lines attributed to the target character
    # e.g. 'Ross: "Hey guys, does anybody know..."' or 'Ross: (frantically presses...) "Oh my God!"'
    char_first = character.split()[0]  # "Ross" from "Ross Geller"
    # Pattern: CharName: "dialogue" or CharName: (action) "dialogue"
    attributed = re.finditer(
        rf'{char_first}:\s*(?:\(.*?\)\s*)?["\u201c](.+?)["\u201d]',
        content,
        re.DOTALL,
    )
    attributed_texts = set()
    for match in attributed:
        text = match.group(1).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > 10:
            attributed_texts.add(text)
            results.append({
                "character": character,
                "dialogue": text,
                "source": source_name,
            })

    # Also extract standalone quotes (not attributed to a specific speaker)
    # These are typically solo lines: "I'm the holiday armadillo!"
    # Match numbered entries: N. "quote" where there's no CharName: prefix
    standalone = re.finditer(
        r'(?:^|\n)\s*\d+\.\s*["\u201c](.+?)["\u201d]',
        content,
        re.DOTALL,
    )
    for match in standalone:
        text = match.group(1).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > 10 and text not in attributed_texts:
            results.append({
                "character": character,
                "dialogue": text,
                "source": source_name,
            })

    # Also extract from un-numbered plain quoted strings (for harvey-style files)
    if not results:
        plain_quotes = re.finditer(r'["\u201c](.+?)["\u201d]', content, re.DOTALL)
        for match in plain_quotes:
            text = match.group(1).strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) > 15:
                results.append({
                    "character": character,
                    "dialogue": text,
                    "source": source_name,
                })

    return results


def parse_profile(content: str, source_name: str, character: str) -> dict:
    """
    Extract a structured character profile from a wiki/bio file.

    Returns a dict with:
      - character: canonical name
      - personality_traits: list of trait descriptions
      - speech_style: description of how they talk
      - iconic_lines: any actual quotes/lines found
      - raw_description: full text for synthetic data generation
      - source: filename
    """
    profile = {
        "character": character,
        "personality_traits": [],
        "speech_style": "",
        "iconic_lines": [],
        "raw_description": "",
        "source": source_name,
    }

    # Split into sections by common headers
    sections = re.split(
        r"\n(?=(?:Character Overview|Personality Traits|Speech Style|"
        r"Iconic Moments|Key Relationships|Cultural Legacy|"
        r"Dialogues and Speech Style|Character biography|"
        r"His Relationship|Her Relationship))",
        content,
        flags=re.IGNORECASE,
    )

    for section in sections:
        section_lower = section[:50].lower()

        if "personality" in section_lower:
            # Extract trait lines (typically "Trait name -- description")
            traits = re.findall(r"(?:^|\n)(.+?)\s*[-–—]+\s*(.+?)(?=\n|$)", section)
            for trait_name, trait_desc in traits:
                trait_name = trait_name.strip()
                trait_desc = trait_desc.strip()
                if len(trait_desc) > 10:
                    profile["personality_traits"].append(
                        f"{trait_name}: {trait_desc}"
                    )

        elif "speech style" in section_lower or "dialogues and speech" in section_lower:
            # Grab the whole section as speech style info
            # Remove the header line
            lines = section.strip().split("\n")
            style_text = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
            profile["speech_style"] = style_text

        elif "iconic moments" in section_lower:
            # Try to find actual quoted lines
            quoted = re.findall(r'["\u201c](.+?)["\u201d]', section)
            for q in quoted:
                q = q.strip()
                if len(q) > 10:
                    profile["iconic_lines"].append(q)

        elif "character biography" in section_lower:
            # For wiki articles (like Noddy), the biography section has
            # personality/behavior descriptions we can use
            profile["personality_traits"].append(
                f"Character biography: {section[:500].strip()}"
            )

    # If no structured sections were found (pure wiki article), extract
    # whatever useful character info we can from the full text
    if not profile["personality_traits"] and not profile["speech_style"]:
        profile["personality_traits"].append(
            f"General description: {content[:1000].strip()}"
        )

    # Store raw text for synthetic generation
    profile["raw_description"] = content[:5000]

    return profile


def parse_miranda_dialogue(content: str, source_name: str) -> list[dict]:
    """
    Special parser for miranada_quotes.rtf which is scene-by-scene Miranda
    dialogue without speaker labels — it's all Miranda.
    The content is individual lines/paragraphs of her speech.
    """
    results = []
    # Split on newlines — each line or paragraph is a separate speech
    lines = content.split("\n")

    for line in lines:
        line = line.strip()
        if not line or len(line) < 20:
            continue

        # Skip obvious non-dialogue (section headers, descriptions)
        skip_starts = ("character overview", "personality", "speech style",
                       "cultural legacy", "iconic moments")
        if line.lower().startswith(skip_starts):
            continue

        dialogue = clean_dialogue(line)
        if len(dialogue) > 20:
            results.append({
                "character": "Miranda Priestly",
                "dialogue": dialogue,
                "source": source_name,
            })

    return results


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def parse_file(filepath: Path) -> tuple[list[dict], dict | None]:
    """
    Parse a single file. Returns (dialogue_entries, profile_or_none).
    """
    content = read_file_text(filepath)
    if not content:
        return [], None

    stem = filepath.stem.lower()
    file_type = classify_file(filepath, content)
    # Normalize stem for hint lookup (spaces/dots → underscores)
    stem_norm = re.sub(r"[\s.]+", "_", stem)
    hint_char = FILE_CHARACTER_HINTS.get(stem_norm)
    # Fuzzy fallback: check if any hint key is a substring
    if hint_char is None:
        for key, val in FILE_CHARACTER_HINTS.items():
            key_norm = re.sub(r"[\s.]+", "_", key)
            if key_norm in stem_norm or stem_norm in key_norm:
                hint_char = val
                break

    print(f"  → classified as: {file_type}", end="")
    if hint_char:
        print(f" (character: {hint_char})", end="")
    print()

    dialogue_entries = []
    profile = None

    if file_type == "profile":
        char = hint_char or "Unknown"
        profile = parse_profile(content, filepath.name, char)
        # Also extract any iconic lines as dialogue entries
        for line in profile.get("iconic_lines", []):
            dialogue_entries.append({
                "character": char,
                "dialogue": line,
                "source": filepath.name,
            })

    elif file_type == "quotes":
        char = hint_char or "Unknown"
        dialogue_entries = parse_quotes(content, filepath.name, char)

    elif file_type == "script":
        # Special case: miranada_quotes is all Miranda dialogue without labels
        if "miranada_quotes" in stem or "miranda_quotes" in stem:
            dialogue_entries = parse_miranda_dialogue(content, filepath.name)
        else:
            dialogue_entries = parse_script(content, filepath.name)

    return dialogue_entries, profile


def main():
    parser = argparse.ArgumentParser(description="Parse raw data into structured dialogue and profiles")
    parser.add_argument("--input", type=str, help="Parse a specific file instead of all files in data/raw/")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory with raw files")
    parser.add_argument("--out-dir", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    raw_dir = project_root / args.raw_dir
    out_dir = project_root / args.out_dir
    profiles_dir = out_dir / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        files = [Path(args.input)]
    else:
        files = sorted(raw_dir.iterdir())
        # Include regular files and .rtfd bundles (which are directories)
        files = [
            f for f in files
            if not f.name.startswith(".")
            and (f.is_file() or f.suffix.lower() == ".rtfd")
        ]

    if not files:
        print(f"No files found in {raw_dir}")
        return

    total_dialogue = 0
    total_profiles = 0
    character_counts: dict[str, int] = {}
    all_profiles: dict[str, dict] = {}

    for filepath in files:
        print(f"\nParsing: {filepath.name}...")
        dialogue_entries, profile = parse_file(filepath)

        # Save dialogue
        if dialogue_entries:
            out_path = out_dir / f"{filepath.stem}.json"
            with open(out_path, "w") as f:
                json.dump(dialogue_entries, f, indent=2)
            print(f"  → {len(dialogue_entries)} dialogue lines → {out_path.name}")
            total_dialogue += len(dialogue_entries)
            for entry in dialogue_entries:
                char = entry["character"]
                character_counts[char] = character_counts.get(char, 0) + 1
        else:
            print(f"  → 0 dialogue lines extracted")

        # Save profile
        if profile:
            profile_path = profiles_dir / f"{profile['character'].lower().replace(' ', '_')}.json"
            # Merge with existing profile if we already have one for this character
            if profile["character"] in all_profiles:
                existing = all_profiles[profile["character"]]
                existing["personality_traits"].extend(profile["personality_traits"])
                existing["iconic_lines"].extend(profile["iconic_lines"])
                if profile["speech_style"] and not existing["speech_style"]:
                    existing["speech_style"] = profile["speech_style"]
                existing["raw_description"] += "\n\n" + profile["raw_description"]
            else:
                all_profiles[profile["character"]] = profile
            total_profiles += 1

    # Write merged profiles
    for char, profile in all_profiles.items():
        profile_path = profiles_dir / f"{char.lower().replace(' ', '_')}.json"
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)
        trait_count = len(profile["personality_traits"])
        iconic_count = len(profile["iconic_lines"])
        print(f"\n  Profile saved: {profile_path.name} ({trait_count} traits, {iconic_count} iconic lines)")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total dialogue lines: {total_dialogue}")
    print(f"Character profiles:   {total_profiles}")
    print(f"\nDialogue per character:")
    for char, count in sorted(character_counts.items(), key=lambda x: -x[1]):
        flag = " ⚠️  LOW — needs synthetic data" if count < 30 else ""
        print(f"  {char}: {count}{flag}")

    missing = {"Harvey Specter", "Bob the Builder", "Logan Roy", "Ross Geller",
               "Miranda Priestly", "Noddy"} - set(character_counts.keys())
    if missing:
        print(f"\n⚠️  No dialogue at all for: {', '.join(sorted(missing))}")
        print("   → Run generate_synthetic.py to create training data from profiles")

    print(f"\nDialogue → {out_dir}")
    print(f"Profiles → {profiles_dir}")


if __name__ == "__main__":
    main()
