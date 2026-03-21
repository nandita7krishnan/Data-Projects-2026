"""
Debate orchestration: runs 3 phases of character responses.

Phase 1 (Opening): Each character reacts to the pitch
Phase 2 (Cross-debate): 2 rounds where characters respond to each other
Phase 3 (Vote): Each character gives a verdict and yes/no/abstain vote
"""

import re

from .characters import CHARACTERS, CHARACTER_ORDER
from .ollama_client import generate_response


async def run_debate(pitch: str) -> dict:
    """Run a full 3-phase debate on the given pitch."""
    conversation_history = []
    phases = []

    # Phase 1: Opening Reactions
    phase1_messages = []
    for char_key in CHARACTER_ORDER:
        char = CHARACTERS[char_key]
        messages = [
            {"role": "user", "content": f"Someone just pitched this idea: {pitch}\n\nGive your initial reaction."}
        ]

        response = await generate_response(
            system_prompt=char["system_prompt"],
            messages=messages,
            character_name=char["name"],
        )

        entry = {
            "character": char_key,
            "name": char["name"],
            "content": response,
            "initials": char["initials"],
            "color": char["color"],
        }
        phase1_messages.append(entry)
        conversation_history.append({
            "role": "assistant",
            "content": f"{char['name']}: {response}",
        })

    phases.append({"name": "Opening Reactions", "messages": phase1_messages})

    # Phase 2: Final Vote
    vote_messages = []
    votes = {"yes": 0, "no": 0, "abstain": 0}

    for char_key in CHARACTER_ORDER:
        char = CHARACTERS[char_key]

        history_text = "\n".join(
            msg["content"] for msg in conversation_history
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"The pitch: {pitch}\n\n"
                    f"The full debate:\n{history_text}\n\n"
                    f"Give your final verdict on this pitch and cast your vote: YES, NO, or ABSTAIN. "
                    f"End your response with exactly one of: [VOTE: YES], [VOTE: NO], or [VOTE: ABSTAIN]"
                ),
            }
        ]

        response = await generate_response(
            system_prompt=char["system_prompt"],
            messages=messages,
            character_name=char["name"],
        )

        # Parse vote
        vote = parse_vote(response)
        votes[vote] += 1

        entry = {
            "character": char_key,
            "name": char["name"],
            "content": response,
            "vote": vote,
            "initials": char["initials"],
            "color": char["color"],
        }
        vote_messages.append(entry)

    phases.append({"name": "Final Vote", "messages": vote_messages})

    return {
        "pitch": pitch,
        "phases": phases,
        "votes": votes,
    }


def parse_vote(response: str) -> str:
    """Extract vote from response text."""
    response_upper = response.upper()

    # Look for explicit vote markers
    vote_match = re.search(r"\[VOTE:\s*(YES|NO|ABSTAIN)\]", response_upper)
    if vote_match:
        return vote_match.group(1).lower()

    # Fallback: look for the last occurrence of yes/no/abstain
    if "ABSTAIN" in response_upper:
        return "abstain"
    if "YES" in response_upper.split(".")[-1]:
        return "yes"
    if "NO" in response_upper.split(".")[-1]:
        return "no"

    return "abstain"  # Default if we can't parse
