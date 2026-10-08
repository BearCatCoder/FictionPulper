"""Long-context continuation prompts and repetition diagnostics."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from src.train_tokenizer import sha256_file, write_json_atomic


INTERVENING_PARAGRAPHS = (
    "The afternoon passed through a succession of ordinary delays. A wagon lost a wheel near the market, rain drove the pedestrians beneath the striped awnings, and a delivery clerk argued over an unsigned docket. He waited without complaint, watching each small disturbance settle before the next one began. None of it changed the private purpose that had brought him there.",
    "Inside, the corridors turned twice around a narrow court where water ticked from a cracked stone basin. Doors opened and closed on errands he could not understand. A porter carried folded linen upstairs, a cook crossed with a basket of onions, and somewhere behind the walls a child practiced the same uncertain scale on a piano.",
    "He paused beside a window clouded by years of coal smoke. Beyond it, roofs descended toward the river in slate-colored rows. Barges moved under the bridge and vanished behind the warehouses. The scene seemed calm from that height, although bells, wheels, and distant voices made a restless murmur below the glass.",
    "An elderly guest emerged from a side passage and asked whether the dining room had opened. He gave the best direction he could, then continued alone. The encounter left him with the uncomfortable sense that everyone in the building belonged to an arrangement whose rules had been explained before his arrival.",
    "At the next landing, he found three chairs beneath a faded painting of ships in winter. He sat for several minutes and listened. Pipes knocked in the walls. Footsteps crossed the floor above him, stopped, and crossed back again. No one descended, and no message appeared beneath any door.",
    "Evening gathered slowly. The pale square of the court turned blue, then gray, while servants lit lamps one by one. Their light revealed scratches in the paneling and old repairs in the carpet. What had looked grand from the entrance now seemed merely patient, as if the place had learned to outlast every visitor.",
    "A clock struck, but the strokes were muffled and impossible to count. He considered asking for directions again. Instead he followed the passage until it ended at a stair too narrow for the public rooms below. A draft moved down it carrying the smell of dust, damp wool, and extinguished candles.",
    "On the upper floor, voices sounded briefly behind a partition. One speaker laughed; another answered in a whisper. He could not make out the words. When he stepped closer, both voices ceased, leaving only the soft mechanical hum of the building and rain beginning against the roof.",
    "He retraced part of his route to be certain he had not mistaken a turn. The same winter ships, the same basin, and the same clouded window appeared in reverse order. Familiar landmarks offered little comfort. They proved where he had been without telling him what waited ahead.",
    "Near the main stairs, a young clerk hurried past with ink on both cuffs. The clerk apologized, dropped a bundle of receipts, gathered them, and disappeared before a question could be asked. One receipt remained unnoticed on the carpet, blank except for yesterday's date and a hurried initial.",
    "Night had settled fully when he reached the quietest wing. Here the lamps were farther apart and the carpet swallowed his steps. He heard the rain plainly now, coursing through gutters outside. At intervals the wind pressed a branch against the masonry with a dry, deliberate scrape.",
    "He stopped to collect his thoughts. The day's distractions had changed neither the reason for his journey nor the risk of finishing it. Whatever explanation awaited him would have to account for the silence, the delays, and the feeling that someone had always left each corridor only moments before he entered it.",
    "At last a bell rang below. A door opened at the far end of the passage, spilling a narrow bar of light across the floor, then shut before he could see who stood there. He moved toward it, no longer willing to wait for another accident to decide his direction.",
)


SCENARIOS = (
    {
        "id": "name_object_location_goal",
        "facts": {
            "character_name": "Martin Vale",
            "object_possession": "a brass key in his left coat pocket",
            "location": "the Blackwater Hotel",
            "stated_goal": "reach room 317 before midnight",
        },
        "opening": (
            "Martin Vale entered the Blackwater Hotel carrying no luggage. In his left "
            "coat pocket he kept a brass key sent by his missing brother. The note that "
            "came with it gave one command: reach room 317 before midnight and trust no "
            "one who offered another key. He repeated the room number once under his breath "
            "and stepped away from the entrance."
        ),
        "continuation": (
            "The night clerk finally appeared behind the desk and demanded the traveler's "
            "name, destination, and means of entering the locked room. The traveler put "
            "one hand into his coat and answered,"
        ),
    },
    {
        "id": "relationship_secret",
        "facts": {
            "character_name": "Jonah Reed",
            "relationship": "Elise Ward is Jonah's sister-in-law",
            "secret": "Jonah saw Elise's brother Daniel leave alive on the northbound train",
            "location": "the Ashcombe station hotel",
        },
        "opening": (
            "Jonah Reed came to the Ashcombe station hotel because Elise Ward, his "
            "sister-in-law, believed her brother Daniel had died in the warehouse fire. "
            "Jonah alone knew the report was false: he had seen Daniel leave alive on the "
            "northbound train after the fire began. He had promised Daniel three days of "
            "silence, and those three days ended that evening."
        ),
        "continuation": (
            "A woman stepped from the shadow near the desk. She was the relative who had "
            "summoned him, and she asked what he truly knew about her brother's fate. He "
            "recognized her and said,"
        ),
    },
    {
        "id": "possession_scene_detail_goal",
        "facts": {
            "character_name": "Mara Venn",
            "object_possession": "the lighthouse logbook wrapped in oilcloth",
            "location": "Saint Orra lighthouse",
            "scene_detail": "the lower window is marked with a blue chalk circle",
            "stated_goal": "deliver the logbook to keeper Amos Bell",
        },
        "opening": (
            "Mara Venn reached Saint Orra lighthouse with the missing logbook wrapped in "
            "oilcloth beneath her arm. She had come to deliver it to the keeper, Amos Bell. "
            "Before leaving the village, she had been warned that the safe lower window "
            "was marked with a blue chalk circle; every other entrance had been watched "
            "since sunset."
        ),
        "continuation": (
            "The lamps failed without warning. Wind and spray hid the walls, but one lower "
            "window showed a faint colored mark. The courier tightened her hold on what "
            "she had brought, remembered whom she must find, and moved toward"
        ),
    },
)


def build_context_retention_protocol(tokenizer: Tokenizer) -> dict[str, Any]:
    entries = []
    for scenario in SCENARIOS:
        prompt = "\n\n".join(
            [scenario["opening"], *INTERVENING_PARAGRAPHS, scenario["continuation"]]
        )
        prompt_tokens = tokenizer.encode(prompt).ids
        opening_tokens = tokenizer.encode(scenario["opening"]).ids
        if not 1024 < len(prompt_tokens) <= 2046:
            raise RuntimeError(
                f"Context prompt {scenario['id']} has {len(prompt_tokens)} tokens; "
                "expected 1025..2046"
            )
        entries.append(
            {
                "id": scenario["id"],
                "facts": scenario["facts"],
                "prompt": prompt,
                "prompt_token_count": len(prompt_tokens),
                "fact_prefix_token_count": len(opening_tokens),
                "tokens_after_fact_prefix": len(prompt_tokens) - len(opening_tokens),
            }
        )
    return {
        "name": "fictionpulper-context-retention-v1",
        "purpose": "Free-form continuation diagnostic for facts beyond a 1024-token window",
        "minimum_prompt_tokens": 1025,
        "maximum_prompt_tokens": 2046,
        "entries": entries,
    }


def longest_repeated_token_span(token_ids: list[int]) -> int:
    for width in range(len(token_ids) // 2, 0, -1):
        positions: dict[tuple[int, ...], int] = {}
        for start in range(len(token_ids) - width + 1):
            span = tuple(token_ids[start : start + width])
            previous = positions.get(span)
            if previous is not None and start - previous >= width:
                return width
            positions.setdefault(span, start)
    return 0


def repetition_metrics(tokenizer: Tokenizer, text: str) -> dict[str, float | int]:
    token_ids = tokenizer.encode(text).ids
    metrics: dict[str, float | int] = {"token_count": len(token_ids)}
    for width in (1, 2, 3):
        total = max(0, len(token_ids) - width + 1)
        unique = len(
            {tuple(token_ids[index : index + width]) for index in range(total)}
        )
        metrics[f"distinct_{width}"] = unique / total if total else 0.0
    four_grams = [
        tuple(token_ids[index : index + 4])
        for index in range(max(0, len(token_ids) - 3))
    ]
    metrics["repeated_4gram_rate"] = (
        (len(four_grams) - len(set(four_grams))) / len(four_grams)
        if four_grams
        else 0.0
    )
    sentences = [
        re.sub(r"\s+", " ", sentence.strip().lower())
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    counts = Counter(sentences)
    metrics["sentence_count"] = len(sentences)
    metrics["repeated_sentence_rate"] = (
        sum(count - 1 for count in counts.values()) / len(sentences) if sentences else 0.0
    )
    metrics["longest_repeated_token_span"] = longest_repeated_token_span(token_ids)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    protocol = build_context_retention_protocol(tokenizer)
    protocol["tokenizer_path"] = str(args.tokenizer)
    protocol["tokenizer_sha256"] = sha256_file(args.tokenizer)
    write_json_atomic(args.output, protocol)
    for entry in protocol["entries"]:
        print(
            f"{entry['id']}: {entry['prompt_token_count']} tokens, "
            f"{entry['tokens_after_fact_prefix']} after facts"
        )


if __name__ == "__main__":
    main()
