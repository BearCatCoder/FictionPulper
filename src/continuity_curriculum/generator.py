"""Generate deterministic, varied continuity fiction and replayable state sidecars."""

from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from tokenizers import Tokenizer

from src.continuity_curriculum.audits import (
    artifact_manifest,
    curriculum_stats,
    distance_audit,
    exact_duplicate_audit,
    quality_audit,
    similarity_audit,
    state_audit,
    template_audit,
)
from src.continuity_curriculum.common import (
    canonical_json,
    normalized_text_hash,
    normalized_words,
    sha256_bytes,
    sha256_file,
    stable_seed,
    write_json_atomic,
    write_jsonl,
)
from src.continuity_curriculum.validation import validate_example, validate_split_isolation


SPLITS = ("train", "validation", "test", "generalization_holdout")
MAIN_SPLITS = ("train", "validation", "test")
SCHEMA_VERSION = 1
STATE_TYPES = (
    "character_identity_role_relationship",
    "ownership_transfer_hiding_retrieval",
    "physical_scene_location_movement",
    "goal_persistence_completion_failure",
    "knowledge_ignorance_secret",
    "cause_effect",
    "promise_debt_obligation",
    "persistent_injury",
    "temporal_ordering",
)
GENRES = (
    "crime_noir", "mystery", "horror", "science_fiction", "western",
    "adventure", "fantasy_weird",
)
GENRE_CONTROL_CANDIDATES = {
    "crime_noir": ("<|noir|>", "<|crime|>"),
    "mystery": ("<|mystery|>",),
    "horror": ("<|horror|>",),
    "science_fiction": ("<|science_fiction|>",),
    "western": ("<|western|>",),
    "adventure": ("<|adventure|>",),
    "fantasy_weird": ("<|weird|>", "<|fantasy|>"),
}
GENRE_SCENES = {
    "crime_noir": (
        "Rain polished the alley behind the hotel, and every window looked blind.",
        "A tram groaned through the midnight district while the desk clerk watched the door.",
        "The office smelled of wet wool, old paper, and coffee left too long on the burner.",
    ),
    "mystery": (
        "The manor clock had stopped at nine, though someone had wound it before supper.",
        "Dust lay evenly across the archive except beside one narrow cabinet.",
        "The village was quiet enough for footsteps to carry from the locked conservatory.",
    ),
    "horror": (
        "Wind pressed its palms against the chapel glass and worried the loose panes.",
        "The abandoned house seemed smaller by day and immeasurably deep after sunset.",
        "Something moved in the reeds whenever the moon disappeared behind cloud.",
    ),
    "science_fiction": (
        "The survey station turned beneath a violet planet and the instruments whispered softly.",
        "Beyond the pressure glass, the colony lights made a second constellation on the plain.",
        "The courier ship had carried no voice traffic since entering the magnetic storm.",
    ),
    "western": (
        "Heat trembled above the main street, where three horses waited in a strip of shade.",
        "The stage road crossed dry country before dropping toward the lights of the settlement.",
        "At sundown the wind carried dust, wood smoke, and a distant harmonica across the yard.",
    ),
    "adventure": (
        "The river narrowed between black cliffs, forcing the expedition into single file.",
        "Their map ended at the ridge, but the ruined watchtower was visible through the mist.",
        "The launch rose and fell against the pier while the last crates came aboard.",
    ),
    "fantasy_weird": (
        "Blue moths circled the ruined gate, although winter had silvered every branch.",
        "At the edge of the marsh, the old road continued beneath water without bending.",
        "The market opened only at moonrise, under bells that rang without being touched.",
    ),
}
FAMILY_BRIDGES = {
    "interrupted_errand": "What began as an errand became uncertain when the usual route was closed.",
    "witness_exchange": "Each witness remembered a different detail, so they compared accounts carefully.",
    "night_watch": "They agreed to keep watch in turns and to wake the others at any change.",
    "delayed_departure": "Departure was delayed while workers examined the road ahead.",
    "quiet_investigation": "Questions were asked softly, without alarming the people nearby.",
    "storm_shelter": "A sudden storm drew strangers under the same narrow shelter.",
    "border_crossing": "The checkpoint required patience, papers, and a convincing explanation.",
    "missing_courier": "The expected courier had not arrived, and no message explained the delay.",
    "sealed_room": "No one admitted entering the sealed room since the previous evening.",
    "rescue_route": "They reviewed the rescue route before committing themselves to the descent.",
    "false_trail": "A convincing trail led east before vanishing on bare stone.",
    "dangerous_bargain": "The bargain sounded simple until its final condition was spoken aloud.",
    "return_journey": "The return journey forced them to reconsider every choice made on the way out.",
    "public_ceremony": "The public ceremony left little room for private conversation.",
    "failed_signal": "A signal failed at the appointed hour, changing the meaning of their instructions.",
    "unexpected_guest": "An unexpected guest arrived with mud on both boots and a plausible story.",
    "hidden_passage": "A draft behind the shelves suggested that the wall concealed more than plaster.",
    "last_transport": "The last transport would leave before dawn whether they were ready or not.",
    "remote_outpost": "At the remote outpost, supplies and trustworthy news were equally scarce.",
    "double_message": "Two messages bearing the same signature gave opposite instructions.",
}
FILLER_DETAILS = (
    "A clock marked the quarter hour while shadows lengthened across the floor.",
    "Outside, two travelers passed in conversation and disappeared beyond the gate.",
    "They checked the weather, the available light, and the condition of the road.",
    "A kettle began to murmur in the next room before anyone remembered to lift it.",
    "For several minutes, ordinary work continued around them without interruption.",
    "A distant bell sounded once, paused, and then sounded twice more.",
    "Someone brought a lamp, trimmed its wick, and set it safely away from the papers.",
    "The youngest porter counted the remaining crates and started again to be certain.",
    "Clouds moved over the rooftops, changing the color of the light at every window.",
    "Their conversation turned briefly to food, lodging, and the next day's weather.",
    "A loose shutter knocked until a passerby crossed the yard and fastened it.",
    "No new message arrived, though footsteps sounded several times in the corridor.",
    "The room gradually emptied as workers completed their duties and went home.",
    "They waited through another quiet interval, listening for movement beyond the wall.",
    "A cart rolled past slowly, its wheels finding every hollow in the road.",
    "Fresh water was brought in, and the maps were moved to make space on the table.",
    "The fire settled lower, leaving a red line beneath the darkened logs.",
    "Nothing in the street appeared urgent, but nobody suggested abandoning the plan.",
)
OBSERVATIONS = (
    "a torn notice", "fresh wheel marks", "a shutter moving in the wind",
    "three lights in an upper room", "a trail of muddy water", "an unattended coat",
    "a distant column of smoke", "a door left slightly open", "new scratches on the railing",
    "a bird settling on the chimney", "a lantern behind frosted glass", "footprints near the wall",
)
MINOR_ACTIONS = (
    "made a note of it", "asked a quiet question", "checked the nearby entrance",
    "waited before moving on", "compared it with the map", "mentioned it to the others",
    "looked again from another angle", "kept it in mind", "examined the ground nearby",
    "listened for a second sound", "marked the place for later", "returned to the immediate task",
)
SHORT_ACTIONS = (
    "paused to listen", "checked the time", "studied the road", "lowered the lamp",
    "reviewed the map", "watched the doorway", "rested briefly", "called to the others",
)


def _choice(rng: random.Random, values: list[str], label: str) -> str:
    if not values:
        raise ValueError(f"No configured values for {label}")
    return values[rng.randrange(len(values))]


def _event_specs(quote: str, state_type: str, changes: list[tuple[str, str, Any]]) -> list[dict]:
    return [
        {"quote": quote, "state_type": state_type, "entity": entity,
         "attribute": attribute, "value": value}
        for entity, attribute, value in changes
    ]


def _state_plan(state_type: str, values: dict[str, Any], variant: int) -> dict[str, Any]:
    a, b, c = values["characters"]
    obj, second_obj = values["objects"]
    origin, destination = values["locations"]
    goal = values["goal"]
    secret = values["secret"]
    injury = values["injury"]
    role = values["role"]
    failed = values["outcome"] == "failed"

    if state_type == "character_identity_role_relationship":
        intros = (
            f"{a}, the {role}, introduced {b} as a trusted cousin.",
            f"Everyone knew {a} as the local {role}, while {b} was a trusted cousin.",
            f"As the district's {role}, {a} vouched for {b}, a trusted cousin.",
        )
        resolves = (
            f"Later, {c} addressed {a} as the {role} and welcomed {b} as the trusted cousin.",
            f"At the door, {a}'s work as a {role} and {b}'s place as a trusted cousin were confirmed.",
            f"The register identified {a} as the {role} and {b} as the trusted cousin, exactly as stated.",
        )
        intro, resolve = intros[variant], resolves[variant]
        intro_changes = [
            (a, "identity", a), (b, "identity", b), (a, "role", role),
            (b, f"relationship_to:{a}", "trusted cousin"),
        ]
        resolve_changes = intro_changes
    elif state_type == "ownership_transfer_hiding_retrieval":
        intros = (
            f"{a} gave the {obj} to {b}, who hid it behind a loose brick at the {origin}.",
            f"For safekeeping, the {obj} passed from {a} to {b} and into a niche at the {origin}.",
            f"At {a}'s request, {b} took possession of the {obj} and concealed it at the {origin}.",
        )
        resolves = (
            f"At last {b} retrieved the {obj} from its hiding place and returned it to {a}.",
            f"When it was needed, {b} recovered the hidden {obj} and placed it in {a}'s hands.",
            f"The niche was opened, the {obj} retrieved, and ownership restored to {a}.",
        )
        intro, resolve = intros[variant], resolves[variant]
        intro_changes = [(obj, "holder", b), (obj, "hidden_at", origin), (obj, "retrieved", False)]
        resolve_changes = [(obj, "holder", a), (obj, "hidden_at", "none"), (obj, "retrieved", True)]
    elif state_type == "physical_scene_location_movement":
        intros = (
            f"Leaving the {origin}, {a} and {b} carried the {second_obj} to the {destination}.",
            f"{a} led {b} away from the {origin} and toward the {destination} with the {second_obj}.",
            f"The pair moved from the {origin} to the {destination}, taking the {second_obj} with them.",
        )
        resolves = (
            f"By nightfall {a}, {b}, and the {second_obj} were still together at the {destination}.",
            f"When darkness came, both travelers remained at the {destination} beside the {second_obj}.",
            f"Their journey ended at the {destination}, where {a} set down the {second_obj} near {b}.",
        )
        intro, resolve = intros[variant], resolves[variant]
        intro_changes = [
            (a, "previous_location", origin), (b, "previous_location", origin),
            (a, "location", destination), (b, "location", destination),
            (second_obj, "location", destination),
        ]
        resolve_changes = intro_changes
    elif state_type == "goal_persistence_completion_failure":
        intros = (
            f"{a}'s purpose was clear: {goal}, and no delay had changed that goal.",
            f"Despite the confusion, {a} continued to pursue one goal: {goal}.",
            f"{a} repeated the plan to {b}: the goal remained to {goal}.",
        )
        status = "failed" if failed else "completed"
        ending = "failed before the deadline" if failed else "was completed before the deadline"
        resolves = (
            f"In the end, the attempt to {goal} {ending}.",
            f"By dawn, {a}'s goal to {goal} had {status}.",
            f"The final report marked the goal to {goal} as {status}.",
        )
        intro, resolve = intros[variant], resolves[variant]
        intro_changes = [(a, "goal", goal), (a, "goal_status", "active")]
        resolve_changes = [(a, "goal", goal), (a, "goal_status", status)]
    elif state_type == "knowledge_ignorance_secret":
        intros = (
            f"{a} did not know the secret that {b} guarded: {secret}.",
            f"The fact that {secret} remained secret from {a}, though {b} knew it.",
            f"Only {b} knew that {secret}; {a} remained entirely ignorant of it.",
        )
        resolves = (
            f"When {b} finally revealed that {secret}, {a} understood the secret.",
            f"{b} then told {a} that {secret}, ending the earlier ignorance.",
            f"The secret that {secret} was disclosed by {b}, and {a} learned the truth.",
        )
        intro, resolve = intros[variant], resolves[variant]
        intro_changes = [(a, f"knows:{secret}", False), (b, f"knows:{secret}", True)]
        resolve_changes = [(a, f"knows:{secret}", True), (b, f"knows:{secret}", True)]
    elif state_type == "cause_effect":
        cause = values["cause"]
        effect = values["effect"]
        intros = (
            f"Because {cause}, {effect}.",
            f"The immediate result of the fact that {cause} was that {effect}.",
            f"{effect.capitalize()}, directly because {cause}.",
        )
        resolves = (
            f"The later inquiry confirmed the chain: {cause}, and therefore {effect}.",
            f"No other explanation was needed; {cause} had caused the fact that {effect}.",
            f"They recorded {cause} as the cause and the fact that {effect} as its effect.",
        )
        intro, resolve = intros[variant], resolves[variant]
        intro_changes = [(cause, "occurred", True), (effect, "caused_by", cause)]
        resolve_changes = intro_changes
    elif state_type == "promise_debt_obligation":
        intros = (
            f"{a} owed {b} a debt and promised to repay it with the {second_obj}.",
            f"The {second_obj} represented {a}'s unpaid obligation to {b}.",
            f"Before witnesses, {a} promised {b} the {second_obj} in payment of a debt.",
        )
        status = "outstanding" if failed else "fulfilled"
        resolves = (
            f"{a} {'could not deliver' if failed else 'delivered'} the {second_obj}, leaving the obligation {status}.",
            f"At settlement, the debt to {b} was marked {status} when {a} {'failed to produce' if failed else 'produced'} the {second_obj}.",
            f"The promise concerning the {second_obj} remained {status} at the final accounting.",
        )
        intro, resolve = intros[variant], resolves[variant]
        intro_changes = [(a, f"obligation_to:{b}", "outstanding")]
        resolve_changes = [(a, f"obligation_to:{b}", status)]
    elif state_type == "persistent_injury":
        intros = (
            f"A bandaged {injury} forced {a} to walk with a careful limp.",
            f"{a}'s {injury} was still painful and plainly had not healed.",
            f"Every step reminded {a} of the persistent {injury}.",
        )
        resolves = (
            f"Hours later, the same {injury} still made {a} limp.",
            f"At the journey's end, {a}'s {injury} remained unhealed.",
            f"Nothing had cured the {injury}; {a} continued to favor it.",
        )
        intro, resolve = intros[variant], resolves[variant]
        intro_changes = [(a, "injury", injury), (a, "injury_status", "unhealed")]
        resolve_changes = intro_changes
    elif state_type == "temporal_ordering":
        intros = (
            f"First {a} rang the bell; only afterward did {b} open the gate.",
            f"{b} opened the gate after, never before, {a} rang the bell.",
            f"The order was fixed: {a} rang the bell, then {b} opened the gate.",
        )
        resolves = (
            f"The log preserved that order and placed {c}'s arrival after the gate opened.",
            f"Witnesses agreed that {c} arrived last, following the bell and the opened gate.",
            f"The final timeline listed the bell first, the gate second, and {c}'s arrival third.",
        )
        intro, resolve = intros[variant], resolves[variant]
        intro_changes = [("timeline", "first", f"{a} rang bell"), ("timeline", "second", f"{b} opened gate")]
        resolve_changes = [("timeline", "third", f"{c} arrived")]
    else:
        raise ValueError(f"Unknown state type: {state_type}")
    return {
        "state_type": state_type,
        "intro": intro,
        "resolve": resolve,
        "intro_events": _event_specs(intro, state_type, intro_changes),
        "resolve_events": _event_specs(resolve, state_type, resolve_changes),
    }


def _document_token_ids(tokenizer: Tokenizer, title: str, text: str, control: str | None) -> list[int]:
    special = [tokenizer.token_to_id("<|story|>")]
    if control is not None:
        special.append(tokenizer.token_to_id(control))
    special.append(tokenizer.token_to_id("<|bos|>"))
    end = tokenizer.token_to_id("<|eos|>")
    if any(value is None for value in [*special, end]):
        raise RuntimeError("tokenizer-v1 is missing required story/BOS/EOS tokens")
    if not text.startswith(title + "\n\n"):
        raise ValueError("Curriculum text must begin with its title")
    return [*special, *tokenizer.encode(text).ids, end]  # type: ignore[list-item]


def _selected_states(difficulty: int, ordinal: int) -> list[str]:
    if difficulty == 5:
        required = list(STATE_TYPES[1:4])
        remainder = [item for item in STATE_TYPES if item not in required]
        return required + [remainder[(ordinal + offset) % len(remainder)] for offset in range(2)]
    start = (ordinal * 3 + difficulty) % len(STATE_TYPES)
    return [STATE_TYPES[(start + offset) % len(STATE_TYPES)] for offset in range(difficulty)]


def _control_for_genre(tokenizer: Tokenizer, genre: str) -> str | None:
    return next(
        (token for token in GENRE_CONTROL_CANDIDATES[genre] if tokenizer.token_to_id(token) is not None),
        None,
    )


def generate_example(
    config: dict[str, Any], tokenizer: Tokenizer, *, split: str, ordinal: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(stable_seed(int(config["seed"]), split, ordinal))
    pools = config["holdouts"][split]
    characters = rng.sample(pools["names"], 3)
    objects = rng.sample(pools["objects"], 2)
    locations = rng.sample(pools["locations"], 2)
    difficulty = ordinal % 5 + 1
    compatible = config["difficulty_distance_buckets"][str(difficulty)]
    distance_bucket = compatible[(ordinal // 5) % len(compatible)]
    distance_definition = config["distance_buckets"][distance_bucket]
    genre = GENRES[(ordinal + SPLITS.index(split)) % len(GENRES)]
    family = pools["template_families"][ordinal % len(pools["template_families"])]
    lexical = _choice(rng, pools["lexical_combinations"], "lexical combinations")
    adjective, atmosphere = lexical.split("|", 1)
    values = {
        "characters": characters,
        "objects": objects,
        "locations": locations,
        "goal": _choice(rng, pools["goals"], "goals"),
        "secret": _choice(rng, pools["secrets"], "secrets"),
        "injury": _choice(rng, pools["injuries"], "injuries"),
        "role": _choice(rng, pools["roles"], "roles"),
        "cause": _choice(rng, pools["causes"], "causes"),
        "effect": _choice(rng, pools["effects"], "effects"),
        "outcome": "failed" if ordinal % 4 == 0 else "completed",
    }
    state_types = _selected_states(difficulty, ordinal)
    plans = [_state_plan(state_type, values, (ordinal + index) % 3) for index, state_type in enumerate(state_types)]
    primary = plans[-1]
    title = f"The {adjective.title()} {objects[0].title()}"
    opening = GENRE_SCENES[genre][ordinal % len(GENRE_SCENES[genre])]
    setup = [opening, FAMILY_BRIDGES[family]]
    setup.extend(plan["intro"] for plan in plans)
    setup.append(f"The air seemed {atmosphere}, but {characters[2]} kept attention on the work ahead.")

    filler: list[str] = []
    for index in range(128):
        resolution = [primary["resolve"], *(plan["resolve"] for plan in plans[:-1])]
        text = "\n\n".join((" ".join(setup), " ".join(filler), " ".join(resolution))).replace("\n\n\n\n", "\n\n")
        introduced_end = text.index(primary["intro"]) + len(primary["intro"])
        resolved_start = text.index(primary["resolve"])
        distance = len(tokenizer.encode(text[introduced_end:resolved_start]).ids)
        if distance_definition["minimum"] <= distance <= distance_definition["maximum"]:
            break
        if distance > distance_definition["maximum"]:
            raise RuntimeError(f"Could not fit distance bucket {distance_bucket}")
        observer = characters[(ordinal + index) % len(characters)]
        observation = OBSERVATIONS[rng.randrange(len(OBSERVATIONS))]
        action = MINOR_ACTIONS[rng.randrange(len(MINOR_ACTIONS))]
        place = locations[(ordinal + index) % len(locations)]
        remaining = distance_definition["target"] - distance
        if remaining <= 32:
            short_action = SHORT_ACTIONS[(ordinal + index) % len(SHORT_ACTIONS)]
            filler.append(f"Near the {place}, {observer} {short_action}.")
        elif index % 3 == 0:
            detail = FILLER_DETAILS[(ordinal * 5 + index) % len(FILLER_DETAILS)]
            filler.append(detail)
        else:
            filler.append(f"Near the {place}, {observer} noticed {observation} and {action}.")
    else:
        raise RuntimeError(f"Could not reach distance bucket {distance_bucket}")

    text = f"{title}\n\n{text}"
    introduced_end = text.index(primary["intro"]) + len(primary["intro"])
    resolved_start = text.index(primary["resolve"])

    event_specs = [event for plan in plans for event in (*plan["intro_events"], *plan["resolve_events"])]
    events = []
    for spec in event_specs:
        start = text.index(spec["quote"])
        events.append({
            "operation": "set",
            "state_type": spec["state_type"],
            "entity": spec["entity"],
            "attribute": spec["attribute"],
            "value": spec["value"],
            "evidence_quote": spec["quote"],
            "character_start": start,
            "character_end": start + len(spec["quote"]),
            "token_start": len(tokenizer.encode(text[:start]).ids),
            "token_end": len(tokenizer.encode(text[: start + len(spec["quote"])]).ids),
        })
    events.sort(key=lambda event: (event["character_start"], event["character_end"]))
    expected: dict[str, dict[str, Any]] = {}
    for event in events:
        expected.setdefault(event["entity"], {})[event["attribute"]] = event["value"]

    anchors = []
    for plan in plans:
        intro_start = text.index(plan["intro"])
        resolve_start = text.index(plan["resolve"])
        anchors.append({
            "state_type": plan["state_type"],
            "introduced_token": len(tokenizer.encode(text[:intro_start]).ids),
            "resolved_token": len(tokenizer.encode(text[:resolve_start]).ids),
            "introduced_quote": plan["intro"],
            "resolved_quote": plan["resolve"],
        })
    control = _control_for_genre(tokenizer, genre)
    token_count = len(_document_token_ids(tokenizer, title, text, control))
    if token_count > int(config["maximum_document_tokens"]):
        raise RuntimeError(
            f"Generated {token_count}-token story exceeds {config['maximum_document_tokens']} tokens"
        )
    identifier = "ncc-v1-" + sha256_bytes(canonical_json({
        "seed": config["seed"], "split": split, "ordinal": ordinal,
        "genre": genre, "states": state_types, "family": family,
    }).encode("utf-8"))[:20]
    record = {
        "id": identifier,
        "split": split,
        "title": title,
        "text": text,
        "genre": genre,
        "genre_control_token": control,
        "state_types": state_types,
        "difficulty": difficulty,
        "distance_bucket": distance_bucket,
        "anchor_distance_tokens": distance,
        "template_family": family,
        "paraphrase_variant": ordinal % 3,
        "token_count": token_count,
        "word_count": len(normalized_words(text)),
    }
    sidecar = {
        "id": identifier,
        "split": split,
        "schema_version": SCHEMA_VERSION,
        "genre": genre,
        "state_types": state_types,
        "anchors": anchors,
        "events": events,
        "expected_final_state": expected,
        "primary_distance": {
            "state_type": primary["state_type"],
            "tokens": distance,
            "introduced_character_end": introduced_end,
            "resolved_character_start": resolved_start,
        },
        "generation_values": {
            "names": characters,
            "objects": objects,
            "locations": locations,
            "lexical_combination": lexical,
            "template_family": family,
        },
    }
    validate_example(record, sidecar, config=config, tokenizer=tokenizer)
    return record, sidecar


def _validate_config(config: dict[str, Any]) -> None:
    if config["output_path"] != "data/narrative_curriculum_v1":
        raise ValueError("Narrative Curriculum v1 output_path must be data/narrative_curriculum_v1")
    if tuple(config["splits"]["main"].keys()) != MAIN_SPLITS:
        raise ValueError("Main splits must be train, validation, and test in that order")
    if abs(sum(config["splits"]["main"].values()) - 1.0) > 1e-9:
        raise ValueError("Main split fractions must sum to one")
    if config["splits"]["main"] != {"train": 0.9, "validation": 0.05, "test": 0.05}:
        raise ValueError("Main split fractions must be exactly 90/5/5")
    if set(config["holdouts"]) != set(SPLITS):
        raise ValueError(f"Holdout pools must define exactly {SPLITS}")
    dimensions = ("names", "objects", "locations", "lexical_combinations", "template_families")
    for dimension in dimensions:
        seen: dict[str, str] = {}
        for split in SPLITS:
            for value in config["holdouts"][split][dimension]:
                if value in seen:
                    raise ValueError(f"{dimension} value {value!r} belongs to both {seen[value]} and {split}")
                seen[value] = split
    for split in SPLITS:
        pools = config["holdouts"][split]
        if len(pools["names"]) < 3 or len(pools["objects"]) < 2 or len(pools["locations"]) < 2:
            raise ValueError(f"{split} lacks enough values for Level 5 examples")
        unknown = set(pools["template_families"]) - FAMILY_BRIDGES.keys()
        if unknown:
            raise ValueError(f"{split} has unknown template families: {sorted(unknown)}")
    if set(config["distance_buckets"]) != {
        "d064", "d128", "d256", "d384", "d512", "d768", "d900"
    }:
        raise ValueError("Distance buckets must target 64, 128, 256, 384, 512, 768, and 900")
    target = int(config["target"]["main_tokens"])
    minimum, maximum = config["target"]["allowed_main_token_range"]
    if not minimum <= target <= maximum:
        raise ValueError("Main token target is outside the allowed 4-5M range")


def _build_records(config: dict[str, Any], tokenizer: Tokenizer, max_examples: int | None) -> tuple[list[dict], list[dict], int]:
    records: list[dict] = []
    sidecars: list[dict] = []
    exact_hashes: dict[str, str] = {}
    rejected = 0
    main_target = int(config["target"]["main_tokens"])
    targets = {
        split: round(main_target * config["splits"]["main"][split]) for split in MAIN_SPLITS
    }
    targets["generalization_holdout"] = int(config["target"]["generalization_holdout_tokens"])
    ordinals = Counter()
    observed_tokens = Counter()
    cycle = 0
    while True:
        if max_examples is not None:
            if len(records) >= max_examples:
                break
            split = SPLITS[cycle % len(SPLITS)]
            cycle += 1
        else:
            remaining = [split for split in SPLITS if observed_tokens[split] < targets[split]]
            if not remaining:
                break
            split = max(remaining, key=lambda item: targets[item] - observed_tokens[item])
        record, sidecar = generate_example(config, tokenizer, split=split, ordinal=ordinals[split])
        ordinals[split] += 1
        digest = normalized_text_hash(record["text"])
        if digest in exact_hashes:
            rejected += 1
            continue
        exact_hashes[digest] = record["id"]
        records.append(record)
        sidecars.append(sidecar)
        observed_tokens[split] += record["token_count"]
    records.sort(key=lambda record: (SPLITS.index(record["split"]), record["id"]))
    sidecars_by_id = {sidecar["id"]: sidecar for sidecar in sidecars}
    return records, [sidecars_by_id[record["id"]] for record in records], rejected


def build_curriculum(config_path: Path, *, max_examples: int | None = None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    tokenizer_path = Path(config["tokenizer"]["path"])
    tokenizer_hash = sha256_file(tokenizer_path)
    if tokenizer_hash != config["tokenizer"]["expected_sha256"]:
        raise RuntimeError("tokenizer-v1 hash changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    records, sidecars, rejected_duplicates = _build_records(config, tokenizer, max_examples)
    for record, sidecar in zip(records, sidecars, strict=True):
        validate_example(record, sidecar, config=config, tokenizer=tokenizer)
    isolation = validate_split_isolation(records, sidecars)

    exact = exact_duplicate_audit(records)
    similarity = similarity_audit(records, **config["audits"]["similarity"])
    duplicates = {"passed": exact["passed"] and similarity["passed"], "exact": exact,
                  "similarity": similarity, "rejected_during_generation": rejected_duplicates}
    templates = template_audit(records, float(config["audits"]["maximum_template_family_share"]))
    states = state_audit(records, sidecars, list(STATE_TYPES))
    distances = distance_audit(records, config["distance_buckets"])
    quality = quality_audit(
        records,
        required_genres=list(GENRES),
        maximum_document_tokens=int(config["maximum_document_tokens"]),
        minimum_words=int(config["audits"]["minimum_story_words"]),
    )
    audit_reports = {
        "template-audit.json": templates,
        "state-audit.json": states,
        "distance-audit.json": distances,
        "duplicate-audit.json": duplicates,
        "quality-audit.json": quality,
    }
    if max_examples is None:
        failed = [name for name, report in audit_reports.items() if not report["passed"]]
        if failed:
            raise RuntimeError(f"Required curriculum audits failed: {failed}")
        main_tokens = sum(record["token_count"] for record in records if record["split"] in MAIN_SPLITS)
        allowed_minimum, allowed_maximum = config["target"]["allowed_main_token_range"]
        if not allowed_minimum <= main_tokens <= allowed_maximum:
            raise RuntimeError(f"Main curriculum has {main_tokens} tokens outside configured range")

    output = Path(config["output_path"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite curriculum output: {output}")
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    written: list[Path] = []
    for split in SPLITS:
        for suffix, values in (
            (".jsonl", [record for record in records if record["split"] == split]),
            (".state.jsonl", [sidecar for sidecar in sidecars if sidecar["split"] == split]),
        ):
            relative = Path(f"{split}{suffix}")
            write_jsonl(staging / relative, values)
            written.append(relative)

    stats = curriculum_stats(records)
    stats["main_split_tokens"] = sum(
        record["token_count"] for record in records if record["split"] in MAIN_SPLITS
    )
    stats["generalization_holdout_tokens"] = sum(
        record["token_count"] for record in records if record["split"] == "generalization_holdout"
    )
    reports = {"stats.json": stats, **audit_reports}
    for filename, report in reports.items():
        relative = Path(filename)
        write_json_atomic(staging / relative, report)
        written.append(relative)
    manifest = artifact_manifest(staging, written, {
        "curriculum": "Narrative Continuity Curriculum v1",
        "schema_version": SCHEMA_VERSION,
        "seed": config["seed"],
        "config_sha256": sha256_file(config_path),
        "tokenizer_sha256": tokenizer_hash,
        "output_path": config["output_path"],
        "splits": {
            split: {
                "story_count": sum(record["split"] == split for record in records),
                "token_count": sum(record["token_count"] for record in records if record["split"] == split),
                "ids": [record["id"] for record in records if record["split"] == split],
            }
            for split in SPLITS
        },
        "holdout_policy": "all configured names, objects, locations, lexical combinations, and template families are split-exclusive",
        "split_isolation": isolation,
        "development_limit": max_examples,
        "audits_passed": all(report["passed"] for report in audit_reports.values()),
    })
    write_json_atomic(staging / "manifest.json", manifest)
    staging.replace(output)
    return manifest
