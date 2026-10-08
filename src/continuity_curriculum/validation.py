"""Deterministic prose-evidence and event-state validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


class ValidationError(ValueError):
    """Raised when generated prose and its sidecar cannot be reconciled."""


def _document_length(tokenizer: Any, record: dict[str, Any]) -> int:
    ids = [tokenizer.token_to_id("<|story|>")]
    control = record.get("genre_control_token")
    if control is not None:
        control_id = tokenizer.token_to_id(control)
        if control_id is None:
            raise ValidationError(f"{record['id']} declares a nonexistent genre control token")
        ids.append(control_id)
    ids.append(tokenizer.token_to_id("<|bos|>"))
    if not record["text"].startswith(record["title"] + "\n\n"):
        raise ValidationError(f"{record['id']} text does not begin with its title")
    ids.extend(tokenizer.encode(record["text"]).ids)
    ids.append(tokenizer.token_to_id("<|eos|>"))
    if any(value is None for value in ids):
        raise ValidationError("Tokenizer is missing required document tokens")
    return len(ids)


def _validate_state_semantics(record_id: str, state_type: str, events: list[dict]) -> None:
    attributes = {event["attribute"] for event in events}
    values_by_attribute: dict[str, list[Any]] = defaultdict(list)
    for event in events:
        values_by_attribute[event["attribute"]].append(event["value"])
    if state_type == "character_identity_role_relationship":
        valid = {"identity", "role"} <= attributes and any(item.startswith("relationship_to:") for item in attributes)
    elif state_type == "ownership_transfer_hiding_retrieval":
        valid = {"holder", "hidden_at", "retrieved"} <= attributes and len(values_by_attribute["holder"]) >= 2
    elif state_type == "physical_scene_location_movement":
        valid = {"previous_location", "location"} <= attributes and len(events) >= 2
    elif state_type == "goal_persistence_completion_failure":
        valid = "goal" in attributes and values_by_attribute["goal_status"][0] == "active" and values_by_attribute["goal_status"][-1] in {"completed", "failed"}
    elif state_type == "knowledge_ignorance_secret":
        knowledge_values = [value for attribute, values in values_by_attribute.items() if attribute.startswith("knows:") for value in values]
        valid = False in knowledge_values and True in knowledge_values
    elif state_type == "cause_effect":
        valid = "occurred" in attributes and "caused_by" in attributes
    elif state_type == "promise_debt_obligation":
        obligations = [value for attribute, values in values_by_attribute.items() if attribute.startswith("obligation_to:") for value in values]
        valid = obligations[:1] == ["outstanding"] and obligations[-1:] in (["fulfilled"], ["outstanding"])
    elif state_type == "persistent_injury":
        valid = "injury" in attributes and set(values_by_attribute["injury_status"]) == {"unhealed"}
    elif state_type == "temporal_ordering":
        valid = {"first", "second", "third"} <= attributes
    else:
        valid = False
    if not valid:
        raise ValidationError(f"{record_id} does not reconstruct declared state {state_type}")


def validate_example(
    record: dict[str, Any],
    sidecar: dict[str, Any],
    *,
    config: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    required = {
        "id", "split", "title", "text", "genre", "genre_control_token", "state_types",
        "difficulty", "distance_bucket", "anchor_distance_tokens", "template_family",
        "paraphrase_variant", "token_count", "word_count",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValidationError(f"{record.get('id', '<unknown>')} missing fields: {missing}")
    if record["id"] != sidecar.get("id") or record["split"] != sidecar.get("split"):
        raise ValidationError(f"{record['id']} record/sidecar identity mismatch")
    if record["state_types"] != sidecar.get("state_types") or record["genre"] != sidecar.get("genre"):
        raise ValidationError(f"{record['id']} record/sidecar metadata mismatch")
    if record["difficulty"] not in range(1, 6) or len(record["state_types"]) != record["difficulty"]:
        raise ValidationError(f"{record['id']} difficulty does not match declared state count")
    if len(record["state_types"]) != len(set(record["state_types"])):
        raise ValidationError(f"{record['id']} repeats a declared state type")
    if "\n\n" not in record["text"] or "<|" in record["text"]:
        raise ValidationError(f"{record['id']} is not ordinary paragraph-form fiction")
    if record["token_count"] != _document_length(tokenizer, record):
        raise ValidationError(f"{record['id']} token count is incorrect")
    if record["token_count"] > int(config["maximum_document_tokens"]):
        raise ValidationError(f"{record['id']} exceeds model context")

    events = sidecar.get("events")
    if not isinstance(events, list) or not events:
        raise ValidationError(f"{record['id']} has no replayable events")
    replayed: dict[str, dict[str, Any]] = {}
    prior_start = -1
    prior_end = -1
    events_by_type: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        start = event.get("character_start")
        end = event.get("character_end")
        quote = event.get("evidence_quote")
        if event.get("operation") != "set" or not isinstance(start, int) or not isinstance(end, int):
            raise ValidationError(f"{record['id']} has malformed event data")
        shared = start == prior_start and end == prior_end
        if (start < prior_end and not shared) or record["text"][start:end] != quote:
            raise ValidationError(f"{record['id']} event evidence is absent or out of order")
        token_start = len(tokenizer.encode(record["text"][:start]).ids)
        token_end = len(tokenizer.encode(record["text"][:end]).ids)
        if (event.get("token_start"), event.get("token_end")) != (token_start, token_end):
            raise ValidationError(f"{record['id']} event token offsets are incorrect")
        state_type = event.get("state_type")
        if state_type not in record["state_types"]:
            raise ValidationError(f"{record['id']} event has undeclared state type")
        events_by_type[state_type].append(event)
        replayed.setdefault(event["entity"], {})[event["attribute"]] = event["value"]
        prior_start, prior_end = start, end
    if replayed != sidecar.get("expected_final_state"):
        raise ValidationError(f"{record['id']} final state differs from event replay")
    for state_type in record["state_types"]:
        _validate_state_semantics(record["id"], state_type, events_by_type[state_type])

    anchors = sidecar.get("anchors")
    if not isinstance(anchors, list) or {item.get("state_type") for item in anchors} != set(record["state_types"]):
        raise ValidationError(f"{record['id']} anchors do not cover all declared states")
    for anchor in anchors:
        intro_character = record["text"].find(anchor["introduced_quote"])
        resolved_character = record["text"].find(anchor["resolved_quote"])
        if intro_character < 0 or resolved_character <= intro_character:
            raise ValidationError(f"{record['id']} anchor evidence is absent or reversed")
        introduced_token = len(tokenizer.encode(record["text"][:intro_character]).ids)
        resolved_token = len(tokenizer.encode(record["text"][:resolved_character]).ids)
        if (anchor["introduced_token"], anchor["resolved_token"]) != (introduced_token, resolved_token):
            raise ValidationError(f"{record['id']} anchor token positions are incorrect")

    primary = sidecar.get("primary_distance", {})
    start = primary.get("introduced_character_end")
    end = primary.get("resolved_character_start")
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        raise ValidationError(f"{record['id']} primary distance offsets are invalid")
    distance = len(tokenizer.encode(record["text"][start:end]).ids)
    definition = config["distance_buckets"][record["distance_bucket"]]
    if distance != primary.get("tokens") or distance != record["anchor_distance_tokens"]:
        raise ValidationError(f"{record['id']} anchor distance is incorrect")
    if not definition["minimum"] <= distance <= definition["maximum"]:
        raise ValidationError(f"{record['id']} anchor distance is outside its bucket")

    if record["difficulty"] == 5:
        mandatory = {
            "ownership_transfer_hiding_retrieval",
            "physical_scene_location_movement",
            "goal_persistence_completion_failure",
        }
        values = sidecar["generation_values"]
        if not mandatory <= set(record["state_types"]):
            raise ValidationError(f"{record['id']} Level 5 lacks mandatory compound states")
        if len(values["names"]) < 3 or len(values["objects"]) < 2 or len(values["locations"]) < 2:
            raise ValidationError(f"{record['id']} Level 5 lacks multiple entities")
    return {"id": record["id"], "passed": True, "event_count": len(events)}


def validate_split_isolation(records: list[dict], sidecars: list[dict]) -> dict[str, Any]:
    if len(records) != len(sidecars):
        raise ValidationError("Record and sidecar counts differ")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValidationError("Record IDs are not unique")
    by_id = {sidecar["id"]: sidecar for sidecar in sidecars}
    if set(ids) != set(by_id):
        raise ValidationError("Record and sidecar IDs differ")
    dimensions = ("names", "objects", "locations", "lexical_combination", "template_family")
    leakage: dict[str, list[dict]] = {}
    for dimension in dimensions:
        owners: dict[str, set[str]] = defaultdict(set)
        for record in records:
            values = by_id[record["id"]]["generation_values"]
            observed = values[dimension] if isinstance(values[dimension], list) else [values[dimension]]
            for value in observed:
                owners[value].add(record["split"])
        leakage[dimension] = [
            {"value": value, "splits": sorted(splits)}
            for value, splits in sorted(owners.items()) if len(splits) > 1
        ]
    failed = {dimension: items for dimension, items in leakage.items() if items}
    if failed:
        raise ValidationError(f"Split-exclusive value leakage: {failed}")
    return {"passed": True, "dimensions": leakage}
