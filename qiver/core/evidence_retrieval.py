#!/usr/bin/env python3
"""Retrieve natural-language identity tendencies from the v2 WVS knowledge base."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


MISSING = {"", "Not Answered", "Unknown", "Not specified"}
STRENGTH = {"negligible": 0, "slight": 1, "moderate": 2, "marked": 3}
CONSENSUS = {"mixed": 0, "leading": 1, "clear": 2}
RELIABILITY = {"cautious": 0, "moderate": 1, "strong": 2}

# These fields describe strongly overlapping aspects of the same underlying
# identity.  Treating each as independent evidence can amplify one signal just
# because WVS records it in several columns.
IDENTITY_FAMILIES = {
    **{feature: "migration_background" for feature in ("Q263", "Q264", "Q265", "Q266", "Q267", "Q268", "Q269")},
    "Q289": "religion",
    "Q289CS9": "religion",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _load_protocol(path: str | Path | None) -> tuple[dict[str, Any], list[str]]:
    if not path:
        return {}, []
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return {}, []
    by_question = value.get("by_question", {})
    return (by_question if isinstance(by_question, dict) else {}), [str(q) for q in value.get("source_qids", [])]


def _load_transfer_graph(path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    edges = payload.get("edges", []) if isinstance(payload, dict) else []
    by_target: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        if not isinstance(edge, dict) or not edge.get("transferable", False):
            continue
        target = str(edge.get("target_Q_id", ""))
        source = str(edge.get("source_Q_id", ""))
        if target and source and target != source:
            by_target.setdefault(target, []).append(dict(edge))
    for target in by_target:
        by_target[target].sort(key=lambda edge: (int(edge.get("priority", 999)), str(edge.get("source_Q_id", ""))))
    return by_target


class IdentityTendencyGuidance:
    """Match identities, using stored semantic neighbours for cross-question transfer."""

    def __init__(
        self,
        knowledge_dir: str | Path,
        *,
        cross_question_mapping: str | Path | None = None,
        top_k_questions: int = 3,
        max_combinations: int | None = None,
        max_singles: int | None = None,
        min_cross_question_similarity: float = -1.0,
        deduplicate_identity_families: bool = False,
        transfer_graph: str | Path | None = None,
    ) -> None:
        root = Path(knowledge_dir)
        self.questions = {str(x["Q_id"]): x for x in _load_jsonl(root / "questions.jsonl")}
        self.documents = _load_jsonl(root / "tendency_evidence.jsonl")
        self.by_question: dict[str, list[dict[str, Any]]] = {}
        for document in self.documents:
            self.by_question.setdefault(str(document["Q_id"]), []).append(document)
        neighbor_payload = json.loads((root / "semantic_neighbors.json").read_text(encoding="utf-8"))
        self.neighbors = neighbor_payload.get("by_question", {})
        self.protocol_by_question, self.protocol_sources = _load_protocol(cross_question_mapping)
        self.top_k_questions = max(0, int(top_k_questions))
        self.max_combinations = None if max_combinations is None else max(0, int(max_combinations))
        self.max_singles = None if max_singles is None else max(0, int(max_singles))
        self.min_cross_question_similarity = float(min_cross_question_similarity)
        self.deduplicate_identity_families = bool(deduplicate_identity_families)
        self.transfer_edges = _load_transfer_graph(transfer_graph)

    @staticmethod
    def _matches(document: dict[str, Any], identities: dict[str, Any]) -> bool:
        items = document.get("identities", [])
        if not items:
            return False
        return all(
            str(identities.get(str(item.get("feature", "")), "")) not in MISSING
            and str(identities.get(str(item.get("feature", "")), "")) == str(item.get("value", ""))
            for item in items
        )

    @staticmethod
    def _score(document: dict[str, Any]) -> tuple[int, int, int]:
        return (
            STRENGTH.get(str(document.get("shift_strength", "")), -1),
            RELIABILITY.get(str(document.get("evidence_reliability", "")), -1),
            CONSENSUS.get(str(document.get("group_consensus", "")), -1),
        )

    def _allowed_sources(self, target_qid: str) -> set[str] | None:
        entry = self.protocol_by_question.get(target_qid, {})
        if isinstance(entry, dict) and isinstance(entry.get("source_qids"), list):
            return {str(q) for q in entry["source_qids"]} - {target_qid}
        if self.protocol_sources:
            return set(self.protocol_sources) - {target_qid}
        # No protocol file means leave-one-out over every question in the KB.
        return set(self.questions) - {target_qid}

    def source_qids(self, target_qid: str, *, cross_question: bool) -> list[str]:
        if not cross_question:
            return [target_qid]
        allowed = self._allowed_sources(target_qid)
        if self.transfer_edges:
            return [
                str(edge["source_Q_id"])
                for edge in self.transfer_edges.get(target_qid, [])
                if allowed is None or str(edge["source_Q_id"]) in allowed
            ][: self.top_k_questions]
        ranked = self.neighbors.get(target_qid, [])
        return [
            str(item["Q_id"]) for item in ranked
            if str(item.get("Q_id", "")) != target_qid and (allowed is None or str(item.get("Q_id", "")) in allowed)
            and float(item.get("similarity", -1.0)) >= self.min_cross_question_similarity
        ][: self.top_k_questions]

    @staticmethod
    def _identity_family(document: dict[str, Any]) -> str:
        features = [str(item.get("feature", "")) for item in document.get("identities", [])]
        if len(features) != 1:
            return "|".join(sorted(features))
        return IDENTITY_FAMILIES.get(features[0], features[0])

    def _deduplicate_singles(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.deduplicate_identity_families:
            return documents
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for document in documents:
            family = self._identity_family(document)
            if family in seen:
                continue
            seen.add(family)
            selected.append(document)
        return selected

    def _transfer_edge(self, target_qid: str, source_qid: str) -> dict[str, Any]:
        for edge in self.transfer_edges.get(target_qid, []):
            if str(edge.get("source_Q_id", "")) == source_qid:
                return edge
        return {}

    @staticmethod
    def _orientation_tendency(document: dict[str, Any], edge: dict[str, Any]) -> str:
        values = [str(item.get("value", "")).strip() for item in document.get("identities", [])]
        values = [value for value in values if value]
        subject = (
            f"People identified as {values[0]}" if len(values) == 1
            else f"People matching both {values[0]} and {values[1]}" if len(values) == 2
            else "People in this identity group"
        )
        viewpoint = str(document.get("leading_viewpoint", document.get("leading_option", ""))).strip()
        orientation = str(edge.get("orientation_label", "the shared underlying value orientation")).strip()
        return (
            f"On the related source question, {subject[0].lower() + subject[1:]} most commonly show “{viewpoint}”. "
            f"Treat this only as indirect evidence about {orientation}; it does not determine the current answer. "
            "Do not copy any source-question answer label."
        )

    def render(self, qid: Any, identity_json: Any, *, cross_question: bool = False) -> str:
        target_qid = str(qid).strip()
        identities = _mapping(identity_json)
        blocks: list[str] = []
        for source_qid in self.source_qids(target_qid, cross_question=cross_question):
            matched = [
                d for d in self.by_question.get(source_qid, [])
                if d.get("shift_strength") != "negligible" and self._matches(d, identities)
            ]
            combinations = sorted(
                (d for d in matched if d.get("evidence_type") == "identity_combination"),
                key=self._score, reverse=True,
            )[: self.max_combinations]
            singles = self._deduplicate_singles(sorted(
                (d for d in matched if d.get("evidence_type") == "single_identity"),
                key=self._score, reverse=True,
            ))[: self.max_singles]
            selected = combinations + singles
            if not selected:
                continue
            question = self.questions.get(source_qid, {})
            transfer_edge = self._transfer_edge(target_qid, source_qid) if cross_question else {}
            transfer_type = str(transfer_edge.get("relation_type", "semantic_neighbor"))
            heading = (
                f"Evidence about the current question ({source_qid}):"
                if source_qid == target_qid
                else f"Related source question ({source_qid}, {transfer_type} transfer): {question.get('question_text', '')}"
            )
            lines = [heading]
            for document in selected:
                dimensions = [str(value).lower() for value in document.get("identity_dimensions", []) if str(value)]
                dimension_text = " and ".join(dimensions) or "matching identity"
                prefix = "More specific evidence from" if document.get("evidence_type") == "identity_combination" else "Evidence from"
                tendency = (
                    self._orientation_tendency(document, transfer_edge)
                    if cross_question and transfer_type == "orientation"
                    else str(document.get("tendency_text", ""))
                )
                lines.extend([f"{prefix} {dimension_text}:", tendency])
            blocks.append("\n".join(lines))

        if not blocks:
            return ""
        mode = "indirect evidence transferred from semantically related questions" if cross_question else "evidence learned for the same question"
        return "\n".join([
            "### [Train-only Identity Tendency Knowledge]",
            f"This is {mode}. It describes uncertain group-level tendencies, not this person's known answer.",
            "Use compatible clues together, give combined-identity evidence priority when appropriate, and retain uncertainty when clues conflict.",
            *blocks,
        ])
