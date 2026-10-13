#!/usr/bin/env python3
"""Retrieve train-only WVS identity-distribution evidence for prompting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


MISSING = {"", "Not Answered", "Unknown", "Not specified"}


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_cross_mapping(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("by_question"), dict):
        return value["by_question"]
    return value if isinstance(value, dict) else {}


class IdentityKnowledgeGuidance:
    """Match a respondent's identities to single and combination documents."""

    def __init__(
        self,
        knowledge_dir: str | Path,
        *,
        cross_question_mapping: str | Path | None = None,
        max_combinations: int | None = None,
        max_singles: int | None = None,
    ) -> None:
        root = Path(knowledge_dir)
        self.documents = _load_jsonl(root / "evidence.jsonl")
        self.baselines = {
            str(item["Q_id"]): item
            for item in _load_jsonl(root / "question_baselines.jsonl")
        }
        self.by_question: dict[str, list[dict[str, Any]]] = {}
        for document in self.documents:
            self.by_question.setdefault(str(document["Q_id"]), []).append(document)
        self.cross_mapping = _load_cross_mapping(cross_question_mapping)
        self.max_combinations = None if max_combinations is None else max(0, int(max_combinations))
        self.max_singles = None if max_singles is None else max(0, int(max_singles))

    @staticmethod
    def _score(document: dict[str, Any]) -> tuple[float, float, int]:
        shift = max((abs(float(value)) for value in document.get("distribution_shift", {}).values()), default=0.0)
        return float(document.get("reliability", 0.0)), shift, int(document.get("sample_size", 0))

    @staticmethod
    def _matches(document: dict[str, Any], identities: dict[str, Any]) -> bool:
        for item in document.get("identities", []):
            feature, expected = str(item.get("feature", "")), str(item.get("value", ""))
            actual = str(identities.get(feature, ""))
            if actual in MISSING or actual != expected:
                return False
        return bool(document.get("identities"))

    def _source_qids(self, qid: str, cross_question: bool) -> list[str]:
        if not cross_question:
            return [qid]
        entry = self.cross_mapping.get(qid, {})
        if isinstance(entry, dict):
            qids = entry.get("retrieved_qids", [])
            if isinstance(qids, list):
                return [str(value) for value in qids if str(value) != qid]
        if isinstance(entry, list):
            return [str(value) for value in entry if str(value) != qid]
        # Never fall back to target-question evidence in cross-question mode.
        return []

    @staticmethod
    def _format_distribution(values: Mapping[str, Any]) -> str:
        return "; ".join(f"{option}: {100 * float(probability):.1f}%" for option, probability in values.items())

    def render(self, qid: Any, identity_json: Any, *, cross_question: bool = False) -> str:
        target_qid = str(qid).strip()
        identities = _mapping(identity_json)
        source_qids = self._source_qids(target_qid, cross_question)
        if not source_qids:
            return ""

        blocks: list[str] = []
        for source_qid in source_qids:
            matched = [d for d in self.by_question.get(source_qid, []) if self._matches(d, identities)]
            combinations = sorted(
                (d for d in matched if d.get("evidence_type") == "identity_combination"),
                key=self._score,
                reverse=True,
            )[: self.max_combinations]
            singles = sorted(
                (d for d in matched if d.get("evidence_type") == "single_identity"),
                key=self._score,
                reverse=True,
            )[: self.max_singles]
            baseline = self.baselines.get(source_qid)
            if not combinations and not singles and not baseline:
                continue

            heading = f"Evidence for current question {source_qid}:" if source_qid == target_qid else f"Related source question {source_qid}: {baseline.get('question_text', '') if baseline else ''}"
            lines = [heading]
            if baseline:
                lines.append(
                    f"- Training-population baseline (n={baseline.get('sample_size', 0)}): "
                    + self._format_distribution(baseline.get("baseline_distribution", {}))
                )
            for document in combinations + singles:
                label = " × ".join(f"{item['feature']}={item['value']}" for item in document["identities"])
                lines.append(
                    f"- Matching {document['evidence_type']} [{label}] (n={document['sample_size']}, reliability={document['reliability']:.2f}): "
                    + self._format_distribution(document.get("answer_distribution", {}))
                )
            blocks.append("\n".join(lines))

        if not blocks:
            return ""
        mode = "related-question transfer evidence" if cross_question else "same-question group evidence"
        return "\n".join([
            "### [Train-only Identity Distribution Evidence]",
            f"This is {mode}. It describes probabilistic group tendencies, not this person's known answer.",
            "Combine matching combination, single-identity, and baseline evidence with the complete profile. Do not mechanically choose the largest percentage.",
            *blocks,
        ])
