#!/usr/bin/env python3
"""Build train-only identity evidence documents for WVS value questions.

Only identity signals confirmed by the two-stage permutation analysis are
materialized. Single-identity and combination evidence are separate documents,
so retrieval can fall back from an unseen combination to its components.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
MISSING = {"", "Not Answered"}


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_signals(path: Path) -> dict[str, dict[str, set[Any]]]:
    signals: dict[str, dict[str, set[Any]]] = defaultdict(
        lambda: {"single": set(), "combination": set()}
    )
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Q_id", "view", "feature", "supported"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Signal summary lacks columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            if not truthy(row["supported"]):
                continue
            qid, view = row["Q_id"], row["view"]
            # Country is a control variable, not retrieval evidence. Questions
            # supported only by country_direct are intentionally left out.
            if view == "country_controlled":
                signals[qid]["single"].add(row["feature"])
            elif view == "combination_synergy":
                parts = tuple(part.strip() for part in row["feature"].split("+") if part.strip())
                if len(parts) == 2:
                    signals[qid]["combination"].add(parts)
    return signals


def load_question_metadata(cleaned_path: Path, rich_path: Path) -> dict[str, dict[str, Any]]:
    cleaned = json.loads(cleaned_path.read_text(encoding="utf-8"))
    rich = {item["Q_id"]: item for item in json.loads(rich_path.read_text(encoding="utf-8"))}
    output = {}
    for item in cleaned.get("target_questions", []):
        qid = str(item["qid"])
        extra = rich.get(qid, {})
        output[qid] = {
            "Q_id": qid,
            "question_text": item.get("question", extra.get("question_text", "")),
            "value_domain": item.get("category", extra.get("value_domain", "")),
            "answer_options": [str(value) for value in item.get("options", [])],
            "model_scale": extra.get("model_scale", {}),
            "simplified": bool(item.get("bucketed_wide_scale", extra.get("simplified", False))),
        }
    return output


def distribution(counts: Counter, options: list[str], total: int) -> dict[str, float]:
    return {option: counts[option] / total for option in options}


def evidence_document(
    qmeta: dict[str, Any], kind: str, identities: list[dict[str, str]], counts: Counter,
    baseline: dict[str, float], options: list[str], prior_strength: float,
) -> dict[str, Any]:
    n = sum(counts.values())
    raw = distribution(counts, options, n)
    smoothed = {
        option: (counts[option] + prior_strength * baseline[option]) / (n + prior_strength)
        for option in options
    }
    identity_key = "|".join(f"{item['feature']}={item['value']}" for item in identities)
    return {
        "doc_id": f"{qmeta['Q_id']}::{kind}::{identity_key}",
        "Q_id": qmeta["Q_id"],
        "question_text": qmeta["question_text"],
        "value_domain": qmeta["value_domain"],
        "answer_options": options,
        "evidence_type": kind,
        "identities": identities,
        "sample_size": n,
        "reliability": n / (n + prior_strength),
        "raw_answer_distribution": raw,
        "answer_distribution": smoothed,
        "baseline_distribution": baseline,
        "distribution_shift": {option: smoothed[option] - baseline[option] for option in options},
        "usage_note": "Probabilistic group evidence, not a deterministic answer; combine with other retrieved evidence.",
    }


def build_documents(
    rows: Iterable[dict[str, str]], signals: dict[str, dict[str, set[Any]]],
    questions: dict[str, dict[str, Any]], min_group_size: int = 30, prior_strength: float = 20.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    documents, baselines = [], []
    for qid, selected in sorted(signals.items()):
        if qid not in questions:
            continue
        qmeta, answer_column = questions[qid], f"answer__{qid}"
        options = list(qmeta["answer_options"])
        valid = [row for row in rows if row.get(answer_column, "") not in MISSING]
        counts = Counter(row[answer_column] for row in valid)
        # Preserve metadata order, but retain an unexpected decoded category for auditability.
        options.extend(sorted(set(counts) - set(options)))
        if not valid:
            continue
        baseline = distribution(counts, options, len(valid))
        baselines.append({**qmeta, "sample_size": len(valid), "baseline_distribution": baseline})

        for feature in sorted(selected["single"]):
            column = f"identity__{feature}"
            groups: dict[str, Counter] = defaultdict(Counter)
            for row in valid:
                value = row.get(column, "")
                if value not in MISSING:
                    groups[value][row[answer_column]] += 1
            for value, group_counts in sorted(groups.items()):
                if sum(group_counts.values()) >= min_group_size:
                    documents.append(evidence_document(qmeta, "single_identity", [{"feature": feature, "value": value}], group_counts, baseline, options, prior_strength))

        for first, second in sorted(selected["combination"]):
            columns = f"identity__{first}", f"identity__{second}"
            groups: dict[tuple[str, str], Counter] = defaultdict(Counter)
            for row in valid:
                values = row.get(columns[0], ""), row.get(columns[1], "")
                if all(value not in MISSING for value in values):
                    groups[values][row[answer_column]] += 1
            for values, group_counts in sorted(groups.items()):
                if sum(group_counts.values()) >= min_group_size:
                    identities = [{"feature": first, "value": values[0]}, {"feature": second, "value": values[1]}]
                    documents.append(evidence_document(qmeta, "identity_combination", identities, group_counts, baseline, options, prior_strength))
    return documents, baselines


def write_outputs(output_dir: Path, documents: list[dict[str, Any]], baselines: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in [("evidence.jsonl", documents), ("question_baselines.jsonl", baselines)]:
        with (output_dir / name).open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    database = output_dir / "identity_value_knowledge.sqlite"
    if database.exists():
        database.unlink()
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE evidence (doc_id TEXT PRIMARY KEY, q_id TEXT NOT NULL, evidence_type TEXT NOT NULL, identity_key TEXT NOT NULL, sample_size INTEGER NOT NULL, payload_json TEXT NOT NULL);
            CREATE INDEX evidence_lookup ON evidence(q_id, evidence_type, identity_key);
            CREATE TABLE question_baselines (q_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL);
        """)
        connection.executemany(
            "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?)",
            [(d["doc_id"], d["Q_id"], d["evidence_type"], "|".join(f"{x['feature']}={x['value']}" for x in d["identities"]), d["sample_size"], json.dumps(d, ensure_ascii=False)) for d in documents],
        )
        connection.executemany("INSERT INTO question_baselines VALUES (?, ?)", [(d["Q_id"], json.dumps(d, ensure_ascii=False)) for d in baselines])
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--signal-summary", type=Path, required=True)
    parser.add_argument("--cleaned-metadata", type=Path, required=True)
    parser.add_argument("--question-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-group-size", type=int, default=30)
    parser.add_argument("--prior-strength", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signals = load_signals(args.signal_summary)
    questions = load_question_metadata(args.cleaned_metadata, args.question_metadata)
    with args.train_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    documents, baselines = build_documents(rows, signals, questions, args.min_group_size, args.prior_strength)
    manifest = {
        "knowledge_scope": "train_only", "train_csv": str(args.train_csv), "signal_summary": str(args.signal_summary),
        "respondent_count": len(rows), "question_selection": "at_least_one_supported_non_country_identity_or_combination",
        "country_role": "control_only_not_retrieval_evidence",
        "question_count": len(baselines), "evidence_document_count": len(documents),
        "single_document_count": sum(d["evidence_type"] == "single_identity" for d in documents),
        "combination_document_count": sum(d["evidence_type"] == "identity_combination" for d in documents),
        "min_group_size": args.min_group_size, "prior_strength": args.prior_strength,
    }
    write_outputs(args.output_dir, documents, baselines, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"saved: {args.output_dir}")


if __name__ == "__main__":
    main()
