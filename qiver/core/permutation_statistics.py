#!/usr/bin/env python3
"""Permutation-based, country-controlled identity significance analysis.

For every non-demographic question and identity field, this script compares the
observed country-controlled JSD with a null distribution obtained by shuffling
that identity *within each country*. Answer labels are never shuffled. Identity
category counts and missing-value patterns are therefore preserved while the
identity/answer association is removed.

The output contains an empirical p-value, a per-question Benjamini-Hochberg
q-value, and the question/identity-specific 95th and 99th percentile null
thresholds. This is deliberately independent of the previously selected
36-question subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEANED = None
DEFAULT_METADATA = None

FEATURES = ["Country"] + [f"Q{i}" for i in range(260, 291)] + ["Q289CS9"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument(
        "--min-effect-js",
        type=float,
        default=0.005,
        help="Practical JSD floor for the strong label; significance alone is not enough.",
    )
    parser.add_argument("--min-question-n", type=int, default=800)
    parser.add_argument("--max-majority", type=float, default=0.90)
    parser.add_argument("--min-country-n", type=int, default=100)
    parser.add_argument("--min-country-group-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--only-qids", default="", help="Comma-separated Q IDs for a smoke test.")
    parser.add_argument("--question-limit", type=int, default=0)
    return parser.parse_args()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = (p + q) / 2.0
    p_mask = p > 0
    q_mask = q > 0
    value = 0.0
    if np.any(p_mask):
        value += 0.5 * float(np.sum(p[p_mask] * np.log(p[p_mask] / midpoint[p_mask])))
    if np.any(q_mask):
        value += 0.5 * float(np.sum(q[q_mask] * np.log(q[q_mask] / midpoint[q_mask])))
    return value


def controlled_jsd(
    identities: np.ndarray,
    answers: np.ndarray,
    countries: np.ndarray,
    answer_count: int,
    *,
    min_country_n: int,
    min_country_group_n: int,
) -> float:
    """Match analyze_identity_distribution_alignment.country_controlled_assoc."""
    weighted = 0.0
    valid_n = 0
    for country in np.unique(countries):
        row_mask = countries == country
        country_answers = answers[row_mask]
        country_n = len(country_answers)
        if country_n < min_country_n:
            continue
        baseline_counts = np.bincount(country_answers, minlength=answer_count)
        baseline = baseline_counts / country_n
        country_ids = identities[row_mask]
        for group in np.unique(country_ids):
            group_answers = country_answers[country_ids == group]
            group_n = len(group_answers)
            if group_n < min_country_group_n:
                continue
            group_counts = np.bincount(group_answers, minlength=answer_count)
            group_dist = group_counts / group_n
            weighted += group_n * js_divergence(group_dist, baseline)
            valid_n += group_n
    return weighted / valid_n if valid_n else 0.0


def empirical_p(observed: float, null_values: np.ndarray) -> float:
    return (1.0 + float(np.count_nonzero(null_values >= observed))) / (len(null_values) + 1.0)


def bh_qvalues(pvalues: Iterable[float]) -> list[float]:
    values = np.asarray(list(pvalues), dtype=float)
    if len(values) == 0:
        return []
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for rank in range(len(values), 0, -1):
        index = order[rank - 1]
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def load_rows(path: Path, qids: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    columns = ["country_name"] + [f"identity__{feature}" for feature in FEATURES]
    columns += [f"answer__{qid}" for qid in qids]
    result = {column: [] for column in columns}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [column for column in columns if column not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(f"Missing columns in {path}: {missing[:10]}")
        for row in reader:
            for column in columns:
                result[column].append(row.get(column, "Not Answered"))
    return result, columns


def encode(values: list[str]) -> np.ndarray:
    # Stable coding makes smoke-test output reproducible across Python runs.
    mapping: dict[str, int] = {}
    encoded = np.empty(len(values), dtype=np.int32)
    for index, value in enumerate(values):
        if value not in mapping:
            mapping[value] = len(mapping)
        encoded[index] = mapping[value]
    return encoded


def question_ids(metadata_path: Path, answer_columns: list[str]) -> list[str]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    questions = metadata.get("target_questions", metadata) if isinstance(metadata, dict) else metadata
    demographic = {
        item.get("Q_id", item.get("qid"))
        for item in questions
        if item.get("value_dim") == "DEMOGRAPHICS" or item.get("category") == "DEMOGRAPHICS"
    }
    demographic.discard(None)
    available = {column.removeprefix("answer__") for column in answer_columns}
    return sorted(available - demographic)


def analyse_question(
    qid: str,
    rows: dict[str, list[str]],
    *,
    permutations: int,
    quantile: float,
    min_question_n: int,
    max_majority: float,
    min_country_n: int,
    min_country_group_n: int,
    min_effect_js: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    raw_answers = np.asarray(rows[f"answer__{qid}"], dtype=object)
    valid = (raw_answers != "") & (raw_answers != "Not Answered")
    if not np.any(valid):
        return [], None
    answers = encode(raw_answers[valid].tolist())
    countries = encode(np.asarray(rows["country_name"], dtype=object)[valid].tolist())
    answer_counts = np.bincount(answers)
    n = len(answers)
    majority = float(answer_counts.max() / n)
    if n < min_question_n or majority >= max_majority:
        return [], {"Q_id": qid, "n": n, "majority": majority, "eligible": False}

    output: list[dict[str, object]] = []
    tested_pvalues: list[float] = []
    tested_features: list[int] = []
    for feature_index, feature in enumerate(FEATURES):
        identity_values = np.asarray(rows[f"identity__{feature}"], dtype=object)[valid].tolist()
        identities = encode(identity_values)
        if feature == "Country":
            output.append(
                {
                    "Q_id": qid,
                    "feature": feature,
                    "observed_controlled_js": 0.0,
                    "null_mean": "",
                    "null_std": "",
                    "null_threshold": "",
                    "null_threshold_99": "",
                    "empirical_p": "",
                    "fdr_q_value": "",
                    "signal": "not_tested_country",
                    "n": n,
                    "answers": len(answer_counts),
                    "majority": majority,
                }
            )
            continue

        observed = controlled_jsd(
            identities, answers, countries, len(answer_counts),
            min_country_n=min_country_n,
            min_country_group_n=min_country_group_n,
        )
        null_values = np.empty(permutations, dtype=float)
        for permutation_index in range(permutations):
            shuffled = identities.copy()
            for country in np.unique(countries):
                country_indices = np.flatnonzero(countries == country)
                shuffled[country_indices] = identities[rng.permutation(country_indices)]
            null_values[permutation_index] = controlled_jsd(
                shuffled, answers, countries, len(answer_counts),
                min_country_n=min_country_n,
                min_country_group_n=min_country_group_n,
            )
        pvalue = empirical_p(observed, null_values)
        tested_features.append(len(output))
        tested_pvalues.append(pvalue)
        output.append(
            {
                "Q_id": qid,
                "feature": feature,
                "observed_controlled_js": observed,
                "null_mean": float(null_values.mean()),
                "null_std": float(null_values.std(ddof=1)) if permutations > 1 else 0.0,
                "null_threshold": float(np.quantile(null_values, quantile)),
                "null_threshold_99": float(np.quantile(null_values, 0.99)),
                "empirical_p": pvalue,
                "fdr_q_value": "",
                "signal": "not_significant",
                "n": n,
                "answers": len(answer_counts),
                "majority": majority,
            }
        )

    qvalues = bh_qvalues(tested_pvalues)
    for output_index, qvalue in zip(tested_features, qvalues):
        output[output_index]["fdr_q_value"] = qvalue
        observed = float(output[output_index]["observed_controlled_js"])
        threshold = float(output[output_index]["null_threshold"])
        output[output_index]["signal"] = "supported" if qvalue < 0.05 and observed > threshold else "not_significant"
        if qvalue < 0.05 and observed > threshold and observed >= min_effect_js:
            output[output_index]["signal"] = "strong"

    summary = {
        "Q_id": qid,
        "n": n,
        "majority": majority,
        "eligible": True,
        "tested_identities": len(tested_pvalues),
        "supported_identities": sum(row["signal"] in {"supported", "strong"} for row in output),
        "strong_identities": sum(row["signal"] == "strong" for row in output),
    }
    return output, summary


def main() -> None:
    args = parse_args()
    if args.permutations < 99:
        raise ValueError("Use at least 99 permutations; 1999 is recommended for the formal analysis.")
    if not 0.5 < args.quantile < 1.0:
        raise ValueError("--quantile must be between 0.5 and 1.0")
    if not args.cleaned_csv.is_file():
        raise FileNotFoundError(args.cleaned_csv)
    if not args.metadata.is_file():
        raise FileNotFoundError(args.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.cleaned_csv.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        answer_columns = [column for column in (reader.fieldnames or []) if column.startswith("answer__")]
    qids = question_ids(args.metadata, answer_columns)
    if args.only_qids:
        allowed = {qid.strip() for qid in args.only_qids.split(",") if qid.strip()}
        qids = [qid for qid in qids if qid in allowed]
    if args.question_limit:
        qids = qids[: args.question_limit]
    if not qids:
        raise ValueError("No questions selected")

    rows, _ = load_rows(args.cleaned_csv, qids)
    rng = np.random.default_rng(args.seed)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for index, qid in enumerate(qids, start=1):
        print(f"[{index}/{len(qids)}] {qid}: {args.permutations} permutations × {len(FEATURES) - 1} identities", flush=True)
        result_rows, summary = analyse_question(
            qid, rows,
            permutations=args.permutations,
            quantile=args.quantile,
            min_question_n=args.min_question_n,
            max_majority=args.max_majority,
            min_country_n=args.min_country_n,
            min_country_group_n=args.min_country_group_n,
            min_effect_js=args.min_effect_js,
            rng=rng,
        )
        all_rows.extend(result_rows)
        if summary is not None:
            summaries.append(summary)

    result_fields = [
        "Q_id", "feature", "observed_controlled_js", "null_mean", "null_std",
        "null_threshold", "null_threshold_99", "empirical_p", "fdr_q_value",
        "signal", "n", "answers", "majority",
    ]
    with (args.output_dir / "permutation_identity_significance.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output_dir / "question_permutation_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["Q_id", "n", "majority", "eligible", "tested_identities", "supported_identities", "strong_identities"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    config = {
        "cleaned_csv": str(args.cleaned_csv),
        "metadata": str(args.metadata),
        "permutations": args.permutations,
        "quantile": args.quantile,
        "min_effect_js": args.min_effect_js,
        "seed": args.seed,
        "within_country_shuffle": True,
        "answer_labels_shuffled": False,
        "features": FEATURES,
        "multiple_testing": "Benjamini-Hochberg within each question over the 32 non-Country identities",
        "selection": "fdr_q_value < 0.05 and observed_controlled_js > null_threshold",
        "strong_selection": "selection plus observed_controlled_js >= min_effect_js",
    }
    (args.output_dir / "permutation_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {args.output_dir / 'permutation_identity_significance.csv'}")
    print(f"saved: {args.output_dir / 'question_permutation_summary.csv'}")


if __name__ == "__main__":
    main()
