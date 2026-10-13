#!/usr/bin/env python3
"""Unified subsampled permutation analysis for three identity signals.

The script uses the same country-stratified respondent subsamples for three
views of identity information:

1. country-direct JSD: country groups versus the overall answer distribution;
2. direct JSD: non-country identity groups versus the overall answer distribution;
3. country-controlled JSD: the same statistic computed within countries;
4. combination synergy: a country-controlled identity combination minus its
   strongest single component.

For country-direct and direct JSD, answers are permuted globally. For country-controlled JSD,
the tested identity is permuted within country. For synergy, answers are
permuted within country while all identity fields remain fixed; this preserves
the correlation structure among identity fields while removing their answer
association.

This is an exploratory/screening implementation. Combination tests are run
for pairs among the top ``--combo-top-k`` single identities in each question.
For a confirmatory result, pre-register a small set of combinations or rerun
the reported candidates on the full data with more permutations.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .permutation_statistics import (
    FEATURES,
    controlled_jsd,
    encode,
    js_divergence,
    load_rows,
    question_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
NON_COUNTRY_FEATURES = tuple(feature for feature in FEATURES if feature != "Country")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-per-country", type=int, default=1000)
    parser.add_argument("--subsample-repeats", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise significance level used by the multiple-testing correction.",
    )
    parser.add_argument(
        "--correction",
        choices=("bonferroni", "bh"),
        default="bonferroni",
        help="Multiple-testing correction within each question/subsample/view.",
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--min-question-n", type=int, default=800)
    parser.add_argument("--max-majority", type=float, default=0.90)
    parser.add_argument("--min-country-n", type=int, default=100)
    parser.add_argument("--min-country-group-n", type=int, default=30)
    parser.add_argument("--min-group-n", type=int, default=50)
    parser.add_argument("--combo-top-k", type=int, default=6)
    parser.add_argument("--min-combo-group-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--min-support-rate",
        type=float,
        default=0.60,
        help="Minimum fraction of subsamples in which an identity is supported.",
    )
    parser.add_argument(
        "--min-effect-js",
        type=float,
        default=0.005,
        help="Minimum median observed JSD for the compact 'relevant' summary.",
    )
    parser.add_argument(
        "--top-identities",
        type=int,
        default=5,
        help="Number of identities/combinations listed per question and view.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-question progress details.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes. 0 automatically uses min(CPU count, question count).",
    )
    parser.add_argument("--only-qids", default="", help="Comma-separated Q IDs for a smoke test.")
    parser.add_argument("--question-limit", type=int, default=0)
    return parser.parse_args()


_WORKER_ROWS: dict[str, list[str]] | None = None


def _init_worker(rows: dict[str, list[str]]) -> None:
    """Install the read-only row table once per forked worker."""
    global _WORKER_ROWS
    _WORKER_ROWS = rows


def _analyse_task(task: tuple[str, np.ndarray, int, argparse.Namespace, int]) -> list[dict[str, object]]:
    qid, sample_indices, repeat, args, seed = task
    if _WORKER_ROWS is None:
        raise RuntimeError("parallel worker was not initialized")
    return analyse_q(
        qid,
        _WORKER_ROWS,
        sample_indices,
        repeat=repeat,
        args=args,
        rng=np.random.default_rng(seed),
    )


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


def direct_jsd(
    identities: np.ndarray,
    answers: np.ndarray,
    answer_count: int,
    *,
    min_group_n: int,
) -> float:
    baseline_counts = np.bincount(answers, minlength=answer_count)
    baseline = baseline_counts / len(answers)
    weighted = 0.0
    valid_n = 0
    for group in np.unique(identities):
        group_answers = answers[identities == group]
        if len(group_answers) < min_group_n:
            continue
        group_dist = np.bincount(group_answers, minlength=answer_count) / len(group_answers)
        weighted += len(group_answers) * js_divergence(group_dist, baseline)
        valid_n += len(group_answers)
    return weighted / valid_n if valid_n else 0.0


def shuffle_within_groups(values: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = values.copy()
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        shuffled[indices] = values[rng.permutation(indices)]
    return shuffled


def sample_question_by_country(
    rows: dict[str, list[str]], qid: str, n_per_country: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample valid answers per country so each question has balanced exposure."""
    answers = rows[f"answer__{qid}"]
    countries = rows["country_name"]
    by_country: dict[str, list[int]] = defaultdict(list)
    for index, (answer, country) in enumerate(zip(answers, countries)):
        if answer not in ("", "Not Answered"):
            by_country[country].append(index)
    selected: list[int] = []
    for country in sorted(by_country):
        indices = np.asarray(by_country[country], dtype=np.int64)
        count = min(n_per_country, len(indices))
        selected.extend(rng.choice(indices, size=count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def pvalue(observed: float, null_values: np.ndarray) -> float:
    return (1.0 + float(np.count_nonzero(null_values >= observed))) / (len(null_values) + 1.0)


def metric_row(
    *, qid: str, repeat: int, view: str, feature: str, observed: float,
    null_values: np.ndarray | None, quantile: float, n: int, majority: float,
    combo_components: str = "",
) -> dict[str, object]:
    if null_values is None:
        return {
            "Q_id": qid, "subsample": repeat, "view": view, "feature": feature,
            "components": combo_components, "n": n, "majority": majority,
            "observed": observed, "null_mean": "", "null_std": "",
            "null_threshold": "", "null_threshold_99": "", "empirical_p": "",
            "fdr_q_value": "", "signal": "observed_only",
        }
    return {
        "Q_id": qid, "subsample": repeat, "view": view, "feature": feature,
        "components": combo_components, "n": n, "majority": majority,
        "observed": observed, "null_mean": float(null_values.mean()),
        "null_std": float(null_values.std(ddof=1)) if len(null_values) > 1 else 0.0,
        "null_threshold": float(np.quantile(null_values, quantile)),
        "null_threshold_99": float(np.quantile(null_values, 0.99)),
        "empirical_p": pvalue(observed, null_values), "fdr_q_value": "",
        "signal": "not_significant",
    }


def analyse_q(
    qid: str,
    rows: dict[str, list[str]],
    sample_indices: np.ndarray,
    *,
    repeat: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    raw_answers = np.asarray(rows[f"answer__{qid}"], dtype=object)[sample_indices]
    valid = (raw_answers != "") & (raw_answers != "Not Answered")
    if not np.any(valid):
        return []
    answers = encode(raw_answers[valid].tolist())
    countries = encode(np.asarray(rows["country_name"], dtype=object)[sample_indices][valid].tolist())
    n = len(answers)
    answer_counts = np.bincount(answers)
    majority = float(answer_counts.max() / n)
    if n < args.min_question_n or majority >= args.max_majority:
        return []

    identities = {
        feature: encode(np.asarray(rows[f"identity__{feature}"], dtype=object)[sample_indices][valid].tolist())
        for feature in NON_COUNTRY_FEATURES
    }
    results: list[dict[str, object]] = []
    country_direct_observed = direct_jsd(
        countries, answers, len(answer_counts), min_group_n=args.min_group_n
    )
    direct_observed: dict[str, float] = {}
    controlled_observed: dict[str, float] = {}

    for feature, feature_values in identities.items():
        direct_observed[feature] = direct_jsd(feature_values, answers, len(answer_counts), min_group_n=args.min_group_n)
        controlled_observed[feature] = controlled_jsd(
            feature_values, answers, countries, len(answer_counts),
            min_country_n=args.min_country_n,
            min_country_group_n=args.min_country_group_n,
        )

    country_direct_null = np.empty(args.permutations, dtype=float)
    direct_null = {feature: np.empty(args.permutations, dtype=float) for feature in NON_COUNTRY_FEATURES}
    controlled_null = {feature: np.empty(args.permutations, dtype=float) for feature in NON_COUNTRY_FEATURES}
    for permutation_index in range(args.permutations):
        shuffled_answers = answers[rng.permutation(len(answers))]
        shuffled_country_answers = shuffle_within_groups(answers, countries, rng)
        country_direct_null[permutation_index] = direct_jsd(
            countries, shuffled_answers, len(answer_counts), min_group_n=args.min_group_n
        )
        for feature, feature_values in identities.items():
            direct_null[feature][permutation_index] = direct_jsd(
                feature_values, shuffled_answers, len(answer_counts), min_group_n=args.min_group_n
            )
            shuffled_identity = shuffle_within_groups(feature_values, countries, rng)
            controlled_null[feature][permutation_index] = controlled_jsd(
                shuffled_identity, answers, countries, len(answer_counts),
                min_country_n=args.min_country_n,
                min_country_group_n=args.min_country_group_n,
            )

    results.append(metric_row(
        qid=qid, repeat=repeat, view="country_direct", feature="Country",
        observed=country_direct_observed, null_values=country_direct_null,
        quantile=args.quantile, n=n, majority=majority,
    ))
    for feature in NON_COUNTRY_FEATURES:
        results.append(metric_row(
            qid=qid, repeat=repeat, view="direct", feature=feature,
            observed=direct_observed[feature], null_values=direct_null[feature],
            quantile=args.quantile, n=n, majority=majority,
        ))
        results.append(metric_row(
            qid=qid, repeat=repeat, view="country_controlled", feature=feature,
            observed=controlled_observed[feature], null_values=controlled_null[feature],
            quantile=args.quantile, n=n, majority=majority,
        ))

    ranked = sorted(NON_COUNTRY_FEATURES, key=lambda feature: controlled_observed[feature], reverse=True)
    combo_features = ranked[: args.combo_top_k]
    combos = list(itertools.combinations(combo_features, 2))
    combo_observed: dict[tuple[str, str], float] = {}
    for first, second in combos:
        combo_identity = encode([
            f"{left}\x1f{right}"
            for left, right in zip(
                np.asarray(rows[f"identity__{first}"], dtype=object)[sample_indices][valid],
                np.asarray(rows[f"identity__{second}"], dtype=object)[sample_indices][valid],
            )
        ])
        combo_score = controlled_jsd(
            combo_identity, answers, countries, len(answer_counts),
            min_country_n=args.min_country_n,
            min_country_group_n=args.min_combo_group_n,
        )
        combo_observed[(first, second)] = combo_score - max(controlled_observed[first], controlled_observed[second])

    combo_null = {combo: np.empty(args.permutations, dtype=float) for combo in combos}
    for permutation_index in range(args.permutations):
        shuffled_answers = shuffle_within_groups(answers, countries, rng)
        shuffled_single: dict[str, float] = {}
        for feature, feature_values in identities.items():
            shuffled_single[feature] = controlled_jsd(
                feature_values, shuffled_answers, countries, len(answer_counts),
                min_country_n=args.min_country_n,
                min_country_group_n=args.min_country_group_n,
            )
        for first, second in combos:
            combo_identity = encode([
                f"{left}\x1f{right}"
                for left, right in zip(
                    np.asarray(rows[f"identity__{first}"], dtype=object)[sample_indices][valid],
                    np.asarray(rows[f"identity__{second}"], dtype=object)[sample_indices][valid],
                )
            ])
            combo_score = controlled_jsd(
                combo_identity, shuffled_answers, countries, len(answer_counts),
                min_country_n=args.min_country_n,
                min_country_group_n=args.min_combo_group_n,
            )
            combo_null[(first, second)][permutation_index] = combo_score - max(
                shuffled_single[first], shuffled_single[second]
            )
    for first, second in combos:
        results.append(metric_row(
            qid=qid, repeat=repeat, view="combination_synergy",
            feature=f"{first}+{second}", observed=combo_observed[(first, second)],
            null_values=combo_null[(first, second)], quantile=args.quantile,
            n=n, majority=majority, combo_components=f"{first},{second}",
        ))
    return results


def apply_fdr(rows: list[dict[str, object]], *, alpha: float, correction: str) -> None:
    groups: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["empirical_p"] != "":
            view = str(row["view"])
            # Country-direct and non-country direct identities are the same
            # family of 33 direct association hypotheses for each question.
            # Country-controlled and synergy use different null models and
            # therefore remain separate correction families.
            family = "direct_identity" if view in {"country_direct", "direct"} else view
            groups[(str(row["Q_id"]), int(row["subsample"]), family)].append(index)
    for indices in groups.values():
        pvalues = [float(rows[index]["empirical_p"]) for index in indices]
        if correction == "bonferroni":
            qvalues = [min(1.0, value * len(pvalues)) for value in pvalues]
        else:
            qvalues = bh_qvalues(pvalues)
        for index, qvalue in zip(indices, qvalues):
            rows[index]["fdr_q_value"] = qvalue
            if qvalue < alpha and float(rows[index]["observed"]) > float(rows[index]["null_threshold"]):
                rows[index]["signal"] = "supported"


def write_question_summary(
    rows: list[dict[str, object]],
    output_dir: Path,
    *,
    min_support_rate: float,
    min_effect_js: float,
    top_identities: int,
) -> None:
    """Write a compact, human-readable summary aggregated over subsamples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["Q_id"]), str(row["view"]), str(row["feature"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (qid, view, feature), feature_rows in sorted(grouped.items()):
        observed = np.asarray([float(row["observed"]) for row in feature_rows])
        support_rate = sum(row["signal"] == "supported" for row in feature_rows) / len(feature_rows)
        summary_rows.append({
            "Q_id": qid,
            "view": view,
            "feature": feature,
            "subsamples": len(feature_rows),
            "observed_median": float(np.median(observed)),
            "observed_mean": float(np.mean(observed)),
            "support_rate": support_rate,
            "permutation_supported": support_rate >= min_support_rate,
            "practical_strong": support_rate >= min_support_rate and float(np.median(observed)) >= min_effect_js,
        })

    summary_fields = [
        "Q_id", "view", "feature", "subsamples", "observed_median",
        "observed_mean", "support_rate", "permutation_supported", "practical_strong",
    ]
    summary_path = output_dir / "question_identity_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    by_question_view: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_question_view[(str(row["Q_id"]), str(row["view"]))].append(row)
    compact_rows: list[dict[str, object]] = []
    for (qid, view), feature_rows in sorted(by_question_view.items()):
        supported = [row for row in feature_rows if row["permutation_supported"]]
        supported.sort(key=lambda row: float(row["observed_median"]), reverse=True)
        strong = [row for row in feature_rows if row["practical_strong"]]
        strong.sort(key=lambda row: float(row["observed_median"]), reverse=True)
        ranked = sorted(feature_rows, key=lambda row: float(row["observed_median"]), reverse=True)
        top_supported = supported[:top_identities]
        top_strong = strong[:top_identities]
        top_any = ranked[:top_identities]
        selected = top_supported or top_any
        names = ", ".join(
            f"{row['feature']}({float(row['observed_median']):.4f})" for row in selected
        ) or "none"
        compact_rows.append({
            "Q_id": qid,
            "view": view,
            "permutation_supported_count": len(supported),
            "practical_strong_count": len(strong),
            "top_supported": ",".join(str(row["feature"]) for row in top_supported),
            "top_strong": ",".join(str(row["feature"]) for row in top_strong),
            "top_ranked": ",".join(str(row["feature"]) for row in top_any),
        })

    # One line per question: this is the human-facing summary log.
    text_lines: list[str] = []
    combined_rows: list[dict[str, object]] = []
    qids = sorted({str(row["Q_id"]) for row in summary_rows})
    for qid in qids:
        parts = [qid]
        combined: dict[str, object] = {"Q_id": qid}
        for view in ("country_direct", "direct", "country_controlled", "combination_synergy"):
            view_rows = [row for row in compact_rows if row["Q_id"] == qid and row["view"] == view]
            if not view_rows:
                parts.append(f"{view}: none")
                combined[f"{view}_supported_count"] = 0
                combined[f"{view}_strong_count"] = 0
                combined[f"{view}_top_supported"] = ""
                combined[f"{view}_top_strong"] = ""
                continue
            row = view_rows[0]
            parts.append(
                f"{view}: supported={row['permutation_supported_count']} "
                f"strong={row['practical_strong_count']} "
                f"top={row['top_strong'] or row['top_supported'] or 'none'}"
            )
            combined[f"{view}_supported_count"] = row["permutation_supported_count"]
            combined[f"{view}_strong_count"] = row["practical_strong_count"]
            combined[f"{view}_top_supported"] = row["top_supported"]
            combined[f"{view}_top_strong"] = row["top_strong"]
        text_lines.append(" | ".join(parts))
        combined_rows.append(combined)

    (output_dir / "question_identity_summary.txt").write_text(
        "\n".join(text_lines) + "\n", encoding="utf-8"
    )
    compact_path = output_dir / "question_identity_summary_compact.csv"
    with compact_path.open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "Q_id", "view", "permutation_supported_count", "practical_strong_count",
            "top_supported", "top_strong", "top_ranked",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(compact_rows)
    combined_path = output_dir / "question_identity_summary_by_question.csv"
    combined_fields = ["Q_id"]
    for view in ("country_direct", "direct", "country_controlled", "combination_synergy"):
        combined_fields.extend([
            f"{view}_supported_count", f"{view}_strong_count",
            f"{view}_top_supported", f"{view}_top_strong",
        ])
    with combined_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=combined_fields)
        writer.writeheader()
        writer.writerows(combined_rows)
    print(f"saved: {summary_path}")
    print(f"saved: {output_dir / 'question_identity_summary.txt'}")
    print(f"saved: {combined_path}")


def main() -> None:
    args = parse_args()
    if args.permutations < 99:
        raise ValueError("Use at least 99 permutations; 999 is recommended for formal analysis.")
    if not args.cleaned_csv.is_file() or not args.metadata.is_file():
        raise FileNotFoundError("cleaned CSV or metadata file does not exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.cleaned_csv.open(encoding="utf-8", newline="") as stream:
        answer_columns = [column for column in (csv.DictReader(stream).fieldnames or []) if column.startswith("answer__")]
    qids = question_ids(args.metadata, answer_columns)
    if args.only_qids:
        allowed = {qid.strip() for qid in args.only_qids.split(",") if qid.strip()}
        qids = [qid for qid in qids if qid in allowed]
    if args.question_limit:
        qids = qids[: args.question_limit]
    rows, _ = load_rows(args.cleaned_csv, qids)
    rng = np.random.default_rng(args.seed)
    all_results: list[dict[str, object]] = []
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    worker_count = args.workers or min(os.cpu_count() or 1, max(1, len(qids)))
    worker_count = min(worker_count, max(1, len(qids)))
    print(f"parallel workers: {worker_count} (cpu_count={os.cpu_count() or 1})", flush=True)
    tasks: list[tuple[str, np.ndarray, int, argparse.Namespace, int]] = []
    for repeat in range(args.subsample_repeats):
        print(f"subsample {repeat + 1}/{args.subsample_repeats}", flush=True)
        for index, qid in enumerate(qids, start=1):
            # Sampling is question-specific because missing answers differ by question.
            # The same selected rows are then shared by all three views for this qid.
            sample_indices = sample_question_by_country(rows, qid, args.sample_per_country, rng)
            if args.verbose:
                print(f"  [{index}/{len(qids)}] {qid}", flush=True)
            # Each task gets its own deterministic RNG, so scheduling order does
            # not affect null distributions or the final summary.
            task_seed = int(args.seed + repeat * 1_000_003 + index * 9_176)
            tasks.append((qid, sample_indices, repeat, args, task_seed))

    if worker_count == 1:
        _init_worker(rows)
        task_results = list(
            tqdm(
                (_analyse_task(task) for task in tasks),
                total=len(tasks),
                desc="analyzing questions",
                unit="task",
            )
        )
    else:
        # On Linux fork shares the large read-only row table copy-on-write,
        # avoiding one full CSV copy per question task.
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=(rows,),
        ) as executor:
            task_results = list(
                tqdm(
                    executor.map(_analyse_task, tasks, chunksize=1),
                    total=len(tasks),
                    desc="analyzing questions",
                    unit="task",
                )
            )
    for result in task_results:
        all_results.extend(result)
    apply_fdr(all_results, alpha=args.alpha, correction=args.correction)
    write_question_summary(
        all_results,
        args.output_dir,
        min_support_rate=args.min_support_rate,
        min_effect_js=args.min_effect_js,
        top_identities=args.top_identities,
    )
    fields = [
        "Q_id", "subsample", "view", "feature", "components", "n", "majority",
        "observed", "null_mean", "null_std", "null_threshold", "null_threshold_99",
        "empirical_p", "fdr_q_value", "signal",
    ]
    output_file = args.output_dir / "identity_three_views_permutation.csv"
    with output_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)
    config = {
        "sample_per_country": args.sample_per_country,
        "subsample_repeats": args.subsample_repeats,
        "permutations": args.permutations,
        "alpha": args.alpha,
        "multiple_testing_correction": args.correction,
        "seed": args.seed,
        "country_direct_null": "global answer permutation with Country as the tested grouping",
        "direct_null": "global answer permutation for non-country identities",
        "controlled_null": "identity permutation within country",
        "synergy_null": "answer permutation within country; identities and their correlations fixed",
        "multiple_testing": (
            f"{args.correction} within each question/subsample/family at alpha={args.alpha}; "
            "country_direct and direct share one 33-identity family; "
            "country_controlled and combination_synergy use separate families"
        ),
        "combo_search": f"pairs among top {args.combo_top_k} controlled-JSD identities",
        "compact_summary_supported": f"support_rate >= {args.min_support_rate}; based only on permutation/FDR results",
        "compact_summary_practical_strong": f"supported plus median observed JSD >= {args.min_effect_js}; practical effect filter, not the permutation threshold",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {output_file}")


if __name__ == "__main__":
    main()
