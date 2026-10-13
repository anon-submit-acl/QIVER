#!/usr/bin/env python3
"""Two-stage permutation testing for country-controlled identity synergy.

Stage 1 screens every valid non-country identity pair with a small number of
permutations.  Stage 2 evaluates only the screen candidates on an independent
subsample with more permutations.  Country is the stratification variable and
is therefore intentionally excluded from the tested identity pairs.
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

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

from .association_statistics import (
    NON_COUNTRY_FEATURES,
    controlled_jsd,
    direct_jsd,
    encode,
    load_rows,
    pvalue,
    question_ids,
    sample_question_by_country,
    shuffle_within_groups,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cleaned-csv", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-per-country", type=int, default=100)
    p.add_argument("--subsample-repeats", type=int, default=10)
    p.add_argument("--screen-permutations", type=int, default=99)
    p.add_argument("--permutations", type=int, default=999)
    p.add_argument("--screen-max-candidates", type=int, default=25)
    p.add_argument("--screen-quantile", type=float, default=0.95)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-question-n", type=int, default=800)
    p.add_argument("--max-majority", type=float, default=0.90)
    p.add_argument("--min-country-n", type=int, default=100)
    p.add_argument("--min-country-group-n", type=int, default=30)
    p.add_argument("--min-combo-group-n", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--only-qids", default="")
    p.add_argument("--question-limit", type=int, default=0)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes; 0 uses min(CPU count, task count).",
    )
    return p.parse_args()


_WORKER_ROWS = None


def _init_worker(rows):
    global _WORKER_ROWS
    _WORKER_ROWS = rows


def _run_task(task):
    qid, repeat, sample_a, sample_b, args, pairs, seed_a, seed_b = task
    if _WORKER_ROWS is None:
        raise RuntimeError("worker rows were not initialized")
    screen = evaluate_pairs(
        _WORKER_ROWS, qid, sample_a, args, args.screen_permutations, pairs,
        np.random.default_rng(seed_a + 17),
    )
    candidates = [
        r for r in screen
        if r["view"] == "combination_synergy" and r["observed"] > r["null_threshold"]
    ]
    candidates.sort(
        key=lambda r: r["observed"] - r["null_threshold"], reverse=True
    )
    candidates = candidates[:args.screen_max_candidates]
    candidate_pairs = [tuple(r["components"].split(",")) for r in candidates]
    output = []
    for row in candidates:
        row.update({
            "subsample": repeat,
            "stage": "screen",
            "screen_candidate": True,
            "fdr_q_value": "",
            "signal": "screen_candidate",
        })
        output.append(row)

    confirmed = evaluate_pairs(
        _WORKER_ROWS, qid, sample_b, args, args.permutations, candidate_pairs,
        np.random.default_rng(seed_b + 17), include_singles=True,
    )
    candidate_count = max(1, len(candidate_pairs))
    for row in confirmed:
        correction_count = (
            33
            if row["view"] in {"country_direct", "country_controlled"}
            else candidate_count
        )
        q_value = min(1.0, float(row["empirical_p"]) * correction_count)
        row.update({
            "subsample": repeat,
            "stage": "confirm",
            "screen_candidate": row["view"] == "combination_synergy",
            "fdr_q_value": q_value,
            "signal": (
                "supported"
                if q_value < args.alpha and row["observed"] > row["null_threshold"]
                else "not_significant"
            ),
        })
        output.append(row)
    return output


def controlled_single(values, answers, countries, answer_count, args):
    return controlled_jsd(
        values, answers, countries, answer_count,
        min_country_n=args.min_country_n,
        min_country_group_n=args.min_country_group_n,
    )


def combo_values(rows, sample_indices, valid, first, second):
    left = np.asarray(rows[f"identity__{first}"], dtype=object)[sample_indices][valid]
    right = np.asarray(rows[f"identity__{second}"], dtype=object)[sample_indices][valid]
    return encode([f"{a}\x1f{b}" for a, b in zip(left, right)])


def evaluate_pairs(rows, qid, sample_indices, args, permutations, pairs, rng, *, include_singles=False):
    raw = np.asarray(rows[f"answer__{qid}"], dtype=object)[sample_indices]
    valid = (raw != "") & (raw != "Not Answered")
    if not np.any(valid):
        return []
    answers = encode(raw[valid].tolist())
    countries = encode(np.asarray(rows["country_name"], dtype=object)[sample_indices][valid].tolist())
    n = len(answers)
    counts = np.bincount(answers)
    majority = float(counts.max() / n)
    if n < args.min_question_n or majority >= args.max_majority:
        return []
    identities = {
        f: encode(np.asarray(rows[f"identity__{f}"], dtype=object)[sample_indices][valid].tolist())
        for f in NON_COUNTRY_FEATURES
    }
    answer_count = len(counts)
    singles = {
        f: controlled_single(v, answers, countries, answer_count, args)
        for f, v in identities.items()
    }
    observed = {}
    combo_encoded = {}
    for first, second in pairs:
        encoded = combo_values(rows, sample_indices, valid, first, second)
        combo_encoded[(first, second)] = encoded
        observed[(first, second)] = controlled_single(
            encoded, answers, countries, answer_count, args
        ) - max(singles[first], singles[second])

    country_direct_null = np.empty(permutations, dtype=float)
    null = {pair: np.empty(permutations, dtype=float) for pair in pairs}
    single_null = {feature: np.empty(permutations, dtype=float) for feature in identities}
    for index in range(permutations):
        shuffled_global_answers = answers[rng.permutation(len(answers))]
        shuffled_answers = shuffle_within_groups(answers, countries, rng)
        country_direct_null[index] = direct_jsd(
            countries, shuffled_global_answers, answer_count, min_group_n=args.min_country_group_n
        )
        shuffled_singles = {
            f: controlled_single(v, shuffled_answers, countries, answer_count, args)
            for f, v in identities.items()
        }
        for feature in identities:
            single_null[feature][index] = shuffled_singles[feature]
        for pair in pairs:
            score = controlled_single(
                combo_encoded[pair], shuffled_answers, countries,
                answer_count, args,
            )
            null[pair][index] = score - max(
                shuffled_singles[pair[0]], shuffled_singles[pair[1]]
            )

    output = []
    if include_singles:
        output.append({
            "Q_id": qid,
            "observed": direct_jsd(
                countries, answers, answer_count, min_group_n=args.min_country_group_n
            ),
            "null_mean": float(country_direct_null.mean()),
            "null_threshold": float(np.quantile(country_direct_null, 0.95)),
            "null_threshold_99": float(np.quantile(country_direct_null, 0.99)),
            "empirical_p": pvalue(
                direct_jsd(countries, answers, answer_count, min_group_n=args.min_country_group_n),
                country_direct_null,
            ),
            "n": n,
            "majority": majority,
            "feature": "Country",
            "components": "",
            "view": "country_direct",
        })
        for feature in identities:
            values = single_null[feature]
            output.append({
                "Q_id": qid,
                "observed": singles[feature],
                "null_mean": float(values.mean()),
                "null_threshold": float(np.quantile(values, 0.95)),
                "null_threshold_99": float(np.quantile(values, 0.99)),
                "empirical_p": pvalue(singles[feature], values),
                "n": n,
                "majority": majority,
                "feature": feature,
                "components": "",
                "view": "country_controlled",
            })
    for first, second in pairs:
        values = null[(first, second)]
        output.append({
            "Q_id": qid,
            "observed": observed[(first, second)],
            "null_mean": float(values.mean()),
            "null_threshold": float(np.quantile(values, 0.95)),
            "null_threshold_99": float(np.quantile(values, 0.99)),
            "empirical_p": pvalue(observed[(first, second)], values),
            "n": n,
            "majority": majority,
            "feature": f"{first}+{second}",
            "components": f"{first},{second}",
            "view": "combination_synergy",
        })
    return output


def main() -> None:
    args = parse_args()
    if args.screen_permutations < 19 or args.permutations < 99:
        raise ValueError("Use at least 19 screen and 99 confirmation permutations.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.cleaned_csv.open(encoding="utf-8", newline="") as stream:
        answer_columns = [c for c in (csv.DictReader(stream).fieldnames or []) if c.startswith("answer__")]
    qids = question_ids(args.metadata, answer_columns)
    if args.only_qids:
        allowed = {x.strip() for x in args.only_qids.split(",") if x.strip()}
        qids = [q for q in qids if q in allowed]
    if args.question_limit:
        qids = qids[:args.question_limit]
    rows, _ = load_rows(args.cleaned_csv, qids)
    pairs = list(itertools.combinations(NON_COUNTRY_FEATURES, 2))
    print(f"tested identities: {len(NON_COUNTRY_FEATURES)} non-country identities + Country direct signal")
    print(f"all candidate pairs: {len(pairs)}")

    tasks = []
    for repeat in range(args.subsample_repeats):
        for qid in qids:
            qid_seed = sum((index + 1) * ord(char) for index, char in enumerate(qid))
            seed_a = int(args.seed + repeat * 1_000_003 + qid_seed)
            seed_b = seed_a + 500_000_009
            sample_a = sample_question_by_country(
                rows, qid, args.sample_per_country, np.random.default_rng(seed_a)
            )
            sample_b = sample_question_by_country(
                rows, qid, args.sample_per_country, np.random.default_rng(seed_b)
            )
            tasks.append((qid, repeat, sample_a, sample_b, args, pairs, seed_a, seed_b))

    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    worker_count = args.workers or min(os.cpu_count() or 1, max(1, len(tasks)))
    worker_count = min(worker_count, max(1, len(tasks)))
    print(f"parallel workers: {worker_count} (cpu_count={os.cpu_count() or 1})", flush=True)
    if worker_count == 1:
        _init_worker(rows)
        task_results = list(tqdm((_run_task(task) for task in tasks), total=len(tasks), desc="analyzing", unit="task"))
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=(rows,),
        ) as executor:
            task_results = list(tqdm(executor.map(_run_task, tasks, chunksize=1), total=len(tasks), desc="analyzing", unit="task"))
    records = [row for result in task_results for row in result]

    fields = ["Q_id", "subsample", "stage", "view", "feature", "components", "screen_candidate", "n", "majority", "observed", "null_mean", "null_threshold", "null_threshold_99", "empirical_p", "fdr_q_value", "signal"]
    with (args.output_dir / "identity_combination_two_stage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)

    confirmed = [r for r in records if r["stage"] == "confirm"]
    by_feature = defaultdict(list)
    for r in confirmed:
        by_feature[(r["Q_id"], r["feature"])].append(r)
    summary = []
    for (qid, feature), items in sorted(by_feature.items()):
        supported = sum(r["signal"] == "supported" for r in items)
        summary.append({"Q_id": qid, "view": items[0]["view"], "feature": feature, "confirmed_subsamples": len(items), "supported_subsamples": supported, "support_rate": supported / len(items), "median_observed": float(np.median([float(r["observed"]) for r in items])), "supported": supported >= max(1, int(np.ceil(args.subsample_repeats * 0.60)))})
    with (args.output_dir / "question_combination_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Q_id", "view", "feature", "confirmed_subsamples", "supported_subsamples", "support_rate", "median_observed", "supported"])
        writer.writeheader(); writer.writerows(summary)
    analysis_countries = sorted(
        {
            str(country)
            for country in np.asarray(rows["country_name"], dtype=object).tolist()
            if str(country).strip()
        }
    )
    config = vars(args).copy(); config.update({"analysis_split": "train", "analysis_respondent_count": len(np.asarray(rows["country_name"])), "all_candidate_pairs": len(pairs), "country_excluded_from_pairs": True, "country_direct_included": True, "direct_identity_hypotheses": 33, "country_controlled_correction_hypotheses": 33, "stage2_independent_subsample": True, "stage2_correction": "Bonferroni: 33 for country_direct/country_controlled; screen candidates for combination_synergy per question/subsample", "analysis_countries": analysis_countries, "analysis_country_count": len(analysis_countries)})
    serializable_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }
    (args.output_dir / "config.json").write_text(json.dumps(serializable_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {args.output_dir / 'identity_combination_two_stage.csv'}")
    print(f"saved: {args.output_dir / 'question_combination_summary.csv'}")


if __name__ == "__main__":
    main()
