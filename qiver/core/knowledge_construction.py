#!/usr/bin/env python3
"""Build the conservative natural-language identity knowledge base.

The renderer binds every semantic viewpoint to an exact legal answer option,
suppresses weak non-leading relative shifts, and records retrieval defaults.
Numeric distributions remain audit-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .tendency_knowledge_base import (
    REPO_ROOT,
    build_similarity_records,
    consensus_label,
    distribution_shape,
    encode_questions,
    load_identity_names,
    load_jsonl,
    option_semantics,
    plain_group_phrase,
    question_embedding_text,
    reliability_label,
    semantic_question_text,
    strength_label,
    write_outputs,
)


def _option_reference(option: str, viewpoint: str) -> str:
    """Name the legal option first so the model does not copy a paraphrase."""
    if not option:
        return ""
    return f"the answer option “{option}” (meaning {viewpoint})"


def conservative_tendency_document(
    document: dict[str, Any],
    question: dict[str, Any],
    identity_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    distribution = {str(k): float(v) for k, v in document.get("answer_distribution", {}).items()}
    shifts = {str(k): float(v) for k, v in document.get("distribution_shift", {}).items()}
    if not distribution:
        raise ValueError(f"Evidence document has no answer distribution: {document.get('doc_id')}")

    semantics = option_semantics(question)
    leading = max(distribution, key=distribution.get)
    favored = max(shifts, key=shifts.get) if shifts else ""
    magnitude = max((abs(value) for value in shifts.values()), default=0.0)
    strength = strength_label(magnitude)
    consensus = consensus_label(distribution)
    shape = distribution_shape(question, leading, consensus)
    identities = document.get("identities", [])
    subject = plain_group_phrase(identities)
    group = subject[0].lower() + subject[1:]
    leading_viewpoint = semantics.get(leading, leading)
    leading_ref = _option_reference(leading, leading_viewpoint)

    if shape == "mixed":
        absolute = f"Responses among {group} are mixed. The most common response is {leading_ref}, but it does not clearly dominate."
    elif shape == "middle_dominant":
        absolute = f"Responses among {group} are centered on {leading_ref}, which is the most common response."
    elif shape == "concentrated":
        absolute = f"The clearly most common response among {group} is {leading_ref}."
    else:
        absolute = f"The most common response among {group} is {leading_ref}, although other responses remain common."

    # Weak non-leading shifts caused Q51/Q54-style errors: the model selected
    # the identity-distinctive minority instead of the absolute group mode.
    # Retain relative evidence only when it reinforces the mode, or when a
    # non-leading shift is marked (>= .20) and therefore substantively large.
    include_relative = bool(
        favored
        and strength in {"moderate", "marked"}
        and (favored == leading or strength == "marked")
    )
    sentences = [absolute]
    if include_relative:
        adverb = {"moderate": "noticeably", "marked": "much"}[strength]
        if favored == leading:
            sentences.append(f"This same leading response is {adverb} more common here than among respondents overall.")
        else:
            favored_viewpoint = semantics.get(favored, favored)
            favored_ref = _option_reference(favored, favored_viewpoint)
            sentences.append(
                f"Compared with respondents overall, {favored_ref} is {adverb} more common in this group. "
                f"It remains a minority pattern and should not replace “{leading}” as the most likely response."
            )

    names = identity_names or {}
    dimensions = [names.get(str(x.get("feature", "")), str(x.get("feature", ""))) for x in identities]
    reliability = reliability_label(float(document.get("reliability", 0.0)))
    return {
        "doc_id": document["doc_id"],
        "Q_id": document["Q_id"],
        "evidence_type": document["evidence_type"],
        "identities": identities,
        "identity_dimensions": dimensions,
        "tendency_text": " ".join(sentences),
        "leading_option_label": leading,
        "leading_viewpoint": leading_viewpoint,
        "relative_option_label": favored if include_relative else "",
        "relative_viewpoint": semantics.get(favored, favored) if include_relative else "",
        "relative_statement_included": include_relative,
        "distribution_shape": shape,
        "shift_strength": strength,
        "group_consensus": consensus,
        "evidence_reliability": reliability,
        "source_sample_size_band": (
            "small" if int(document.get("sample_size", 0)) < 60
            else "medium" if int(document.get("sample_size", 0)) < 200
            else "large"
        ),
        "source_numeric_doc_id": document["doc_id"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-knowledge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--identity-metadata", type=Path, required=True)
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument(
        "--reuse-similarity-from", type=Path, default=None,
        help="Optional existing matrix built from the same questions; omit to recompute embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest = json.loads((args.source_knowledge_dir / "manifest.json").read_text(encoding="utf-8"))
    numeric_documents = load_jsonl(args.source_knowledge_dir / "evidence.jsonl")
    baselines = load_jsonl(args.source_knowledge_dir / "question_baselines.jsonl")
    questions: list[dict[str, Any]] = []
    for baseline in sorted(baselines, key=lambda item: str(item["Q_id"])):
        question = {
            "Q_id": str(baseline["Q_id"]),
            "question_text": baseline.get("question_text", ""),
            "value_domain": baseline.get("value_domain", ""),
            "answer_options": baseline.get("answer_options", []),
            "model_scale": baseline.get("model_scale", {}),
            "simplified": bool(baseline.get("simplified", False)),
        }
        question["option_semantics"] = option_semantics(question)
        question["semantic_question_text"] = semantic_question_text(question)
        question["embedding_text"] = question_embedding_text(question)
        questions.append(question)

    by_qid = {q["Q_id"]: q for q in questions}
    identity_names = load_identity_names(args.identity_metadata)
    tendencies = [conservative_tendency_document(d, by_qid[str(d["Q_id"])], identity_names) for d in numeric_documents]
    similarity_source = args.reuse_similarity_from
    if similarity_source and (similarity_source / "semantic_similarity_matrix.json").exists():
        matrix_payload = json.loads((similarity_source / "semantic_similarity_matrix.json").read_text(encoding="utf-8"))
        neighbor_payload = json.loads((similarity_source / "semantic_neighbors.json").read_text(encoding="utf-8"))
        expected_qids = [q["Q_id"] for q in questions]
        if [str(q) for q in matrix_payload.get("Q_ids", [])] != expected_qids:
            raise ValueError("Reused similarity matrix question order does not match v4.5 questions")
        matrix = matrix_payload["matrix"]
        neighbours = neighbor_payload["by_question"]
    else:
        embeddings = encode_questions(
            [q["embedding_text"] for q in questions], args.embedding_model,
            batch_size=args.embedding_batch_size, max_length=args.embedding_max_length,
            device=args.embedding_device,
        )
        matrix, neighbours = build_similarity_records(questions, embeddings)
    manifest = {
        "schema_version": "4.5",
        "knowledge_scope": source_manifest.get("knowledge_scope", "train_only"),
        "source_numeric_knowledge": str(args.source_knowledge_dir),
        "source_numeric_manifest": source_manifest,
        "question_count": len(questions),
        "similarity_matrix_shape": [len(questions), len(questions)],
        "similarity_metric": "cosine_similarity",
        "embedding_model": str(args.embedding_model),
        "embedding_input": "scale_instruction_stripped_question_text_only",
        "runtime_similarity_computation": False,
        "similarity_reused_from": str(similarity_source) if similarity_source else None,
        "tendency_document_count": len(tendencies),
        "single_document_count": sum(d["evidence_type"] == "single_identity" for d in tendencies),
        "combination_document_count": sum(d["evidence_type"] == "identity_combination" for d in tendencies),
        "numeric_statistics_in_inference_documents": False,
        "numeric_statistics_retained_for_audit": True,
        "identity_metadata": str(args.identity_metadata),
        "inference_evidence_style": "absolute_first_option_bound_conservative_relative_v45",
        "relative_policy": {
            "omit_slight": True,
            "include_moderate_only_when_reinforcing_leader": True,
            "include_marked_non_leading_with_minority_warning": True,
        },
        "recommended_retrieval": {
            "top_k_questions": 2,
            "minimum_cross_question_similarity": 0.90,
            "max_combinations_per_question": None,
            "max_singles_per_question": None,
            "deduplicate_correlated_identity_families": True,
        },
    }
    write_outputs(args.output_dir, questions, tendencies, numeric_documents, matrix, neighbours, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"saved: {args.output_dir}")


if __name__ == "__main__":
    main()
