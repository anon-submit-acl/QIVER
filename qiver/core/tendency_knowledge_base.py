#!/usr/bin/env python3
"""Build a natural-language WVS identity-tendency knowledge base.

It performs two offline transformations:

1. Convert each numeric identity distribution into deterministic natural-
   language tendency evidence.  Numeric distributions are retained only in a
   separate audit file and are not stored in inference-facing documents.
2. Encode the selected questions once and store their full cosine-similarity
   matrix plus pre-sorted neighbours.  Inference therefore only performs a
   matrix/table lookup; it never runs an embedding model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
# Inference-facing meanings are propositions, not answer labels.  These are
# deliberately curated question by question: a generic Yes/No expansion can
# easily invert the meaning of a negatively worded survey item.
OPTION_VIEWPOINTS: dict[str, dict[str, str]] = {
    "Q5": {
        "Very important": "placing very high importance on work in one's life", "Rather important": "placing fairly high importance on work in one's life",
        "Not very important": "placing limited importance on work in one's life", "Not at all important": "placing no importance on work in one's life",
    },
    "Q6": {
        "Very important": "placing very high importance on religion in one's life", "Rather important": "placing fairly high importance on religion in one's life",
        "Not very important": "placing limited importance on religion in one's life", "Not at all important": "placing no importance on religion in one's life",
    },
    "Q15": {"Yes": "viewing religious faith as especially important for children to learn at home", "No": "not viewing religious faith as especially important for children to learn at home"},
    "Q49": {"Low": "low overall life satisfaction", "Medium": "moderate overall life satisfaction", "High": "high overall life satisfaction"},
    "Q50": {"Low": "low satisfaction with the household's financial situation", "Medium": "moderate satisfaction with the household's financial situation", "High": "high satisfaction with the household's financial situation"},
    "Q51": {"Often": "frequently lacking enough food for the family", "Sometimes": "sometimes lacking enough food for the family", "Rarely": "rarely lacking enough food for the family", "Never": "not lacking enough food for the family"},
    "Q54": {"Often": "frequently going without a cash income", "Sometimes": "sometimes going without a cash income", "Rarely": "rarely going without a cash income", "Never": "not going without a cash income"},
    "Q64": {"A great deal": "having very high confidence in places of worship", "Quite a lot": "having substantial confidence in places of worship", "Not very much": "having little confidence in places of worship", "None at all": "having no confidence in places of worship"},
    "Q95": {"Don't belong": "not belonging to a sports or recreational organization", "Inactive member": "being an inactive member of a sports or recreational organization", "Active member": "actively participating in a sports or recreational organization"},
    "Q97": {"Don't belong": "not belonging to a labor union", "Inactive member": "being an inactive member of a labor union", "Active member": "actively participating in a labor union"},
    "Q140": {"Yes": "avoiding going out at night because of security concerns", "No": "not avoiding going out at night because of security concerns"},
    "Q142": {"Very much": "being extremely worried about losing or not finding a job", "A good deal": "being substantially worried about losing or not finding a job", "Not much": "having little worry about losing or not finding a job", "Not at all": "having no worry about losing or not finding a job"},
    "Q160": {"Low": "rejecting the view that society depends too much on science and not enough on faith", "Medium": "taking a moderate position on whether society depends too much on science and not enough on faith", "High": "endorsing the view that society depends too much on science and not enough on faith"},
    "Q164": {"Low": "placing low importance on God in one's life", "Medium": "placing moderate importance on God in one's life", "High": "placing high importance on God in one's life"},
    "Q165": {"Yes": "believing that God exists", "No": "not believing that God exists"},
    "Q166": {"Yes": "believing in life after death", "No": "not believing in life after death"},
    "Q167": {"Yes": "believing that Hell exists", "No": "not believing that Hell exists"},
    "Q168": {"Yes": "believing that Heaven exists", "No": "not believing that Heaven exists"},
    "Q169": {"Strongly agree": "strongly prioritizing religious authority when science and religion conflict", "Agree": "prioritizing religious authority when science and religion conflict", "Disagree": "not generally prioritizing religious authority when science and religion conflict", "Strongly disagree": "strongly rejecting religious priority when science and religion conflict"},
    "Q171": {"More than once a week": "attending religious services more than once a week", "Once a week": "attending religious services weekly", "Once a month": "attending religious services monthly", "Only on special holy days": "attending religious services only on special holy days", "Once a year": "attending religious services yearly", "Less often": "attending religious services less than yearly", "Never, practically never": "practically never attending religious services"},
    "Q172": {"Several times a day": "praying several times a day", "Once a day": "praying daily", "Several times each week": "praying several times a week", "Only when attending religious services": "praying only during religious services", "Only on special holy days": "praying only on special holy days", "Once a year": "praying yearly", "Less often": "praying less than yearly", "Never, practically never": "practically never praying"},
    "Q173": {"A religious person": "identifying as a religious person", "Not a religious person": "identifying as non-religious", "An atheist": "identifying as an atheist"},
    "Q186": {"Low": "viewing sex before marriage as difficult to justify", "Medium": "taking a conditional or moderate view of sex before marriage", "High": "viewing sex before marriage as broadly justifiable"},
    "Q193": {"Low": "viewing casual sex as difficult to justify", "Medium": "taking a conditional or moderate view of casual sex", "High": "viewing casual sex as broadly justifiable"},
    "Q199": {"Very interested": "having very high interest in politics", "Somewhat interested": "having some interest in politics", "Not very interested": "having little interest in politics", "Not at all interested": "having no interest in politics"},
    "Q200": {"Frequently": "frequently discussing politics with friends", "Occasionally": "occasionally discussing politics with friends", "Never": "never discussing politics with friends"},
    "Q206": {"Daily": "using the internet daily to obtain information", "Weekly": "using the internet weekly to obtain information", "Monthly": "using the internet monthly to obtain information", "Less than monthly": "using the internet less than monthly to obtain information", "Never": "never using the internet to obtain information"},
    "Q207": {"Daily": "using social media daily to obtain information", "Weekly": "using social media weekly to obtain information", "Monthly": "using social media monthly to obtain information", "Less than monthly": "using social media less than monthly to obtain information", "Never": "never using social media to obtain information"},
    "Q221": {"Always": "consistently voting in local elections", "Usually": "usually voting in local elections", "Never": "not voting in local elections", "Not allowed to vote": "being ineligible to vote in local elections"},
    "Q222": {"Always": "consistently voting in national elections", "Usually": "usually voting in national elections", "Never": "not voting in national elections", "Not allowed to vote": "being ineligible to vote in national elections"},
    "Q254": {"Very proud": "feeling very proud of one's nationality", "Quite proud": "feeling fairly proud of one's nationality", "Not very proud": "feeling little pride in one's nationality", "Not at all proud": "feeling no pride in one's nationality", "I am not of this nationality": "not identifying with the surveyed nationality"},
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def option_semantics(question: dict[str, Any]) -> dict[str, str]:
    """Map stored answer labels to human-readable response meanings."""
    answer_options = [str(value) for value in question.get("answer_options", [])]
    curated = OPTION_VIEWPOINTS.get(str(question.get("Q_id", "")), {})
    if curated and set(answer_options).issubset(curated):
        return {option: curated[option] for option in answer_options}
    model_options = question.get("model_scale", {}).get("options", [])
    labels = [str(item.get("option_text", "")).strip() for item in model_options]
    if len(answer_options) == len(labels) and all(labels):
        return dict(zip(answer_options, labels))
    return {option: option for option in answer_options}


def semantic_question_text(question: dict[str, Any]) -> str:
    """Remove survey scale instructions while preserving the core question.

    The original question remains untouched elsewhere in the database.  This
    normalized form is used only for question-to-question embeddings.
    """
    text = str(question.get("question_text", "")).strip()
    # These WVS clauses describe how to operate the response scale rather than
    # the construct being measured.  Keeping them makes unrelated 1--10 items
    # appear artificially similar.
    text = re.split(r"\s+using\s+this\s+card\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"\s+on\s+a\s+scale\s+from\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.rstrip(" ,;:")


def question_embedding_text(question: dict[str, Any]) -> str:
    """Represent only the scale-free core question."""
    return semantic_question_text(question)


def strength_label(absolute_shift: float) -> str:
    if absolute_shift < 0.05:
        return "negligible"
    if absolute_shift < 0.10:
        return "slight"
    if absolute_shift < 0.20:
        return "moderate"
    return "marked"


def consensus_label(distribution: dict[str, float]) -> str:
    ordered = sorted((float(value) for value in distribution.values()), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else (ordered[0] if ordered else 0.0)
    if margin < 0.10:
        return "mixed"
    if margin < 0.20:
        return "leading"
    return "clear"


def reliability_label(reliability: float) -> str:
    if reliability < 0.75:
        return "cautious"
    if reliability < 0.90:
        return "moderate"
    return "strong"


def distribution_shape(
    question: dict[str, Any], leading_option: str, consensus: str
) -> str:
    if consensus == "mixed":
        return "mixed"
    options = [str(value) for value in question.get("answer_options", [])]
    if len(options) >= 3 and len(options) % 2 == 1 and options[len(options) // 2] == leading_option:
        return "middle_dominant"
    return "concentrated" if consensus == "clear" else "leading"


def identity_phrase(identities: Iterable[dict[str, Any]]) -> str:
    values = [str(item.get("value", "")).strip() for item in identities]
    values = [value for value in values if value]
    if not values:
        return "this identity group"
    if len(values) == 1:
        return f"respondents identified as {values[0]}"
    return "respondents with the combined identities " + " and ".join(values)


def load_identity_names(path: Path) -> dict[str, str]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["feature"]): str(item.get("name", item["feature"]))
        for item in metadata.get("identity_features", [])
        if isinstance(item, dict) and item.get("feature")
    }


def plain_group_phrase(identities: list[dict[str, Any]]) -> str:
    values = [str(item.get("value", "")).strip() for item in identities if str(item.get("value", "")).strip()]
    if len(values) == 1:
        return f"People identified as {values[0]}"
    if len(values) == 2:
        return f"People matching both {values[0]} and {values[1]}"
    return "People in this identity group"


def tendency_document(
    document: dict[str, Any], question: dict[str, Any], identity_names: dict[str, str] | None = None
) -> dict[str, Any]:
    """Create an inference-facing record without numeric probabilities."""
    shifts = {str(key): float(value) for key, value in document.get("distribution_shift", {}).items()}
    distribution = {str(key): float(value) for key, value in document.get("answer_distribution", {}).items()}
    semantics = option_semantics(question)
    favored = max(shifts, key=shifts.get) if shifts else ""
    disfavored = min(shifts, key=shifts.get) if shifts else ""
    leading = max(distribution, key=distribution.get) if distribution else ""
    magnitude = max((abs(value) for value in shifts.values()), default=0.0)
    strength = strength_label(magnitude)
    consensus = consensus_label(distribution)
    reliability = reliability_label(float(document.get("reliability", 0.0)))
    identities = document.get("identities", [])
    subject = plain_group_phrase(identities)
    favored_text = semantics.get(favored, favored)
    disfavored_text = semantics.get(disfavored, disfavored)
    leading_text = semantics.get(leading, leading)
    shape = distribution_shape(question, leading, consensus)

    sentences: list[str] = []
    if strength == "negligible" or not favored:
        sentences.append(f"{subject} show no meaningful tendency away from the overall pattern.")
    else:
        group = subject[0].lower() + subject[1:]
        if shape == "mixed":
            sentences.append(
                f"Responses among {group} are mixed; the most common pattern is “{leading_text}”, but it does not clearly dominate."
            )
        elif shape == "middle_dominant":
            sentences.append(
                f"Responses among {group} are centered on “{leading_text}”, which is the most common pattern."
            )
        elif shape == "concentrated":
            sentences.append(f"The most common pattern among {group} is clearly “{leading_text}”.")
        else:
            sentences.append(
                f"The most common pattern among {group} is “{leading_text}”, although other patterns remain common."
            )

        relative_strength = {"slight": "slightly", "moderate": "noticeably", "marked": "much"}[strength]
        if favored == leading:
            sentences.append(
                f"Compared with respondents overall, this pattern is {relative_strength} more common in the group."
            )
        else:
            sentences.append(
                f"Compared with respondents overall, “{favored_text}” is {relative_strength} more common in this group."
            )

    names = identity_names or {}
    dimensions = [names.get(str(item.get("feature", "")), str(item.get("feature", ""))) for item in identities]

    return {
        "doc_id": document["doc_id"],
        "Q_id": document["Q_id"],
        "evidence_type": document["evidence_type"],
        "identities": identities,
        "identity_dimensions": dimensions,
        "tendency_text": " ".join(sentences),
        "favored_option": favored_text,
        "disfavored_option": disfavored_text if disfavored != favored else "",
        "leading_option": leading_text,
        "relative_shift_viewpoint": favored_text,
        "same_leading_and_shift_direction": favored == leading,
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


def cosine_matrix(embeddings: Sequence[Sequence[float]]) -> list[list[float]]:
    """Compute cosine similarity without requiring numpy (also easy to test)."""
    vectors = [[float(value) for value in row] for row in embeddings]
    norms = [math.sqrt(sum(value * value for value in row)) for row in vectors]
    if any(norm == 0 for norm in norms):
        raise ValueError("Question embedding contains a zero vector")
    return [
        [sum(a * b for a, b in zip(left, right)) / (norms[i] * norms[j]) for j, right in enumerate(vectors)]
        for i, left in enumerate(vectors)
    ]


def encode_questions(
    texts: list[str], model_path: Path, *, batch_size: int = 32, max_length: int = 512, device: str = "auto"
) -> list[list[float]]:
    """Encode all questions with a local E5-compatible Transformers model."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("Semantic matrix construction requires torch and transformers") from exc

    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(resolved_device).eval()
    outputs: list[list[float]] = []
    # E5 uses a task prefix.  The same prefix is deliberately used for every
    # item because this is a symmetric question-to-question comparison.
    prefixed = ["query: " + text for text in texts]
    with torch.inference_mode():
        for start in range(0, len(prefixed), batch_size):
            batch = tokenizer(
                prefixed[start : start + batch_size], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(resolved_device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.extend(pooled.cpu().float().tolist())
    return outputs


def build_similarity_records(
    questions: list[dict[str, Any]], embeddings: Sequence[Sequence[float]]
) -> tuple[list[list[float]], dict[str, list[dict[str, Any]]]]:
    matrix = cosine_matrix(embeddings)
    qids = [str(question["Q_id"]) for question in questions]
    neighbours: dict[str, list[dict[str, Any]]] = {}
    for i, target in enumerate(qids):
        ranked = sorted(
            ({"Q_id": source, "similarity": matrix[i][j]} for j, source in enumerate(qids) if j != i),
            key=lambda item: (-float(item["similarity"]), str(item["Q_id"])),
        )
        neighbours[target] = [{**item, "rank": rank} for rank, item in enumerate(ranked, start=1)]
    return matrix, neighbours


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_outputs(
    output_dir: Path,
    questions: list[dict[str, Any]],
    tendencies: list[dict[str, Any]],
    audit_documents: list[dict[str, Any]],
    matrix: list[list[float]],
    neighbours: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "questions.jsonl", questions)
    write_jsonl(output_dir / "tendency_evidence.jsonl", tendencies)
    write_jsonl(output_dir / "audit_numeric_evidence.jsonl", audit_documents)
    qids = [str(question["Q_id"]) for question in questions]
    matrix_payload = {
        "schema_version": 1,
        "metric": "cosine_similarity",
        "embedding_model": manifest["embedding_model"],
        "Q_ids": qids,
        "matrix": matrix,
    }
    (output_dir / "semantic_similarity_matrix.json").write_text(
        json.dumps(matrix_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "semantic_neighbors.json").write_text(
        json.dumps({"by_question": neighbours}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    database = output_dir / "identity_tendency_knowledge.sqlite"
    if database.exists():
        database.unlink()
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE questions (
                q_id TEXT PRIMARY KEY, question_text TEXT NOT NULL,
                value_domain TEXT NOT NULL, embedding_text TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE tendency_evidence (
                doc_id TEXT PRIMARY KEY, q_id TEXT NOT NULL,
                evidence_type TEXT NOT NULL, identity_key TEXT NOT NULL,
                tendency_text TEXT NOT NULL, shift_strength TEXT NOT NULL,
                group_consensus TEXT NOT NULL, evidence_reliability TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX tendency_lookup ON tendency_evidence(q_id, evidence_type, identity_key);
            CREATE TABLE question_similarity (
                target_q_id TEXT NOT NULL, source_q_id TEXT NOT NULL,
                similarity REAL NOT NULL, rank INTEGER NOT NULL,
                PRIMARY KEY(target_q_id, source_q_id)
            );
            CREATE INDEX similarity_rank_lookup ON question_similarity(target_q_id, rank);
        """)
        connection.executemany(
            "INSERT INTO questions VALUES (?, ?, ?, ?, ?)",
            [(q["Q_id"], q["question_text"], q["value_domain"], q["embedding_text"], json.dumps(q, ensure_ascii=False)) for q in questions],
        )
        connection.executemany(
            "INSERT INTO tendency_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(
                d["doc_id"], d["Q_id"], d["evidence_type"],
                "|".join(f"{x['feature']}={x['value']}" for x in d["identities"]),
                d["tendency_text"], d["shift_strength"], d["group_consensus"],
                d["evidence_reliability"], json.dumps(d, ensure_ascii=False),
            ) for d in tendencies],
        )
        connection.executemany(
            "INSERT INTO question_similarity VALUES (?, ?, ?, ?)",
            [(target, item["Q_id"], item["similarity"], item["rank"]) for target, items in neighbours.items() for item in items],
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-knowledge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--identity-metadata", type=Path, required=True)
    parser.add_argument("--embedding-device", default="auto", help="auto, cpu, cuda, or a CUDA device such as cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest = json.loads((args.source_knowledge_dir / "manifest.json").read_text(encoding="utf-8"))
    numeric_documents = load_jsonl(args.source_knowledge_dir / "evidence.jsonl")
    baselines = load_jsonl(args.source_knowledge_dir / "question_baselines.jsonl")
    questions = []
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
    by_qid = {question["Q_id"]: question for question in questions}
    identity_names = load_identity_names(args.identity_metadata)
    tendencies = [tendency_document(document, by_qid[str(document["Q_id"])], identity_names) for document in numeric_documents]

    embeddings = encode_questions(
        [question["embedding_text"] for question in questions], args.embedding_model,
        batch_size=args.embedding_batch_size, max_length=args.embedding_max_length,
        device=args.embedding_device,
    )
    matrix, neighbours = build_similarity_records(questions, embeddings)
    manifest = {
        "schema_version": 2,
        "knowledge_scope": source_manifest.get("knowledge_scope", "train_only"),
        "source_numeric_knowledge": str(args.source_knowledge_dir),
        "source_numeric_manifest": source_manifest,
        "question_count": len(questions),
        "similarity_matrix_shape": [len(questions), len(questions)],
        "similarity_metric": "cosine_similarity",
        "embedding_model": str(args.embedding_model),
        "embedding_input": "scale_instruction_stripped_question_text_only",
        "original_question_text_preserved": True,
        "runtime_similarity_computation": False,
        "tendency_document_count": len(tendencies),
        "single_document_count": sum(d["evidence_type"] == "single_identity" for d in tendencies),
        "combination_document_count": sum(d["evidence_type"] == "identity_combination" for d in tendencies),
        "numeric_statistics_in_inference_documents": False,
        "numeric_statistics_retained_for_audit": True,
        "identity_metadata": str(args.identity_metadata),
        "inference_evidence_style": "plain_absolute_group_pattern_plus_relative_identity_shift",
        "tendency_thresholds_absolute_shift": {"omit_below": 0.05, "slight_below": 0.10, "moderate_below": 0.20},
        "group_consensus_thresholds_top_margin": {"mixed_below": 0.10, "leading_below": 0.20},
    }
    write_outputs(args.output_dir, questions, tendencies, numeric_documents, matrix, neighbours, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"saved: {args.output_dir}")


if __name__ == "__main__":
    main()
