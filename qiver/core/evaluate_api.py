#!/usr/bin/env python3
"""Evaluate WVS records through an OpenAI-compatible API endpoint."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

from .prompt_renderer import WVSPromptRenderer
from .identity_knowledge_guidance import IdentityKnowledgeGuidance
from .evidence_retrieval import IdentityTendencyGuidance


DEFAULT_PROMPT_CONFIG = REPO_ROOT / "configs/evidence_grounded_reasoning.yaml"
IDENTITY_FIELDS = [
    "Country",
    "Gender",
    "Marital Status",
    "Has Children",
    "Education Level",
    "Occupation",
    "Work Nature",
    "Religion",
    "Life Stage",
    "Income Bracket",
    "Common Language",
]
CSV_FIELDS = [
    "UID",
    "Q_id",
    "prompt_template",
    "prompt_template_hash",
    "rendered_prompt_hash",
    "prompt",
    "identity_knowledge",
    *IDENTITY_FIELDS,
    "Information",
    "Category",
    "Question",
    "Options",
    "answer",
    "frequency",
    "response",
    "predicted",
    "is_correct",
]


def parse_options(options: Any) -> list[str]:
    if isinstance(options, (list, tuple)):
        return [str(option).strip() for option in options if str(option).strip()]
    if hasattr(options, "tolist"):
        return [str(option).strip() for option in options.tolist() if str(option).strip()]
    text = str(options).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(option).strip() for option in parsed if str(option).strip()]
    return [option.strip() for option in text.split(",") if option.strip()]


def answer_is_in_options(answer: str, options: Any) -> bool:
    answer_norm = normalize_answer_for_match(answer)
    parsed_options = parse_options(options)
    if any(normalize_answer_for_match(option) == answer_norm for option in parsed_options):
        return True
    return answer_norm in normalize_answer_for_match(options)


def extract_tagged_answer(text: str | None, answer_start: str, answer_end: str) -> str | None:
    if not text:
        return None
    start = text.rfind(answer_start)
    if start < 0:
        return None
    start += len(answer_start)
    end = text.find(answer_end, start)
    if end < 0:
        return None
    answer = text[start:end].strip()
    return answer or None


def is_correct_answer(predicted: str | None, expected: str, options: Any) -> bool:
    if predicted is None:
        return False
    predicted = predicted.strip()
    expected = expected.strip()
    if predicted == expected:
        return True
    predicted_norm = normalize_answer_for_match(predicted)
    expected_norm = normalize_answer_for_match(expected)
    if predicted_norm == expected_norm:
        return True

    other_options = []
    for option in parse_options(options):
        option_norm = normalize_answer_for_match(option)
        if not option_norm or option_norm == expected_norm:
            continue
        if option_norm in expected_norm:
            continue
        other_options.append(option_norm)
    return expected_norm in predicted_norm and all(
        option_norm not in predicted_norm for option_norm in other_options
    )


def is_lenient_tag_exact_option_answer(
    predicted: str | None, expected: str, options: Any
) -> bool:
    """Allow an unclosed tag upstream, but require an exact normalized option."""
    if predicted is None:
        return False
    first_line = next(
        (line.strip() for line in str(predicted).splitlines() if line.strip()),
        "",
    )
    candidate_norm = normalize_answer_for_match(first_line)
    expected_norm = normalize_answer_for_match(expected)
    # The dataset ground truth is itself the canonical option. Comparing with it
    # directly avoids incorrectly splitting option texts that contain commas.
    return bool(candidate_norm) and candidate_norm == expected_norm


def normalize_answer_for_match(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def coerce_dataset_prompt(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                pass
        if isinstance(value, str):
            return [{"role": "user", "content": value}]

    if isinstance(value, list):
        messages = []
        for message in value:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError("Dataset prompt messages require role and content fields.")
            messages.append({"role": str(message["role"]), "content": message["content"]})
        return messages
    raise ValueError("Dataset prompt must be a string or a list of messages.")


def rendered_prompt_hash(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_processed_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(f"Existing output has an incompatible CSV header: {path}")
        return {row["UID"] for row in reader}


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def load_eval_records(path: str, max_samples: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required to read parquet evaluation data.") from exc

    dataset = load_dataset("parquet", data_files=path)["train"]
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return [dict(record) for record in dataset]


def build_examples(
    records: list[dict[str, Any]],
    renderer: WVSPromptRenderer | None,
    use_dataset_prompt: bool,
    identity_knowledge: IdentityKnowledgeGuidance | IdentityTendencyGuidance | None = None,
    identity_knowledge_cross_question: bool = False,
    empty_knowledge_fallback_renderer: WVSPromptRenderer | None = None,
) -> list[dict[str, Any]]:
    examples = []
    for index, record in enumerate(records):
        expected = str(record.get("answer", "")).strip()
        options = record.get("Options", "")
        if not expected or not answer_is_in_options(expected, options):
            continue

        if use_dataset_prompt:
            if "prompt" not in record:
                raise KeyError("--use-dataset-prompt requires a prompt column in the parquet file.")
            messages = coerce_dataset_prompt(record["prompt"])
            template_name = "dataset_prompt"
            template_hash = "dataset_prompt"
            answer_start = "<answer>"
            answer_end = "</answer>"
        else:
            assert renderer is not None
            qid = str(record.get("Q_id", ""))
            knowledge = (
                identity_knowledge.render(
                    qid,
                    record.get("identity_json", {}),
                    cross_question=identity_knowledge_cross_question,
                )
                if identity_knowledge is not None
                else ""
            )
            active_renderer = (
                empty_knowledge_fallback_renderer
                if identity_knowledge_cross_question and identity_knowledge is not None and not knowledge
                and empty_knowledge_fallback_renderer is not None
                else renderer
            )
            messages = active_renderer.render_messages(
                record,
                extra_context={
                    "identity_knowledge": knowledge,
                },
            )
            template_name = active_renderer.name
            template_hash = active_renderer.template_hash
            answer_start = active_renderer.answer_start
            answer_end = active_renderer.answer_end

        prompt_hash = rendered_prompt_hash(messages)
        qid = str(record.get("Q_id", ""))
        uid = f"{index}_{qid}_{prompt_hash[:12]}"
        examples.append(
            {
                "uid": uid,
                "record": record,
                "messages": messages,
                "template_name": template_name,
                "template_hash": template_hash,
                "prompt_hash": prompt_hash,
                "answer_start": answer_start,
                "answer_end": answer_end,
                "identity_knowledge": knowledge if not use_dataset_prompt else "",
            }
        )
    return examples


def request_completion(client, args: argparse.Namespace, messages: list[dict[str, Any]]) -> str | None:
    for attempt in range(args.max_retries):
        try:
            request: dict[str, Any] = {
                "model": args.model,
                "messages": messages,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            extra_body: dict[str, Any] = {}
            if args.send_chat_template_kwargs:
                # Local vLLM/Qwen chat templates consume this nested field.
                extra_body["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}
            if args.send_enable_thinking_param:
                # DashScope's OpenAI-compatible API consumes enable_thinking
                # as a top-level request-body extension.  The OpenAI Python
                # SDK forwards extra_body entries at that level.
                extra_body["enable_thinking"] = args.enable_thinking
            if extra_body:
                request["extra_body"] = extra_body
            response = client.chat.completions.create(**request)
            return response.choices[0].message.content
        except Exception as exc:
            if attempt + 1 == args.max_retries:
                print(f"[ERROR] request failed after {args.max_retries} attempts: {exc}", file=sys.stderr)
                return None
            delay = min(2**attempt, 8)
            print(f"[WARN] request failed: {exc}; retrying in {delay}s", file=sys.stderr)
            time.sleep(delay)
    return None


def result_row(example: dict[str, Any], response: str) -> dict[str, Any]:
    record = example["record"]
    expected = str(record["answer"]).strip()
    predicted = extract_tagged_answer(response, example["answer_start"], example["answer_end"])
    row = {
        "UID": example["uid"],
        "Q_id": record.get("Q_id", ""),
        "prompt_template": example["template_name"],
        "prompt_template_hash": example["template_hash"],
        "rendered_prompt_hash": example["prompt_hash"],
        "prompt": json.dumps(example["messages"], ensure_ascii=False),
        "identity_knowledge": example.get("identity_knowledge", ""),
        "Information": _csv_value(record.get("Information", "")),
        "Category": record.get("Category", ""),
        "Question": record.get("Question", ""),
        "Options": record.get("Options", ""),
        "answer": expected,
        "frequency": record.get("frequency", ""),
        "response": response,
        "predicted": predicted or "",
        "is_correct": is_correct_answer(predicted, expected, record.get("Options", "")),
    }
    for field in IDENTITY_FIELDS:
        row[field] = _csv_value(record.get(field, ""))
    return row


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _progress_bar(done: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = min(width, int(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def evaluate(args: argparse.Namespace) -> None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for API evaluation.") from exc

    renderer = None if args.use_dataset_prompt else WVSPromptRenderer(args.prompt_config)
    empty_knowledge_fallback_renderer = (
        WVSPromptRenderer(args.identity_tendency_empty_fallback_prompt_config)
        if args.identity_tendency_empty_fallback_prompt_config else None
    )
    records = load_eval_records(args.eval_data, args.max_samples)
    if args.num_shards > 1:
        records = records[args.shard_index :: args.num_shards]
    identity_knowledge = None
    if args.identity_knowledge_dir:
        identity_knowledge = IdentityKnowledgeGuidance(
            args.identity_knowledge_dir,
            cross_question_mapping=args.identity_knowledge_cross_question_mapping,
            max_combinations=args.identity_knowledge_max_combinations,
            max_singles=args.identity_knowledge_max_singles,
        )
    if args.identity_tendency_knowledge_dir:
        if identity_knowledge is not None:
            raise ValueError("Numeric and tendency knowledge cannot be enabled in the same evaluator call")
        identity_knowledge = IdentityTendencyGuidance(
            args.identity_tendency_knowledge_dir,
            cross_question_mapping=args.identity_tendency_cross_question_mapping,
            top_k_questions=args.identity_tendency_top_k_questions,
            max_combinations=args.identity_knowledge_max_combinations,
            max_singles=args.identity_knowledge_max_singles,
            min_cross_question_similarity=args.identity_tendency_min_cross_question_similarity,
            deduplicate_identity_families=args.identity_tendency_deduplicate_identity_families,
            transfer_graph=args.identity_tendency_transfer_graph,
        )
    examples = build_examples(
        records,
        renderer,
        args.use_dataset_prompt,
        identity_knowledge=identity_knowledge,
        identity_knowledge_cross_question=(args.identity_knowledge_cross_question or args.identity_tendency_cross_question),
        empty_knowledge_fallback_renderer=empty_knowledge_fallback_renderer,
    )
    output_path = Path(args.output).expanduser()
    if args.overwrite and output_path.exists():
        output_path.unlink()
    elif not args.resume and output_path.exists():
        raise FileExistsError(f"Output already exists; pass --overwrite or choose another path: {output_path}")
    processed = load_processed_uids(output_path) if args.resume else set()
    pending = [example for example in examples if example["uid"] not in processed]

    print(f"records={len(records)} valid={len(examples)} pending={len(pending)} resumed={len(processed)}")
    if renderer is not None:
        print(f"prompt_template={renderer.metadata()}")
    if not pending:
        return

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    correct = 0
    completed = 0
    finished = 0
    failed = 0
    progress_started = time.monotonic()

    def print_progress(*, end: str = "\r") -> None:
        total = len(pending)
        percent = 100.0 * finished / total if total else 100.0
        accuracy = correct / completed if completed else 0.0
        elapsed = max(time.monotonic() - progress_started, 1e-9)
        rate = finished / elapsed
        eta_seconds = (total - finished) / rate if rate > 0 else None
        eta = (
            f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
            if eta_seconds is not None else "--:--"
        )
        print(
            f"{_progress_bar(finished, total)} {finished}/{total} ({percent:5.1f}%) "
            f"success={completed} failed={failed} acc={accuracy:.4f} "
            f"rate={rate:.2f}/s eta={eta}",
            end=end,
            flush=True,
        )

    print_progress()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            futures = {
                pool.submit(request_completion, client, args, example["messages"]): example for example in batch
            }
            rows = []
            for future in as_completed(futures):
                example = futures[future]
                response = future.result()
                if response is None:
                    failed += 1
                    finished += 1
                    print_progress()
                    continue
                row = result_row(example, response)
                rows.append(row)
                completed += 1
                finished += 1
                correct += int(bool(row["is_correct"]))
                print_progress()
            append_rows(output_path, rows)
    print_progress(end="\n")

    if failed:
        raise RuntimeError(
            f"Evaluation incomplete: {failed} of {len(pending)} pending examples failed. "
            "Rerun with the same output to resume the missing records."
        )

    accuracy = correct / completed if completed else 0.0
    print(f"finished={completed} correct={correct} accuracy={accuracy:.6f} output={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name/path exposed by the API server.")
    parser.add_argument("--eval-data", required=True, help="Structured WVS parquet file.")
    parser.add_argument("--output", required=True, help="Result CSV path.")
    parser.add_argument("--prompt-config", default=str(DEFAULT_PROMPT_CONFIG), help="Runtime prompt YAML/JSON.")
    parser.add_argument("--identity-knowledge-dir", default=None)
    parser.add_argument("--identity-knowledge-cross-question-mapping", default=None)
    parser.add_argument("--identity-knowledge-cross-question", action="store_true")
    parser.add_argument(
        "--identity-knowledge-max-combinations", type=int, default=None,
        help="Optional evidence-scope cap; omit to retain all matching combination evidence.",
    )
    parser.add_argument(
        "--identity-knowledge-max-singles", type=int, default=None,
        help="Optional evidence-scope cap; omit to retain all matching single-identity evidence.",
    )
    parser.add_argument("--identity-tendency-knowledge-dir", default=None)
    parser.add_argument("--identity-tendency-cross-question-mapping", default=None)
    parser.add_argument("--identity-tendency-cross-question", action="store_true")
    parser.add_argument("--identity-tendency-top-k-questions", type=int, default=3)
    parser.add_argument("--identity-tendency-min-cross-question-similarity", type=float, default=-1.0)
    parser.add_argument("--identity-tendency-deduplicate-identity-families", action="store_true")
    parser.add_argument("--identity-tendency-transfer-graph", default=None)
    parser.add_argument("--identity-tendency-empty-fallback-prompt-config", default=None)
    parser.add_argument(
        "--use-dataset-prompt",
        action="store_true",
        help="Compatibility mode: evaluate the static prompt column instead of rendering from structured fields.",
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:5456/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--send-chat-template-kwargs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--send-enable-thinking-param",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Send DashScope-style top-level enable_thinking through OpenAI extra_body.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing result CSV before evaluation.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
