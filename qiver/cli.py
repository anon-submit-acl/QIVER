"""Stable entry points for the paper's final QIVER pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path

from .paths import qiver_root


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _run(module: str, arguments: list[str]) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _core_module(name: str) -> str:
    return f"{__package__}.core.{name}"


def _discovery_command(args: argparse.Namespace) -> None:
    command = [
        "--cleaned-csv", str(args.train_csv),
        "--metadata", str(args.metadata),
        "--output-dir", str(args.output_dir),
        "--sample-per-country", str(args.sample_per_country),
        "--subsample-repeats", str(args.subsample_repeats),
        "--screen-permutations", str(args.screen_permutations),
        "--permutations", str(args.confirm_permutations),
        "--seed", str(args.seed),
        "--workers", str(args.workers),
        *args.extra,
    ]
    _run(_core_module("association_discovery"), command)


def _construct_command(args: argparse.Namespace) -> None:
    numeric_dir = args.output_dir / "numeric"
    tendency_dir = args.output_dir / "tendency"
    summary = args.association_dir / "question_combination_summary.csv"
    _run(_core_module("numeric_knowledge"), [
        "--train-csv", str(args.train_csv),
        "--signal-summary", str(summary),
        "--cleaned-metadata", str(args.metadata),
        "--question-metadata", str(args.question_metadata),
        "--output-dir", str(numeric_dir),
        "--min-group-size", str(args.min_group_size),
        "--prior-strength", str(args.prior_strength),
    ])
    # A project-local missing cache path forces recomputation and avoids a
    # hidden dependency on historical experiment artifacts.
    _run(_core_module("knowledge_construction"), [
        "--source-knowledge-dir", str(numeric_dir),
        "--output-dir", str(tendency_dir),
        "--embedding-model", str(args.embedding_model),
        "--identity-metadata", str(args.metadata),
        "--embedding-device", args.embedding_device,
        "--embedding-batch-size", str(args.embedding_batch_size),
        "--reuse-similarity-from", str(args.output_dir / "_no_similarity_cache"),
    ])


def _reason_command(args: argparse.Namespace) -> None:
    prompt = args.prompt_config or (qiver_root() / "configs/evidence_grounded_reasoning.yaml")
    command = [
        "--model", args.model,
        "--eval-data", str(args.eval_data),
        "--output", str(args.output),
        "--prompt-config", str(prompt),
        "--identity-tendency-knowledge-dir", str(args.knowledge_dir),
        "--identity-tendency-top-k-questions", str(args.top_k_questions),
        "--identity-tendency-min-cross-question-similarity", str(args.min_similarity),
        "--identity-tendency-deduplicate-identity-families",
        "--temperature", str(args.temperature),
        "--max-tokens", str(args.max_tokens),
        "--max-workers", str(args.max_workers),
    ]
    # Omitting a cap retains every matching statement.  Explicit caps exist
    # only for evidence-scope ablations.
    if args.max_combinations is not None:
        command.extend(["--identity-knowledge-max-combinations", str(args.max_combinations)])
    if args.max_singles is not None:
        command.extend(["--identity-knowledge-max-singles", str(args.max_singles)])
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if args.cross_question:
        command.append("--identity-tendency-cross-question")
    if args.transfer_graph:
        command.extend(["--identity-tendency-transfer-graph", str(args.transfer_graph)])
    if args.overwrite:
        command.append("--overwrite")
    command.extend(args.extra)
    _run(_core_module("evaluate_api"), command)


def _wait_for_local_server(base_url: str, timeout_seconds: int, process: subprocess.Popen[object]) -> None:
    """Wait for the vLLM OpenAI-compatible endpoint owned by this command."""
    deadline = time.monotonic() + timeout_seconds
    endpoint = f"{base_url.rstrip('/')}/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Local vLLM server exited before becoming ready (exit code {process.returncode}).")
        try:
            with urlopen(endpoint, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except (URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"Timed out after {timeout_seconds}s waiting for local vLLM at {endpoint}.")


def _reason_local_command(args: argparse.Namespace) -> None:
    served_name = args.served_model_name or args.model_path.name
    server_command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(args.model_path),
        "--served-model-name", served_name,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    server_command.extend(args.vllm_extra)
    print("+", " ".join(server_command), flush=True)
    process = subprocess.Popen(server_command)
    args.model = served_name
    args.base_url = f"http://{args.host}:{args.port}/v1"
    args.api_key = args.api_key or "EMPTY"
    try:
        _wait_for_local_server(args.base_url, args.startup_timeout, process)
        _reason_command(args)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _add_reason_arguments(parser: argparse.ArgumentParser, *, local: bool = False) -> None:
    if local:
        parser.add_argument("--model-path", type=_path, required=True, help="Local Hugging Face/vLLM model directory.")
        parser.add_argument("--served-model-name", help="Name exposed by the local vLLM server (default: model directory name).")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--tensor-parallel-size", type=int, default=1)
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
        parser.add_argument("--startup-timeout", type=int, default=600)
        parser.add_argument("--vllm-extra", nargs=argparse.REMAINDER, default=[], help="Extra vLLM server options; place after --.")
    else:
        parser.add_argument("--model", required=True, help="Model name exposed by the API endpoint.")
        parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL, including /v1.")
    parser.add_argument("--eval-data", type=_path, required=True)
    parser.add_argument("--knowledge-dir", type=_path, required=True)
    parser.add_argument("--output", type=_path, required=True)
    parser.add_argument("--prompt-config", type=_path)
    parser.add_argument("--api-key")
    parser.add_argument("--top-k-questions", type=int, default=2)
    parser.add_argument("--min-similarity", type=float, default=0.90)
    parser.add_argument("--max-combinations", type=int, default=None,
                        help="Maximum combination statements; default retains all retrieved evidence.")
    parser.add_argument("--max-singles", type=int, default=None,
                        help="Maximum single-identity statements; default retains all retrieved evidence.")
    parser.add_argument("--cross-question", action="store_true")
    parser.add_argument("--transfer-graph", type=_path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    if not local:
        parser.add_argument("extra", nargs=argparse.REMAINDER, help="Additional evaluator options; place after --.")
    else:
        # vLLM-specific options are accepted by --vllm-extra; evaluator
        # options are deliberately explicit to keep runs reproducible.
        parser.set_defaults(extra=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Run Association Discovery.")
    discover.add_argument("--train-csv", type=_path, required=True)
    discover.add_argument("--metadata", type=_path, required=True)
    discover.add_argument("--output-dir", type=_path, required=True)
    discover.add_argument("--sample-per-country", type=int, default=100)
    discover.add_argument("--subsample-repeats", type=int, default=10)
    discover.add_argument("--screen-permutations", type=int, default=99)
    discover.add_argument("--confirm-permutations", type=int, default=999)
    discover.add_argument("--seed", type=int, default=42)
    discover.add_argument("--workers", type=int, default=0)
    discover.add_argument("extra", nargs=argparse.REMAINDER, help="Additional discovery options; place after --.")
    discover.set_defaults(handler=_discovery_command)

    construct = subparsers.add_parser("construct", help="Run Knowledge Construction.")
    construct.add_argument("--train-csv", type=_path, required=True)
    construct.add_argument("--metadata", type=_path, required=True)
    construct.add_argument("--question-metadata", type=_path, required=True)
    construct.add_argument("--association-dir", type=_path, required=True)
    construct.add_argument("--output-dir", type=_path, required=True)
    construct.add_argument("--embedding-model", type=_path, required=True)
    construct.add_argument("--embedding-device", default="auto")
    construct.add_argument("--embedding-batch-size", type=int, default=32)
    construct.add_argument("--min-group-size", type=int, default=30)
    construct.add_argument("--prior-strength", type=float, default=20.0)
    construct.set_defaults(handler=_construct_command)

    reason_api = subparsers.add_parser("reason-api", help="Run Evidence-Grounded Reasoning through an external API.")
    _add_reason_arguments(reason_api)
    reason_api.set_defaults(handler=_reason_command)

    reason_local = subparsers.add_parser("reason-local", help="Start vLLM locally, then run Evidence-Grounded Reasoning.")
    _add_reason_arguments(reason_local, local=True)
    reason_local.set_defaults(handler=_reason_local_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
