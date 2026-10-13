# QIVER

This repository is the paper-facing, reproducible implementation of **QIVER**
(Question-conditioned Identity--Value Evidence Retrieval).  It deliberately
contains only the final method used in the paper:

1. **Association Discovery** -- a two-stage, country-stratified
   permutation procedure identifies stable single-identity and combination
   relations.
2. **Knowledge Construction** -- the conservative renderer turns confirmed numeric
   relations into conservative, answer-option-bound natural-language evidence
   and builds the E5 semantic question index.
3. **Evidence-Grounded Reasoning** -- the final prompt retrieves this evidence
   for a profile and asks an LLM to reason in the represented person's
   first-person perspective.

It is self-contained: it contains method code, prompts, configuration, and
execution entry points only. Raw survey data, constructed knowledge bases,
model checkpoints, API keys, and experiment outputs are external inputs or
outputs and are deliberately excluded.

All data, metadata, embedding-model, knowledge, and output locations are
explicit command-line arguments. The lower-level stage modules follow the same
rule, so they cannot silently fall back to paths from an earlier experiment.

## Layout

```
QIVER/
├── configs/
│   ├── association_discovery.md
│   └── evidence_grounded_reasoning.yaml
├── qiver/
│   ├── cli.py
│   └── paths.py
└── scripts/
    └── run_paper_pipeline.sh
```

## Quick start

Install the project, then run the commands below.  The local execution mode
also requires the optional `vllm` dependency.

```bash
pip install -e .
# Optional, for `reason-local`:
pip install -e '.[local]'
```

```bash
# 1. Association Discovery
qiver discover \
  --train-csv /path/to/estimation_respondents.csv \
  --metadata /path/to/identity_metadata.json \
  --output-dir /path/to/qiver_outputs/association_discovery

# 2. Knowledge Construction
qiver construct \
  --train-csv /path/to/estimation_respondents.csv \
  --metadata /path/to/identity_metadata.json \
  --question-metadata /path/to/question_metadata.json \
  --association-dir /path/to/qiver_outputs/association_discovery \
  --output-dir /path/to/qiver_outputs/knowledge \
  --embedding-model /path/to/intfloat__e5-base-v2

# 3a. Evidence-Grounded Reasoning through an external API
qiver reason-api \
  --model YOUR_MODEL \
  --eval-data PATH/TO/test_with_source.parquet \
  --knowledge-dir /path/to/qiver_outputs/knowledge/tendency \
  --output /path/to/qiver_outputs/predictions.csv \
  --base-url YOUR_OPENAI_COMPATIBLE_ENDPOINT

# 3b. Evidence-Grounded Reasoning with a local vLLM deployment
qiver reason-local \
  --model-path /path/to/local_model \
  --eval-data PATH/TO/test_with_source.parquet \
  --knowledge-dir /path/to/qiver_outputs/knowledge/tendency \
  --output /path/to/qiver_outputs/predictions.csv
```

`reason-api` directly calls an OpenAI-compatible endpoint. `reason-local`
starts a local vLLM OpenAI-compatible server, waits for it to become ready,
runs the same evaluation path, and shuts down that server when finished.
Consequently, the two modes share the same prompt, evidence rendering,
retrieval, and answer parsing. Use `--help` after any subcommand to inspect
the execution-specific options.

## Reproducibility boundary

Only the estimation split may be passed to `discover` and `construct`.
The two reasoning modes may read the resulting knowledge directory and an evaluation parquet,
but it never reads evaluation answers to construct or select evidence.  This is
the train-only boundary reported in the paper.

## Outputs

`construct` writes two external directories below `--output-dir`:

- `numeric/`: smoothed conditional distributions and audit records;
- `tendency/`: natural-language evidence, semantic index, retrieval defaults,
  and the SQLite knowledge store used by Evidence-Grounded Reasoning.

The default retrieval configuration uses two source questions, minimum
cross-question similarity 0.90, and correlated-family deduplication. For each
retrieved question, it retains **all** matching combination-identity and
single-identity evidence statements. `--max-combinations` and `--max-singles`
are optional caps intended for controlled ablations.

The repository also retains the numeric `Distribution_Knowledge` renderer
solely to reproduce the paper's knowledge-representation ablation. It is not
the default QIVER inference path.

This repository will continue to be refined and updated.
