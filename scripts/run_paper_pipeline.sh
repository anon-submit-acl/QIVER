#!/usr/bin/env bash
# Reproduce the three final QIVER stages.  Set the data/model variables below
# or export them before running this script.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
TRAIN_CSV=${TRAIN_CSV:?Set TRAIN_CSV to the external estimation CSV.}
METADATA=${METADATA:?Set METADATA to the external identity metadata JSON.}
QUESTION_METADATA=${QUESTION_METADATA:?Set QUESTION_METADATA to the external question metadata JSON.}
OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT to a writable external output directory.}
EMBEDDING_MODEL=${EMBEDDING_MODEL:?Set EMBEDDING_MODEL to an intfloat/e5-base-v2 model path.}

cd "${ROOT}"

"${PYTHON_BIN}" -m qiver.cli discover \
  --train-csv "${TRAIN_CSV}" --metadata "${METADATA}" \
  --output-dir "${OUTPUT_ROOT}/association_discovery"

"${PYTHON_BIN}" -m qiver.cli construct \
  --train-csv "${TRAIN_CSV}" --metadata "${METADATA}" \
  --question-metadata "${QUESTION_METADATA}" \
  --association-dir "${OUTPUT_ROOT}/association_discovery" \
  --output-dir "${OUTPUT_ROOT}/knowledge" \
  --embedding-model "${EMBEDDING_MODEL}"

echo "Knowledge construction completed: ${OUTPUT_ROOT}/knowledge/tendency"
echo "Run external-API reasoning with:"
echo "  ${PYTHON_BIN} -m qiver.cli reason-api --model MODEL --eval-data EVAL.parquet --knowledge-dir ${OUTPUT_ROOT}/knowledge/tendency --output predictions.csv --base-url URL"
echo "Run local-vLLM reasoning with:"
echo "  ${PYTHON_BIN} -m qiver.cli reason-local --model-path MODEL_PATH --eval-data EVAL.parquet --knowledge-dir ${OUTPUT_ROOT}/knowledge/tendency --output predictions.csv"
