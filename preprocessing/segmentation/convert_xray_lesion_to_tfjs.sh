#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'Usage: %s <saved_model_dir> <tfjs_output_dir>\n' "$0" >&2
  exit 2
fi

SAVED_MODEL_DIR="$1"
TFJS_OUTPUT_DIR="$2"

tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  "$SAVED_MODEL_DIR" \
  "$TFJS_OUTPUT_DIR"

printf 'TF.js lesion segmentation model written to %s\n' "$TFJS_OUTPUT_DIR"
