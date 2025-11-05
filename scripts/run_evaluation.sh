#!/usr/bin/env bash
set -euo pipefail

# This script runs a smoke test of the evaluation pipeline using a small
# model (`gemma-2-2b`) and a toy dataset subset (50 instances).
#
# To run a full evaluation, modify the arguments below (e.g., --model-name,
# --rule, --num-shots) and remove the --toy flag.

python -m icl_attribution.cli \
    --exp-date 2025-07-01 \
    --exp-name smoke-test \
    --rule distinct \
    --model-name gemma-2-2b \
    --num-shots 10 \
    --model-accuracy \
    --method-accuracy \
    --toy

