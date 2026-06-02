#!/bin/bash
# Portable GPU test runner for vllm-logits. Works on any SLURM cluster (or locally
# with a GPU — just run it directly). No hardcoded paths; repo is derived from the
# script location, and the Python interpreter is whatever is on PATH (override with
# VLLM_LOGITS_PY=/path/to/python).
#SBATCH --job-name=vllm_logits_gpu_tests
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=gpu_tests_%j.log

set -e
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# repo root = parent of this script's dir (works whether run via sbatch or directly)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY="${VLLM_LOGITS_PY:-python}"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

echo "=================== test_dual_load_qwen ==================="
"$PY" -m pytest "$REPO/tests/test_dual_load_qwen.py" -s -q || echo "QWEN TEST FAILED"

echo "=================== test_dual_load_phi4 ==================="
"$PY" -m pytest "$REPO/tests/test_dual_load_phi4.py" -s -q || echo "PHI4 TEST FAILED"

echo "=================== showcase_three_regimes ==================="
"$PY" "$REPO/examples/showcase_three_regimes.py" || echo "SHOWCASE FAILED"

echo "=================== showcase_clustering ==================="
"$PY" "$REPO/examples/showcase_clustering.py" || echo "CLUSTERING SHOWCASE FAILED"

echo "=================== ALL GPU CHECKS DONE ==================="
