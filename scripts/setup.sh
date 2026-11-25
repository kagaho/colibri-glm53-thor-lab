#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor logs runtime results
engine_commit="e479104ae01bdc40ce81e675299b1d230891b321"
if [[ ! -d vendor/colibri/.git ]]; then
  git clone --depth 1 --branch v2.2.0 https://github.com/JustVugg/colibri.git vendor/colibri
fi
[[ "$(git -C vendor/colibri rev-parse HEAD)" == "$engine_commit" ]] || {
  echo "Unexpected engine revision; refusing to replace an existing checkout." >&2
  exit 1
}
make -C vendor/colibri/c -j 4 glm53 CUDA=1 CUDA_ARCH=sm_110 > logs/build-cuda.log 2>&1
make -C vendor/colibri/c cuda-glm53c-check CUDA=1 CUDA_ARCH=sm_110 > logs/cuda-glm53c-check.log 2>&1
python3 scripts/record_build.py
echo "Build and CUDA checks passed. Download separately with: python3 scripts/download_model.py --workers 4"
