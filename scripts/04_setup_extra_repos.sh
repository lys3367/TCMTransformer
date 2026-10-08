#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p external_repos

clone_if_missing() {
  local url="$1"
  local destination="$2"
  if [[ -d "${destination}" ]]; then
    echo "Found ${destination}; skip clone."
  else
    git clone --depth 1 "${url}" "${destination}"
  fi
}

# These URLs are the common public repositories used by the corresponding papers.
# If your server already has a different official clone, place it under the same
# destination folder name and this script will not overwrite it.
clone_if_missing "${SEGRNN_URL:-https://github.com/lss-1138/SegRNN.git}" \
  "external_repos/SegRNN"
clone_if_missing "${SPARSE_TSF_URL:-https://github.com/lss-1138/SparseTSF.git}" \
  "external_repos/SparseTSF"

# MTS-Mixers and MoLE have multiple community mirrors. Set MTSMIXERS_URL and
# MOLE_URL explicitly on the server if these defaults do not match your source.
if [[ -n "${MTSMIXERS_URL:-}" ]]; then
  clone_if_missing "${MTSMIXERS_URL}" "external_repos/MTS-Mixers"
else
  echo "MTSMIXERS_URL is not set. Put the official MTS-Mixers repo under external_repos/MTS-Mixers manually."
fi

if [[ -n "${MOLE_URL:-}" ]]; then
  clone_if_missing "${MOLE_URL}" "external_repos/MoLE"
else
  echo "MOLE_URL is not set. Put the official MoLE repo under external_repos/MoLE manually."
fi

if [[ -n "${NEURAL_PROCESSES_URL:-}" ]]; then
  clone_if_missing "${NEURAL_PROCESSES_URL}" "external_repos/neural-processes"
else
  echo "NEURAL_PROCESSES_URL is not set. Neural-processes is not wired into the shared forecasting trainer yet."
fi

echo "Extra repositories prepared as far as configured."
