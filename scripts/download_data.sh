#!/usr/bin/env bash
# Download chromatin tracing files listed in metadata/<task>_manifest.tsv from the public
# 4DN open-data S3 bucket (anonymous access, no AWS account needed) and verify their md5.
#
#   ctcf   Hafner et al. (2023) Mol Cell, https://doi.org/10.1016/j.molcel.2023.04.008
#          (12 core tables, ~140 MB)
#   axin2  Llimos et al. (2022) Nat Commun, https://doi.org/10.1038/s41467-022-29625-6
#          (12 FOF-CT tables, ~7 MB)
#
# Layout produced:  $DATA_DIR/<task>-data/<region>/<condition>/<experiment>/<file>.csv
# DATA_DIR defaults to data/ in the repo. Files already present with the right md5 are skipped.
#
# Usage: bash scripts/download_data.sh [ctcf|axin2|all]      (default: all)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
case "${TARGET}" in
  all) TASKS=(axin2 ctcf) ;;
  axin2|ctcf) TASKS=("${TARGET}") ;;
  *) echo "usage: $0 [ctcf|axin2|all]" >&2; exit 2 ;;
esac

command -v aws >/dev/null || { echo "aws CLI not found; install it first (https://aws.amazon.com/cli/)" >&2; exit 1; }

md5_of() {
  if command -v md5sum >/dev/null; then md5sum "$1" | awk '{print $1}'; else md5 -q "$1"; fi
}

for task in "${TASKS[@]}"; do
  manifest="${REPO_ROOT}/metadata/${task}_manifest.tsv"
  out_dir="${DATA_DIR}/${task}-data"
  # emit region, condition, experiment, file, md5, s3_key by header name
  awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)c[$i]=i; next}
              {print $c["region"]"\t"$c["condition"]"\t"$c["experiment"]"\t"$c["file"]"\t"$c["md5"]"\t"$c["s3_key"]}' \
      "${manifest}" |
  while IFS=$'\t' read -r region condition experiment file md5 s3_key; do
    rel="${region}/${condition}/${experiment}/${file}.csv"
    dest="${out_dir}/${rel}"
    mkdir -p "$(dirname "${dest}")"
    if [[ -f "${dest}" && "$(md5_of "${dest}")" == "${md5}" ]]; then
      echo "ok (cached)  ${task}: ${rel}"
      continue
    fi
    aws s3 cp --no-sign-request --only-show-errors "${s3_key}" "${dest}"
    got="$(md5_of "${dest}")"
    if [[ "${got}" != "${md5}" ]]; then
      echo "md5 mismatch for ${file}: expected ${md5}, got ${got}" >&2
      exit 1
    fi
    echo "downloaded   ${task}: ${rel}"
  done
done
echo "Done."
