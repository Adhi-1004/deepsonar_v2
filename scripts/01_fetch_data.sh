#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"

$PY -m phase.data.download --all "$@"

echo
echo "on disk:"
$PY -m phase.data.download --check --all

echo
echo "audit each dataset that landed:"
for name in deepship ds3500 esc50 qiandaoear22; do
    if [ -d "data/raw/$name" ]; then
        $PY -m phase.data.verify --root "data/raw/$name" \
            --report "data/manifests/${name}_audit.json"
        echo
    fi
done
