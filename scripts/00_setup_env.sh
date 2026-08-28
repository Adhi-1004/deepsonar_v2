#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if python -c "import torch" 2>/dev/null; then
    echo "torch already present, leaving the platform build in place"
    EXTRA=""
else
    EXTRA="torch torchaudio"
fi

python -m pip install --upgrade pip
python -m pip install -e ".[dev,data]" ${EXTRA}

mkdir -p data/raw data/manifests data/splits data/cache runs checkpoints results/tables results/figures

python - <<'PY'
import phase
from phase.config import PretrainConfig, load_config
from phase.seed import set_seed
from phase.tracking import git_sha

set_seed(0)
cfg = load_config("configs/pretrain/moco_physics_demon.yaml", PretrainConfig)
print(f"phase {phase.__version__} @ {git_sha(short=True)}")
print(f"config ok: {cfg.name} / {cfg.augment.name} / demon={cfg.demon_loss.enabled}")
PY

python -m pytest -q
echo "environment ready"
