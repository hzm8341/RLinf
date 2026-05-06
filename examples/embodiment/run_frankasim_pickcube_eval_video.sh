#!/usr/bin/env bash
# Roll out Franka PandaPickCube-v0 policy and encode MP4 (headless EGL).
#
# Checkpoint layout after training save:
#   <log>/<experiment>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt
#
# Overrides: EVAL_CKPT_PATH, EVAL_OUT_MP4, EVAL_EPISODES, EVAL_SEED,
#           EVAL_CAMERA=front|wrist, EVAL_LOG_BASE, EVAL_RENDER_WIDTH, EVAL_RENDER_HEIGHT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"

DEFAULT_LOG_BASE="${EVAL_LOG_BASE:-${REPO_PATH}/results/franka_sim_ppo_mlp}"
EVAL_CKPT_PATH="${EVAL_CKPT_PATH:-}"
if [[ -z "${EVAL_CKPT_PATH}" ]]; then
  # Newest checkpoint under default experiment log dir
  CKPT="$(find "${DEFAULT_LOG_BASE}/checkpoints" -path '*/actor/model_state_dict/full_weights.pt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)"
  EVAL_CKPT_PATH="${CKPT:-}"
fi

OUT_MP4="${EVAL_OUT_MP4:-${REPO_PATH}/logs/franka_pickcube_eval_rollout.mp4}"
EPISODES="${EVAL_EPISODES:-5}"
SEED="${EVAL_SEED:-0}"
CAMERA="${EVAL_CAMERA:-front}"
RW="${EVAL_RENDER_WIDTH:-1280}"
RH="${EVAL_RENDER_HEIGHT:-720}"

RANDOM_POLICY=0
if [[ "${1:-}" == "--随机" ]] || [[ "${1:-}" == "--random-policy" ]]; then
  RANDOM_POLICY=1
fi

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

EXTRA_ARGS=(
  --out "${OUT_MP4}"
  --episodes "${EPISODES}"
  --seed "${SEED}"
  --camera "${CAMERA}"
  --width "${RW}"
  --height "${RH}"
)

PY_SCRIPT="${SCRIPT_DIR}/scripts/frankasim_pickcube_rollout_mp4.py"

run_py() {
  if [[ -x "${REPO_PATH}/.venv/bin/python" ]]; then
    exec "${REPO_PATH}/.venv/bin/python" "$@"
  fi
  if command -v uv >/dev/null 2>&1; then
    exec uv --directory="${REPO_PATH}" run python "$@"
  fi
  echo "Need ${REPO_PATH}/.venv or uv."
  exit 1
}

if [[ "$RANDOM_POLICY" -eq 1 ]]; then
  mkdir -p "$(dirname "${OUT_MP4}")"
  run_py "${PY_SCRIPT}" "${EXTRA_ARGS[@]}"
elif [[ -z "${EVAL_CKPT_PATH}" || ! -f "${EVAL_CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${EVAL_CKPT_PATH:-'(empty)'}"
  echo "Set EVAL_CKPT_PATH=/path/to/full_weights.pt, or enable runner.save_ckpt_at_train_end and re-train,"
  echo "or: bash $0 --随机  # random policy rendering smoke test"
  exit 1
else
  mkdir -p "$(dirname "${OUT_MP4}")"
  run_py "${PY_SCRIPT}" --ckpt "${EVAL_CKPT_PATH}" "${EXTRA_ARGS[@]}"
fi
