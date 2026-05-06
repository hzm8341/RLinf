#!/usr/bin/env bash
# 从已有 RLinf checkpoint 恢复训练直至 runner.max_epochs，结束后自动用权重录 MP4。
#
# 在仓库根目录执行:
#   bash examples/embodiment/run_franka_pickcube_resume_to_mp4.sh
#
# 环境变量覆盖:
#   RESUME_DIR   含 actor/ 且路径名含 global_step_*（默认: franka_mini 的 step_1）
#   LOG_PATH     runner.logger.log_path（默认: logs/franka_pickcube_full_run）
#   EXP_NAME     runner.logger.experiment_name（默认: franka_pickcube_full）
#   TRAIN_ENVS   env.train.total_num_envs（默认: 64）
#   MAX_EPOCHS   runner.max_epochs（默认: 200）
#   MP4_OUT      输出视频路径（默认 LOG_PATH/${EXP_NAME}_rollout_full.mp4）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"
EMBODIED_PATH="$SCRIPT_DIR"

RESUME_DIR="${RESUME_DIR:-$REPO_PATH/logs/franka_mini_ckpt_demo/franka_mini/checkpoints/global_step_1}"
LOG_PATH="${LOG_PATH:-$REPO_PATH/logs/franka_pickcube_full_run}"
EXP_NAME="${EXP_NAME:-franka_pickcube_full}"
TRAIN_ENVS="${TRAIN_ENVS:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
MP4_OUT="${MP4_OUT:-$LOG_PATH/${EXP_NAME}_rollout.mp4}"

PY="${REPO_PATH}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "需要 ${PY} ，请先创建 venv。"
  exit 1
fi
export PATH="${REPO_PATH}/.venv/bin:${PATH}"

if ! ray status >/dev/null 2>&1; then
  echo "[resume] Ray 未运行，启动本地 head..."
  ray start --head >/dev/null
fi

if [[ ! -d "$RESUME_DIR/actor" ]]; then
  echo "RESUME_DIR 无效（缺少 actor/）: $RESUME_DIR"
  exit 1
fi

read -r GBS MBS < <(
  "$PY" - <<PY
train_envs = int("${TRAIN_ENVS}")
roll_steps = 200
rollout_size = train_envs * roll_steps
target_gbs = 256
gbs = min(target_gbs, rollout_size)
while gbs > 0 and rollout_size % gbs != 0:
    gbs -= 1
if gbs <= 0:
    raise SystemExit("no valid global_batch_size")
for mbs in (256, 128, 64, 32, 16, 8, 4, 2, 1):
    if mbs <= gbs and gbs % mbs == 0:
        print(gbs, mbs)
        break
PY
)

export EMBODIED_PATH
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "$LOG_PATH"

TRAIN_PY="$SCRIPT_DIR/train_embodied_agent.py"
echo "[resume] RESUME_DIR=$RESUME_DIR"
echo "[resume] LOG_PATH=$LOG_PATH EXP_NAME=$EXP_NAME"
echo "[resume] TRAIN_ENVS=$TRAIN_ENVS GBS=$GBS MBS=$MBS MAX_EPOCHS=$MAX_EPOCHS"

"$PY" "$TRAIN_PY" \
  --config-path "$SCRIPT_DIR/config" --config-name frankasim_ppo_mlp \
  "env.train.total_num_envs=${TRAIN_ENVS}" \
  "actor.global_batch_size=${GBS}" \
  "actor.micro_batch_size=${MBS}" \
  "runner.max_epochs=${MAX_EPOCHS}" \
  "runner.logger.log_path=${LOG_PATH}" \
  "runner.logger.experiment_name=${EXP_NAME}" \
  "runner.resume_dir=${RESUME_DIR}"

# Avoid `cut -d' ' -f2-` breaking paths / parsing under pipefail; strip leading mtime field in bash.
latest_line="$(
  find "${LOG_PATH}/${EXP_NAME}/checkpoints" -path '*/actor/model_state_dict/full_weights.pt' \
    -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 || true
)"
CKPT="${latest_line#*[[:space:]]}"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "训练结束但未找到 full_weights.pt，请检查 LOG_PATH=${LOG_PATH}"
  exit 1
fi

echo "[resume] Checkpoint: $CKPT"
echo "[resume] Encoding MP4 -> $MP4_OUT"

"$PY" "$SCRIPT_DIR/scripts/frankasim_pickcube_rollout_mp4.py" \
  --ckpt "$CKPT" \
  --out "$MP4_OUT" \
  --episodes 8 \
  --seed 0

echo "[resume] Done: $MP4_OUT"
