#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# nanochat base pretraining profiler run
#
# Scheme A:
#   scripts/base_train_profiler.py
#   nanochat/gpt_profiler.py
#
# Usage:
#   bash runs/run_base_profiler_2epochs.sh
#
# Switch mode:
#   RUN_MODE=compile bash runs/run_base_profiler_2epochs.sh
#   RUN_MODE=no_compile bash runs/run_base_profiler_2epochs.sh
#
# RUN_MODE:
#   compile    = keep torch.compile, use nanochat/gpt.py
#   no_compile = disable torch.compile, use nanochat/gpt_profiler.py
# ============================================================

cd "$(dirname "$0")/.."

# -------------------------
# Runtime environment
# -------------------------
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# -------------------------
# Choose profiling mode
# -------------------------
RUN_MODE="${RUN_MODE:-compile}"   # compile | no_compile

# -------------------------
# Basic config
# -------------------------
GPUS="${GPUS:-6}"
NPROC="${NPROC:-1}"

DEPTH="${DEPTH:-24}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-2048}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"

PROFILE_STEPS="${PROFILE_STEPS:-30}"
PROFILE_EPOCHS="${PROFILE_EPOCHS:--1}"

# sampled = run 2 epochs, but only record profiler windows
# continuous = run 2 epochs, record all profiler events; trace may be huge
PROFILE_MODE="${PROFILE_MODE:-sampled}"   # sampled | continuous

# Profiler schedule, only used when PROFILE_MODE=sampled
PROFILE_WAIT="${PROFILE_WAIT:-5}"
PROFILE_WARMUP="${PROFILE_WARMUP:-5}"
PROFILE_ACTIVE="${PROFILE_ACTIVE:-20}"
PROFILE_REPEAT="${PROFILE_REPEAT:-1}"

# Detail switches.
# For timing comparison, keep them off first.
PROFILE_RECORD_SHAPES="${PROFILE_RECORD_SHAPES:-0}"
PROFILE_MEMORY="${PROFILE_MEMORY:-0}"
PROFILE_WITH_STACK="${PROFILE_WITH_STACK:-0}"
PROFILE_WITH_FLOPS="${PROFILE_WITH_FLOPS:-0}"

# -------------------------
# SwanLab
# -------------------------
USE_SWANLAB="${USE_SWANLAB:-1}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-nanochat-profiler}"
SWANLAB_MODE="${SWANLAB_MODE:-local}"   # cloud | local | offline | disabled
SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-./swanlog}"
SWANLAB_TAGS="${SWANLAB_TAGS:-nanochat,profiler,base-pretrain}"

LOG_DIR="${LOG_DIR:-./logs}"
TRACE_ROOT="${TRACE_ROOT:-./profiler_traces}"

mkdir -p "${LOG_DIR}" "${TRACE_ROOT}"

# -------------------------
# Decide compile/model behavior
# -------------------------
EXTRA_ARGS=()

if [[ "${RUN_MODE}" == "compile" ]]; then
  # Real-ish performance mode:
  # - use original gpt.py
  # - keep torch.compile enabled
  MODEL_IMPORT="from nanochat.gpt import GPT, GPTConfig, Linear"
  MODE_TAG="compile"
  DISABLE_COMPILE=0

elif [[ "${RUN_MODE}" == "no_compile" ]]; then
  # Structure-analysis mode:
  # - use gpt_profiler.py with record_function labels
  # - disable torch.compile for clearer module boundaries
  MODEL_IMPORT="from nanochat.gpt_profiler import GPT, GPTConfig, Linear"
  MODE_TAG="no_compile"
  DISABLE_COMPILE=1
  EXTRA_ARGS+=(--disable-compile)

else
  echo "[ERROR] RUN_MODE must be compile or no_compile"
  echo "Example:"
  echo "  RUN_MODE=compile bash runs/run_base_profiler_2epochs.sh"
  echo "  RUN_MODE=no_compile bash runs/run_base_profiler_2epochs.sh"
  exit 1
fi

# -------------------------
# Switch import in base_train_profiler.py
# -------------------------
if [[ "${RUN_MODE}" == "compile" ]]; then
  sed -i \
    -e 's/from nanochat.gpt_profiler import GPT, GPTConfig, Linear/from nanochat.gpt import GPT, GPTConfig, Linear/' \
    -e 's/from nanochat.gpt import GPT, GPTConfig, Linear/from nanochat.gpt import GPT, GPTConfig, Linear/' \
    scripts/base_train_profiler.py
else
  sed -i \
    -e 's/from nanochat.gpt import GPT, GPTConfig, Linear/from nanochat.gpt_profiler import GPT, GPTConfig, Linear/' \
    -e 's/from nanochat.gpt_profiler import GPT, GPTConfig, Linear/from nanochat.gpt_profiler import GPT, GPTConfig, Linear/' \
    scripts/base_train_profiler.py
fi

# -------------------------
# Detail tag
# -------------------------
DETAIL_TAG="timing"
if [[ "${PROFILE_RECORD_SHAPES}" == "1" || "${PROFILE_MEMORY}" == "1" || "${PROFILE_WITH_STACK}" == "1" || "${PROFILE_WITH_FLOPS}" == "1" ]]; then
  DETAIL_TAG="detail"
fi

GPU_COUNT="${NPROC}"
GPU_IDS_TAG="${GPUS//,/x}"
GPU_TAG="gpu${GPU_COUNT}_ids${GPU_IDS_TAG}"

PROFILE_DIR="${PROFILE_DIR:-${TRACE_ROOT}/base_d${DEPTH}_s${MAX_SEQ_LEN}_${PROFILE_STEPS}steps_${GPU_TAG}_${PROFILE_MODE}_${MODE_TAG}_${DETAIL_TAG}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/profile_base_d${DEPTH}_s${MAX_SEQ_LEN}_${PROFILE_STEPS}steps_${GPU_TAG}_${PROFILE_MODE}_${MODE_TAG}_${DETAIL_TAG}.log}"
SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-base_d${DEPTH}_s${MAX_SEQ_LEN}_${PROFILE_STEPS}steps_${GPU_TAG}_${PROFILE_MODE}_${MODE_TAG}_${DETAIL_TAG}}"

mkdir -p "${PROFILE_DIR}"

echo "============================================================"
echo "nanochat profiler run"
echo "RUN_MODE=${RUN_MODE}"
echo "NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR}"
echo "GPUS=${GPUS}"
echo "NPROC=${NPROC}"
echo "DEPTH=${DEPTH}"
echo "MAX_SEQ_LEN=${MAX_SEQ_LEN}"
echo "DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE}"
echo "TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}"
echo "PROFILE_STEPS=${PROFILE_STEPS}"
echo "PROFILE_EPOCHS=${PROFILE_EPOCHS}"
echo "PROFILE_MODE=${PROFILE_MODE}"
echo "PROFILE_WAIT=${PROFILE_WAIT}"
echo "PROFILE_WARMUP=${PROFILE_WARMUP}"
echo "PROFILE_ACTIVE=${PROFILE_ACTIVE}"
echo "PROFILE_REPEAT=${PROFILE_REPEAT}"
echo "DISABLE_COMPILE=${DISABLE_COMPILE}"
echo "PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES}"
echo "PROFILE_MEMORY=${PROFILE_MEMORY}"
echo "PROFILE_WITH_STACK=${PROFILE_WITH_STACK}"
echo "PROFILE_WITH_FLOPS=${PROFILE_WITH_FLOPS}"
echo "PROFILE_DIR=${PROFILE_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "USE_SWANLAB=${USE_SWANLAB}"
echo "SWANLAB_PROJECT=${SWANLAB_PROJECT}"
echo "SWANLAB_MODE=${SWANLAB_MODE}"
echo "SWANLAB_LOGDIR=${SWANLAB_LOGDIR}"
echo "SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME}"
echo "SWANLAB_TAGS=${SWANLAB_TAGS}"
echo "Current model import:"
grep "from nanochat.gpt" scripts/base_train_profiler.py || true
echo "============================================================"

# -------------------------
# Sanity checks: scripts
# -------------------------
if [[ ! -f "scripts/base_train_profiler.py" ]]; then
  echo "[ERROR] scripts/base_train_profiler.py not found"
  exit 1
fi

if [[ ! -f "nanochat/gpt.py" ]]; then
  echo "[ERROR] nanochat/gpt.py not found"
  exit 1
fi

if [[ ! -f "nanochat/gpt_profiler.py" ]]; then
  echo "[ERROR] nanochat/gpt_profiler.py not found"
  exit 1
fi

if ! python -m scripts.base_train_profiler --help | grep -q "profile-stop-after-epochs"; then
  echo "[ERROR] scripts.base_train_profiler does not expose --profile-stop-after-epochs"
  echo "Please check whether you are running the profiler version of base_train."
  exit 1
fi

if ! python -m scripts.base_train_profiler --help | grep -q "profile-stop-after-steps"; then
  echo "[ERROR] scripts.base_train_profiler does not expose --profile-stop-after-steps"
  echo "Please check whether you added the step-stop argument to base_train_profiler.py."
  exit 1
fi

if ! python -m scripts.base_train_profiler --help | grep -q "disable-compile"; then
  echo "[ERROR] scripts.base_train_profiler does not expose --disable-compile"
  echo "Please check whether you are running the profiler version of base_train."
  exit 1
fi

# -------------------------
# Sanity checks: data/tokenizer
# -------------------------
DATA_DIR="${NANOCHAT_BASE_DIR}/base_data_climbmix"
TOKENIZER_DIR="${NANOCHAT_BASE_DIR}/tokenizer"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ERROR] Dataset directory not found: ${DATA_DIR}"
  echo "Please run:"
  echo "  python -m nanochat.dataset -n 8 -w 4"
  exit 1
fi

PARQUET_COUNT="$(find "${DATA_DIR}" -maxdepth 1 -name '*.parquet' | wc -l)"
if [[ "${PARQUET_COUNT}" -lt 2 ]]; then
  echo "[ERROR] Too few parquet files in ${DATA_DIR}: ${PARQUET_COUNT}"
  echo "Please run:"
  echo "  python -m nanochat.dataset -n 8 -w 4"
  exit 1
fi

if [[ ! -f "${TOKENIZER_DIR}/tokenizer.pkl" ]]; then
  echo "[ERROR] tokenizer.pkl not found in ${TOKENIZER_DIR}"
  echo "Please run:"
  echo "  python -m scripts.tok_train"
  exit 1
fi

if [[ ! -f "${TOKENIZER_DIR}/token_bytes.pt" ]]; then
  echo "[ERROR] token_bytes.pt not found in ${TOKENIZER_DIR}"
  echo "Please run:"
  echo "  python -m scripts.tok_train"
  exit 1
fi

# -------------------------
# Sanity checks: batch divisibility
# -------------------------
WORLD_TOKENS_PER_FWDBWD=$(( DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NPROC ))

if (( TOTAL_BATCH_SIZE % WORLD_TOKENS_PER_FWDBWD != 0 )); then
  echo "[ERROR] TOTAL_BATCH_SIZE must be divisible by DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NPROC"
  echo "TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}"
  echo "DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE}"
  echo "MAX_SEQ_LEN=${MAX_SEQ_LEN}"
  echo "NPROC=${NPROC}"
  echo "WORLD_TOKENS_PER_FWDBWD=${WORLD_TOKENS_PER_FWDBWD}"
  exit 1
fi

GRAD_ACCUM_STEPS=$(( TOTAL_BATCH_SIZE / WORLD_TOKENS_PER_FWDBWD ))

echo "PARQUET_COUNT=${PARQUET_COUNT}"
echo "WORLD_TOKENS_PER_FWDBWD=${WORLD_TOKENS_PER_FWDBWD}"
echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"

# -------------------------
# Build profiler args
# -------------------------
if [[ "${PROFILE_MODE}" == "continuous" ]]; then
  EXTRA_ARGS+=(--profile-continuous)
elif [[ "${PROFILE_MODE}" == "sampled" ]]; then
  EXTRA_ARGS+=(
    --profile-wait "${PROFILE_WAIT}"
    --profile-warmup "${PROFILE_WARMUP}"
    --profile-active "${PROFILE_ACTIVE}"
    --profile-repeat "${PROFILE_REPEAT}"
  )
else
  echo "[ERROR] PROFILE_MODE must be sampled or continuous"
  exit 1
fi

if [[ "${PROFILE_RECORD_SHAPES}" == "1" ]]; then
  EXTRA_ARGS+=(--profile-record-shapes)
fi

if [[ "${PROFILE_MEMORY}" == "1" ]]; then
  EXTRA_ARGS+=(--profile-memory)
fi

if [[ "${PROFILE_WITH_STACK}" == "1" ]]; then
  EXTRA_ARGS+=(--profile-with-stack)
fi

if [[ "${PROFILE_WITH_FLOPS}" == "1" ]]; then
  EXTRA_ARGS+=(--profile-with-flops)
fi

if [[ "${USE_SWANLAB}" == "1" ]]; then
  EXTRA_ARGS+=(
    --swanlab
    --swanlab-project "${SWANLAB_PROJECT}"
    --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME}"
    --swanlab-mode "${SWANLAB_MODE}"
    --swanlab-logdir "${SWANLAB_LOGDIR}"
    --swanlab-tags "${SWANLAB_TAGS}"
  )
fi
# -------------------------
# Launch
# -------------------------
CUDA_VISIBLE_DEVICES="${GPUS}" \
torchrun --standalone --nproc_per_node="${NPROC}" -m scripts.base_train_profiler \
  --depth="${DEPTH}" \
  --max-seq-len="${MAX_SEQ_LEN}" \
  --window-pattern="${WINDOW_PATTERN}" \
  --device-batch-size="${DEVICE_BATCH_SIZE}" \
  --total-batch-size="${TOTAL_BATCH_SIZE}" \
  --target-param-data-ratio="${TARGET_PARAM_DATA_RATIO}" \
  --num-iterations=1000000000 \
  --eval-every=-1 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1 \
  --profile \
  --profile-stop-after-steps "${PROFILE_STEPS}" \
  --profile-stop-after-epochs "${PROFILE_EPOCHS}" \
  --profile-dir "${PROFILE_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo "============================================================"
echo "Done."
echo "RUN_MODE=${RUN_MODE}"
echo "Profiler traces saved to: ${PROFILE_DIR}"
echo "Log saved to: ${LOG_FILE}"
echo "Open TensorBoard with:"
echo "  tensorboard --logdir ${TRACE_ROOT} --host 0.0.0.0 --port 6006"
echo "============================================================"