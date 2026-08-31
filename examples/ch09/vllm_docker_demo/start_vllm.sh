#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="m9-vllm-qwen"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/reports/m9/vllm}"
VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not reachable. Check Docker Desktop or the Docker socket permissions." >&2
  exit 1
fi

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container '$CONTAINER_NAME' already exists. Remove it before restarting:" >&2
  echo "  docker rm -f $CONTAINER_NAME" >&2
  exit 1
fi

mkdir -p "$HF_CACHE_DIR"
mkdir -p "$REPORT_DIR"

docker_args=(
  run --detach
  --name "$CONTAINER_NAME"
  --gpus all
  --ipc=host
  --publish "$PORT:8000"
  --volume "$HF_CACHE_DIR:/root/.cache/huggingface"
  --volume "$REPORT_DIR:/reports"
  --env "VLLM_WSL2_ENABLE_PIN_MEMORY=$VLLM_WSL2_ENABLE_PIN_MEMORY"
  --env "VLLM_WORKER_MULTIPROC_METHOD=spawn"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  docker_args+=(--env "HF_TOKEN=$HF_TOKEN")
fi

docker_args+=(
  vllm/vllm-openai:latest
  "$MODEL"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/reports","torch_profiler_record_shapes":true,"torch_profiler_with_memory":true,"torch_profiler_use_gzip":true,"ignore_frontend":true}'
)

if [[ -n "$MAX_NUM_SEQS" ]]; then
  docker_args+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
  docker_args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

echo "Starting vLLM container '$CONTAINER_NAME' with model '$MODEL'..."
container_id="$(docker "${docker_args[@]}")"
echo "Container started: ${container_id:0:12}"
echo "Waiting for the server on http://localhost:$PORT/v1..."

for attempt in $(seq 1 300); do
  if curl --silent --fail "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "vLLM is ready."
    exit 0
  fi

  if ! docker container inspect "$CONTAINER_NAME" --format '{{.State.Running}}' 2>/dev/null | grep -q true; then
    echo "The vLLM container stopped during startup. Recent logs:" >&2
    docker logs --tail 80 "$CONTAINER_NAME" >&2
    exit 1
  fi

  sleep 2
done

echo "Timed out while waiting for vLLM. Recent logs:" >&2
docker logs --tail 80 "$CONTAINER_NAME" >&2
exit 1
