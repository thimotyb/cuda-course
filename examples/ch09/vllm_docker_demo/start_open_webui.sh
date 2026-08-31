#!/usr/bin/env bash
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8001}"
OPEN_WEBUI_CONTAINER_NAME="${OPEN_WEBUI_CONTAINER_NAME:-m9-open-webui}"
OPEN_WEBUI_PORT="${OPEN_WEBUI_PORT:-3000}"
OPEN_WEBUI_VOLUME="${OPEN_WEBUI_VOLUME:-m9-open-webui-data}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not reachable. Check Docker Desktop or the Docker socket permissions." >&2
  exit 1
fi

if ! curl --silent --fail "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
  echo "vLLM is not ready on http://localhost:$VLLM_PORT." >&2
  echo "Start it first with: bash examples/ch09/vllm_docker_demo/start_vllm.sh" >&2
  exit 1
fi

if docker container inspect "$OPEN_WEBUI_CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container '$OPEN_WEBUI_CONTAINER_NAME' already exists. Remove it before restarting:" >&2
  echo "  docker rm -f $OPEN_WEBUI_CONTAINER_NAME" >&2
  exit 1
fi

echo "Starting Open WebUI '$OPEN_WEBUI_CONTAINER_NAME'..."
docker run --detach \
  --name "$OPEN_WEBUI_CONTAINER_NAME" \
  --publish "$OPEN_WEBUI_PORT:8080" \
  --add-host host.docker.internal:host-gateway \
  --volume "$OPEN_WEBUI_VOLUME:/app/backend/data" \
  --env "OPENAI_API_BASE_URL=http://host.docker.internal:$VLLM_PORT/v1" \
  --env "OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}" \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main

echo "Open WebUI started. Open http://localhost:$OPEN_WEBUI_PORT"
