#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-m9-vllm-qwen}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not reachable. Check Docker Desktop or the Docker socket permissions." >&2
  exit 1
fi

if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container '$CONTAINER_NAME' does not exist."
  exit 0
fi

echo "Stopping and removing container '$CONTAINER_NAME'..."
docker rm -f "$CONTAINER_NAME" >/dev/null
echo "Container '$CONTAINER_NAME' removed."
