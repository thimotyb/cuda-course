#!/usr/bin/env bash
set -euo pipefail

OPEN_WEBUI_CONTAINER_NAME="${OPEN_WEBUI_CONTAINER_NAME:-m9-open-webui}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not reachable. Check Docker Desktop or the Docker socket permissions." >&2
  exit 1
fi

if ! docker container inspect "$OPEN_WEBUI_CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container '$OPEN_WEBUI_CONTAINER_NAME' does not exist."
  exit 0
fi

echo "Stopping and removing container '$OPEN_WEBUI_CONTAINER_NAME'..."
docker rm -f "$OPEN_WEBUI_CONTAINER_NAME" >/dev/null
echo "Container '$OPEN_WEBUI_CONTAINER_NAME' removed."
