#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_DIR="$ROOT_DIR/site"
PORT="${1:-8000}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required to serve the site." >&2
  exit 1
fi

if [[ ! -d "$SITE_DIR" ]]; then
  echo "Error: site directory not found: $SITE_DIR" >&2
  exit 1
fi

echo "Serving CUDA course site at http://localhost:$PORT"
echo "Press Ctrl+C to stop."

exec python3 -m http.server "$PORT" --directory "$SITE_DIR"
