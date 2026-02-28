#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash scripts/update_server.sh [branch]

BRANCH="${1:-main}"
APP_DIR="${APP_DIR:-$(pwd)}"

cd "$APP_DIR"

if [[ ! -d ".git" ]]; then
  echo "ERROR: $APP_DIR is not a git repo"
  exit 1
fi

git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

docker compose up -d --build --remove-orphans
docker compose ps
