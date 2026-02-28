#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash scripts/deploy_server.sh <git_repo_url> [branch] [app_dir]
# Example:
#   bash scripts/deploy_server.sh https://github.com/you/humoniod-ai.git main /opt/humoniod-ai

REPO_URL="${1:-}"
BRANCH="${2:-main}"
APP_DIR="${3:-/opt/humoniod-ai}"

if [[ -z "$REPO_URL" && ! -d "$APP_DIR/.git" ]]; then
  echo "ERROR: pass git repo url as first arg (or keep existing repo at $APP_DIR)"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is required."
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: docker compose plugin is required."
  exit 1
fi

if [[ ! -d "$APP_DIR/.git" ]]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
  echo "IMPORTANT: edit .env before production usage."
fi

mkdir -p data

docker compose up -d --build --remove-orphans
docker compose ps

echo "Waiting health check..."
sleep 5

if curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
  echo "Health OK"
else
  echo "Health endpoint not reachable yet. Check logs:"
  echo "docker compose logs --tail=200 humoniod"
fi
