#!/usr/bin/env bash
# Run on the EC2 host inside the cloned repo (invoked by CI after ssh cd).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BRANCH="${DEPLOY_BRANCH:-main}"

git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"

docker compose up -d --build --remove-orphans

docker image prune -f
