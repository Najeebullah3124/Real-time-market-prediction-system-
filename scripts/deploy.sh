#!/usr/bin/env bash
# Run on the EC2 host inside the cloned repo (invoked by CI after ssh cd).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BRANCH="${DEPLOY_BRANCH:-main}"

git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"

# Optional Airflow + Postgres: export COMPOSE_PROFILES=airflow on the server (needs ~20+ GiB free on / for images).
export COMPOSE_PROFILES="${COMPOSE_PROFILES:-}"

sudo COMPOSE_PROFILES="$COMPOSE_PROFILES" docker compose up -d --build --remove-orphans

sudo docker image prune -f
