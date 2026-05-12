#!/usr/bin/env bash
# One-time / repeat server setup: free Docker disk, pull latest, build images, start stack.
# Run on EC2 as ubuntu (with passwordless sudo) from /opt/market-prediction after Docker is installed.
#
# Small root volume (<~20 GiB free): use INSTALL_AIRFLOW=0 (default) — API + MLflow only.
# Full stack: INSTALL_AIRFLOW=1 bash scripts/install_server.sh  (needs space for two large images)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Disk before cleanup"
df -h /

echo "==> Prune Docker build cache (frees space on small volumes)"
sudo docker builder prune -af || true

echo "==> Pull latest code"
BRANCH="${DEPLOY_BRANCH:-main}"
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"

INSTALL_AIRFLOW="${INSTALL_AIRFLOW:-0}"
if [ "$INSTALL_AIRFLOW" = "1" ]; then
  export COMPOSE_PROFILES="${COMPOSE_PROFILES:-airflow}"
else
  export COMPOSE_PROFILES=""
  echo "==> INSTALL_AIRFLOW=0: API + MLflow only. Grow EBS then: INSTALL_AIRFLOW=1 bash scripts/install_server.sh"
fi

echo "==> Build API image"
sudo COMPOSE_PROFILES="$COMPOSE_PROFILES" docker compose build api

if [ "$INSTALL_AIRFLOW" = "1" ]; then
  echo "==> Build Airflow image"
  sudo COMPOSE_PROFILES="$COMPOSE_PROFILES" docker compose build airflow-init
fi

echo "==> Start services"
sudo COMPOSE_PROFILES="$COMPOSE_PROFILES" docker compose up -d --remove-orphans

echo "==> Status"
sudo COMPOSE_PROFILES="$COMPOSE_PROFILES" docker compose ps

echo "==> Disk after"
df -h /
