#!/usr/bin/env bash
# One-time / repeat server setup: free Docker disk, pull latest, build images one-by-one, start stack.
# Run on EC2 as ubuntu (with passwordless sudo) from /opt/market-prediction after Docker is installed.
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

export COMPOSE_PROFILES="${COMPOSE_PROFILES:-airflow}"

echo "==> Build images sequentially (avoids parallel builds filling small disks)"
sudo -E docker compose build --parallel 1

echo "==> Start services"
sudo -E docker compose up -d --remove-orphans

echo "==> Status"
sudo docker compose ps

echo "==> Disk after"
df -h /
