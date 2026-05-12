# Real-time market prediction system

Stack: **FastAPI** inference API, **MLflow** UI, optional **Apache Airflow** (Docker Compose profiles). Training and dataset build scripts live in the repo root.

## CI/CD (GitHub Actions)

Workflow: [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml).

| Stage | What it does |
|--------|----------------|
| **Test** | Install `requirements.txt`, run `pytest` (offline smoke: `tests/test_ci_smoke.py`), byte-compile sources |
| **Compose** | `docker compose --profile airflow config` |
| **Build** | Builds API and (on non-PR) Airflow images with Buildx + GHA cache |
| **Deploy** | On `push` to `main` or manual **Run workflow**: SSH to your server and run `scripts/deploy.sh` (git pull + `docker compose up`) |

### Deploy target (EC2)

Configure these **repository secrets** in GitHub (Settings → Secrets and variables → Actions):

| Secret | Example | Required |
|--------|---------|----------|
| `EC2_HOST` | `16.16.206.204` | Yes for deploy |
| `EC2_USER` | `ubuntu` | Yes for deploy |
| `EC2_SSH_KEY` | Full PEM private key text (same material as your local `.pem`; **never commit** the file) | Yes for deploy |
| `EC2_DEPLOY_PATH` | `/opt/market-prediction` | No (defaults to `/opt/market-prediction`) |

**One-time on the server**

1. Install Docker and Docker Compose plugin; ensure the deploy user can run `sudo docker …`.
2. Clone this repository to `EC2_DEPLOY_PATH` (same path as in secrets), e.g.  
   `git clone https://github.com/<your-org-or-user>/Real-time-market-prediction-system-.git /opt/market-prediction`
3. Open EC2 **security group** SSH (port 22) to GitHub Actions runners, **or** use a self-hosted runner on the instance.

Optional: for a smaller disk footprint on first deploy, on the host run `INSTALL_AIRFLOW=0 bash scripts/install_server.sh` (see script header).

### Optional integration tests

`tests/test_newsapi.py` runs only when `NEWSAPI_KEY` is set (mark `integration`). Default CI does not require it.

## Local quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -v
docker compose up --build
```

API: `http://localhost:8000/health` · MLflow: `http://localhost:5000` · Airflow (with profile): `http://localhost:8080`

Runtime expects trained artifacts (`scaler.pkl`, `mlflow.db`, `mlruns/`) per `docker-compose.yml` comments; they are gitignored.
