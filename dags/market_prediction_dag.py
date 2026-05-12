from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'market_admin',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'market_prediction_pipeline',
    default_args=default_args,
    description='Automated pipeline for market data ingestion and model training',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    # 1. Pull data from DVC (if remote is configured)
    pull_data = BashOperator(
        task_id='dvc_pull',
        bash_command='dvc pull',
    )

    # 2. Run DVC pipeline (repro)
    # This will run build_dataset.py and train.py if dependencies changed
    run_pipeline = BashOperator(
        task_id='dvc_repro',
        bash_command='dvc repro',
    )

    # 3. Check MLflow metrics (optional check)
    check_metrics = BashOperator(
        task_id='check_mlflow',
        bash_command='python check_mlflow.py',
    )

    # 4. Push changes to GitHub (optional, use with caution in automation)
    # git_push = BashOperator(
    #     task_id='git_push',
    #     bash_command='git add . && git commit -m "Auto-update from pipeline" && git push origin main',
    # )

    pull_data >> run_pipeline >> check_metrics
