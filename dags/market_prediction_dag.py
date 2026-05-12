import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

PROJECT_DIR = os.environ.get("AIRFLOW_PROJECT_DIR", "/opt/airflow/project")

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

    pull_data = BashOperator(
        task_id='dvc_pull',
        bash_command=f'cd "{PROJECT_DIR}" && dvc pull',
    )

    run_pipeline = BashOperator(
        task_id='dvc_repro',
        bash_command=f'cd "{PROJECT_DIR}" && dvc repro',
    )

    check_metrics = BashOperator(
        task_id='check_mlflow',
        bash_command=f'cd "{PROJECT_DIR}" && python check_mlflow.py',
    )

    pull_data >> run_pipeline >> check_metrics
