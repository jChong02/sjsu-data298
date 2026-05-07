# This dag can only be triggered manually
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
import os
import sys
import logging
from airflow.models import Variable

sys.path.append('/opt/airflow')

import pipeline.preprocessing as preprocessing


def run_preprocessing_task(**kwargs):
    logging.info("Starting preprocessing task")
    try:
        preprocessing.run_preprocessing()
    except Exception:
        logging.exception("Preprocessing failed")
        raise
    logging.info("Preprocessing finished")


def upload_to_s3(**kwargs):
    hook = S3Hook(aws_conn_id='aws_default')

    try:
        bucket_name = Variable.get('AWS_BUCKET_S3')
    except Exception:
        bucket_name = os.environ.get('AWS_BUCKET_S3', 'sjsu-data298a-bucket')

    if not bucket_name:
        raise RuntimeError("No S3 bucket configured. Set Airflow Variable 'AWS_BUCKET_S3' or env var AWS_BUCKET_S3")

    logging.info(f"upload_to_s3: using bucket '{bucket_name}'")

    output_dir = '/tmp'
    expected_files = [
        os.path.join(output_dir, 'compiled_df.parquet'),
        os.path.join(output_dir, 'mcq_df.parquet'),
        os.path.join(output_dir, 'yn_df.parquet'),
        os.path.join(output_dir, 'huatuo_df.parquet'),
    ]

    found_files = []
    for p in expected_files:
        if os.path.isfile(p):
            size = -1
            try:
                size = os.path.getsize(p)
            except OSError:
                pass
            logging.info(f"Found cleaned file: {p} (size={size})")
            found_files.append(p)
        else:
            logging.warning(f"Expected cleaned file not found: {p}")

    if not found_files:
        logging.error("No cleaned parquet files found in /tmp. Ensure preprocessing wrote the outputs.")
        raise RuntimeError("No cleaned parquet files to upload")

    uploaded = 0
    for path in found_files:
        key = f"cleaned/{os.path.basename(path)}"
        try:
            logging.info(f"Uploading {path} -> s3://{bucket_name}/{key}")
            hook.load_file(filename=path, key=key, bucket_name=bucket_name, replace=True)
            uploaded += 1
            logging.info(f"Uploaded {path} successfully")
        except Exception:
            logging.exception(f"Failed to upload {path} to s3://{bucket_name}/{key}")
            raise

    logging.info(f"upload_to_s3: uploaded {uploaded} file(s)")


with DAG(
    dag_id='s3_preprocessing_pipeline',
    start_date=datetime(2025, 11, 4),
    schedule_interval=None,
    catchup=False,
    tags=['s3', 'preprocessing'],
) as dag:

    preprocess = PythonOperator(
        task_id='run_preprocessing',
        python_callable=run_preprocessing_task,
    )

    upload = PythonOperator(
        task_id='upload_cleaned_data',
        python_callable=upload_to_s3,
    )

    preprocess >> upload