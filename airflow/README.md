# Airflow Pipeline Setup Guide

> **Note:** This pipeline is documentation / lineage for how
> `data/compiled_df.parquet` was built. It is **not required** to run the
> Streamlit demo - the preprocessed outputs already ship in the top-level
> `data/` directory. Follow the steps below only if you want to re-run the
> ingestion + preprocessing on AWS S3 yourself.

## Initialize Airflow (Run First Time Only)
```bash
docker-compose run airflow-webserver airflow db init
```

##  Start Airflow Services
```bash
docker-compose up --build
```

Once all services are running, access the Airflow web UI at:  
 **http://localhost:8080**

---

##  Useful Docker Commands
```bash
# Stop all running services
docker-compose down

# Rebuild and restart services
docker-compose up --build

# Restart existing services
docker-compose restart
```

---

## Environment Configuration (.env)
Make sure the following environment variables are set in your `.env` file:

```env
AIRFLOW__WEBSERVER__SECRET_KEY=<generate_a_new_secret_key>
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=securepassword123456
AWS_BUCKET_S3=sjsu-data298a-bucket
AWS_ACCESS_KEY_ID=<your_aws_access_key>
AWS_SECRET_ACCESS_KEY=<your_aws_secret_key>
AWS_DEFAULT_REGION=<your_region>
```

> You can generate a new secret key using Python:
> ```bash
> python -c "import secrets; print(secrets.token_hex(16))"
> ```

---

## S3 Bucket Structure

### Raw Data (Manually Stored Before Running the DAG)
```
s3://sjsu-data298a-bucket/raw/
    ├── raw_huatuo_df.parquet
    ├── raw_medmcqa_df.parquet
    ├── raw_pubmedqa_df.parquet
    ├── raw_medqa_df.parquet
    └── raw_mmlu_df.parquet
```

### Generated Cleaned Data (Uploaded by DAG)
```
s3://sjsu-data298a-bucket/cleaned/
    ├── compiled_df.parquet
    ├── mcq_df.parquet
    ├── huatuo_df.parquet
    └── yn_df.parquet
```

---

## Airflow Connection Setup (Before Running the DAG)

In the Airflow web UI:
1. Go to **Admin → Connections**  
2. Create a new connection:
   - **Conn Id:** `aws_default`  
   - **Conn Type:** `Amazon Web Services`  
   - **Login:** *Your AWS Access Key ID*  
   - **Password:** *Your AWS Secret Access Key* 

This connection allows Airflow’s **S3Hook** to securely read and write files to your S3 bucket.

---
 
After completing these steps, trigger your DAG in the Airflow UI to begin ingestion and preprocessing. The pipeline will read from `raw/` in S3, process data, and upload cleaned outputs to `cleaned/`.
