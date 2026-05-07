import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
import logging
import os

def run_preprocessing():
    """Main entry point for preprocessing.

    This function reads raw data from S3, processes it, and saves cleaned data to local /tmp.
    The cleaned data will then be uploaded back to S3 by the DAG.
    """
    logging.info("Starting run_preprocessing")
    try:
        from airflow.hooks.S3_hook import S3Hook
        from airflow.models import Variable
        import io

        # Initialize S3 connection
        hook = S3Hook(aws_conn_id='aws_default')
        try:
            bucket_name = Variable.get('AWS_BUCKET_S3')
        except Exception:
            bucket_name = os.environ.get('AWS_BUCKET_S3', 'sjsu-data298a-bucket')

        if not bucket_name:
            raise ValueError("No S3 bucket name configured. Set AWS_BUCKET_S3 variable or AWS_BUCKET_S3 env var.")

        logging.info(f"Reading raw data from bucket: {bucket_name}")

        # List of expected raw parquet files in s3://<bucket>/raw/
        raw_files = {
            'medmcqa': 'raw_medmcqa_df.parquet',
            'pubmedqa': 'raw_pubmedqa_df.parquet',
            'medqa': 'raw_medqa_df.parquet',
            'mmlu': 'raw_mmlu_df.parquet',
            'huatuo': 'raw_huatuo_df.parquet',
        }

        def _read_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
            logging.info(f"Reading s3://{bucket}/{key}")
            obj = hook.get_key(key=key, bucket_name=bucket)
            if obj is None:
                raise FileNotFoundError(f"S3 object not found: s3://{bucket}/{key}")
            data = obj.get()['Body'].read()
            return pd.read_parquet(io.BytesIO(data))

        # MedMCQA
        medmcqa_qs = _read_parquet_from_s3(bucket_name, f"raw/{raw_files['medmcqa']}")
        logging.info(f"Loaded medmcqa: {len(medmcqa_qs)} rows")
        medmcqa_qs['num_choices'] = medmcqa_qs['options'].apply(count_answer_choices)
        medmcqa_qs['answer_label'] = medmcqa_qs.apply(
            lambda row: extract_mcq_label_fuzzy_medmcqa(row['answer'], row['options']), axis=1
        )
        medmcqa_qs["question_type"] = "MCQ"

        # PubMedQA
        pubmedqa_qs = _read_parquet_from_s3(bucket_name, f"raw/{raw_files['pubmedqa']}")
        logging.info(f"Loaded pubmedqa: {len(pubmedqa_qs)} rows")
        pubmedqa_qs['num_choices'] = 2
        pubmedqa_qs['answer_label'] = pubmedqa_qs['answer'].apply(extract_decision_pubmedqa)
        pubmedqa_qs = pubmedqa_qs[pubmedqa_qs['answer_label'] != 'maybe']
        pubmedqa_qs["question_type"] = "Y/N"

        # MedQA
        medqa_qs = _read_parquet_from_s3(bucket_name, f"raw/{raw_files['medqa']}")
        logging.info(f"Loaded medqa: {len(medqa_qs)} rows")
        medqa_qs['num_choices'] = medqa_qs['options'].apply(count_answer_choices)
        medqa_qs['answer_label'] = medqa_qs.apply(
            lambda row: extract_mcq_label_fuzzy_medqa(row['answer'], row['options'], 0.80), axis=1
        )
        medqa_qs["question_type"] = "MCQ"

        # MMLU
        mmlu_qs = _read_parquet_from_s3(bucket_name, f"raw/{raw_files['mmlu']}")
        logging.info(f"Loaded mmlu: {len(mmlu_qs)} rows")
        mmlu_qs['num_choices'] = mmlu_qs['options'].apply(count_answer_choices)
        mmlu_qs['answer_label'] = mmlu_qs.apply(
            lambda row: extract_mcq_label_fuzzy_mmlu(row['answer'], row['options'], 0.9), axis=1
        )
        mmlu_qs["question_type"] = "MCQ"

        # Huatuo
        huatuo_qs = _read_parquet_from_s3(bucket_name, f"raw/{raw_files['huatuo']}")
        logging.info(f"Loaded huatuo: {len(huatuo_qs)} rows")
        huatuo_qs["question_type"] = "Free-response"
        huatuo_qs["options"] = np.nan
        huatuo_qs["answer_label"] = np.nan

        # Compile all datasets
        desired_cols = [
            "dataset_name",
            "id_in_dataset",
            "question_type",
            "question",
            "options",
            "answer_label",
            "answer",
            "reasoning",
        ]

        compiled_df = pd.concat([
            medmcqa_qs[desired_cols],
            pubmedqa_qs[desired_cols],
            medqa_qs[desired_cols],
            mmlu_qs[desired_cols],
            huatuo_qs[desired_cols],
        ])

        compiled_df = compiled_df.reset_index(drop=True)

        # Create prompt text column
        compiled_df["prompt_text"] = compiled_df.apply(make_prompt, axis=1)

        # Ensure output directory exists and export to /tmp for Airflow task pickup
        output_dir = '/tmp'
        os.makedirs(output_dir, exist_ok=True)

        compiled_path = os.path.join(output_dir, "compiled_df.parquet")
        compiled_df.to_parquet(compiled_path)
        logging.info(f"Wrote compiled_df -> {compiled_path}")

        # Split by question type and export cleaned data
        mcq_df = compiled_df[compiled_df['question_type'] == 'MCQ'].reset_index(drop=True)
        yn_df = compiled_df[compiled_df['question_type'] == 'Y/N'].reset_index(drop=True)
        huatuo_df = compiled_df[compiled_df['question_type'] == 'Free-response'].reset_index(drop=True)

        mcq_path = os.path.join(output_dir, "mcq_df.parquet")
        yn_path = os.path.join(output_dir, "yn_df.parquet")
        huatuo_path = os.path.join(output_dir, "huatuo_df.parquet")

        mcq_df.to_parquet(mcq_path)
        yn_df.to_parquet(yn_path)
        huatuo_df.to_parquet(huatuo_path)

        logging.info(f"Wrote mcq_df -> {mcq_path}")
        logging.info(f"Wrote yn_df -> {yn_path}")
        logging.info(f"Wrote huatuo_df -> {huatuo_path}")

        logging.info(f"Total questions: {len(compiled_df)}")
        logging.info(f"MCQ questions: {len(mcq_df)}")
        logging.info(f"Y/N questions: {len(yn_df)}")
        logging.info(f"Free-response questions: {len(huatuo_df)}")

    except Exception:
        logging.exception("Error during preprocessing")
        raise

def count_answer_choices(text):
    """Count number of multiple choice options (A, B, C, D format)"""
    if pd.isnull(text):
        return 0
    lines = text.strip().split('\n')
    choice_lines = lines[1:]
    count = sum(bool(re.match(r'^[A-Z]\.', line.strip())) for line in choice_lines)
    return count

def clean_text(text):
    """Lowercase, remove newlines/tabs/extra spaces, strip, remove trailing period."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.rstrip('.')
    return text

def extract_mcq_label_fuzzy_medmcqa(answer_text, options_text, threshold=0.9):
    """Extract MCQ letter (A-D) for MedMCQA using fuzzy matching"""
    answer_main = clean_text(answer_text.split("Explanation:")[0])
    matches = re.findall(r'([A-D])\.\s*(.*)', options_text)

    best_letter = None
    best_ratio = 0

    for letter, text in matches:
        option_text = clean_text(text)
        ratio = SequenceMatcher(None, answer_main, option_text).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_letter = letter

    return best_letter

def extract_mcq_label_fuzzy_medqa(answer_text, options_text, threshold=0.8):
    """Extract MCQ letter (A-D) for MedQA using fuzzy matching"""
    answer_main = clean_text(answer_text)
    matches = re.findall(r'([A-D])\.\s*(.*)', options_text)

    best_letter = None
    best_ratio = 0

    for letter, text in matches:
        option_text = clean_text(text)
        ratio = SequenceMatcher(None, answer_main, option_text).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_letter = letter

    return best_letter

def extract_mcq_label_fuzzy_mmlu(answer_text, options_text, threshold=0.9):
    """Extract MCQ letter (A-D) for MMLU using fuzzy matching"""
    answer_main = clean_text(answer_text)
    matches = re.findall(r'([A-D])\.\s*(.*)', options_text)

    best_letter = None
    best_ratio = 0

    for letter, text in matches:
        option_text = clean_text(text)
        ratio = SequenceMatcher(None, answer_main, option_text).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_letter = letter

    return best_letter

def extract_decision_pubmedqa(text):
    """Extract yes/no decision from PubMedQA answer"""
    match = re.search(r"The final decision is:\s*(yes|no|maybe)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None

def make_prompt(row):
    """Create prompt text from question and options"""
    q = str(row.get("question", "")).strip()
    opts = row.get("options", "")

    if pd.notnull(opts) and str(opts).strip():
        return f"Question:\n{q}\n\n{str(opts).strip()}"
    else:
        return f"Question:\n{q}"

if __name__ == "__main__":
    run_preprocessing()