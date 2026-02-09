#!/usr/bin/env python3

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from mysql.connector import Error

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
print(PROJECT_DIR)
load_dotenv(PROJECT_DIR / ".env")

from config.db_config import DBConfig  # noqa: E402

try:
    from scripts.run_production_batch import ProductionBatchRunner  # noqa: E402
except ImportError:
    sys.path.append(str(PROJECT_DIR / "scripts"))
    from run_production_batch import ProductionBatchRunner  # noqa: E402

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_DIR / "db_batch.log"),
    ],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def get_env_int(key: str, default: int) -> int:
    """Safely read int value from environment."""
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        logger.warning(f"Invalid env value for {key}, using default {default}")
        return default


def get_env_str(key: str, default: str) -> str:
    """Safely read string value from environment."""
    value = os.getenv(key)
    if value is None or not str(value).strip():
        logger.warning(
            f"Env value for {key} not set or empty, using default '{default}'"
        )
        return default
    return value.strip()


# ---------------------------------------------------------------------
# Batch Runner
# ---------------------------------------------------------------------


class DBProductionBatchRunner(ProductionBatchRunner):
    """
    Extends ProductionBatchRunner to read/write from MySQL database.
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.db_config = DBConfig.get_config()
        logger.info(
            f"Initialized DB Batch Runner (env={os.getenv('APP_ENV', 'LOCAL')})"
        )

    def get_db_connection(self):
        """Create a new database connection."""
        try:
            connection = mysql.connector.connect(**self.db_config)
            if connection.is_connected():
                return connection
        except Error as e:
            logger.error(f"MySQL connection failed: {e}")
            raise

    def run_db_batch(self, limit: int):
        """
        Fetch pending records, process them, update DB,
        and export results to CSV.
        """
        connection = None
        processed_count = 0
        error_count = 0
        batch_results = []
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            connection = self.get_db_connection()
            cursor = connection.cursor(dictionary=True)

            table_name = "imagine_records"

            logger.info(
                f"Fetching up to {limit} records from `{table_name}` where status='CREATED'"
            )

            cursor.execute(
                f"SELECT * FROM {table_name} WHERE status = 'CREATED' LIMIT %s",
                (limit,),
            )

            records = cursor.fetchall()
            logger.info(f"Found {len(records)} records to process")

            for idx, row in enumerate(records, start=1):
                record_id = row["id"]
                report_text = row.get("report")

                csv_row = {
                    "S.No": idx,
                    "Patient ID": row.get("patient_id", ""),
                    "Patient Name": row.get("patient_name", ""),
                    "DOS": row.get("dos", ""),
                    "Reports": str(report_text)[:500] if report_text else "",
                    "batch_id": batch_id,
                    "record_idx": idx,
                    "source_file": row.get("file_name", ""),
                    "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                if not report_text or len(str(report_text).strip()) < 10:
                    logger.warning(f"Record {record_id}: Report empty or too short")
                    self._update_record(
                        connection,
                        table_name,
                        record_id,
                        status="ERROR",
                        error="Report empty or too short",
                    )
                    csv_row.update(self._error_csv_payload("Report empty or too short"))
                    batch_results.append(csv_row)
                    error_count += 1
                    continue

                try:
                    if self.verbose:
                        logger.info(f"Processing record {record_id}")

                    result = self.predict_single(report=str(report_text), exam_desc="")

                    if result.get("status") == "error":
                        self._update_record(
                            connection,
                            table_name,
                            record_id,
                            status="ERROR",
                            error=result.get("prediction_error"),
                        )
                        csv_row.update(
                            self._error_csv_payload(result.get("prediction_error"))
                        )
                        error_count += 1
                    else:
                        self._update_record(
                            connection,
                            table_name,
                            record_id,
                            status="PROCESSED",
                            result=result,
                        )
                        csv_row.update(self._success_csv_payload(result))
                        processed_count += 1

                    batch_results.append(csv_row)

                except Exception as exc:
                    import traceback

                    error_msg = f"{exc}\n{traceback.format_exc(limit=2)}"
                    logger.error(f"Record {record_id} failed: {error_msg}")
                    self._update_record(
                        connection,
                        table_name,
                        record_id,
                        status="ERROR",
                        error=error_msg,
                    )
                    csv_row.update(self._error_csv_payload(error_msg))
                    batch_results.append(csv_row)
                    error_count += 1

        except Error as e:
            logger.error(f"Database error: {e}")

        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()
                logger.info("Database connection closed")

        self._export_csv(batch_results, batch_id)

        logger.info(
            f"Batch complete → processed={processed_count}, errors={error_count}"
        )

    # -----------------------------------------------------------------

    def _export_csv(self, batch_results, batch_id: str):
        if not batch_results:
            return

        results_dir = PROJECT_DIR / "results"
        results_dir.mkdir(exist_ok=True)

        csv_path = results_dir / f"batch_{batch_id}.csv"
        pd.DataFrame(batch_results).to_csv(csv_path, index=False)

        logger.info(f"CSV exported → {csv_path}")

    def _error_csv_payload(self, message: Optional[str]):
        return {
            "CPT": "ERROR",
            "Modifier": "",
            "ICD10_Diagnosis": "",
            "Confidence_Score": 0,
            "RAG_Match_Score": 0,
            "CPT_Was_Extracted": False,
            "Has_Medical_Necessity": False,
            "Medical_Necessity_Warning": (message or "")[:200],
            "LLM_Raw_Response": "",
            "Preprocess_Time_ms": 0,
            "LLM_Time_ms": 0,
            "Postprocess_Time_ms": 0,
            "status": "error",
        }

    def _success_csv_payload(self, result: dict):
        return {
            "CPT": result.get("predicted_cpt", ""),
            "Modifier": result.get("predicted_modifier", ""),
            "ICD10_Diagnosis": result.get("predicted_icd", ""),
            "Confidence_Score": result.get("confidence_score", 0),
            "RAG_Match_Score": result.get("rag_match_score", 0),
            "CPT_Was_Extracted": result.get("cpt_was_extracted", False),
            "Has_Medical_Necessity": result.get("has_medical_necessity", True),
            "Medical_Necessity_Warning": result.get("medical_necessity_warning", ""),
            "LLM_Raw_Response": result.get("llm_raw_response", ""),
            "Preprocess_Time_ms": result.get("preprocess_time_ms", 0),
            "LLM_Time_ms": result.get("llm_time_ms", 0),
            "Postprocess_Time_ms": result.get("postprocess_time_ms", 0),
            "status": "processed",
        }

    def _update_record(
        self,
        connection,
        table_name: str,
        record_id,
        status: str,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ):
        cursor = connection.cursor()
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if status == "ERROR":
            cursor.execute(
                f"""
                UPDATE {table_name}
                SET status=%s, log=%s, updated_at=%s
                WHERE id=%s
                """,
                ("ERROR", error, updated_at, record_id),
            )

        elif status == "PROCESSED" and result:
            log_data = {
                k: v
                for k, v in result.items()
                if k
                not in {
                    "closest_samples",
                    "rag_cpt_candidates",
                    "rag_icd_candidates",
                    "prediction_error",
                    "status",
                }
            }
            log_data["output_summary"] = (
                f"CPT={result.get('predicted_cpt')}, "
                f"MOD={result.get('predicted_modifier')}, "
                f"ICD={result.get('predicted_icd')}"
            )

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET status=%s, log=%s, updated_at=%s
                WHERE id=%s
                """,
                ("PROCESSED", json.dumps(log_data), updated_at, record_id),
            )

        connection.commit()
        cursor.close()


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------


def main():
    import argparse
    from apscheduler.schedulers.blocking import BlockingScheduler

    env_limit = get_env_int("BATCH_LIMIT", 50)
    env_interval = get_env_int("BATCH_INTERVAL_MINUTES", 10)

    parser = argparse.ArgumentParser(
        description="NYXMed Production Database Batch Runner"
    )
    parser.add_argument("--limit", type=int, default=env_limit)
    parser.add_argument("--interval", type=int, default=env_interval)
    parser.add_argument("--cron", action="store_true")

    args = parser.parse_args()

    if args.cron:
        args.interval = 0

    logger.info(
        f"Effective batch config -> limit={args.limit}, interval={args.interval} min"
    )

    hf_api_token = get_env_str("HF_API_TOKEN", "")

    if not hf_api_token:
        logger.error("HF_API_TOKEN is not set or empty")
        sys.exit(1)

    runner = DBProductionBatchRunner(verbose=True)

    if args.interval > 0:
        scheduler = BlockingScheduler()
        scheduler.add_job(
            func=runner.run_db_batch,
            trigger="interval",
            minutes=args.interval,
            kwargs={"limit": args.limit},
            id="db_batch_job",
            replace_existing=True,
            next_run_time=datetime.now(),
        )

        try:
            logger.info("Batch scheduler started (Ctrl+C to stop)")
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped")

    else:
        logger.info("Running batch once (cron mode)")
        runner.run_db_batch(limit=args.limit)


if __name__ == "__main__":
    main()
