import json
import logging
from datetime import datetime

from db.connection import mysql_connection
from db.repository import fetch_pending, mark_error, mark_processed
from batch.processor import RecordProcessor

logger = logging.getLogger(__name__)


class DBBatchRunner:
    def __init__(self, predictor):
        self.processor = RecordProcessor(predictor)

    def run(self, limit: int):
        batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        processed = 0
        errors = 0
        csv_rows = []

        with mysql_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            rows = fetch_pending(cursor, limit)

            logger.info("Fetched %d records", len(rows))

            for idx, row in enumerate(rows, start=1):
                record_id = row["id"]
                now = datetime.utcnow().isoformat()

                try:
                    csv_row, result = self.processor.process(row, idx, batch_id)
                    mark_processed(cursor, record_id, json.dumps(result), now)
                    processed += 1
                    csv_rows.append(csv_row)

                except Exception as exc:
                    mark_error(cursor, record_id, str(exc), now)
                    errors += 1
                    csv_rows.append({
                        "S.No": idx,
                        "status": "error",
                        "error": str(exc),
                    })

        logger.info(
            "Batch finished: processed=%d errors=%d",
            processed,
            errors
        )

        return csv_rows, batch_id
