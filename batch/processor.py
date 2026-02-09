from datetime import datetime


class RecordProcessor:
    def __init__(self, predictor):
        self.predictor = predictor

    def process(self, row: dict, idx: int, batch_id: str) -> tuple:
        report = row.get("report")

        if not report or len(str(report).strip()) < 10:
            raise ValueError("Report empty or too short")

        result = self.predictor.predict_single(
            report=str(report),
            exam_desc=""
        )

        csv_row = {
            "S.No": idx,
            "Patient ID": row.get("patient_id", ""),
            "Patient Name": row.get("patient_name", ""),
            "DOS": row.get("dos", ""),
            "Reports": str(report)[:500],
            "batch_id": batch_id,
            "status": result.get("status", "processed"),
            "processed_at": datetime.utcnow().isoformat(),
        }

        return csv_row, result
