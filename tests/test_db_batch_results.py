
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Mock the database config and other imports that might fail
sys.modules['config.db_config'] = MagicMock()
sys.modules['mysql.connector'] = MagicMock()

from scripts.run_production_db_batch import DBProductionBatchRunner

class TestDBBatchInsertion(unittest.TestCase):
    def setUp(self):
        # Patch the init to avoid real DB connection or config loading issues during init
        with patch('scripts.run_production_db_batch.DBProductionBatchRunner.__init__', return_value=None):
            self.runner = DBProductionBatchRunner()
            # Manually set attributes that init would have set if needed
            self.runner.db_config = {}
            self.runner.verbose = True

    def test_insert_result_record_success_columns(self):
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # Test data
        record_data = {
            "Patient ID": "PAT123",
            "Patient Name": "John Doe",
            "DOS": "2023-10-27",
            "Reports": "Sample Report Text",
            "CPT": "99213",
            "Modifier": "25",
            "ICD10_Diagnosis": "R05, J01.90",
            "Confidence_Score": 0.95,
            "RAG_Match_Score": 0.88,
            "CPT_Was_Extracted": True,
            "Has_Medical_Necessity": True,
            "Medical_Necessity_Warning": "",
            "LLM_Raw_Response": '{"cpt": "99213"}',
            "Preprocess_Time_ms": 100,
            "LLM_Time_ms": 200,
            "Postprocess_Time_ms": 50,
            "batch_id": "BATCH001",
            "record_idx": 1,
            "processed_at": "2023-10-27 10:00:00",
            "status": "processed",
            "source_file": "file.txt"
        }

        # Call the method
        self.runner._insert_result_record(mock_conn, record_data)

        # Verify SQL execution
        self.assertTrue(mock_cursor.execute.called)
        args, _ = mock_cursor.execute.call_args
        query, values = args

        # Verify query structure
        self.assertIn("INSERT INTO imagine_records_results", query)
        self.assertIn("patient_id", query)
        self.assertIn("batch_id", query)

        # Verify values mapping
        # Order: patient_id, patient_name, date_of_service, report, cpt, modifier, icd10, conf, rag, cpt_ext, med_nec, warn, raw, pre, llm, post, batch, idx, proc_at, status, source
        expected_values = (
            "PAT123", "John Doe", "2023-10-27", "Sample Report Text",
            "99213", "25", "R05, J01.90",
            0.95, 0.88, 1, 1, "", '{"cpt": "99213"}',
            100, 200, 50,
            "BATCH001", 1, "2023-10-27 10:00:00", "processed", "file.txt"
        )
        
        self.assertEqual(values, expected_values)

    def test_insert_result_record_date_parsing_formats(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        test_cases = [
            ("2023-10-27", "2023-10-27"),
            ("10/27/2023", "2023-10-27"),
            ("27-10-2023", "2023-10-27"),
            ("2023/10/27", "2023-10-27"),
            ("InvalidDate", None),
            (None, None)
        ]

        for input_date, expected_date in test_cases:
            record = {"DOS": input_date}
            self.runner._insert_result_record(mock_conn, record)
            
            # Get the values passed to execute
            # call_args returns (args, kwargs)
            args, _ = mock_cursor.execute.call_args
            # args is (query, values_tuple)
            query_values = args[1]
            
            # date_of_service is the 3rd element (index 2)
            actual_date = query_values[2]
            
            self.assertEqual(actual_date, expected_date, f"Failed for input: {input_date}")

if __name__ == '__main__':
    unittest.main()
