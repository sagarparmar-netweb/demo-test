import logging
import sys
from pathlib import Path

def configure_logging(project_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(project_dir / "db_batch.log"),
        ],
    )
