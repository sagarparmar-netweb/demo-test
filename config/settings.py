from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_DIR / ".env")


def _get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    batch_limit: int = int(os.getenv("BATCH_LIMIT", "50"))
    batch_interval_min: int = int(os.getenv("BATCH_INTERVAL_MINUTES", "10"))
    hf_api_token: str = os.getenv("HF_API_TOKEN", "")
    app_env: str = os.getenv("APP_ENV", "LOCAL")

    def validate(self) -> None:
        if not self.hf_api_token:
            raise RuntimeError("HF_API_TOKEN is missing")

