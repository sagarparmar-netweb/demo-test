from pathlib import Path
import pandas as pd


def export(rows: list, batch_id: str, base_dir: Path):
    if not rows:
        return

    results = base_dir / "results"
    results.mkdir(exist_ok=True)

    path = results / f"batch_{batch_id}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
