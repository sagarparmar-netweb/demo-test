from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler

from config.settings import Settings
from config.logging import configure_logging
from batch.runner import DBBatchRunner
from scripts.run_production_batch import ProductionBatchRunner


PROJECT_DIR = Path(__file__).resolve().parents[1]


def main():
    configure_logging(PROJECT_DIR)

    settings = Settings()
    settings.validate()

    predictor = ProductionBatchRunner(verbose=True)
    runner = DBBatchRunner(predictor)

    if settings.batch_interval_min > 0:
        scheduler = BlockingScheduler()
        scheduler.add_job(
            func=runner.run,
            trigger="interval",
            minutes=settings.batch_interval_min,
            kwargs={"limit": settings.batch_limit},
            id="db_batch_job",
            replace_existing=True,
        )

        scheduler.start()

    else:
        runner.run(settings.batch_limit)


if __name__ == "__main__":
    main()
