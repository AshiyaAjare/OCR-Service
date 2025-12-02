# app/celery_app.py
import os
from celery import Celery


CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")


celery = Celery(
    "ocr_indexer",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

# Register task modules explicitly (so workers know about them)
celery.conf.update(
    include=[
        "app.tasks.indexing_tasks",
    ],
)

# Optional: tune serializers, timezone
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=False,
)
