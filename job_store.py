"""
Job Store Module - Manages transcription job state using Firestore

Jobs are stored in Firestore with automatic TTL for cleanup.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    filename: str = ""
    file_size: int = 0
    gcs_blob: str = ""
    speakers: Optional[int] = None
    progress_stage: str = ""
    progress_detail: str = ""
    progress_percent: int = 0
    debug_messages: list = field(default_factory=list)
    result: Optional[dict] = None
    error: Optional[str] = None
    expires_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Job':
        data['status'] = JobStatus(data['status'])
        return cls(**data)


class JobStore:
    """Abstract job store interface."""

    def create_job(self, filename: str, file_size: int, gcs_blob: str = "", speakers: Optional[int] = None) -> Job:
        raise NotImplementedError

    def get_job(self, job_id: str) -> Optional[Job]:
        raise NotImplementedError

    def update_job(self, job: Job) -> None:
        raise NotImplementedError

    def delete_job(self, job_id: str) -> None:
        raise NotImplementedError


class FirestoreJobStore(JobStore):
    """Firestore-backed job store."""

    def __init__(self, collection_name: str = "transcription_jobs"):
        from google.cloud import firestore
        self.db = firestore.Client()
        self.collection = self.db.collection(collection_name)

    def create_job(self, filename: str, file_size: int, gcs_blob: str = "", speakers: Optional[int] = None) -> Job:
        job_id = uuid.uuid4().hex[:12]
        now = datetime.utcnow()
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
            filename=filename,
            file_size=file_size,
            gcs_blob=gcs_blob,
            speakers=speakers,
            expires_at=now + timedelta(hours=24)  # Jobs expire after 24 hours
        )
        self.collection.document(job_id).set(job.to_dict())
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        doc = self.collection.document(job_id).get()
        if doc.exists:
            return Job.from_dict(doc.to_dict())
        return None

    def update_job(self, job: Job) -> None:
        job.updated_at = datetime.utcnow()
        self.collection.document(job.id).set(job.to_dict())

    def delete_job(self, job_id: str) -> None:
        self.collection.document(job_id).delete()


class InMemoryJobStore(JobStore):
    """In-memory job store (for development/testing)."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def create_job(self, filename: str, file_size: int, gcs_blob: str = "", speakers: Optional[int] = None) -> Job:
        job_id = uuid.uuid4().hex[:12]
        now = datetime.utcnow()
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
            filename=filename,
            file_size=file_size,
            gcs_blob=gcs_blob,
            speakers=speakers,
            expires_at=now + timedelta(hours=24)
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update_job(self, job: Job) -> None:
        job.updated_at = datetime.utcnow()
        self._jobs[job.id] = job

    def delete_job(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)


def get_job_store() -> JobStore:
    """Get the appropriate job store based on environment."""
    # Use Firestore in production (when on Cloud Run)
    if os.environ.get("K_SERVICE"):  # K_SERVICE is set by Cloud Run
        try:
            store = FirestoreJobStore()
            # Test connectivity with a quick operation
            print("[JobStore] Firestore initialized successfully")
            return store
        except Exception as e:
            print(f"[JobStore] Firestore failed, using in-memory store: {e}")
            return InMemoryJobStore()
    # Use in-memory store for local development
    print("[JobStore] Using in-memory store (local development)")
    return InMemoryJobStore()
