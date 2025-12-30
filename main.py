#!/usr/bin/env python3
"""
Audio Transcription Web App - FastAPI Backend

A web application for transcribing audio files with speaker diarization
using the Gemini API. Designed for deployment on Google Cloud Run.

Environment Variables:
    GEMINI_API_KEY - Your Gemini API key (use Secret Manager in Cloud Run)
    GCS_BUCKET - Google Cloud Storage bucket for large file uploads (optional)
"""

import asyncio
import json
import os
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from transcriber import transcribe_audio_with_progress, TranscriptionResult
from job_store import get_job_store, Job, JobStatus

# App version - increment with each deployment
APP_VERSION = "2.0.1"

app = FastAPI(
    title="Audio Transcription",
    description="Transcribe audio files with speaker diarization using Gemini AI",
    version=APP_VERSION
)

# Supported audio formats
SUPPORTED_FORMATS = {'.mp3', '.m4a', '.wav', '.flac', '.ogg', '.webm', '.mp4', '.mpeg', '.mpga', '.aac'}
MAX_DIRECT_UPLOAD = 30 * 1024 * 1024  # 30MB for direct upload (under Cloud Run's 32MB limit)
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB max via GCS

# GCS bucket for large file uploads
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global job store instance
_job_store = None

def get_store():
    global _job_store
    if _job_store is None:
        _job_store = get_job_store()
    return _job_store


def get_gcs_client():
    """Get Google Cloud Storage client."""
    try:
        from google.cloud import storage
        return storage.Client()
    except Exception:
        return None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse(content="<h1>Static files not found</h1>", status_code=404)


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "api_key_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "gcs_configured": bool(GCS_BUCKET)
    }


@app.get("/api/version")
async def get_version():
    """Get app version."""
    return {"version": APP_VERSION}


@app.get("/api/upload-config")
async def get_upload_config():
    """Get upload configuration including size limits."""
    return {
        "max_direct_upload": MAX_DIRECT_UPLOAD,
        "max_file_size": MAX_FILE_SIZE if GCS_BUCKET else MAX_DIRECT_UPLOAD,
        "gcs_enabled": bool(GCS_BUCKET),
        "supported_formats": list(SUPPORTED_FORMATS)
    }


@app.post("/api/get-upload-url")
async def get_upload_url(
    filename: str = Form(...),
    content_type: str = Form(...)
):
    """
    Get a signed URL for uploading large files directly to GCS.
    This bypasses Cloud Run's 32MB request limit.
    """
    if not GCS_BUCKET:
        raise HTTPException(
            status_code=400,
            detail="Large file uploads not configured. Please set GCS_BUCKET environment variable."
        )

    # Validate file extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}"
        )

    try:
        import google.auth
        from google.auth.transport import requests
        from google.cloud import storage

        # Get credentials and create signing credentials for Cloud Run
        credentials, project = google.auth.default()

        # Refresh credentials to ensure they're valid
        auth_request = requests.Request()
        credentials.refresh(auth_request)

        # Create storage client
        client = storage.Client(credentials=credentials, project=project)
        bucket = client.bucket(GCS_BUCKET)
        blob_name = f"uploads/{uuid.uuid4().hex}{file_ext}"
        blob = bucket.blob(blob_name)

        # Use v4 signing with service account email
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="PUT",
            content_type=content_type,
            service_account_email=credentials.service_account_email,
            access_token=credentials.token,
        )

        return {
            "upload_url": url,
            "blob_name": blob_name,
            "bucket": GCS_BUCKET
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate upload URL: {str(e)}"
        )


def process_transcription_job_sync(job_id: str, temp_path: Path, speakers: Optional[int], gcs_blob: Optional[str]):
    """Background task to process transcription job (sync wrapper for async code)."""
    import asyncio

    async def run_job():
        store = get_store()
        job = store.get_job(job_id)
        if not job:
            print(f"[Job {job_id}] Job not found, aborting")
            return

        print(f"[Job {job_id}] Starting processing...")
        job.status = JobStatus.PROCESSING
        store.update_job(job)

        try:
            # Progress callback - updates job in store
            async def progress_callback(stage: str, detail: str = "", percent: int = 0):
                job.progress_stage = stage
                job.progress_detail = detail
                job.progress_percent = percent
                store.update_job(job)

            # Debug callback - appends to debug messages
            async def debug_callback(message: str):
                job.debug_messages.append(message)
                # Limit debug messages to last 50
                if len(job.debug_messages) > 50:
                    job.debug_messages = job.debug_messages[-50:]
                store.update_job(job)

            # Run transcription
            result = await transcribe_audio_with_progress(
                file_path=temp_path,
                num_speakers=speakers,
                progress_callback=progress_callback,
                debug_callback=debug_callback
            )

            # Store result
            job.status = JobStatus.COMPLETED
            job.progress_percent = 100
            job.progress_stage = "Complete"
            job.progress_detail = f"Transcribed {len(result.segments)} segments"
            job.result = {
                "language": result.language,
                "summary": result.summary,
                "speaker_count": result.speaker_count,
                "segments": [
                    {
                        "speaker": seg.speaker,
                        "timestamp": seg.timestamp,
                        "text": seg.text
                    }
                    for seg in result.segments
                ]
            }
            store.update_job(job)
            print(f"[Job {job_id}] Completed successfully with {len(result.segments)} segments")

        except Exception as e:
            print(f"[Job {job_id}] Failed with error: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            store.update_job(job)

        finally:
            # Clean up temp file
            if temp_path.exists():
                try:
                    temp_path.unlink()
                    print(f"[Job {job_id}] Cleaned up temp file")
                except Exception:
                    pass

            # Clean up GCS blob
            if gcs_blob and GCS_BUCKET:
                try:
                    client = get_gcs_client()
                    if client:
                        bucket = client.bucket(GCS_BUCKET)
                        blob = bucket.blob(gcs_blob)
                        blob.delete()
                        print(f"[Job {job_id}] Cleaned up GCS blob")
                except Exception:
                    pass

    # Run the async job in a new event loop
    asyncio.run(run_job())


@app.post("/api/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    gcs_blob: Optional[str] = Form(None),
    speakers: Optional[int] = Form(None),
    output_format: str = Form("text"),
    debug: bool = Form(False)
):
    """
    Start a transcription job.

    Returns immediately with a job_id that can be polled for status.

    Supports two modes:
    - Direct upload: file parameter (for files < 30MB)
    - GCS upload: gcs_blob parameter (for larger files)
    """
    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured. Please set up the API key in Secret Manager."
        )

    # Determine file source
    use_gcs = bool(gcs_blob)
    file_ext = None
    content = None
    filename = ""
    file_size = 0

    if use_gcs:
        if not GCS_BUCKET:
            raise HTTPException(status_code=400, detail="GCS not configured")
        file_ext = Path(gcs_blob).suffix.lower()
        filename = gcs_blob.split("/")[-1]
    elif file:
        file_ext = Path(file.filename).suffix.lower()
        filename = file.filename
        content = await file.read()
        file_size = len(content)

        # Check file size for direct upload
        if file_size > MAX_DIRECT_UPLOAD:
            raise HTTPException(
                status_code=400,
                detail=f"File too large for direct upload. Maximum is {MAX_DIRECT_UPLOAD // (1024*1024)}MB. Use GCS upload for larger files."
            )
    else:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file extension
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Create job
    store = get_store()
    job = store.create_job(
        filename=filename,
        file_size=file_size,
        gcs_blob=gcs_blob or "",
        speakers=speakers
    )

    # Save file to temp location
    temp_dir = tempfile.gettempdir()
    temp_filename = f"job_{job.id}{file_ext}"
    temp_path = Path(temp_dir) / temp_filename

    if use_gcs:
        # Download from GCS
        try:
            client = get_gcs_client()
            if not client:
                raise Exception("Could not initialize GCS client")
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(gcs_blob)
            blob.download_to_filename(str(temp_path))
        except Exception as e:
            store.delete_job(job.id)
            raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")
    else:
        # Save direct upload
        temp_path.write_bytes(content)

    # Start background processing
    background_tasks.add_task(process_transcription_job_sync, job.id, temp_path, speakers, gcs_blob)

    return {
        "job_id": job.id,
        "status": job.status.value,
        "message": "Transcription job started"
    }


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a transcription job.

    Poll this endpoint to get progress updates and final result.
    """
    store = get_store()
    job = store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "job_id": job.id,
        "status": job.status.value,
        "filename": job.filename,
        "progress": {
            "stage": job.progress_stage,
            "detail": job.progress_detail,
            "percent": job.progress_percent
        },
        "debug_messages": job.debug_messages[-10:],  # Last 10 messages
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }

    if job.status == JobStatus.COMPLETED and job.result:
        response["result"] = job.result

    if job.status == JobStatus.FAILED and job.error:
        response["error"] = job.error

    return response


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its data."""
    store = get_store()
    job = store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    store.delete_job(job_id)
    return {"message": "Job deleted"}


@app.get("/manifest.json")
async def manifest():
    """Serve PWA manifest."""
    manifest_path = static_dir / "manifest.json"
    if manifest_path.exists():
        return FileResponse(manifest_path, media_type="application/json")
    raise HTTPException(status_code=404, detail="Manifest not found")


@app.get("/sw.js")
async def service_worker():
    """Serve service worker from root path."""
    sw_path = static_dir / "sw.js"
    if sw_path.exists():
        return FileResponse(sw_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service worker not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
