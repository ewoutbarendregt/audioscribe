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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from transcriber import transcribe_audio_with_progress, TranscriptionResult

# App version - increment with each deployment
APP_VERSION = "1.2.0"

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

    client = get_gcs_client()
    if not client:
        raise HTTPException(
            status_code=500,
            detail="Could not initialize GCS client"
        )

    try:
        bucket = client.bucket(GCS_BUCKET)
        blob_name = f"uploads/{uuid.uuid4().hex}{file_ext}"
        blob = bucket.blob(blob_name)

        # Generate signed URL for upload (valid for 15 minutes)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="PUT",
            content_type=content_type,
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


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(None),
    gcs_blob: Optional[str] = Form(None),
    speakers: Optional[int] = Form(None),
    output_format: str = Form("text"),
    debug: bool = Form(False)
):
    """
    Transcribe an audio file with streaming progress updates.

    Supports two modes:
    - Direct upload: file parameter (for files < 30MB)
    - GCS upload: gcs_blob parameter (for larger files)

    Returns Server-Sent Events (SSE) with progress updates, then final result.
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

    if use_gcs:
        if not GCS_BUCKET:
            raise HTTPException(status_code=400, detail="GCS not configured")
        file_ext = Path(gcs_blob).suffix.lower()
    elif file:
        file_ext = Path(file.filename).suffix.lower()
        content = await file.read()

        # Check file size for direct upload
        if len(content) > MAX_DIRECT_UPLOAD:
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

    async def generate_sse():
        """Generate Server-Sent Events for progress and result."""
        temp_dir = tempfile.gettempdir()
        temp_filename = f"upload_{uuid.uuid4().hex}{file_ext}"
        temp_path = Path(temp_dir) / temp_filename
        gcs_client = None
        bucket = None
        blob = None

        try:
            if use_gcs:
                # Download from GCS
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'Downloading', 'detail': 'Fetching file from cloud storage...', 'percent': 5})}\n\n"

                gcs_client = get_gcs_client()
                if not gcs_client:
                    raise Exception("Could not initialize GCS client")

                bucket = gcs_client.bucket(GCS_BUCKET)
                blob = bucket.blob(gcs_blob)
                blob.download_to_filename(str(temp_path))

                yield f"data: {json.dumps({'type': 'progress', 'stage': 'Downloaded', 'detail': 'File ready for processing', 'percent': 10})}\n\n"
            else:
                # Save direct upload
                temp_path.write_bytes(content)
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'Starting', 'detail': 'Preparing audio file...', 'percent': 5})}\n\n"

            # Transcribe with progress updates
            result: TranscriptionResult = None
            progress_queue = asyncio.Queue()

            async def progress_callback(stage: str, detail: str = "", percent: int = 0):
                await progress_queue.put({"type": "progress", "stage": stage, "detail": detail, "percent": percent})

            async def debug_callback(message: str):
                if debug:
                    await progress_queue.put({"type": "debug", "message": message})

            # Start transcription in background
            transcribe_task = asyncio.create_task(
                transcribe_audio_with_progress(
                    file_path=temp_path,
                    num_speakers=speakers,
                    progress_callback=progress_callback,
                    debug_callback=debug_callback
                )
            )

            # Stream progress updates while transcription runs
            while not transcribe_task.done():
                try:
                    event = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                    event_data = json.dumps(event)
                    yield f"data: {event_data}\n\n"
                except asyncio.TimeoutError:
                    pass

            # Get result
            result = await transcribe_task

            # Drain any remaining progress/debug updates
            while not progress_queue.empty():
                event = await progress_queue.get()
                event_data = json.dumps(event)
                yield f"data: {event_data}\n\n"

            # Send final result
            result_data = {
                "type": "result",
                "success": True,
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
            yield f"data: {json.dumps(result_data)}\n\n"

        except Exception as e:
            error_data = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_data}\n\n"

        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

            # Clean up GCS blob
            if use_gcs and blob:
                try:
                    blob.delete()
                except Exception:
                    pass

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


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
