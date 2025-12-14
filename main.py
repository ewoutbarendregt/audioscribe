#!/usr/bin/env python3
"""
Audio Transcription Web App - FastAPI Backend

A web application for transcribing audio files with speaker diarization
using the Gemini API. Designed for deployment on Google Cloud Run.

Environment Variables:
    GEMINI_API_KEY - Your Gemini API key (use Secret Manager in Cloud Run)
"""

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from transcriber import transcribe_audio_with_progress, TranscriptionResult

# App version - increment with each deployment
APP_VERSION = "1.0.5"

app = FastAPI(
    title="Audio Transcription",
    description="Transcribe audio files with speaker diarization using Gemini AI",
    version=APP_VERSION
)

# Supported audio formats
SUPPORTED_FORMATS = {'.mp3', '.m4a', '.wav', '.flac', '.ogg', '.webm', '.mp4', '.mpeg', '.mpga', '.aac'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


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
        "api_key_configured": bool(os.environ.get("GEMINI_API_KEY"))
    }


@app.get("/api/version")
async def get_version():
    """Get app version."""
    return {"version": APP_VERSION}


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    speakers: Optional[int] = Form(None),
    output_format: str = Form("text")
):
    """
    Transcribe an uploaded audio file with streaming progress updates.

    Returns Server-Sent Events (SSE) with progress updates, then final result.
    """
    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured. Please set up the API key in Secret Manager."
        )

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    async def generate_sse():
        """Generate Server-Sent Events for progress and result."""
        temp_dir = tempfile.gettempdir()
        temp_filename = f"upload_{uuid.uuid4().hex}{file_ext}"
        temp_path = Path(temp_dir) / temp_filename

        try:
            # Save file
            temp_path.write_bytes(content)

            # Progress callback
            async def on_progress(stage: str, detail: str = "", percent: int = 0):
                event_data = json.dumps({
                    "type": "progress",
                    "stage": stage,
                    "detail": detail,
                    "percent": percent
                })
                yield f"data: {event_data}\n\n"

            # Send initial progress
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'Starting', 'detail': 'Preparing audio file...', 'percent': 5})}\n\n"

            # Transcribe with progress updates
            result: TranscriptionResult = None
            progress_queue = asyncio.Queue()

            async def progress_callback(stage: str, detail: str = "", percent: int = 0):
                await progress_queue.put({"stage": stage, "detail": detail, "percent": percent})

            # Start transcription in background
            transcribe_task = asyncio.create_task(
                transcribe_audio_with_progress(
                    file_path=temp_path,
                    num_speakers=speakers,
                    progress_callback=progress_callback
                )
            )

            # Stream progress updates while transcription runs
            while not transcribe_task.done():
                try:
                    progress = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                    event_data = json.dumps({"type": "progress", **progress})
                    yield f"data: {event_data}\n\n"
                except asyncio.TimeoutError:
                    pass

            # Get result
            result = await transcribe_task

            # Drain any remaining progress updates
            while not progress_queue.empty():
                progress = await progress_queue.get()
                event_data = json.dumps({"type": "progress", **progress})
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
