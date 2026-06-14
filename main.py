#!/usr/bin/env python3
"""
Audio Transcription Web App - FastAPI Backend

A web application for transcribing audio files with speaker diarization
using the Gemini API. Deployed on the Trustable.nl staging AND prod VPSes
under /projects/audioscribe (Caddy strips the prefix before requests reach
this app).

Environment Variables (set in /opt/trustable/audioscribe.env on each VPS):
    GEMINI_API_KEY - Gemini API key (this env file is the single source of truth)
    API_TOKEN      - Bearer token clients must send; unset = open (local dev only)
    RATE_LIMIT     - Requests per hour per IP for /api/transcribe (default: 10)
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import base64
from typing import List
from pydantic import BaseModel
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from starlette.middleware.base import BaseHTTPMiddleware
from google import genai
from google.genai import types
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from transcriber import transcribe_audio_with_progress, TranscriptionResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# App version - increment with each deployment
APP_VERSION = "1.4.0"

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
_rate_limit = os.environ.get("RATE_LIMIT", "10")
RATE_LIMIT_RULE = f"{_rate_limit}/hour"

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Audio Transcription",
    description="Transcribe audio files with speaker diarization using Gemini AI",
    version=APP_VERSION
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' blob:; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self' ws: wss:; "
            "worker-src 'self' blob:;"
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
_bearer = HTTPBearer(auto_error=False)


async def require_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> None:
    """Validate bearer token when API_TOKEN env var is configured."""
    expected = os.environ.get("API_TOKEN")
    if not expected:
        return  # Token auth disabled — dev/local mode
    if not credentials or credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API token.")


# ---------------------------------------------------------------------------
# Supported formats / limits
# ---------------------------------------------------------------------------
SUPPORTED_FORMATS = {'.mp3', '.m4a', '.wav', '.flac', '.ogg', '.webm', '.mp4', '.mpeg', '.mpga', '.aac'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

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


@app.post("/api/transcribe", dependencies=[Depends(require_token)])
@limiter.limit(RATE_LIMIT_RULE)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    speakers: Optional[int] = Form(None, ge=1, le=20),
    output_format: str = Form("text"),
):
    """
    Transcribe an uploaded audio file with streaming progress updates.

    Returns Server-Sent Events (SSE) with progress updates, then final result.
    """
    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured. Set it in /opt/trustable/audioscribe.env on the host."
        )

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Check file size before loading fully into memory to prevent OOM
    await file.seek(0, 2)
    file_size = await file.tell()
    await file.seek(0)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    # Read file content
    content = await file.read()

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
                await progress_queue.put({"type": "progress", "stage": stage, "detail": detail, "percent": percent})

            async def debug_callback(message: str):
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
            logger.error("Transcription failed: %s", e, exc_info=True)
            error_data = json.dumps({"type": "error", "message": "Transcription failed. Please try again."})
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


class TranscriptSegmentIn(BaseModel):
    speaker: str
    text: str

class SummarizeRequest(BaseModel):
    segments: List[TranscriptSegmentIn]

@app.post("/api/summarize-live", dependencies=[Depends(require_token)])
@limiter.limit(RATE_LIMIT_RULE)
async def summarize_live(request: Request, payload: SummarizeRequest):
    """
    Generate a meeting summary, action points, and action holders from the live transcript.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured."
        )

    if not payload.segments:
        return {
            "summary": "No speech was recorded in this session.",
            "action_points": []
        }

    transcript_text = ""
    for seg in payload.segments:
        transcript_text += f"{seg.speaker}: {seg.text}\n"

    client = genai.Client(api_key=api_key)
    
    prompt = f"""You are an executive assistant. Please review the following meeting transcript:

{transcript_text}

CRITICAL RULES:
1. Focus ONLY on the actual content, decisions, and topics discussed by the human speakers (e.g., Speaker 1, Speaker 2, etc.).
2. Ignore any meta-commentary, system messages, acknowledgements, or confirmations from the AI/Assistant (e.g., "I have transcribed your speech", "The assistant prepares the action points", etc.). Do not include these in the summary or action points.
3. The summary must be a synthesis of the meeting's subject matter.

Provide the following outputs:
1. A concise summary of the conversation.
2. A list of action points, including who is responsible (action holder) for each point.

Format the output clearly using Markdown. The summary must be in the same language as the conversation."""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        return {
            "summary": response.text
        }
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate summary from the transcript."
        )

class AmendSummaryRequest(BaseModel):
    segments: List[TranscriptSegmentIn]
    current_summary: str
    correction: str

@app.post("/api/amend-summary", dependencies=[Depends(require_token)])
@limiter.limit(RATE_LIMIT_RULE)
async def amend_summary(request: Request, payload: AmendSummaryRequest):
    """
    Amend the meeting summary based on a user correction.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured."
        )

    transcript_text = ""
    for seg in payload.segments:
        transcript_text += f"{seg.speaker}: {seg.text}\n"

    client = genai.Client(api_key=api_key)
    
    prompt = f"""You are an executive assistant.
Below is the meeting transcript:
{transcript_text}

Below is the CURRENT generated summary and action points:
{payload.current_summary}

The user has provided the following correction:
"{payload.correction}"

Please update the summary and list of action points + action holders to incorporate this user correction. 
CRITICAL RULES:
1. ONLY apply changes relevant to the user's correction. Keep all other details and formatting identical.
2. Output the fully updated, corrected summary and action points.
3. Keep the same Markdown formatting.
4. The output must be written in the same language as the conversation/current summary.
5. Focus only on the actual meeting content discussed by human speakers; ignore any AI/Assistant acknowledgements or system logs."""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        return {
            "summary": response.text
        }
    except Exception as e:
        logger.error(f"Error amending summary: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to amend the summary."
        )

@app.websocket("/api/live-record")
async def live_record(websocket: WebSocket, token: Optional[str] = None):
    """
    Handles real-time audio recording, streaming to Gemini Live API,
    performing real-time transcription and speaker diarization.
    """
    expected = os.environ.get("API_TOKEN")
    if expected:
        if not token or token != expected:
            logger.warning("WebSocket auth failed: invalid or missing token.")
            await websocket.close(code=1008)
            return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("WebSocket failed: GEMINI_API_KEY not set.")
        await websocket.close(code=1011)
        return

    await websocket.accept()
    logger.info("WebSocket connection accepted for live-record")

    client = genai.Client(api_key=api_key)
    
    model_id = "gemini-2.5-flash-native-audio-latest"
    
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Aoede"
                )
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(text="""You are a silent meeting observer.
Your only job is to listen to the meeting audio.
1. DO NOT transcribe the speakers' voices or output transcripts of what they say. The client already has an automatic transcription system.
2. Remain completely silent in both audio and text. Do not speak or output any text responses unless a speaker directly addresses you (e.g. "Assistant, ...") or you need to ask a critical clarifying question because something said was completely unintelligible.
3. If you are directly addressed, respond concisely in both text and audio. Prefix your text response with "Assistant: ".
""")]
        ),
        input_audio_transcription=types.AudioTranscriptionConfig()
    )

    try:
        async with client.aio.live.connect(model=model_id, config=config) as session:
            logger.info("Connected to Gemini Live API")
            
            async def receive_from_client():
                try:
                    while True:
                           data = await websocket.receive_bytes()
                           await session.send_realtime_input(
                               audio=types.Blob(
                                   data=data,
                                   mime_type="audio/pcm;rate=16000"
                               )
                           )
                except WebSocketDisconnect:
                    logger.info("Client disconnected from WebSocket")
                except Exception as e:
                    logger.error(f"Error receiving from client: {e}")
                    raise

            async def send_to_client():
                from websockets import ConnectionClosed
                try:
                    while True:
                        response = await session._receive()
                        logger.info(f"Gemini Response: server_content={bool(response.server_content)}, tool_call={bool(response.tool_call)}, tool_call_cancellation={bool(response.tool_call_cancellation)}")
                        if response.server_content:
                            logger.info(f"  server_content fields: turn_complete={response.server_content.turn_complete}, interrupted={response.server_content.interrupted}, model_turn={bool(response.server_content.model_turn)}")
                            if response.server_content.input_transcription:
                                logger.info(f"  input_transcription: {response.server_content.input_transcription.text} (finished={response.server_content.input_transcription.finished})")
                                await websocket.send_json({
                                    "type": "user_transcript",
                                    "text": response.server_content.input_transcription.text,
                                    "finished": response.server_content.input_transcription.finished
                                })
                            
                            model_turn = response.server_content.model_turn
                            if model_turn:
                                for part in model_turn.parts:
                                    if part.text:
                                        logger.info(f"  model_turn part text: {part.text}")
                                        await websocket.send_json({
                                            "type": "model_text",
                                            "text": part.text
                                        })
                                    if part.inline_data:
                                        await websocket.send_json({
                                            "type": "audio_chunk",
                                            "data": base64.b64encode(part.inline_data.data).decode('utf-8')
                                        })
                            
                            if response.server_content.turn_complete:
                                await websocket.send_json({
                                    "type": "turn_complete"
                                })

                            if response.server_content.interrupted:
                                await websocket.send_json({
                                    "type": "interrupted"
                                })
                except ConnectionClosed:
                    logger.info("Gemini Live connection closed by server.")
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    raise

            receive_task = asyncio.create_task(receive_from_client(), name="receive_from_client")
            send_task = asyncio.create_task(send_to_client(), name="send_to_client")
            
            done, pending = await asyncio.wait(
                [receive_task, send_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                try:
                    exc = task.exception()
                    if exc:
                        logger.error(f"Task {task.get_name()} failed with exception: {exc}")
                    else:
                        logger.info(f"Task {task.get_name()} completed normally.")
                except Exception as e:
                    logger.info(f"Task {task.get_name()} completed: {e}")
            
            for task in pending:
                task.cancel()

    except Exception as e:
        logger.error(f"Live record WebSocket session error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WebSocket connection closed for live-record")

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
