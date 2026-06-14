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
import base64
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import (
    Depends, FastAPI, File, Form, HTTPException, Request, UploadFile,
    WebSocket, WebSocketDisconnect,
)
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from transcriber import transcribe_audio_with_progress, TranscriptionResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# App version - increment with each deployment
APP_VERSION = "1.5.0"

# Gemini models.
#   TEXT_MODEL: latest GA Flash (transcription model lives in transcriber.py).
#   LIVE_MODEL: realtime audio for the live conversation. MUST be the
#   native-audio line — it's the only Live model that emits input-audio
#   transcription. The 3.x Live preview (gemini-3.1-flash-live-preview) only
#   supports AUDIO output and returns NO input transcription, which silently
#   breaks the live transcript (verified against the API 2026-06-14).
TEXT_MODEL = "gemini-3.5-flash"
LIVE_MODEL = "gemini-2.5-flash-native-audio-latest"

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


# ---------------------------------------------------------------------------
# Structured summarization — produces the review "blocks" the new UI renders:
# summary points + action items (with owner + due date). Shared by both the
# upload path and the live-conversation path.
# ---------------------------------------------------------------------------
class TranscriptSegmentIn(BaseModel):
    speaker: str
    text: str


class SummarizeRequest(BaseModel):
    segments: List[TranscriptSegmentIn]
    title: Optional[str] = None


class AmendRequest(BaseModel):
    block_type: str          # 'summary' | 'action'
    block_text: str
    owner: Optional[str] = None
    due: Optional[str] = None
    objection: str           # what the participant said
    transcript: Optional[str] = None


def _transcript_text(segments: List[TranscriptSegmentIn]) -> str:
    return "".join(f"{seg.speaker}: {seg.text}\n" for seg in segments)


_SUMMARY_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "title": types.Schema(
            type=types.Type.STRING,
            description="A short 2-5 word meeting title.",
        ),
        "overview": types.Schema(
            type=types.Type.STRING,
            description="A complete prose summary of the whole conversation — what was discussed, decided, and concluded — in 1-3 short paragraphs.",
        ),
        "summary_points": types.Schema(
            type=types.Type.ARRAY,
            description="Key decisions/conclusions, each a single self-contained sentence.",
            items=types.Schema(type=types.Type.STRING),
        ),
        "action_items": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "text": types.Schema(type=types.Type.STRING, description="The action, as an imperative sentence."),
                    "owner": types.Schema(type=types.Type.STRING, description="Person responsible, or empty string if unclear."),
                    "due": types.Schema(type=types.Type.STRING, description="Due date if mentioned (e.g. 'Fri Jun 19'), else empty string."),
                },
                required=["text", "owner", "due"],
            ),
        ),
    },
    required=["overview", "summary_points", "action_items"],
)


@app.post("/api/summarize-live", dependencies=[Depends(require_token)])
@limiter.limit(RATE_LIMIT_RULE)
async def summarize_live(request: Request, payload: SummarizeRequest):
    """Turn a transcript into structured review blocks (summary points + action items)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    if not payload.segments:
        return {"title": payload.title or "Conversation", "blocks": []}

    transcript_text = _transcript_text(payload.segments)
    prompt = f"""You are an expert meeting assistant. Review this transcript and extract a structured summary.

TRANSCRIPT:
{transcript_text}

RULES:
1. Focus ONLY on the substance discussed by the human speakers. Ignore any assistant/system acknowledgements or meta-commentary.
2. overview: a complete, readable prose summary of the whole conversation (1-3 short paragraphs) — what was discussed, the context, decisions, and outcomes.
3. summary_points: the key decisions and conclusions distilled from the overview, each one clear self-contained sentence.
4. action_items: concrete follow-ups. Set owner to the responsible person (or "" if unclear) and due to any mentioned deadline (or "").
5. Write ALL output in the SAME LANGUAGE as the conversation.
6. Be concise — prefer 2-5 summary points and only genuine action items."""

    client = genai.Client(api_key=api_key)
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=TEXT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_SUMMARY_SCHEMA,
            ),
        )
        data = json.loads(response.text)
    except Exception as e:
        logger.error("Error generating summary: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate summary from the transcript.")

    blocks = []
    for i, point in enumerate(data.get("summary_points", []), start=1):
        if point and point.strip():
            blocks.append({"id": f"s{i}", "type": "summary", "text": point.strip(),
                           "owner": "", "due": "", "agreement": "pending"})
    for i, item in enumerate(data.get("action_items", []), start=1):
        text = (item.get("text") or "").strip()
        if text:
            blocks.append({"id": f"a{i}", "type": "action", "text": text,
                           "owner": (item.get("owner") or "").strip(),
                           "due": (item.get("due") or "").strip(),
                           "agreement": "pending"})
    return {
        "title": data.get("title") or payload.title or "Conversation",
        "overview": (data.get("overview") or "").strip(),
        "blocks": blocks,
    }


_AMEND_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "text": types.Schema(type=types.Type.STRING, description="The rewritten point/action text."),
        "owner": types.Schema(type=types.Type.STRING, description="Owner for action items, else empty string."),
        "due": types.Schema(type=types.Type.STRING, description="Due date for action items, else empty string."),
    },
    required=["text"],
)


@app.post("/api/amend-summary", dependencies=[Depends(require_token)])
@limiter.limit(RATE_LIMIT_RULE)
async def amend_summary(request: Request, payload: AmendRequest):
    """Revise a single review block based on a participant's spoken objection."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    is_action = payload.block_type == "action"
    context = f"\nFULL TRANSCRIPT (for reference):\n{payload.transcript}\n" if payload.transcript else ""
    owner_line = f"\nCurrent owner: {payload.owner}\nCurrent due date: {payload.due}" if is_action else ""
    prompt = f"""You are a meeting assistant revising ONE item after a participant objected.

CURRENT ITEM ({payload.block_type}):
"{payload.block_text}"{owner_line}
{context}
THE PARTICIPANT SAID:
"{payload.objection}"

RULES:
1. Rewrite ONLY this item to incorporate the correction. Keep everything else faithful to the original meaning.
2. {"If the objection changes the owner or due date, update owner/due accordingly; otherwise keep them." if is_action else "This is a summary point — leave owner and due as empty strings."}
3. Write in the SAME LANGUAGE as the current item.
4. Return the rewritten item only."""

    client = genai.Client(api_key=api_key)
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=TEXT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_AMEND_SCHEMA,
            ),
        )
        data = json.loads(response.text)
    except Exception as e:
        logger.error("Error amending summary: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to amend the item.")

    return {
        "text": (data.get("text") or payload.block_text).strip(),
        "owner": (data.get("owner") or payload.owner or "").strip() if is_action else "",
        "due": (data.get("due") or payload.due or "").strip() if is_action else "",
    }


@app.websocket("/api/live-record")
async def live_record(websocket: WebSocket, token: Optional[str] = None):
    """Stream mic PCM to the Gemini Live API and relay real-time transcription back.

    The model acts as a silent observer (it only speaks when directly addressed);
    the client receives `user_transcript` events to build the live transcript, and
    `model_text` / `audio_chunk` when the assistant is asked something directly.
    """
    expected = os.environ.get("API_TOKEN")
    if expected and token != expected:
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
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        ),
        system_instruction=types.Content(parts=[types.Part(text="""You are a silent meeting observer.
Your only job is to listen to the meeting audio.
1. DO NOT transcribe or repeat the speakers' words. The client already transcribes them.
2. Remain completely silent — no audio, no text — unless a speaker directly addresses you (e.g. "Assistant, ...") or something was completely unintelligible and needs a clarifying question.
3. If directly addressed, respond concisely in both text and audio, and prefix your text with "Assistant: ".""")]),
        input_audio_transcription=types.AudioTranscriptionConfig(),
    )

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            logger.info("Connected to Gemini Live API")

            async def receive_from_client():
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        await session.send_realtime_input(
                            audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                        )
                except WebSocketDisconnect:
                    logger.info("Client disconnected from WebSocket")
                except Exception as e:
                    logger.error("Error receiving from client: %s", e)
                    raise

            async def send_to_client():
                try:
                    async for response in session.receive():
                        sc = response.server_content
                        if not sc:
                            continue
                        if sc.input_transcription and sc.input_transcription.text:
                            await websocket.send_json({
                                "type": "user_transcript",
                                "text": sc.input_transcription.text,
                                "finished": bool(getattr(sc.input_transcription, "finished", False)),
                            })
                        if sc.model_turn:
                            for part in sc.model_turn.parts:
                                if part.text:
                                    await websocket.send_json({"type": "model_text", "text": part.text})
                                if part.inline_data and part.inline_data.data:
                                    await websocket.send_json({
                                        "type": "audio_chunk",
                                        "data": base64.b64encode(part.inline_data.data).decode("utf-8"),
                                    })
                        if sc.turn_complete:
                            await websocket.send_json({"type": "turn_complete"})
                        if sc.interrupted:
                            await websocket.send_json({"type": "interrupted"})
                except Exception as e:
                    logger.error("Error sending to client: %s", e)
                    raise

            receive_task = asyncio.create_task(receive_from_client(), name="receive_from_client")
            send_task = asyncio.create_task(send_to_client(), name="send_to_client")
            done, pending = await asyncio.wait(
                [receive_task, send_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
    except Exception as e:
        logger.error("Live record WebSocket session error: %s", e)
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
