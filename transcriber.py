"""
Audio Transcription Module using Gemini API

Handles audio transcription with speaker diarization, including
automatic chunking for long audio files.
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Maximum chunk duration in minutes (Gemini works best with <15 min chunks)
MAX_CHUNK_MINUTES = 15

# MIME types for formats that Gemini can't auto-detect (especially .m4a on Cloud Run)
MIME_TYPES = {
    '.m4a': 'audio/mp4',
    '.aac': 'audio/aac',
}


@dataclass
class TranscriptSegment:
    """A segment of transcribed text with timing and speaker info."""
    timestamp: str
    text: str
    speaker: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = "unknown"
    summary: str = ""
    speaker_count: int = 0


def get_audio_duration_ffprobe(file_path: Path) -> Optional[float]:
    """Get audio duration using ffprobe if available."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def get_audio_duration_pydub(file_path: Path) -> Optional[float]:
    """Get audio duration using pydub."""
    if not PYDUB_AVAILABLE:
        return None
    try:
        audio = AudioSegment.from_file(str(file_path))
        return len(audio) / 1000.0
    except Exception:
        return None


def split_audio_into_chunks(file_path: Path, chunk_minutes: int = MAX_CHUNK_MINUTES) -> list[Path]:
    """Split audio file into chunks of specified duration."""
    if not PYDUB_AVAILABLE:
        raise ImportError("pydub is required for splitting audio")

    audio = AudioSegment.from_file(str(file_path))
    duration_ms = len(audio)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []
    temp_dir = tempfile.gettempdir()

    for i, start_ms in enumerate(range(0, duration_ms, chunk_ms)):
        end_ms = min(start_ms + chunk_ms, duration_ms)
        chunk = audio[start_ms:end_ms]

        chunk_path = Path(temp_dir) / f"transcribe_chunk_{os.getpid()}_{i}.mp3"
        chunk.export(str(chunk_path), format="mp3")
        chunks.append(chunk_path)

    return chunks


def format_timestamp_offset(timestamp_str: str, offset_minutes: int) -> str:
    """Add offset to a timestamp string (MM:SS or HH:MM:SS format)."""
    parts = timestamp_str.split(':')
    try:
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
            hours = 0
        elif len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            return timestamp_str

        total_minutes = hours * 60 + minutes + offset_minutes
        hours = total_minutes // 60
        minutes = total_minutes % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    except (ValueError, IndexError):
        return timestamp_str


def transcribe_chunk_sync(client, file_path: Path, num_speakers: Optional[int] = None,
                          is_continuation: bool = False,
                          previous_speaker_descriptions: Optional[str] = None) -> tuple[list[TranscriptSegment], str, str, int, str]:
    """
    Transcribe a single audio chunk using Gemini API (synchronous).
    Returns: (segments, detected_language, summary, speaker_count, speaker_descriptions)
    """
    file_ext = file_path.suffix.lower()
    mime_type = MIME_TYPES.get(file_ext)  # Only set for problematic formats

    print(f"[UPLOAD] Starting upload: {file_path.name}" + (f" ({mime_type})" if mime_type else ""))

    # Upload the audio file - only specify MIME type for formats Gemini can't auto-detect
    if mime_type:
        uploaded_file = client.files.upload(file=str(file_path), config={'mimeType': mime_type})
    else:
        uploaded_file = client.files.upload(file=str(file_path))
    print(f"[UPLOAD] Upload complete: {uploaded_file.name}")

    # Build the prompt
    speaker_hint = ""
    if num_speakers:
        speaker_hint = f"There are exactly {num_speakers} different speakers in this audio."

    continuation_hint = ""
    if is_continuation and previous_speaker_descriptions:
        continuation_hint = f"""IMPORTANT: This is a continuation of a longer recording.
The following speakers were identified in previous chunks - you MUST use the same speaker labels for the same voices:

{previous_speaker_descriptions}

Match voices to the speakers above based on their voice characteristics. Use the SAME speaker numbers."""

    prompt = f"""Process this audio file and generate a detailed transcription with accurate speaker diarization.

{speaker_hint}
{continuation_hint}

CRITICAL REQUIREMENTS FOR SPEAKER IDENTIFICATION:
1. Listen carefully to distinguish EACH unique voice by their characteristics:
   - Pitch (high, medium, low)
   - Tone (warm, sharp, nasal, deep, bright)
   - Speaking pace (fast, slow, measured)
   - Accent or speech patterns
   - Voice texture (smooth, gravelly, clear, breathy)
2. Even similar-sounding voices (e.g., multiple male or female speakers) MUST be distinguished - pay attention to subtle differences.
3. Label speakers consistently as "Speaker 1", "Speaker 2", etc.
4. When in doubt between two similar voices, listen for unique speech patterns or verbal habits.

OTHER REQUIREMENTS:
5. Provide accurate timestamps for each segment (Format: MM:SS).
6. Detect the primary language of the audio.
7. Create a brief summary of the conversation IN THE SAME LANGUAGE as the audio.
8. Transcribe ALL speech accurately, preserving the original language.
9. Provide a brief description of each speaker's voice characteristics in the speaker_descriptions field.

Output the transcription with clear speaker labels and timestamps."""

    print(f"[API] Calling Gemini API for transcription...")

    # Use structured output for consistent JSON responses
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, uploaded_file],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "language": types.Schema(
                        type=types.Type.STRING,
                        description="The primary language of the audio (e.g., 'Dutch', 'English', 'nl-NL')"
                    ),
                    "summary": types.Schema(
                        type=types.Type.STRING,
                        description="A brief summary of the conversation in the same language as the audio"
                    ),
                    "speaker_count": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of distinct speakers identified in the audio"
                    ),
                    "speaker_descriptions": types.Schema(
                        type=types.Type.STRING,
                        description="Description of each speaker's voice characteristics, e.g., 'Speaker 1: Male, deep voice, slow pace. Speaker 2: Male, higher pitch, fast talker. Speaker 3: Female, warm tone.'"
                    ),
                    "segments": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "speaker": types.Schema(
                                    type=types.Type.STRING,
                                    description="Speaker identifier (e.g., 'Speaker 1', 'Speaker 2')"
                                ),
                                "timestamp": types.Schema(
                                    type=types.Type.STRING,
                                    description="Timestamp in MM:SS format marking when this segment starts"
                                ),
                                "text": types.Schema(
                                    type=types.Type.STRING,
                                    description="The transcribed text for this segment"
                                ),
                            },
                            required=["speaker", "timestamp", "text"],
                        ),
                    ),
                },
                required=["language", "summary", "speaker_count", "speaker_descriptions", "segments"],
            ),
        ),
    )

    print(f"[API] Gemini API response received ({len(response.text)} chars)")

    # Clean up uploaded file
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass

    # Parse the JSON response
    result = None
    try:
        result = json.loads(response.text)
    except json.JSONDecodeError:
        # Try to salvage partial JSON
        try:
            text = response.text
            if '"segments"' in text:
                last_brace = text.rfind('}')
                if last_brace > 0:
                    fixed_text = text[:last_brace + 1]
                    open_brackets = fixed_text.count('[') - fixed_text.count(']')
                    open_braces = fixed_text.count('{') - fixed_text.count('}')
                    fixed_text += ']' * open_brackets
                    fixed_text += '}' * open_braces
                    result = json.loads(fixed_text)
        except json.JSONDecodeError:
            pass

    if result is None:
        # Last resort: regex extraction
        segments = []
        segment_pattern = r'\{\s*"speaker"\s*:\s*"([^"]+)"\s*,\s*"timestamp"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
        matches = re.findall(segment_pattern, response.text)

        for speaker, timestamp, text in matches:
            text = text.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
            segments.append(TranscriptSegment(timestamp=timestamp, text=text, speaker=speaker))

        lang_match = re.search(r'"language"\s*:\s*"([^"]+)"', response.text)
        summary_match = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', response.text)
        desc_match = re.search(r'"speaker_descriptions"\s*:\s*"((?:[^"\\]|\\.)*)"', response.text)
        detected_language = lang_match.group(1) if lang_match else "unknown"
        summary = summary_match.group(1) if summary_match else ""
        speaker_descriptions = desc_match.group(1) if desc_match else ""
        if summary:
            summary = summary.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
        if speaker_descriptions:
            speaker_descriptions = speaker_descriptions.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')

        return segments, detected_language, summary, len(set(s.speaker for s in segments if s.speaker)), speaker_descriptions

    # Convert to TranscriptSegment objects
    segments = []
    for seg in result.get("segments", []):
        segments.append(TranscriptSegment(
            timestamp=seg.get("timestamp", "00:00"),
            text=seg.get("text", ""),
            speaker=seg.get("speaker")
        ))

    detected_language = result.get("language", "unknown")
    summary = result.get("summary", "")
    speaker_count = result.get("speaker_count", 0)
    speaker_descriptions = result.get("speaker_descriptions", "")

    return segments, detected_language, summary, speaker_count, speaker_descriptions


# Type alias for progress callback
ProgressCallback = Optional[callable]


async def transcribe_audio_with_progress(
    file_path: Path,
    num_speakers: Optional[int] = None,
    progress_callback: ProgressCallback = None
) -> TranscriptionResult:
    """
    Transcribe audio using Gemini API with speaker diarization.
    Automatically chunks long audio files.
    Sends progress updates via callback.
    """
    async def report_progress(stage: str, detail: str = "", percent: int = 0):
        print(f"[PROGRESS] {percent}% - {stage}: {detail}")  # Console logging
        if progress_callback:
            await progress_callback(stage, detail, percent)

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    await report_progress("Connecting", "Initializing Gemini API...", 10)
    client = genai.Client(api_key=api_key)

    # Get audio duration
    await report_progress("Analyzing", "Checking audio duration...", 15)
    duration = get_audio_duration_ffprobe(file_path)
    if duration is None:
        duration = get_audio_duration_pydub(file_path)

    duration_str = ""
    if duration:
        mins = int(duration // 60)
        secs = int(duration % 60)
        duration_str = f" ({mins}:{secs:02d})"

    # Check if we need to split the audio
    duration_minutes = (duration or 0) / 60
    needs_chunking = duration_minutes > MAX_CHUNK_MINUTES and PYDUB_AVAILABLE

    if needs_chunking:
        # Split and process chunks
        num_chunks = int(duration_minutes / MAX_CHUNK_MINUTES) + 1
        await report_progress("Splitting", f"Splitting into {num_chunks} chunks...", 20)
        chunks = split_audio_into_chunks(file_path, MAX_CHUNK_MINUTES)

        all_segments = []
        detected_language = "unknown"
        summaries = []
        total_speaker_count = 0
        speaker_descriptions = ""  # Track speaker descriptions across chunks

        try:
            for i, chunk_path in enumerate(chunks):
                chunk_num = i + 1
                base_percent = 25 + int((i / len(chunks)) * 65)

                await report_progress(
                    "Transcribing",
                    f"Processing chunk {chunk_num}/{len(chunks)}...",
                    base_percent
                )

                # Run sync transcription in thread pool to not block
                loop = asyncio.get_event_loop()
                chunk_segments, chunk_lang, chunk_summary, chunk_speakers, chunk_descriptions = await loop.run_in_executor(
                    None,
                    lambda: transcribe_chunk_sync(
                        client, chunk_path, num_speakers,
                        is_continuation=(i > 0),
                        previous_speaker_descriptions=speaker_descriptions if i > 0 else None
                    )
                )

                if i == 0 and chunk_lang != "unknown":
                    detected_language = chunk_lang

                # Save speaker descriptions from first chunk to use in subsequent chunks
                if i == 0 and chunk_descriptions:
                    speaker_descriptions = chunk_descriptions

                if chunk_summary:
                    summaries.append(chunk_summary)

                total_speaker_count = max(total_speaker_count, chunk_speakers)

                # Adjust timestamps for chunk offset
                offset_minutes = i * MAX_CHUNK_MINUTES
                for seg in chunk_segments:
                    seg.timestamp = format_timestamp_offset(seg.timestamp, offset_minutes)
                    all_segments.append(seg)

                await report_progress(
                    "Transcribing",
                    f"Chunk {chunk_num}/{len(chunks)} complete - {len(chunk_segments)} segments",
                    base_percent + 10
                )

        finally:
            # Clean up chunk files
            for chunk_path in chunks:
                try:
                    chunk_path.unlink()
                except Exception:
                    pass

        await report_progress("Finalizing", "Combining results...", 95)
        combined_summary = " ".join(summaries) if summaries else ""

        await report_progress("Complete", f"Transcribed {len(all_segments)} segments", 100)

        return TranscriptionResult(
            segments=all_segments,
            language=detected_language,
            summary=combined_summary,
            speaker_count=total_speaker_count
        )

    else:
        # Process single file
        await report_progress("Uploading", f"Sending audio to Gemini{duration_str}...", 25)

        await report_progress("Transcribing", "AI is processing audio (this may take a minute)...", 40)

        loop = asyncio.get_event_loop()
        segments, detected_language, summary, speaker_count, _ = await loop.run_in_executor(
            None,
            lambda: transcribe_chunk_sync(client, file_path, num_speakers, False, None)
        )

        await report_progress("Complete", f"Transcribed {len(segments)} segments from {speaker_count} speakers", 100)

        return TranscriptionResult(
            segments=segments,
            language=detected_language,
            summary=summary,
            speaker_count=speaker_count
        )


# Keep old function for backwards compatibility
async def transcribe_audio(file_path: Path, num_speakers: Optional[int] = None) -> TranscriptionResult:
    """Transcribe audio without progress updates."""
    return await transcribe_audio_with_progress(file_path, num_speakers, None)
