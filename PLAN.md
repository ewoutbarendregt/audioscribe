# PLAN

## What we're building
Audioscribe — a web app that turns conversations into confirmed, shareable summaries.

Two entry paths, one review experience:
1. **Upload** an audio file → transcribe (Gemini) → structured summary.
2. **Record live** → real-time transcription (Gemini Live) → structured summary.

Both land on a **Review & confirm** screen where the app reads each summary point and
action item aloud (browser TTS, karaoke word-highlight), and the room can **agree** or
**object & revise** any item on the spot before exporting.

## Goals
- Make the live-conversation experience attractive and convenient ("warm dark,
  voice-first" design from the Claude Design handoff).
- Keep transcription accurate and the summary faithful to the actual discussion.
- Easy operator controls: walk through points, resume reading from any line, capture a
  participant's objection and revise that exact item.
- Works on desktop and mobile; deployable behind `/projects/audioscribe` on the
  Trustable VPS fleet.

## Approach
- FastAPI backend, Gemini for transcription/summarization, Gemini Live for realtime audio.
- Single-file vanilla-JS frontend (`static/index.html`) recreating the design.
- Latest Gemini models: `gemini-3.5-flash` (text), `gemini-2.5-flash-native-audio-latest`
  (live audio — the native-audio line is required for realtime I/O).
