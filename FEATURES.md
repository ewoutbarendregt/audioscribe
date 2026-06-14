# FEATURES

Status: ✅ done · 🟡 partial · ⬜ planned

## Upload / transcription
- ✅ **Live activity log** — a small 3-line scrollable panel during transcription showing
  the system's real-time progress + debug messages (chunking, upload, model calls) streamed
  from the `/api/transcribe` SSE `progress`/`debug` events.
- ✅ **Read-only document review** — uploads land on a document (summary → key points →
  action items → diarized transcript → export), with the live-only read-aloud transport,
  Interrupt, Agree/Object and "confirm with the room" controls suppressed.

## Home
- ✅ Record orb + "Start recording" → live mode
- ✅ Drag-and-drop / click audio upload (MP3, M4A, WAV, FLAC, OGG, WEBM, MP4, AAC; 100MB)
- ✅ Unsupported-format error message
- ✅ "Set API token" link (localStorage)

## Live conversation
- ✅ Reactive record orb (pulse rings + animated waveform), tap-to-pause/resume
- ✅ REC indicator + elapsed timer
- ✅ Live caption with active-speaker chip
- ✅ Real-time transcript feed (turns append live, auto-scroll)
- ✅ "In the room" participant rail
- ✅ Stop & summarize → Review
- ✅ **Speaker diarization in live mode** — server buffers the PCM and re-transcribes it
  every ~6s with `gemini-3.5-flash` to produce Speaker 1/2/… turns with timestamps
  (interim during recording + an authoritative final pass on Stop). The Live API provides
  the instant flat caption alongside it. (Long meetings: see BUG-003.)

## Review & confirm
- ✅ **Complete prose summary** of the conversation at the top, above the points to confirm
- ✅ Structured summary points + action items (owner chip + due date)
- ✅ Read-aloud (browser TTS) with word-by-word karaoke highlight
- ✅ Transport bar: prev / play-pause / next / Interrupt / reading-speed / language
- ✅ Tap any line to start reading from there
- ✅ Agree per item; progress bar + confirmed count
- ✅ Object & revise: "Listening" sheet captures the objection (mic or typed) → rewrites
  that item via `amend-summary`, badges it "Revised", resumes reading
- ✅ "Everyone's aligned" state + Export summary / Export transcript
- ✅ **Full transcript panel** — diarized, timestamped turns on the Review screen
  (expanded by default for uploads, collapsed for live); Export transcript/summary
  always available, not gated on the done state
- ✅ Responsive (desktop + mobile breakpoint at 640px)

## Cross-cutting
- ✅ Bearer/`?token=` auth, rate limiting, security headers
- ✅ PWA manifest + service worker, subpath-safe relative URLs
