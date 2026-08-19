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

## Sign-in
- ✅ **Email one-time-code login** — enter your address, get a 6-digit code, receive an
  httpOnly session cookie (30 days). Self-contained: no dependency on trustable's login.
- ✅ **Local user store** — permitted addresses live in a `users` table on the data
  volume, managed with `python auth.py add|remove|list` (no redeploy). `ALLOWED_EMAILS`
  seeds it once, on an empty database only
- ✅ Unregistered addresses are told so and pointed at `SUPPORT_EMAIL`, with a
  "Request access" mailto — no code is sent. Trade-off: the user list is enumerable,
  bounded by the 5/hour per-IP limit
- ✅ Removing a user drops their sessions immediately, not at cookie expiry
- ✅ Codes and session tokens stored sha256-hashed; single-use, 10-min TTL, 5-attempt cap
- ✅ WebSocket authenticates from the cookie on the handshake — no token in any URL
- ✅ "Signed in as … · Sign out"; any 401 returns you to the login screen
- ✅ Legacy `Bearer API_TOKEN` still accepted for scripted API access

## Cross-cutting
- ✅ Rate limiting, security headers
- ✅ PWA manifest + service worker, subpath-safe relative URLs
