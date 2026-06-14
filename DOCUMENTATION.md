# DOCUMENTATION

Auto-updated by agents as they work. Newest entries first.

## [2026-06-14] — Near-real-time live transcript (faster diarization)

**Session**: claude/feat/live-transcript-realtime
**Changed**: main.py, static/index.html
**Summary**: After the stall fix, the whole recording was processed but live text barely
appeared during recording. Diagnosed via acoustic test (spoke into the mic through the
Mac speakers + browser instrumentation): the Live API instant caption is sparse/unreliable
in-browser (only ~2 fragments for 12s — likely the worklet's no-filter downsampling
aliases the audio, which the streaming ASR handles poorly while the batch diarizer is
robust to it), and the diarized transcript — though correct — only updated ~every 11s.
Fixes: (1) disable thinking on the diarization call (`ThinkingConfig(thinking_budget=0)`,
~2.5s→~1.7s); (2) tighten the diarize loop (sleep 1.5s, 1.5s new-audio threshold) → turns
now append ~every 3-4s; (3) mirror the latest diarized turn into the caption box so recent
text is always visible. Verified in the real browser with real speech: updates at
5.6/8.6/11.7/15.4/19.4s…, transcript box populated, caption shows recent text.

**Prompts used**:
- "the entire audio was processed but the live transcription was not showing in the text
  box. the interpreted texts must be appended to the recording as they become available in
  real time"

## [2026-06-14] — Fix: live recording stalled after ~1s (worklet socket flood)

**Session**: master (hotfix — committed directly; should have branched)
**Changed**: static/index.html, main.py
**Summary**: Live recording broke ~1s into real speech. Root cause: the AudioWorklet
posted every 128-sample render quantum (~375 tiny WebSocket frames/sec), each awaiting
`session.send_realtime_input` server-side; the read loop fell behind and the socket
stalled. Fix: the worklet now accumulates and emits ~250ms chunks (~4 frames/sec of
~8KB), matching the cadence proven in testing (verified in the real browser: 375/s → 4/s,
WS stays open). Backend hardening: a `gone` flag + `safe_send` (checks `WebSocketState`)
so nothing is sent after close; the final diarization runs only on an explicit client
stop (not on disconnect); the caption relay stops once the client is gone — fixes the
"Unexpected ASGI message 'websocket.send' after 'websocket.close'" error seen in logs.
Client also now surfaces an unexpected WS drop (shows code, pauses) instead of freezing.
Diagnosis ruled out: Caddy WS upgrade (fine), the model talking back (stays silent on
real speech), and the connection path (Python client holds 30s+).

**Prompts used**:
- "the recording starts correctly, but breaks off after 1 second."

## [2026-06-14] — Live meeting diarization (speaker-labelled live transcript)

**Session**: claude/feat/live-diarization
**Changed**: main.py, static/index.html, BUGS.md, FEATURES.md, TEST.md
**Summary**: Live meetings are now diarized (BUG-001 fixed). The Gemini Live API only
returns a flat input transcript, so `live_record` now ALSO buffers the raw PCM and a
background loop re-transcribes the full buffer every ~6s with `gemini-3.5-flash` (`_diarize_pcm`,
inline audio < ~6 min else Files API), emitting `diarized_transcript` events with
`Speaker 1/2/…` + mm:ss timestamps. Re-running on the whole buffer keeps speaker labels
self-consistent across updates. The Live session is kept only for the instant flat
caption (immediate feedback while speaking). Stop is now a handshake: client sends
`{"type":"stop"}` → server runs a final authoritative diarization → emits it + a `final`
event → client summarizes from the diarized turns. Client: `handleLiveMessage` handles
`diarized_transcript`/`final`, no longer commits flat "Speaker" turns; `stopAndSummarize`
awaits `waitForFinalDiarization()` (30s timeout). Researched/validated against the real
API: enumerated all 5 Live models (none diarize input transcription natively; 3.1-live
only does AUDIO out); batch `gemini-3.5-flash` diarizes a 2-speaker sample perfectly;
end-to-end live test showed an interim diarized update at +11s and a correct 3-turn final.
See BUG-003 for the long-meeting full-buffer-cost limitation.

**Prompts used**:
- "we do need diarization for the live meetings as well, which must be possible with the
  latest Gemini API's. please research further until you have a solution"

## [2026-06-14] — Fix: live transcription empty (wrong Live model)

**Session**: claude/fix/live-model-transcription
**Changed**: main.py, BUGS.md
**Summary**: Live recording showed no transcript as you spoke. Root cause: the earlier
"latest models everywhere" change set `LIVE_MODEL = gemini-3.1-flash-live-preview`, which
only supports AUDIO output and emits NO input-audio transcription — so `user_transcript`
events never fired. Verified by streaming real 16kHz speech PCM straight to the Live API:
`gemini-3.1-flash-live-preview` → `''`; `gemini-2.5-flash-native-audio-latest` →
"Hello, this is a test of the live transcription system." (and 3.1 rejects TEXT modality
entirely: 1007 "response modalities (TEXT) is not supported"). Reverted `LIVE_MODEL` to
`gemini-2.5-flash-native-audio-latest` — the only Live line that transcribes input. The
WebSocket path itself (app + Caddy proxy upgrade) was confirmed healthy during diagnosis.
Text/summarize/upload-transcription stay on `gemini-3.5-flash`.

**Prompts used**:
- "for the live conversation transcription: when I hit record, no text is being shown in
  the text box as I speak"

## [2026-06-14] — Upload review is a read-only document (no live-confirm controls)

**Session**: claude/feat/upload-document-review
**Changed**: static/index.html, FEATURES.md
**Summary**: The Review screen previously rendered identically for both sources, so a
pre-recorded **upload** inherited the whole live-meeting apparatus — the floating
read-aloud transport bar (play/Interrupt/speed/language), per-item Agree / Object & revise
buttons, the "confirm with the room" framing, the confirmed-count chip and progress bar —
none of which make sense when there's no room to read back to. Per user decision, the
upload review is now a clean document: a `doc = S.source === 'upload'` flag gates all of
that off. Upload shows H1 "Summary & transcript" + "Transcribed from <file>", then the
prose Summary, "Key points" (not "…to confirm"), action items, and the full diarized
transcript; cards are static (no jump/karaoke/agree/object) and full-brightness; exports
remain available in the transcript-panel header. The live flow is unchanged (transport,
Interrupt, agree/object, confirm chip all intact — verified).

**Prompts used**:
- "why am I seeing the button controls at the bottom of the page when I am reviewing a
  pre-recorded audio session?" → chose "Document, no controls"

## [2026-06-14] — Complete conversation summary above the points

**Session**: claude/feat/conversation-overview
**Changed**: main.py, static/index.html, FEATURES.md
**Summary**: Added a complete prose summary of the conversation at the top of the Review
screen, above the points to confirm. `/api/summarize-live` now also returns an `overview`
field (added to the structured schema + prompt as a required 1-3 paragraph narrative). The
frontend stores `S.overview` and renders it in a "Summary" card (Newsreader serif,
paragraph-split) above the point cards; when an overview is present the point-cards header
is relabelled "Key points to confirm" to disambiguate from the prose summary. Resets with
each new upload / live summarize / home.

**Prompts used**:
- "for the upload functionality ... I would like to see a complete summary of the
  conversation appearing above the points to agree on"

## [2026-06-14] — Transcription activity log + latest Gemini models everywhere

**Session**: claude/feat/transcription-activity-log (+ claude/chore/latest-gemini-models)
**Changed**: static/index.html, transcriber.py, main.py, FEATURES.md
**Summary**: (1) Added a small scrollable (~3-line) "Activity" log to the Review loading
state. The `/api/transcribe` SSE stream already emitted `progress` and `debug` events; the
frontend now captures both into `S.log`, renders them in a fixed-height monospace box
(id `asLog`) that auto-scrolls to the newest line, and resets it per run. Surfaces exactly
what the system is doing during upload transcription (connect, duration check, chunking,
upload, model call, segment count). (2) Moved every Gemini call to the latest model line —
verified available models against the live API: `transcriber.py` transcription
`gemini-2.5-flash` → `gemini-3.5-flash` (new `TRANSCRIBE_MODEL`); live conversation
`gemini-2.5-flash-native-audio-latest` → `gemini-3.1-flash-live-preview` (latest 3.x Live
model; there is no 3.x native-audio model — confirmed the new model accepts our
LiveConnectConfig). Summarize/amend were already on `gemini-3.5-flash`. No `2.5` refs left.

**Prompts used**:
- "why did it call gemini 2.5 flash, we need to be calling the latest gemini api's everywhere"
- "during the transcription of an uploaded audiofile, i need to be able to see what the
  system is doing, show a small section of 3 lines with the log (scrollable)"

## [2026-06-14] — Restore diarized transcript view + fix stale service worker

**Session**: claude/fix/restore-diarized-transcript
**Changed**: static/index.html, static/sw.js
**Summary**: Fixed two regressions from the redesign. (1) The upload flow routed straight
into the summary-only Review screen, hiding the original app's primary output — the full
**diarized, timestamped transcript**. Added a "Full transcript" panel to the Review
screen (speaker chips + timestamps, turn/speaker counts, expanded by default for uploads,
collapsed for live) with always-available Export transcript / Export summary buttons (no
longer gated on the "everyone's aligned" state). The SSE result now preserves each
segment's `timestamp` string. Empty results now show a clear "No speech could be
detected" message. (2) `sw.js` was cache-first on the HTML document with background
revalidate, so every deploy was invisible to returning users until a second reload —
switched the document/navigation requests to **network-first** (cache fallback offline)
and bumped `CACHE_NAME` to `audioscribe-v3`. This stale-cache behaviour was a major
contributor to "the new version doesn't work" after a deploy.

Diagnosis note: staging logs confirmed the backend transcribe pipeline (upload → Gemini
`gemini-2.5-flash` → diarization, chunking check) works end-to-end; the one failing test
returned 0 segments because Gemini judged that audio file distorted/silent — not a code
fault. The real issue was the UI hiding the transcript + the SW serving stale HTML.

**Prompts used**:
- "the original transcribe function (with diarization and chunking of larger files while
  maintaining the attribution of sentences to the same people in the conversation) does
  not seem to work now"

## [2026-06-14] — Live Conversation UI redesign (Audioscribe design card)

**Session**: claude/feat/audioscribe-design-redesign
**Changed**: main.py, requirements.txt, static/index.html, ../.claude/launch.json (new), PLAN.md/BUILD.md/FEATURES.md/TEST.md/BUGS.md (new scaffold)
**Summary**: Implemented the "warm dark, voice-first" Audioscribe design (fetched from
the Claude Design handoff bundle at design link `PRb-w-Im86T1ASM03E4SNg`). Full
frontend rewrite of `static/index.html` as a faithful vanilla-JS recreation of the
three-screen flow — Home (record orb + drag/drop upload), Live (reactive record orb,
animated waveform/pulse rings, live caption with speaker chip, real-time transcript,
participant rail), and Review (read-aloud summary with word-by-word karaoke
highlighting, agree / object-and-revise per item, floating transport bar, and a
"Listening to the room" sheet for spoken objections). Dark warm palette (#100F0D /
#E9A23B amber / #F4EFE6 cream), Hanken Grotesque + Newsreader serif + JetBrains Mono
via Google Fonts.

Backend (main.py, v1.5.0):
- Restored & re-developed the live-conversation endpoints that the 2026-06-13 rollback
  removed: `WS /api/live-record` (streams 16kHz Int16 PCM to the Gemini Live API
  `gemini-2.5-flash-native-audio-latest`, relays `user_transcript` / `model_text` /
  `audio_chunk` events; model is a silent observer), `POST /api/summarize-live`, and
  `POST /api/amend-summary`.
- Summarization now follows the new design: `summarize-live` returns **structured
  blocks** (`summary` points + `action` items with owner/due) via a Gemini structured-
  output schema on `gemini-3.5-flash`; `amend-summary` rewrites a single block from a
  participant's objection. Both share the model constant `TEXT_MODEL`.
- Original upload+transcription (`POST /api/transcribe`, SSE) is unchanged; the frontend
  now feeds its result segments into `summarize-live` so uploads land on the same Review
  screen.
- CSP updated to allow Google Fonts (`fonts.googleapis.com` style-src,
  `fonts.gstatic.com` font-src) + `ws:`/`wss:` connect-src + `blob:` worker/script for
  the PCM AudioWorklet. `google-genai` pin bumped 1.0.0 → 1.47.0 (Live API support).

Frontend live audio: in-browser AudioWorklet downsamples mic to 16kHz Int16 PCM and
streams it over the WebSocket; the assistant's 24kHz PCM replies are decoded and played
via Web Audio. Object → mic-captures the objection into the textarea (operator can also
type) → `amend-summary` → revised block + read-aloud resumes. Auth token (Bearer for
HTTP, `?token=` for WS), relative URLs (subpath-safe for `/projects/audioscribe`), and
service-worker registration preserved.

Verified locally (dummy key): all routes register, server healthy on v1.5.0, CSP header
correct, `summarize-live` empty-segments path returns `{blocks:[]}`. Frontend verified
in the preview — Home, Live, Review and the Listening sheet all render pixel-faithfully
to the design with zero console errors; karaoke highlight advances correctly. NOT yet
verified end-to-end against a real GEMINI_API_KEY (live mic→Gemini and upload→summarize
need a real key on the VPS).

**Prompts used**:
- "implement this design, use the latest gemini models (e.g. gemini 3.5 flash), maintain
  the original upload and transcription functionality (the summarization function should
  be according to the new design) and include/redevelop the live conversation logic.
  Here is the design: … PRb-w-Im86T1ASM03E4SNg … Implement: Audioscribe.dc.html"

## [2026-06-09] — Security hardening + Trustable VPS deployment

**Session**: claude/security-hardening-and-vps-deploy
**Changed**: main.py, requirements.txt, Dockerfile, static/index.html, .gitignore, README.md, deploy.sh (new), api-key-flow.drawio (new)
**Summary**: Hardened the API after a security review and moved deployment off
Google Cloud Run onto the Trustable.nl staging + prod VPSes (Docker + Caddy).
The Gemini key now lives only in `/opt/trustable/audioscribe.env` on each host
(no Secret Manager dependency).

Security changes (main.py):
- Bearer-token auth on `POST /api/transcribe` via the `API_TOKEN` env var
  (unset = open, for local dev only).
- Per-IP rate limiting with `slowapi` (`RATE_LIMIT`/hour, default 10).
- Generic client-facing error messages; full exceptions logged server-side only.
- Bounded `speakers` form field (`ge=1, le=20`).
- Removed the client-controlled `debug` flag.
- `SecurityHeadersMiddleware` (CSP, X-Frame-Options, X-Content-Type-Options,
  Referrer-Policy). NB: `BaseHTTPMiddleware` imports from `starlette.middleware.base`,
  not `fastapi.middleware.base`.

Frontend (static/index.html): one-time API-token prompt stored in localStorage,
sent as `Authorization: Bearer`; 401/429 handling; "Set API token" footer link.

Deployment:
- App is a profile-gated `audioscribe` service inside the trustable
  docker-compose stack on each VPS (profile keeps it invisible to trustable's
  own CI deploys). Caddy strips the `/projects/audioscribe` prefix
  (`flush_interval -1` for SSE).
- `deploy.sh` builds `linux/amd64` (VPSes are x86_64, dev machine is arm64),
  pushes to `ghcr.io/ewoutbarendregt/audioscribe`, then SSHes to the
  `trustable-staging` / `trustable-prod` aliases and runs
  `docker compose --profile audioscribe up -d`.
- Dockerfile: added `curl` (used by the container healthcheck), explicit COPY.
- Subpath fix: frontend now uses relative URLs (`api/transcribe`, `manifest.json`,
  `sw.js`) instead of absolute (`/api/...`), which bypassed the `/projects/audioscribe`
  prefix and hit the trustable web app (returned HTML → "Unexpected token '<'").
  Also fixed manifest `start_url` and the service worker (relative cache paths,
  `/api/` detection, cache bumped to v2).

Deployed and verified live on 2026-06-09: both hosts healthy, public `/health`
200, auth gate returns 401 without/with-wrong token.

**Prompts used**:
- "run a security check on this code, particularly around the google API and the API keys"
- "fix #1,2,3" then "fix # 4-7" (the numbered findings from the review)
- "how can we securely handle the Gemini API key in this project?"
- "we will deploy the application on the trustable staging vps ... under /projects/audioscribe and we will not use cloud run"
- "the trustable code is deployed to both staging and prod ... audioscribe also needs to be deployed to both"
- "I am no longer using secret manager" → removed all Secret Manager references
