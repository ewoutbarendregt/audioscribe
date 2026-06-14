# DOCUMENTATION

Auto-updated by agents as they work. Newest entries first.

## [2026-06-13] — Live Conversation UI Redesign

**Session**: claude/live-conversation-ui-redesign
**Changed**: static/index.html, main.py
**Summary**: Full UI overhaul implementing the "Audioscribe Live Conversation UI Redesign" design card. Replaced the previous grey/white UI with a dark warm palette (`#100F0D` bg, `#E9A23B` amber accent, `#F4EFE6` cream text) using Google Fonts (Hanken Grotesque, Newsreader, JetBrains Mono). Added a two-panel live recording screen (recording card with pulse ring + wave bars on the left, live transcript feed on the right) and a new review screen with per-block TTS karaoke word highlighting, agree/object/revise actions, transport bar, and a "Listening" modal for spoken corrections. CSP updated in main.py to allow `fonts.googleapis.com` (style-src) and `fonts.gstatic.com` (font-src).
**Prompts used**:
- "Fetch this design file, read its readme, and implement the relevant aspects of the design. https://api.anthropic.com/v1/design/h/AEN80I1FKDUiZ77QVv9edw?open_file=Audioscribe.dc.html — Implement: Audioscribe.dc.html"

## [2026-06-12] — Live Conversation Mode (Real-time Recording & Diarization)

**Session**: antigravity/live-conversation-recording-and-transcription
**Changed**: main.py, static/index.html
**Summary**: Added a new "Live Conversation Mode" allowing users to record meetings. Streams 16kHz Int16 PCM mono audio from the browser to Gemini Live API (`gemini-2.5-flash-native-audio-latest`) over WebSockets. Performs real-time speaker diarization and transcription, plays back AI clarification prompts in a warm female voice (`Aoede`), cancels audio buffers on barge-in/interruption, and generates meeting summaries/action items using `gemini-3.5-flash` with the ability to read them aloud via browser-native text-to-speech.

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
