# DOCUMENTATION

Auto-updated by agents as they work. Newest entries first.

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
