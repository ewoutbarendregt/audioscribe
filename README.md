# Audio Transcription Web App

A Progressive Web App for transcribing audio files with speaker diarization using Google's Gemini AI.

## Features

- Upload audio files (MP3, M4A, WAV, FLAC, OGG, AAC)
- Automatic speaker diarization (identifies different speakers)
- Language detection
- Conversation summary
- Download transcripts as TXT or JSON
- PWA: installable on mobile/desktop

## Local Development

### Prerequisites

- Python 3.11+
- ffmpeg installed
- Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Setup

```bash
cd webapp

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set API key
set GEMINI_API_KEY=your_api_key_here  # Windows
# export GEMINI_API_KEY=your_api_key_here  # Linux/Mac

# Run locally
python main.py
```

Open http://localhost:8080 in your browser.

## Deploy to the Trustable.nl VPS fleet (staging + prod)

The app runs as a Docker container on **both** the staging and prod Trustable
VPSes, behind Caddy:

- Staging: `https://staging.trustable.nl/projects/audioscribe/`
- Prod:    `https://trustable.nl/projects/audioscribe/`

Caddy strips the `/projects/audioscribe` prefix before forwarding to the
container, so the FastAPI app sees plain `/` paths — no app changes needed.

### How it fits the trustable stack

AudioScribe is defined in the trustable `docker-compose.yml` under a dedicated
`audioscribe` Compose profile. The trustable web/api deploys never
enable that profile, so they **never touch** this container — it is not pulled,
recreated, or removed as an orphan during a trustable release. AudioScribe has
its own independent release cycle, driven entirely by `deploy.sh`.

The Caddy routes live in the trustable repo (`Caddyfile` for prod,
`Caddyfile.staging` for staging) and are deployed by trustable's own `scripts/deploy.sh`.

### First-time setup on EACH VPS (run once per host)

This env file is the single source of truth for the key — nothing else reads
it (no Secret Manager, no external store). Keep a backup wherever you normally
store secrets (e.g. a password manager).

```bash
ssh trustable-staging      # then repeat for trustable-contabo (prod)

# Write the env file
cat > /opt/trustable/audioscribe.env << 'EOF'
GEMINI_API_KEY=<paste your Gemini key>
API_TOKEN=<generate with: openssl rand -hex 32>

# Interactive sign-in — users receive a one-time code by email
ALLOWED_EMAILS=you@example.com,colleague@example.com
SMTP_HOST=smtp.resend.com
SMTP_PORT=587
SMTP_USER=resend
SMTP_PASS=<Resend API key>
MAIL_FROM=audioscribe@trustable.nl
COOKIE_PATH=/projects/audioscribe
EOF

# Owned by the deploy user (which runs docker compose); 640 is sufficient
chmod 640 /opt/trustable/audioscribe.env
```

### Deploying

Deploys run from Ewout's Mac straight to the servers with `deploy.sh`. There is
no GitHub Actions deploy (Actions is no longer used for deployments anywhere).
Prod is the Contabo VPS (`trustable-contabo`, 144.91.87.103, user `deploy`),
which replaced the Hetzner server (`trustable-prod`) on 2026-11-24. Staging is
`trustable-staging`. Both are `~/.ssh/config` aliases; override with
`PROD_SSH_HOST` / `STAGING_SSH_HOST`.

```bash
./deploy.sh prod --dry-run      # build + server checks, change nothing
./deploy.sh prod                # deploy origin/master to prod (asks to confirm)
./deploy.sh staging             # staging only
./deploy.sh                     # staging and prod
./deploy.sh prod --ref <ref>    # deploy a specific commit/tag (e.g. a rollback)
./deploy.sh prod --yes          # skip the confirmation prompt
```

The script checks out the ref into a temporary git worktree (uncommitted local
work is never deployed), tags the image `v<APP_VERSION>` from that ref's
`main.py` (never `latest`), builds it for `linux/amd64` with `docker buildx`,
and ships it with `docker save | gzip | ssh docker load` — no registry push or
pull, no GHCR login needed. It then copies `docker-compose.audioscribe.yml`,
uploads the server steps as a file and runs them with `ssh -n`:
`docker compose --profile audioscribe up -d audioscribe` (scoped to just this
service, so the trustable containers are never disturbed), writes the tag to
`AUDIOSCRIBE_IMAGE_TAG` in `/opt/trustable/.env`, and waits for the container to
be healthy. The run only counts as done when the server prints
`REMOTE-DEPLOY-OK` and the container actually runs the new tag. All ssh/scp
calls share one multiplexed connection, because the Contabo sshd resets bursts
of new connections.

### Updating the Gemini API key

The key lives in `/opt/trustable/audioscribe.env` on **each** VPS. To rotate it,
edit the file and recreate the container on each host:

```bash
ssh trustable-staging      # then repeat for trustable-contabo (prod)
nano /opt/trustable/audioscribe.env
# Recreate (NOT restart) — env_file is only re-read when the container is recreated
cd /opt/trustable && docker compose -f docker-compose.yml -f docker-compose.audioscribe.yml --profile audioscribe up -d --force-recreate audioscribe
```

### Managing who can sign in

Permitted addresses live in a `users` table in the SQLite database on the
`audioscribe_data` volume, so changes take effect immediately — no redeploy, no
container restart. On the first start of an empty database the table is seeded from
`ALLOWED_EMAILS`; after that the variable is ignored, so removing someone can't be
undone by the next restart.

```bash
ssh trustable-staging
cd /opt/trustable
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.audioscribe.yml"
$COMPOSE --profile audioscribe exec audioscribe python auth.py list
$COMPOSE --profile audioscribe exec audioscribe python auth.py add name@example.com
$COMPOSE --profile audioscribe exec audioscribe python auth.py remove name@example.com
```

`remove` also deletes that person's sessions and any pending code, so access ends at
once rather than whenever their 30-day cookie happens to lapse.

Someone who isn't registered is told so on the sign-in screen and pointed at
`SUPPORT_EMAIL` — they never receive a code. Note this makes the user list
enumerable: a stranger can learn whether an address has access. That's an accepted
trade for a small internal tool, bounded by the 5/hour per-IP limit on code requests.

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Gemini API key — lives only in `audioscribe.env` on each VPS |
| `ALLOWED_EMAILS` | First run only | Comma-separated seed applied only when the `users` table is first created. After that the table is the source of truth — manage it with the CLI below. |
| `SUPPORT_EMAIL` | No | Address shown to people who aren't registered (default: `support@trustable.nl`) |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASS` | Yes | Mail transport for the login code. Resend: `smtp.resend.com` / `587` / `resend` / *(API key)*. Unset = codes are logged to stdout instead (local dev only). |
| `MAIL_FROM` | Yes | Sender address; its domain must be verified with the mail provider |
| `COOKIE_PATH` | Yes in prod | Cookie scope. `/projects/audioscribe` behind Caddy; `/` locally. |
| `API_TOKEN` | No | Legacy bearer token for scripted access to the API. Interactive users sign in instead. |
| `SESSION_TTL_DAYS` | No | How long a sign-in lasts (default: `30`) |
| `OTP_TTL_MINUTES` | No | Login-code lifetime (default: `10`) |
| `COOKIE_SECURE` | No | `false` only for plain-HTTP local dev (default: `true`) |
| `DATA_DIR` | No | Where the auth SQLite database lives (default: `/data`, a named volume) |
| `RATE_LIMIT` | No | Requests per hour per IP for `/api/transcribe` (default: `10`) |

### Signing in

Audioscribe has its own login — it is *hosted* under `trustable.nl/projects/audioscribe`
but shares nothing else with trustable, so it deliberately does not reuse trustable's
session or admin allowlist. Users enter their email, receive a 6-digit code, and get an
httpOnly session cookie valid for `SESSION_TTL_DAYS`. Add or remove people with the
`auth.py` CLI described above; `ALLOWED_EMAILS` is only the first-run seed.

### Troubleshooting

```bash
ssh deploy@<host>

# View logs
cd /opt/trustable && docker compose --profile audioscribe logs audioscribe -f

# Health check (from anywhere)
curl https://staging.trustable.nl/projects/audioscribe/health
curl https://trustable.nl/projects/audioscribe/health

# Restart
docker compose --profile audioscribe restart audioscribe
```

### Large file issues

The app supports files up to 100MB. For larger files, consider:
1. Compressing the audio
2. Converting to a more efficient format (e.g., MP3)

## Project Structure

```
webapp/
├── main.py              # FastAPI application
├── transcriber.py       # Transcription logic
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container definition
├── static/
│   ├── index.html      # PWA frontend
│   ├── manifest.json   # PWA manifest
│   └── sw.js           # Service worker
└── README.md           # This file
```
