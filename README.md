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
`audioscribe` Compose profile. The trustable web/api deploy workflows never
enable that profile, so they **never touch** this container — it is not pulled,
recreated, or removed as an orphan during a trustable release. AudioScribe has
its own independent release cycle, driven entirely by `deploy.sh`.

The Caddy routes live in the trustable repo (`Caddyfile` for prod,
`Caddyfile.staging` for staging) and are deployed by the trustable workflows.

### First-time setup on EACH VPS (run once per host)

This env file is the single source of truth for the key — nothing else reads
it (no Secret Manager, no external store). Keep a backup wherever you normally
store secrets (e.g. a password manager).

```bash
ssh trustable-staging      # then repeat for trustable-prod

# Write the env file
cat > /opt/trustable/audioscribe.env << 'EOF'
GEMINI_API_KEY=<paste your Gemini key>
API_TOKEN=<generate with: openssl rand -hex 32>
EOF

# Owned by the deploy user (which runs docker compose); 640 is sufficient
chmod 640 /opt/trustable/audioscribe.env
```

### Deploying

Make sure you are logged into GHCR:
```bash
echo $GHCR_PAT | docker login ghcr.io -u ewoutbarendregt --password-stdin
```

Then deploy (the script targets the `trustable-staging` / `trustable-prod`
SSH config aliases by default):
```bash
cd audioscribe

./deploy.sh            # build, push, deploy to BOTH staging and prod
./deploy.sh staging    # staging only
./deploy.sh prod       # prod only
TAG=v1.2.0 ./deploy.sh # deploy a specific image tag
```

The script builds the image once, pushes it to
`ghcr.io/ewoutbarendregt/audioscribe`, then SSHs to each target host and runs
`docker compose --profile audioscribe up -d audioscribe` — scoped to just this
service so the trustable containers are never disturbed.

### Updating the Gemini API key

The key lives in `/opt/trustable/audioscribe.env` on **each** VPS. To rotate it,
edit the file and recreate the container on each host:

```bash
ssh trustable-staging      # then repeat for trustable-prod
nano /opt/trustable/audioscribe.env
# Recreate (NOT restart) — env_file is only re-read when the container is recreated
cd /opt/trustable && docker compose --profile audioscribe up -d --force-recreate audioscribe
```

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Gemini API key — lives only in `audioscribe.env` on each VPS |
| `API_TOKEN` | Recommended | Bearer token clients must send to use `/api/transcribe`. Unset = open (dev only). |
| `RATE_LIMIT` | No | Requests per hour per IP for `/api/transcribe` (default: `10`) |

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
