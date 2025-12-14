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

## Deploy to Google Cloud Run

### 1. Prerequisites

- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed
- A Google Cloud project with billing enabled

### 2. Enable APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### 3. Store API Key in Secret Manager

```bash
# Create secret
echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-

# Grant Cloud Run access to the secret
gcloud secrets add-iam-policy-binding gemini-api-key \
    --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

To find your project number:
```bash
gcloud projects describe YOUR_PROJECT_ID --format="value(projectNumber)"
```

### 4. Deploy to Cloud Run

```bash
cd webapp

# Deploy (builds and deploys in one command)
gcloud run deploy audio-transcribe \
    --source . \
    --region europe-west1 \
    --allow-unauthenticated \
    --set-secrets=GEMINI_API_KEY=gemini-api-key:latest \
    --memory=1Gi \
    --timeout=600 \
    --cpu=1
```

The deploy command will:
1. Build the Docker image using Cloud Build
2. Push to Artifact Registry
3. Deploy to Cloud Run

### 5. Access Your App

After deployment, you'll get a URL like:
```
https://audio-transcribe-xxxxx-ew.a.run.app
```

## Configuration Options

### Cloud Run Settings

| Setting | Recommended | Description |
|---------|-------------|-------------|
| Memory | 1Gi | Audio processing needs memory |
| CPU | 1 | Sufficient for API calls |
| Timeout | 600 | Long audio files take time |
| Max instances | 10 | Prevent runaway costs |
| Min instances | 0 | Scale to zero when idle |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Your Gemini API key |
| `PORT` | No | Server port (default: 8080) |

## Cost Estimation

- **Cloud Run**: ~$0 when idle (scales to zero)
- **Gemini API**: Check [Google AI pricing](https://ai.google.dev/pricing)
- **Cloud Build**: Free tier: 120 build-minutes/day

## Updating the App

To deploy updates:

```bash
cd webapp
gcloud run deploy audio-transcribe --source .
```

## Troubleshooting

### "GEMINI_API_KEY not configured"

Make sure the secret is properly linked:
```bash
gcloud run services describe audio-transcribe --region europe-west1
```

### Timeout errors

Increase the timeout:
```bash
gcloud run services update audio-transcribe --timeout=900
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
