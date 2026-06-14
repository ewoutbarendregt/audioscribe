# BUILD

Technical implementation & architecture decisions.

## Stack
- **Backend**: FastAPI (`main.py`), `google-genai` 1.47.0, `slowapi` rate limiting,
  `uvicorn`. Audio chunking via `pydub`/`ffmpeg` in `transcriber.py`.
- **Frontend**: single static `static/index.html`, vanilla JS, no build step. Google
  Fonts (Hanken Grotesque, Newsreader, JetBrains Mono).
- **Models**: `TEXT_MODEL = gemini-3.5-flash`, `LIVE_MODEL = gemini-2.5-flash-native-audio-latest`.

## Endpoints
| Route | Purpose |
|---|---|
| `GET /` | Serve the app (`static/index.html`). |
| `POST /api/transcribe` | Upload file → SSE progress → `{segments, summary, language}`. Unchanged from the original app. |
| `POST /api/summarize-live` | Transcript segments → **structured review blocks** (`{id,type,text,owner,due,agreement}`) via a Gemini JSON schema. Used by both live and upload paths. |
| `POST /api/amend-summary` | Revise ONE block from a participant objection → `{text,owner,due}`. |
| `WS /api/live-record` | Stream 16kHz Int16 PCM to Gemini Live; relay `user_transcript`/`model_text`/`audio_chunk`/`turn_complete`/`interrupted`. Model is a silent observer (only speaks when addressed). |
| `GET /health`, `/api/version`, `/manifest.json`, `/sw.js` | infra/PWA. |

## Key decisions
- **Structured summary**: the new design renders discrete cards, so `summarize-live`
  returns typed blocks rather than markdown. Server builds block ids (`s1…`, `a1…`) and
  defaults `agreement: 'pending'`.
- **Upload reuses the summarizer**: `/api/transcribe` stays text-summary-agnostic; the
  frontend posts its result segments to `summarize-live` to land on the Review screen.
- **Live audio in the browser**: an inline `AudioWorkletProcessor` downsamples the mic to
  16kHz Int16 PCM (CSP allows `worker-src blob:`). Gemini's 24kHz PCM replies are decoded
  and scheduled on a Web Audio playback context.
- **Objection capture**: reuses the same live WebSocket + mic to stream the participant's
  words into the listening-sheet textarea; the operator can also type. On revise, the
  text is sent to `amend-summary`.
- **Auth/deploy**: Bearer token for HTTP, `?token=` query for the WebSocket. All
  frontend URLs are relative (`baseDir()`), so the app works under the
  `/projects/audioscribe` Caddy prefix. CSP allows Google Fonts + `ws:`/`wss:`.

## Security headers (CSP)
`default-src 'self'`; `script-src 'self' 'unsafe-inline' blob:`;
`style-src 'self' 'unsafe-inline' https://fonts.googleapis.com`;
`font-src 'self' https://fonts.gstatic.com`; `img-src 'self' data:`;
`connect-src 'self' ws: wss:`; `worker-src 'self' blob:`.
