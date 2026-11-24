# BUGS

Known issues, bug reports, fix status.

### [BUG-007] AudioScribe deploys still targeted the old Hetzner prod and piped server steps over stdin — high
**Location**: deploy.sh
**Status**: fixed
**Found**: 2026-11-24
**Description**: Prod moved to the Contabo VPS (`trustable-contabo`) on 2026-11-24, but
`deploy.sh prod` still defaulted to `trustable-prod` (Hetzner, now only a Caddy forwarder).
The server steps were also sent as a heredoc to `ssh host bash -s`, so any stdin-reading
command could swallow the rest of the script while ssh still exited 0.
**Fix**: prod defaults to `trustable-contabo`; server steps are uploaded as a file, run with
`ssh -n`, every compose call gets `< /dev/null`, and the deploy fails unless the output ends
with `REMOTE-DEPLOY-OK` and the container runs the new tag. Branch `claude/chore/local-deploy-contabo`.

### [BUG-006] README key-rotation recreate dropped the auth-database volume — medium
**Location**: README.md ("Updating the Gemini API key")
**Status**: fixed
**Found**: 2026-11-24
**Description**: The documented `docker compose --profile audioscribe up -d --force-recreate audioscribe`
leaves out `-f docker-compose.audioscribe.yml`, so the recreated container would start without the
`audioscribe_data:/data` mount and lose sessions and the user table until the next deploy.
**Fix**: the command now passes both compose files, like `deploy.sh`.

### [BUG-001] Live-mode turns are not diarized — medium
**Location**: main.py (`live_record`, `_diarize_pcm`), static/index.html
**Status**: fixed
**Found**: 2026-06-14
**Description**: The Gemini Live `input_audio_transcription` is a single combined stream,
so live-recorded turns were all labelled "Speaker" rather than separated per person.
**Fix**: Added server-side chunked diarization — `live_record` now buffers the raw PCM and
periodically (every ~6s) re-transcribes the FULL buffer with `gemini-3.5-flash` (which
diarizes properly), emitting `diarized_transcript` events with Speaker 1/2/… + timestamps;
re-running on the full buffer keeps labels self-consistent. The Gemini Live session is kept
only for the instant flat caption. On Stop the client sends `{"type":"stop"}`, the server
runs a final authoritative diarization, emits it + a `final` event, and the client
summarizes from those diarized turns. Verified against the API with a 2-speaker sample
(interim update at +11s, final 3-turn transcript correctly attributed).
**Limitation**: re-transcribing the full buffer means interim updates slow down on long
(>~15 min) meetings; the final transcript is always complete. Windowing/freezing could
optimize later (see [BUG-003]).

### [BUG-002] Live transcript empty — wrong Live model — high
**Location**: main.py (`LIVE_MODEL`)
**Status**: fixed
**Found**: 2026-06-14
**Description**: No transcript appeared while recording live. `gemini-3.1-flash-live-preview`
(set during "latest models everywhere") only supports AUDIO output and emits no input-audio
transcription, so `user_transcript` never fired.
**Fix**: Reverted `LIVE_MODEL` to `gemini-2.5-flash-native-audio-latest` (the only Live
line that transcribes input — verified against the API). The native-audio line is 2.5-only;
there is no 3.x equivalent.

### [BUG-005] Write-scoped GHCR PAT stored in plaintext on staging — medium
**Location**: /home/deploy/.docker/config.json on trustable-staging
**Status**: open
**Found**: 2026-08-19
**Description**: While debugging the failed GHCR push, `docker login ghcr.io` was run on
the staging VPS (instead of the dev machine, where the push actually happens). Docker
warned it stored the credential **unencrypted**. If that PAT is a classic token with
`write:packages`, it also carries `repo` scope, so a credential that can push images and
read private repos now sits in cleartext on the VPS. Staging only ever needs to *pull*,
and it already had working auth before this.
**Fix**: generate a second classic PAT with **only** `read:packages`, then on staging run
`docker logout ghcr.io && docker login ghcr.io -u ewoutbarendregt` and paste the
read-only token. Since 2026-11-24 `deploy.sh` no longer pulls from GHCR on the hosts
(it ships the image with `docker save | ssh docker load`), so a plain
`docker logout ghcr.io` on staging no longer breaks AudioScribe deploys.

### [BUG-004] Prod will lock out if deployed before its env vars are set — high
**Location**: /opt/trustable/audioscribe.env on trustable-prod, main.py (`require_auth`)
**Status**: open — action needed before the next prod deploy
**Found**: 2026-06-14
**Description**: `require_auth` now **fails closed**, and the frontend no longer sends an
API token. Prod's env file still has only `GEMINI_API_KEY` + `API_TOKEN`, so deploying
the new image there before adding `ALLOWED_EMAILS` / `COOKIE_PATH` (and the SMTP block)
would leave nobody able to sign in. Staging has already been updated.
**Fix**: add the sign-in vars to `/opt/trustable/audioscribe.env` on trustable-prod
before running `./deploy.sh prod` — see the deploy.sh header for the exact block.

### [BUG-003] Live diarization re-transcribes full buffer each interval — low
**Location**: main.py (`live_record` diarize loop)
**Status**: open
**Found**: 2026-06-14
**Description**: For self-consistent speaker labels, each interval re-diarizes the entire
buffer. On long meetings this grows in cost/latency (interim updates get less frequent;
buffers >~6 min switch from inline to the Files API). The final transcript is unaffected.
**Fix**: TBD — sliding window with a frozen prefix + speaker-continuity context, or
silence-based incremental finalization. Not blocking for typical meeting lengths.

_End-to-end Gemini-backed flows — upload transcription, summarize, amend, live
transcription and live diarization — have all been exercised against the real API._
