# BUGS

Known issues, bug reports, fix status.

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
