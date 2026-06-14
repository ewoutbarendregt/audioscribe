# BUGS

Known issues, bug reports, fix status.

### [BUG-001] Live-mode turns are not diarized — low
**Location**: static/index.html (`commitLiveSegment`), main.py (`live_record`)
**Status**: open
**Found**: 2026-06-14
**Description**: The Gemini Live `input_audio_transcription` is a single combined stream,
so live-recorded turns are all labelled "Speaker" rather than separated per person. The
upload path (`/api/transcribe`) still diarizes correctly. The original design assumed
named speakers, which only holds for the upload flow.
**Fix**: TBD — options include a separate diarization pass on the captured audio after
Stop, or speaker-change detection. Not blocking.

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

_End-to-end Gemini-backed flows: upload transcription, summarize, amend and now live
transcription have been exercised against the real API. Live diarization remains BUG-001._
