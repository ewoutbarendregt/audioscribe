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

_No other known bugs. End-to-end Gemini-backed flows are unverified pending a real API
key (see TEST.md) — failures there would be logged here._
