# TEST

Test strategy, prompts, and coverage status. Source of truth for testing.

## Local preview
- Launch config: `.claude/launch.json` → `audioscribe` server on port 8088
  (`venv/bin/python -m uvicorn main:app --app-dir audioscribe`).
- A `GEMINI_API_KEY` is required for real transcription/summarization/live. Without it,
  the UI renders and `summarize-live` with empty segments returns `{blocks:[]}`, but
  Gemini-backed calls return 500.

## Verified (2026-06-14, dummy key)
- ✅ All routes register; `/health` reports v1.5.0; CSP header includes Google Fonts +
  `ws:`/`wss:`.
- ✅ `POST /api/summarize-live` empty segments → `{"title":...,"blocks":[]}`.
- ✅ Frontend renders with zero console errors: Home, Live, Review, Listening sheet.
- ✅ Karaoke highlight advances word-by-word; active card shows amber bar/border.
- ✅ Review cards: summary numbering, action owner/due chips, agreed/revised badges,
  progress bar, transport bar.

## NOT yet verified (needs real GEMINI_API_KEY, ideally on the VPS)
- ⬜ Upload → `/api/transcribe` (SSE) → `/api/summarize-live` end-to-end.
- ⬜ Live mic capture → `WS /api/live-record` → Gemini Live transcription stream.
- ⬜ Assistant audio reply playback (24kHz PCM) when directly addressed.
- ⬜ Object → mic capture → `/api/amend-summary` revision quality.
- ⬜ Same-language summary for non-English audio.

## Manual test prompts
1. Upload a short multi-speaker MP3 → expect Transcribing… → Review with summary points
   + action items; play-aloud reads them with karaoke.
2. Record live, speak a few sentences, Stop & summarize → expect transcript turns and a
   structured summary.
3. On Review, Interrupt on an action item, speak/type "make Semir the owner, due next
   Tuesday" → expect that item rewritten with the new owner/due and a "Revised" badge.
