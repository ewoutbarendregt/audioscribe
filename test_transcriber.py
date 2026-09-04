import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import transcriber


class _FakeChunk:
    def export(self, path, format):
        Path(path).write_bytes(b"audio")


class _FakeAudio:
    def __len__(self):
        return 1_000

    def __getitem__(self, key):
        return _FakeChunk()


class ChunkFileTests(unittest.TestCase):
    def test_concurrent_transcriptions_use_distinct_chunk_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            transcriber.tempfile, "gettempdir", return_value=temp_dir
        ), patch.object(transcriber.AudioSegment, "from_file", return_value=_FakeAudio()):
            first_paths = transcriber.split_audio_into_chunks(Path("first.wav"))
            second_paths = transcriber.split_audio_into_chunks(Path("second.wav"))

            self.assertTrue(set(first_paths).isdisjoint(second_paths))


if __name__ == "__main__":
    unittest.main()
