import unittest
import sys
import os
from collections import deque
from unittest.mock import MagicMock, patch

# Adjust path to import phone.discord_bot.bot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phone.discord_bot.bot import BABYBOT_DISCORD


class TestTrainingBufferSplit(unittest.TestCase):
    def setUp(self):
        # Bypass __init__ to avoid external files and setup dependencies
        self.bot = BABYBOT_DISCORD.__new__(BABYBOT_DISCORD)
        self.bot.training_buffer = deque(maxlen=100)
        self.bot.training_buffer_size = 100
        self.bot.librarian = MagicMock()

        # Mock cleaning and checks to be neutral by default
        self.bot._clean_training_file_text = lambda text: text
        self.bot._normalise_buffer_ingest_text = lambda text: text
        self.bot._normalise_timeline_training_line = lambda text: text
        self.bot._format_for_training_buffer = lambda text: text
        self.bot._sanitise_training_buffer_text = lambda text: text
        self.bot._should_keep_training_buffer_line = lambda text: True
        self.bot._normalise_repetitive_training_signature = lambda text: None
        self.bot._is_recent_duplicate = lambda text, recent, **kwargs: False
        self.bot._should_accept_line = lambda text, recent, **kwargs: True
        self.bot._repeat_admission_probability = lambda text, recent, **kwargs: 1.0
        self.bot._save_training_buffer = MagicMock()

    def test_split_into_sentence_chunks_short_text(self):
        text = "Hello, this is a short sentence."
        chunks = self.bot._split_into_sentence_chunks(text, max_len=100)
        self.assertEqual(chunks, [text])

    def test_split_into_sentence_chunks_punctuation_preserved(self):
        text = "First sentence. Second sentence! Third sentence?"
        # Limit is 20: each sentence is ~15 chars.
        chunks = self.bot._split_into_sentence_chunks(text, max_len=20)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "First sentence.")
        self.assertEqual(chunks[1], "Second sentence!")
        self.assertEqual(chunks[2], "Third sentence?")

    def test_split_into_sentence_chunks_paragraph_preference(self):
        text = "Para 1\nPara 2\nPara 3"
        chunks = self.bot._split_into_sentence_chunks(text, max_len=50)
        # Should split strictly at newlines
        self.assertEqual(chunks, ["Para 1", "Para 2", "Para 3"])

    def test_split_into_sentence_chunks_no_mid_word_cut(self):
        text = "This is a sentence with no punctuation but quite a few words"
        # Split on 30 chars limit. Should split at a space, not mid-word
        chunks = self.bot._split_into_sentence_chunks(text, max_len=30)
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 30)
        self.assertEqual(" ".join(chunks), text)

    def test_split_into_sentence_chunks_extreme_word_split(self):
        # Single word exceeding limit
        text = "Supercalifragilisticexpialidocious"
        chunks = self.bot._split_into_sentence_chunks(text, max_len=10)
        self.assertEqual(chunks, ["Supercalif", "ragilistic", "expialidoc", "ious"])

    def test_preserves_content_modulo_whitespace(self):
        # A 4500-char text (simulated here with repeating sentences)
        sentence = "This is a sentence. "
        text = sentence * 225  # ~4500 characters
        chunks = self.bot._split_into_sentence_chunks(text, max_len=2000)

        # Verify no chunk exceeds 2000
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 2000)

        # Verify total text content modulo whitespace is preserved
        original_clean = "".join(text.split())
        chunks_clean = "".join("".join(chunks).split())
        self.assertEqual(original_clean, chunks_clean)

    def test_training_buffer_add_splits_and_does_not_recurse(self):
        # Spy on _training_buffer_add and _training_buffer_add_single_chunk
        self.bot._training_buffer_add_single_chunk = MagicMock(
            side_effect=self.bot._training_buffer_add_single_chunk
        )

        with patch.object(self.bot, "_training_buffer_add", wraps=self.bot._training_buffer_add) as spy_add:
            text = "Sentence one. Sentence two! Sentence three?"
            # Pass a mock text that will be split
            # We mock sanitise/keep checks to return it directly
            self.bot._sanitise_training_buffer_text = lambda text: text
            self.bot._should_keep_training_buffer_line = lambda text: True

            # If we call it with text that has len > 2000 (we can use max_len=20 inside the test logic,
            # but bot.py has hardcoded 2000. Let's construct a string longer than 2000 chars).
            sentence1 = "A" * 1200 + "."
            sentence2 = "B" * 1200 + "!"
            long_text = sentence1 + " " + sentence2  # 2402 characters

            res = self.bot._training_buffer_add(long_text)

            self.assertTrue(res)
            # _training_buffer_add should only be called once (no recursion)
            spy_add.assert_called_once()
            # _training_buffer_add_single_chunk should be called twice
            self.assertEqual(self.bot._training_buffer_add_single_chunk.call_count, 2)
            # The two chunks should be added to the buffer
            self.assertEqual(len(self.bot.training_buffer), 2)
            self.assertEqual(self.bot.training_buffer[0], sentence1)
            self.assertEqual(self.bot.training_buffer[1], sentence2)

    def test_single_save_operation_batched(self):
        # Construct long text
        sentence1 = "A" * 1200 + "."
        sentence2 = "B" * 1200 + "!"
        long_text = sentence1 + " " + sentence2

        self.bot._training_buffer_add(long_text)
        # Verify that save was called exactly once, not twice
        self.bot._save_training_buffer.assert_called_once()

    def test_pathological_emergency_truncation_logged(self):
        # Construct a massive chunk. Suppose the split function failed and returned a chunk > 2000.
        # We can mock _split_into_sentence_chunks to return a chunk of length 2500.
        self.bot._split_into_sentence_chunks = lambda text, max_len: ["C" * 2500]

        with patch("phone.discord_bot.bot.logger") as mock_logger:
            # Call training buffer add
            self.bot._training_buffer_add("dummy" * 500)

            # The chunk should be truncated to 2000
            self.assertEqual(len(self.bot.training_buffer[0]), 2000)
            # It should have logged an error containing the word pathological/exceeds
            mock_logger.error.assert_called()
            # Verify the warning log about splitting is also called
            mock_logger.warning.assert_called()


if __name__ == "__main__":
    unittest.main()
