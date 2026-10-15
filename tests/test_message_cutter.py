import unittest
import sys
import os
import discord
from unittest.mock import MagicMock, AsyncMock

# Adjust path to import phone.discord_bot.utils and platforms
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phone.discord_bot.utils import split_markdown_message
from phone.discord_bot.platforms.discord_adapter import DiscordAdapter
from phone.discord_bot.platforms.base import PlatformMessage

class TestMessageCutter(unittest.TestCase):
    def test_message_under_limit(self):
        text = "Hello, world!"
        chunks = split_markdown_message(text, max_chunk_len=100)
        self.assertEqual(chunks, [text])

    def test_long_normal_paragraph(self):
        text = "This is the first sentence. This is the second sentence. And here is the third sentence."
        chunks = split_markdown_message(text, max_chunk_len=40)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 40, f"Chunk too long: {len(chunk)} -> {repr(chunk)}")
        self.assertEqual("".join(chunks), text)

    def test_long_bullet_list(self):
        text = "* Item 1 is very cool\n* Item 2 is even cooler\n* Item 3 is the best ever"
        chunks = split_markdown_message(text, max_chunk_len=25)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 25, f"Chunk too long: {len(chunk)} -> {repr(chunk)}")
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(chunk.startswith("* ") or chunk.startswith("\n* ") for chunk in chunks[1:]))

    def test_short_heading_does_not_consume_first_chunk(self):
        heading = "item values! (3902 total ranked items)\n\ntop 420:\n"
        rows = "".join(f"{i}. item {i} is BBY123\n" for i in range(1, 421))
        chunks = split_markdown_message(heading + rows, max_chunk_len=1950)

        self.assertGreater(len(chunks), 1)
        self.assertIn("1. item 1", chunks[0])
        self.assertGreater(len(chunks[0]), 975)

    def test_long_fenced_code_block_with_language(self):
        text = "```python\ndef my_awesome_function():\n    print('Hello World')\n    return 42\n```"
        chunks = split_markdown_message(text, max_chunk_len=75)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 75, f"Chunk too long: {len(chunk)} -> {repr(chunk)}")
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].endswith("```"))
        self.assertTrue(chunks[1].startswith("```python\n"))

    def test_bold_text_spanning_split_point(self):
        text = "This is **extremely bold and long text that will surely cross the split point**"
        chunks = split_markdown_message(text, max_chunk_len=60)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 60, f"Chunk too long: {len(chunk)} -> {repr(chunk)}")
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].endswith("**"))
        self.assertTrue(chunks[1].startswith("**"))

    def test_spoiler_spanning_split_point(self):
        text = "This is ||extremely secret information that will surely cross the split point||"
        chunks = split_markdown_message(text, max_chunk_len=60)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 60, f"Chunk too long: {len(chunk)} -> {repr(chunk)}")
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].endswith("||"))
        self.assertTrue(chunks[1].startswith("||"))

    def test_inline_code_near_split_point(self):
        text = "Some text before and then `atomic_inline_code` is here."
        chunks = split_markdown_message(text, max_chunk_len=35)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 35, f"Chunk too long: {len(chunk)} -> {repr(chunk)}")
        joined = "".join(chunks)
        self.assertEqual(joined, text)
        self.assertTrue(any("`atomic_inline_code`" in chunk for chunk in chunks))

    def test_markdown_link_near_split_point(self):
        text = "Click here [Google Link](https://google.com) to search."
        chunks = split_markdown_message(text, max_chunk_len=35)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 35, f"Chunk too long: {len(chunk)} -> {repr(chunk)}")
        self.assertEqual("".join(chunks), text)
        self.assertTrue(any("[Google Link](https://google.com)" in chunk for chunk in chunks))

    def test_very_long_single_word(self):
        text = "SupercalifragilisticexpialidociousLengthyWordWithoutAnySpaces"
        chunks = split_markdown_message(text, max_chunk_len=15)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 15, f"Chunk too long: {len(chunk)} -> {repr(chunk)}")
        self.assertEqual("".join(chunks), text)

    def test_mixed_markdown_nested(self):
        text = "Text **bold and ||nested spoiler** still bold|| end."
        chunks = split_markdown_message(text, max_chunk_len=30)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 30, f"Chunk too long: {len(chunk)} -> {repr(chunk)}")

        def strip_delims(s):
            return s.replace("**", "").replace("||", "").replace("```", "").replace("\n", " ").replace("  ", " ").strip()
        self.assertEqual(strip_delims("".join(chunks)), strip_delims(text))

    # --- NEW SPECIFIC EDGE/FAILURE CASES ---

    def test_markdown_link_longer_than_limit(self):
        # Link has length 43, limit is 25. Should fall back to hard split inside the link
        text = "[Google Search Link](https://www.google.com)"
        chunks = split_markdown_message(text, max_chunk_len=25)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 25)
        self.assertEqual("".join(chunks), text)
        self.assertEqual(len(chunks), 2)

    def test_inline_code_longer_than_limit(self):
        # Inline code has length 31, limit is 20. Should fall back to hard split inside it
        text = "`super_duper_long_inline_code`"
        chunks = split_markdown_message(text, max_chunk_len=20)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 20)
        self.assertEqual("".join(chunks), text)
        self.assertEqual(len(chunks), 2)

    def test_unclosed_fenced_code_block_reopen_close(self):
        # Case A: Long unclosed code block requiring splitting
        text_long = "```python\ndef run():\n    print('Unclosed code block')"
        chunks_long = split_markdown_message(text_long, max_chunk_len=40)
        for chunk in chunks_long:
            self.assertTrue(len(chunk) <= 40, f"Chunk too long: {len(chunk)}")
        self.assertTrue(chunks_long[-1].endswith("```"))

        # Case B: Under-limit unclosed code block (e.g. max_chunk_len=1950)
        # It must be properly closed to avoid leaking formatting to Discord chat
        text_short = "```python\nprint('x')"
        chunks_short = split_markdown_message(text_short, max_chunk_len=1950)
        self.assertEqual(len(chunks_short), 1)
        self.assertTrue(chunks_short[0].endswith("```"))
        self.assertEqual(chunks_short[0], "```python\nprint('x')\n```")

    def test_asterisks_as_bullets_not_italic(self):
        # A list using * as bullets and * inside words (e.g. *bullet* or similar)
        # It should not try to close/reopen single asterisks
        text = "* Item A\n* Item B\n* Item C"
        chunks = split_markdown_message(text, max_chunk_len=12)
        for chunk in chunks:
            # If it tried to balance single asterisks, it would inject closing/reopening ones
            # Let's ensure no extra asterisks are injected
            self.assertNotIn("**", chunk)
            self.assertEqual(chunk.count("*"), 1)
        self.assertEqual("".join(chunks), text)

    def test_snake_case_underscores_ignored(self):
        # Text containing underscores in snake_case names should not trigger italic/underline
        text = "variable_name_one is very long and database_connection_string is also long"
        chunks = split_markdown_message(text, max_chunk_len=30)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 30)
        # Should be exactly equal to original text because no markers were injected
        self.assertEqual("".join(chunks), text)

    def test_close_reopen_markers_size_clamping(self):
        # Small limit: 25. Text has bold and spoiler open: **||text||** (tags add 4 chars)
        # The splitter should shrink the chunk size to account for the closing/reopening tags
        # and ensure every chunk is <= 25.
        text = "This **||secret bold information is very long and spans multiple lines||** here."
        chunks = split_markdown_message(text, max_chunk_len=25)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 25, f"Chunk exceeded 25: {len(chunk)} -> {repr(chunk)}")

        # Verify that no characters were eaten/lost
        def clean_formatting(s):
            return s.replace("**", "").replace("||", "").replace("\n", " ").replace("  ", " ").strip()
        self.assertEqual(clean_formatting("".join(chunks)), clean_formatting(text))

    def test_preserves_content_no_truncation(self):
        # Test that joining chunks after removing only formatting preserves original text completely
        text = "A very long message containing multiple lines\n```python\nprint('hello')\n```\nand **bold** formatting."
        chunks = split_markdown_message(text, max_chunk_len=30)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 30)

        # Confirm every generated chunk preserves the relative order and content
        def clean_markdown(s):
            return s.replace("**", "").replace("```", "").replace("\n", "").replace(" ", "")
        self.assertEqual(clean_markdown("".join(chunks)), clean_markdown(text))

    # --- INTEGRATION MOCK TESTS ---

    async def _run_discord_adapter_send_splits_integration(self):
        # Mock Discord bot and channel
        mock_bot = MagicMock()
        mock_channel = MagicMock()
        mock_channel.send = AsyncMock()
        mock_bot.get_channel.return_value = mock_channel

        adapter = DiscordAdapter(mock_bot)

        # A message longer than 2000 characters
        long_content = "A" * 3000
        await adapter.send_message("12345", long_content)

        # Verify get_channel was called with channel id
        mock_bot.get_channel.assert_called_with(12345)
        # Verify it split and sent multiple chunks
        self.assertTrue(mock_channel.send.call_count > 1)
        # Verify every chunk was under the 2000 character limit
        for call in mock_channel.send.call_args_list:
            arg = call[0][0]
            self.assertTrue(len(arg) <= 2000)

    async def _run_discord_adapter_reply_splits_integration(self):
        mock_bot = MagicMock()
        mock_discord_msg = MagicMock(spec=discord.Message)
        mock_discord_msg.reply = AsyncMock()
        mock_discord_msg.channel = MagicMock()
        mock_discord_msg.channel.send = AsyncMock()

        adapter = DiscordAdapter(mock_bot)

        message = PlatformMessage(
            content="hello",
            author_id="test_id",
            author_display_name="test",
            channel_id="123",
            platform="discord",
            timestamp=12345.0,
            raw_message=mock_discord_msg
        )

        long_content = "B" * 3500
        await adapter.reply_message(message, long_content)

        # Verify it replied with the first chunk
        mock_discord_msg.reply.assert_called_once()
        reply_arg = mock_discord_msg.reply.call_args[0][0]
        self.assertTrue(len(reply_arg) <= 2000)

        # Verify it sent subsequent chunks in the channel
        self.assertTrue(mock_discord_msg.channel.send.call_count > 0)
        for call in mock_discord_msg.channel.send.call_args_list:
            arg = call[0][0]
            self.assertTrue(len(arg) <= 2000)

    def test_escaped_backticks(self):
        text = "some `world \\` test` here"
        chunks = split_markdown_message(text, max_chunk_len=100)
        self.assertEqual(chunks, [text])

    def test_escaped_delimiters(self):
        text = "this \\**is not bold** but **this is bold** and this is \\||not spoiler||"
        chunks = split_markdown_message(text, max_chunk_len=200)
        self.assertEqual(chunks, [text])

    async def _run_concurrent_sends_serialization(self):
        import asyncio
        mock_bot = MagicMock()
        mock_bot._channel_send_locks = {}

        sent_order = []

        async def mock_send(content):
            sent_order.append(content)
            await asyncio.sleep(0)  # force event loop yield
            return MagicMock()

        mock_channel = MagicMock()
        mock_channel.id = 456
        mock_channel.send = mock_send
        mock_bot.get_channel.return_value = mock_channel

        adapter = DiscordAdapter(mock_bot)

        content_a = "A1" + "a"*1000 + "\n\n" + "A2" + "a"*1000 + "\n\n" + "A3" + "a"*1000
        content_b = "B1" + "b"*1000 + "\n\n" + "B2" + "b"*1000 + "\n\n" + "B3" + "b"*1000

        await asyncio.gather(
            adapter.send_message("456", content_a),
            adapter.send_message("456", content_b)
        )

        self.assertEqual(len(sent_order), 6)

        first_char = sent_order[0][0]
        self.assertIn(first_char, ("A", "B"))

        if first_char == "A":
            expected = [
                "A1" + "a"*1000 + "\n\n",
                "A2" + "a"*1000 + "\n\n",
                "A3" + "a"*1000
            ]
            self.assertEqual(sent_order[:3], expected)
            self.assertTrue(all(x.startswith("B") for x in sent_order[3:]))
        else:
            expected = [
                "B1" + "b"*1000 + "\n\n",
                "B2" + "b"*1000 + "\n\n",
                "B3" + "b"*1000
            ]
            self.assertEqual(sent_order[:3], expected)
            self.assertTrue(all(x.startswith("A") for x in sent_order[3:]))

    async def _run_different_channels_not_blocked(self):
        import asyncio
        mock_bot = MagicMock()
        mock_bot._channel_send_locks = {}

        completed_order = []

        async def mock_send_a(content):
            await asyncio.sleep(0.05)  # slow send
            completed_order.append(content)

        async def mock_send_b(content):
            completed_order.append(content)  # instant send

        mock_channel_a = MagicMock()
        mock_channel_a.id = 111
        mock_channel_a.send = mock_send_a

        mock_channel_b = MagicMock()
        mock_channel_b.id = 222
        mock_channel_b.send = mock_send_b

        def get_channel_mock(channel_id):
            if channel_id == 111:
                return mock_channel_a
            if channel_id == 222:
                return mock_channel_b
            return None

        mock_bot.get_channel.side_effect = get_channel_mock

        adapter = DiscordAdapter(mock_bot)

        await asyncio.gather(
            adapter.send_message("111", "message for A"),
            adapter.send_message("222", "message for B")
        )

        self.assertEqual(completed_order, ["message for B", "message for A"])


# Helper for async tests in unittest
def run_async(coro):
    import asyncio
    return asyncio.run(coro)


class TestAsyncMessageCutter(unittest.TestCase):
    def test_discord_adapter_send_splits_integration(self):
        tc = TestMessageCutter()
        run_async(tc._run_discord_adapter_send_splits_integration())

    def test_discord_adapter_reply_splits_integration(self):
        tc = TestMessageCutter()
        run_async(tc._run_discord_adapter_reply_splits_integration())

    def test_concurrent_sends_serialization(self):
        tc = TestMessageCutter()
        run_async(tc._run_concurrent_sends_serialization())

    def test_different_channels_not_blocked(self):
        tc = TestMessageCutter()
        run_async(tc._run_different_channels_not_blocked())


if __name__ == "__main__":
    unittest.main()
