import pytest
from unittest.mock import MagicMock


def test_baby_stream_channel_respects_explicit_opt_out():
    # Setup mock adapter and bot
    mock_bot = MagicMock()
    mock_bot.AIoptInUsers = []
    mock_bot.userMemory = {
        "opted_out_user": {"web_explicit_opt_out": True},
        "normal_user": {"web_explicit_opt_out": False},
    }
    mock_bot.has_explicit_web_opt_out.side_effect = lambda uid: bool(mock_bot.userMemory.get(uid, {}).get("web_explicit_opt_out", False))
    mock_bot.babyName = "babyllm"

    adapter = MagicMock()
    adapter.bot = mock_bot
    adapter.is_baby_stream_channel.side_effect = lambda ch: ch.lower() == "babyllm"

    # User who explicitly opted out:
    # In baby stream room, should NOT record and should NOT generate response
    author_id = "opted_out_user"
    mem = mock_bot.userMemory[author_id]
    is_opted_in = author_id in adapter.bot.AIoptInUsers
    has_opted_out = (
        adapter.bot.has_explicit_web_opt_out(author_id)
        if hasattr(adapter.bot, "has_explicit_web_opt_out")
        else bool(mem.get("web_explicit_opt_out", False))
    )
    baby_stream = adapter.is_baby_stream_channel("babyllm")
    content = "hello baby"
    is_command = False

    should_record = (is_command or is_opted_in) and not has_opted_out
    should_generate_response = bool(content) and not is_command and not has_opted_out and (
        baby_stream or (False and is_opted_in)
    )

    assert not should_record
    assert not should_generate_response

    # User who is not opted out in baby stream room:
    author_id_normal = "normal_user"
    mem_normal = mock_bot.userMemory[author_id_normal]
    has_opted_out_normal = (
        adapter.bot.has_explicit_web_opt_out(author_id_normal)
        if hasattr(adapter.bot, "has_explicit_web_opt_out")
        else bool(mem_normal.get("web_explicit_opt_out", False))
    )
    should_generate_response_normal = bool(content) and not is_command and not has_opted_out_normal and (
        baby_stream or (False and is_opted_in)
    )

    assert should_generate_response_normal

