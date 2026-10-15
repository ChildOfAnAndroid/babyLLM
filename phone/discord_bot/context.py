# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM // phone/discord_bot/context.py
# v1.1

import inspect
from types import SimpleNamespace
from typing import Optional


class FakeTyping:
    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class FakeMessage:
    def __init__(
        self, content, author, channel=None, guild=None, mentions=None, message_id=0
    ):
        self.content = content
        self.author = author
        self.channel = channel or SimpleNamespace(name="web_channel", id=0)
        self.guild = guild
        self.mentions = mentions or []
        self.reactions = []
        self.id = message_id

    async def add_reaction(self, emoji):
        print(f"[FAKE_CONTEXT] ignored attempt to add reaction: {emoji}")


def create_fake_context(
    bot=None,
    author_id: str = None,
    author_name: str = "kevinonline420",
    channel_id: str = "0",
    message_content: str = "",
    platform: str = "web",
    user_text: str = None,
    reply_sink=None,
):
    """Compatibility signature for the canonical platform context factory."""
    legacy_return = user_text is not None
    if legacy_return:
        message_content = user_text
        author_id = author_id or author_name

    context = create_platform_command_context(
        bot=bot,
        platform=platform,
        author_id=author_id or author_name,
        author_name=author_name,
        channel_id=channel_id,
        message_content=message_content,
        reply_sink=reply_sink,
        send_sink=reply_sink,
    )
    if legacy_return:
        return context, context.get_captured_reply
    return context


def create_platform_command_context(
    *,
    bot=None,
    platform: str,
    author_id: str,
    author_name: str,
    channel_id: str,
    message_content: str,
    command_name: Optional[str] = None,
    message_id: Optional[str] = None,
    reply_sink=None,
    send_sink=None,
    is_mod: bool = False,
):
    """Create a platform-aware command context with unified send/reply semantics."""
    captured_reply = ""
    fake_channel = SimpleNamespace(
        name=str(channel_id),
        id=channel_id,
        send=None,
    )

    async def _emit_via_sink(sink, *, content="", embed=None, **kwargs):
        if sink is None:
            return None
        try:
            result = sink(content=content, embed=embed, **kwargs)
        except TypeError:
            # Some legacy sinks don't accept Discord-style kwargs.
            result = sink(content=content, embed=embed)
        if inspect.isawaitable(result):
            return await result
        return result

    async def platform_reply(content="", embed=None, **kwargs):
        nonlocal captured_reply
        if content:
            captured_reply = (
                str(content.content) if hasattr(content, "content") else str(content)
            )
        elif embed is not None:
            captured_reply = str(
                getattr(embed, "description", "") or getattr(embed, "title", "") or ""
            )

        await _emit_via_sink(reply_sink, content=content, embed=embed, **kwargs)
        return FakeMessage(
            captured_reply,
            SimpleNamespace(name="babyLLM"),
            channel=fake_channel,
            message_id=message_id or 0,
        )

    async def platform_send(content="", embed=None, **kwargs):
        nonlocal captured_reply
        if content:
            captured_reply = (
                str(content.content) if hasattr(content, "content") else str(content)
            )
        elif embed is not None:
            captured_reply = str(
                getattr(embed, "description", "") or getattr(embed, "title", "") or ""
            )

        sink = send_sink or reply_sink
        await _emit_via_sink(sink, content=content, embed=embed, **kwargs)
        return FakeMessage(
            captured_reply,
            SimpleNamespace(name="babyLLM"),
            channel=fake_channel,
            message_id=message_id or 0,
        )

    async def _fake_fetch_member(_member_id):
        return None

    fake_channel.send = platform_send

    fake_guild = SimpleNamespace(
        id=0,
        members=[],
        get_member=lambda member_id: None,
        fetch_member=_fake_fetch_member,
    )

    async def _fake_create_dm():
        return SimpleNamespace(send=platform_send)

    fake_author = SimpleNamespace(
        name=author_id or author_name,
        id=author_id or 0,
        display_name=author_name,
        bot=False,
        is_mod=bool(is_mod),
        create_dm=_fake_create_dm,
    )

    fake_message = FakeMessage(
        message_content,
        fake_author,
        channel=fake_channel,
        guild=fake_guild,
        mentions=[],
        message_id=message_id or 0,
    )

    if command_name is None:
        stripped = (message_content or "").strip()
        if stripped.startswith("!"):
            command_name = stripped.split()[0][1:] if " " in stripped else stripped[1:]
        else:
            command_name = "babyllm"

    fake_command = SimpleNamespace(name=command_name or "babyllm")

    return SimpleNamespace(
        author=fake_author,
        message=fake_message,
        guild=fake_guild,
        reply=platform_reply,
        send=platform_send,
        typing=lambda: FakeTyping(),
        channel=fake_channel,
        command=fake_command,
        bot=bot,
        platform=platform,
        get_captured_reply=lambda: captured_reply,
    )


__all__ = [
    "FakeTyping",
    "FakeMessage",
    "create_fake_context",
    "create_platform_command_context",
]
