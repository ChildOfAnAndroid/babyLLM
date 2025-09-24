# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM // phone/discord_bot/context.py
# v1.1

from types import SimpleNamespace

class FakeTyping:
    async def __aenter__(self): pass
    async def __aexit__(self, exc_type, exc_val, exc_tb): pass

class FakeMessage:
    def __init__(self, content, author, channel=None, guild=None, mentions=None):
        self.content = content
        self.author = author
        self.channel = channel or SimpleNamespace(name='web_channel', id=0)
        self.guild = guild
        self.mentions = mentions or []
        self.reactions = []
        self.id = 0

    async def add_reaction(self, emoji): print(f"[FAKE_CONTEXT] ignored attempt to add reaction: {emoji}")
 
def create_fake_context(user_text: str, author: str = 'kevinonline420'):
    captured_reply = ""

    async def fake_reply_func(content = "", embed=None):
        nonlocal captured_reply
        if content:
            # Ensure we always capture as string, never as object
            if hasattr(content, 'content'):
                captured_reply = str(content.content)  # Extract content from message objects
            else:
                captured_reply = str(content)
        return FakeMessage(captured_reply, SimpleNamespace(name='babyLLM'))

    fake_channel = SimpleNamespace(name='web_channel', id=0)

    async def _fake_fetch_member(_member_id):
        """Return ``None`` for guild member fetches in fake contexts.

        The real :meth:`discord.Guild.fetch_member` coroutine is awaited in
        several places throughout the bot.  The previous synchronous lambda
        caused ``TypeError: object NoneType can't be used in 'await'
        expression`` when the fake context (used for web interactions) hit
        those code paths.  Providing an async stub keeps behaviour aligned with
        the real API and prevents the crash while still signalling that the
        member doesn't exist.
        """

        return None

    fake_guild = SimpleNamespace(
        id=0,
        members=[],
        get_member=lambda member_id: None,
        fetch_member=_fake_fetch_member,
    )
    fake_author = SimpleNamespace(name=author, id=0, display_name=author, bot=False)
    fake_message = FakeMessage(user_text, fake_author, channel=fake_channel, guild=fake_guild, mentions=[])
    fake_command = SimpleNamespace(name='babyllm')  # Make it look like a babyllm command from web
    
    fake_context = SimpleNamespace(
        author=fake_author,
        message=fake_message,
        guild=fake_guild,
        reply=fake_reply_func,
        typing=lambda: FakeTyping(),
        channel=fake_channel,
        command=fake_command,  # Now the fake context has a command attribute!
    )

    return fake_context, lambda: captured_reply
 
__all__ = ["FakeTyping", "FakeMessage", "create_fake_context"]
