# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM // phone/discord_bot/context.py
# v1.2

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
            captured_reply = str(content)
        return FakeMessage(content, SimpleNamespace(name='babyLLM'))

    fake_channel = SimpleNamespace(name='web_channel', id=0)
    fake_guild = SimpleNamespace(id=0, members=[], get_member=lambda id: None, fetch_member=lambda id: None)
    fake_author = SimpleNamespace(name=author, id=0, display_name=author, bot=False)
    fake_message = FakeMessage(user_text, fake_author, channel=fake_channel, guild=fake_guild, mentions=[])
    
    fake_context = SimpleNamespace(
        author=fake_author,
        message=fake_message,
        guild=fake_guild,
        reply=fake_reply_func,
        typing=lambda: FakeTyping(),
        channel=fake_channel,
    )

    return fake_context, lambda: captured_reply
 
__all__ = ["FakeTyping", "FakeMessage", "create_fake_context"]
