self.content = content
self.author = author
self.reactions = []
self.id = 0

async def add_reaction(self, emoji):
    print(f"[FAKE_CONTEXT] Ignored attempt to add reaction: {emoji}")


def create_fake_context(user_text: str):
    """Create a minimal context object for web requests."""
    captured_reply = ""

    async def fake_reply_func(content="", embed=None):
        nonlocal captured_reply
        if content:
            captured_reply = str(content)
        return FakeMessage(content, SimpleNamespace(name="babyLLM"))

    fake_author = SimpleNamespace(name="kevinonline420", id=0, display_name="kevinonline420")
    fake_message = FakeMessage(user_text, fake_author)
    fake_guild = SimpleNamespace(id=0, get_member=lambda _id: None, fetch_member=lambda _id: None)

    fake_context = SimpleNamespace(
        author=fake_author,
        message=fake_message,
        guild=fake_guild,
        reply=fake_reply_func,
        typing=lambda: FakeTyping(),
        channel=SimpleNamespace(name="web_channel", id=0),
    )

    return fake_context, lambda: captured_reply

__all__ = ["FakeTyping", "FakeMessage", "create_fake_context"]