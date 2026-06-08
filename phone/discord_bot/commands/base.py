# CHARIS CAT 2025
# BABYLLM Command Module Base

from __future__ import annotations

from discord.ext import commands


class MainCogCommandCog(commands.Cog):
    """Shared helpers for command cogs that delegate to the main BBY cog."""

    def __init__(self, bot, main_cog):
        self.bot = bot
        self.main_cog = main_cog

    def author_key(self, ctx) -> str:
        author = (
            str(getattr(getattr(ctx, "author", None), "name", "") or "").strip().lower()
        )
        if hasattr(self.bot, "normalise_user_identity"):
            author = self.bot.normalise_user_identity(author)
        return author

    def ensure_user_memory(self, user_key: str) -> dict:
        if not hasattr(self.bot, "userMemory"):
            return {}
        if hasattr(self.bot, "_get_default_user_memory"):
            return self.bot.userMemory.setdefault(
                user_key, self.bot._get_default_user_memory()
            )
        return self.bot.userMemory.setdefault(user_key, {})


async def setup_main_cog_child(bot, cog_cls, *, main_cog_name: str = "BBYCOG") -> bool:
    """Attach a modular command cog that depends on the main BBY cog."""
    main_cog = bot.get_cog(main_cog_name)
    if main_cog is None:
        print(
            f"Warning: Could not load {cog_cls.__name__} - main {main_cog_name} not found"
        )
        return False
    await bot.add_cog(cog_cls(bot, main_cog))
    return True
