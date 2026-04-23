# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM DISCORD ADAPTER // phone/discord_bot/platforms/discord_adapter.py
# v1.0


import discord

from ..utils import to_british_english
from .base import PlatformAdapter, PlatformContext, PlatformMessage


class DiscordAdapter(PlatformAdapter):
    """Discord platform adapter - wraps existing Discord bot functionality"""

    def __init__(self, bot_instance):
        super().__init__(bot_instance)
        self.platform_name = "discord"
        # All Discord commands are allowed
        self.allowed_commands = None  # None means all commands allowed

    async def start(self):
        """Discord bot is already running via the main bot instance"""
        pass

    async def stop(self):
        """Stop Discord bot"""
        if hasattr(self.bot, "close"):
            await self.bot.close()

    async def send_message(self, channel_id: str, content: str):
        """Send a message to a Discord channel"""
        try:
            channel = self.bot.get_channel(int(channel_id))
            if channel:
                # Discord has 2000 char limit
                await channel.send(to_british_english(str(content or ""))[:2000])
        except Exception as e:
            print(f"[DiscordAdapter] Error sending message: {e}")

    async def reply_message(self, message: PlatformMessage, content: str):
        """Reply to a Discord message"""
        try:
            discord_msg = message.raw_message
            if isinstance(discord_msg, discord.Message):
                await discord_msg.reply(to_british_english(str(content or ""))[:2000])
        except Exception as e:
            print(f"[DiscordAdapter] Error replying: {e}")

    def format_message(self, user: str, text: str) -> str:
        """Format a message for Discord (already handled by bot)"""
        return f"{user}: {text}"

    def is_command_allowed(self, command_name: str) -> bool:
        """All commands allowed on Discord"""
        return True

    @staticmethod
    def convert_discord_message(discord_msg: discord.Message) -> PlatformMessage:
        """Convert Discord message to PlatformMessage"""
        return PlatformMessage(
            content=discord_msg.content,
            author_id=str(discord_msg.author.id),
            author_display_name=discord_msg.author.display_name,
            channel_id=str(discord_msg.channel.id),
            platform="discord",
            timestamp=discord_msg.created_at.timestamp(),
            raw_message=discord_msg,
            is_bot=discord_msg.author.bot,
            is_mod=discord_msg.author.guild_permissions.administrator
            if hasattr(discord_msg, "guild") and discord_msg.guild
            else False,
        )

    @staticmethod
    def convert_discord_context(ctx) -> PlatformContext:
        """Convert Discord context to PlatformContext"""
        platform_msg = DiscordAdapter.convert_discord_message(ctx.message)
        return PlatformContext(
            message=platform_msg,
            bot=ctx.bot,
            command=ctx.command.name if ctx.command else None,
            args=ctx.args if hasattr(ctx, "args") else None,
            platform_ctx=ctx,
        )
