# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM DISCORD ADAPTER // phone/discord_bot/platforms/discord_adapter.py
# v1.0


import discord

from ..utils import to_british_english, split_markdown_message
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
                text = to_british_english(str(content or ""))
                chunks = split_markdown_message(text)
                for chunk in chunks:
                    await channel.send(chunk)
        except Exception as e:
            print(f"[DiscordAdapter] Error sending message: {e}")

    async def reply_message(self, message: PlatformMessage, content: str):
        """Reply to a Discord message"""
        try:
            discord_msg = message.raw_message
            if isinstance(discord_msg, discord.Message):
                text = to_british_english(str(content or ""))
                chunks = split_markdown_message(text)
                if chunks:
                    await discord_msg.reply(chunks[0])
                    for chunk in chunks[1:]:
                        await discord_msg.channel.send(chunk)
        except Exception as e:
            print(f"[DiscordAdapter] Error replying: {e}")

    def format_message(self, user: str, text: str) -> str:
        """Format a message for Discord (already handled by bot)"""
        return f"{user}: {text}"

    def is_command_allowed(self, command_name: str) -> bool:
        """All commands allowed on Discord"""
        return True

