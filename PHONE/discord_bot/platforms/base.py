# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM PLATFORM BASE // phone/discord_bot/platforms/base.py
# v1.0

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from ..utils import to_british_english


@dataclass
class PlatformMessage:
    """Unified message representation across platforms"""

    content: str
    author_id: str
    author_display_name: str
    channel_id: str
    platform: str  # "discord" or "twitch"
    timestamp: float
    raw_message: Any  # Original platform-specific message object
    author_colour: Optional[str] = None
    is_bot: bool = False
    is_mod: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class PlatformContext:
    """Unified context for command execution"""

    message: PlatformMessage
    bot: Any
    command: Optional[str] = None
    args: Optional[list] = None
    platform_ctx: Any = None  # Original platform context

    @property
    def author(self):
        """Convenience property to access author info"""

        class Author:
            def __init__(self, msg):
                self.name = msg.author_id
                self.display_name = msg.author_display_name
                self.id = msg.author_id
                self.is_mod = msg.is_mod

        return Author(self.message)

    async def send(self, content: str):
        """Send a message to the channel"""
        await self.bot.platform_send_message(
            self.message.platform,
            self.message.channel_id,
            to_british_english(str(content or "")),
        )

    async def reply(self, content: str):
        """Reply to the message"""
        await self.bot.platform_reply_message(
            self.message.platform,
            self.message,
            to_british_english(str(content or "")),
        )


class PlatformAdapter(ABC):
    """Abstract base class for platform adapters"""

    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.platform_name = "unknown"

    @abstractmethod
    async def start(self):
        """Start the platform bot"""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the platform bot"""
        pass

    @abstractmethod
    async def send_message(self, channel_id: str, content: str):
        """Send a message to a channel"""
        pass

    @abstractmethod
    async def reply_message(self, message: PlatformMessage, content: str):
        """Reply to a specific message"""
        pass

    @abstractmethod
    def format_message(self, user: str, text: str) -> str:
        """Format a message for the buffer in platform-specific way"""
        pass

    @abstractmethod
    def is_command_allowed(self, command_name: str) -> bool:
        """Check if a command is allowed on this platform"""
        pass

    async def handle_message(self, platform_message: PlatformMessage):
        """Handle an incoming message - called by platform-specific code"""
        await self.bot.handle_platform_message(platform_message)

    async def handle_command(self, platform_ctx: PlatformContext):
        """Handle a command - called by platform-specific code"""
        await self.bot.handle_platform_command(platform_ctx)
