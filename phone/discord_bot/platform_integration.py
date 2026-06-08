# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM PLATFORM INTEGRATION MIXIN // phone/discord_bot/platform_integration.py
# v1.0

"""
Platform integration mixin for BABYBOT_DISCORD
This adds multi-platform support without modifying bot.py file directly
"""

import asyncio
import inspect
import shlex
from collections import deque
from typing import Any, Dict


from .context import create_platform_command_context
from .platforms import (
    DiscordAdapter,
    PlatformAdapter,
    PlatformContext,
    PlatformMessage,
    TwitchAdapter,
    WebAdapter,
)
from .utils import to_british_english, embed_to_plain_text


class PlatformIntegrationMixin:
    """Mixin to add multi-platform support to BABYBOT_DISCORD"""

    _TWITCH_COMMAND_ALIASES = {
        "b": "babyllm",
        "ai": "babyllm",
        "aioptin": "bbyoptin",
        "aioptout": "bbyoptout",
        "aioptcheck": "bbyoptcheck",
        "babylim": "babyllm",
        "bbylim": "babyllm",
        "bbyskunk": "babyllm",
        "bbysmink": "bbysminks",
        "sminks": "bbysminks",
        "bsmink": "bbysminks",
        "bbycheers": "bbysminks",
        "sminkboard": "bbysminkboard",
        "bsminkboard": "bbysminkboard",
    }

    def init_platforms(self):
        """Initialize platform adapters - call this in bot __init__"""
        self.platforms: Dict[str, PlatformAdapter] = {}
        self.platform_message_queue = asyncio.Queue()

        # Discord is always active (it's the main bot)
        self.platforms["discord"] = DiscordAdapter(self)

        print("[PlatformIntegration] Multi-platform support initialized")

    def has_explicit_web_opt_out(self, author_id: str) -> bool:
        """Return True only when the user has explicitly opted out on web."""
        author_key = str(author_id or "").strip().lower()
        if not author_key:
            return False
        mem = self.userMemory.get(author_key, {})
        if not isinstance(mem, dict):
            return False
        return bool(mem.get("web_explicit_opt_out", False))

    async def enable_twitch(self, channels=None):
        """Enable Twitch integration

        Args:
            channels: str (single channel) or list of channel names
        """
        if "twitch" in self.platforms:
            print("[PlatformIntegration] Twitch already enabled")
            return

        # Support both single channel and list
        if channels is None:
            from config import twitch_channels

            channels = twitch_channels
        elif isinstance(channels, str):
            channels = [channels]

        print(f"[PlatformIntegration] Enabling Twitch for channels: {channels}")
        twitch_adapter = TwitchAdapter(self, channels)
        self.platforms["twitch"] = twitch_adapter
        await twitch_adapter.start()
        print("[PlatformIntegration] Twitch enabled successfully")

    async def disable_twitch(self):
        """Disable Twitch integration"""
        if "twitch" not in self.platforms:
            return

        await self.platforms["twitch"].stop()
        del self.platforms["twitch"]
        print("[PlatformIntegration] Twitch disabled")

    async def enable_web(self, port: int = 4420):
        """Enable Web API integration (replaces standalone bbyLocal)"""
        if "web" in self.platforms:
            print("[PlatformIntegration] Web API already enabled")
            return

        print(f"[PlatformIntegration] Enabling Web API on port {port}")
        web_adapter = WebAdapter(self, port=port)
        self.platforms["web"] = web_adapter
        await web_adapter.start()
        print("[PlatformIntegration] Web API enabled successfully")

    async def disable_web(self):
        """Disable Web API integration"""
        if "web" not in self.platforms:
            return

        await self.platforms["web"].stop()
        del self.platforms["web"]
        print("[PlatformIntegration] Web API disabled")

    async def handle_platform_message(self, platform_msg: PlatformMessage):
        """Handle incoming message from any platform"""
        author_id = str(platform_msg.author_id or "").strip().lower()
        if not author_id:
            return
        platform_msg.author_id = author_id

        # Skip bot messages
        if platform_msg.is_bot:
            return

        # Update user memory
        if author_id not in self.userMemory:

            def get_default():
                return {
                    "nickname": None,
                    "display_name": None,
                    "timezone": "Europe/London",
                    "BBY": 0.0,
                    "spamMax": 0.3,
                    "bbybook": [],
                    "wins": 0.0,
                    "losses": 0.0,
                    "draws": 0.0,
                    "last_seen": platform_msg.timestamp,
                    "message_count": 0.0,
                    "loyalty": 1,
                    "last_message_words": set(),
                    "creative_combo": 1,
                    "spammer": 1,
                    "inventory": {},
                    "favourites": [],
                    "next_talk_milestone": 50,
                    "translate_wins": 0,
                    "translate_losses": 0,
                    "fave_token_usage": 0,
                    "command_usage": {},
                    "opt_in": False,
                    "last_fact_time": 0,
                    "web_explicit_opt_out": False,
                }

            self.userMemory[author_id] = get_default()

        mem = self.userMemory[author_id]
        mem["display_name"] = platform_msg.author_display_name
        mem["last_seen"] = platform_msg.timestamp
        mem["message_count"] = mem.get("message_count", 0) + 1

        # Add to buffer - privacy checks already done by platform adapters
        # Web users are TREATED as opted in by default
        # BUT: If a web user has explicitly opted out (!bbyoptout), respect that
        is_web_user = platform_msg.platform == "web"
        is_opted_in = author_id in self.AIoptInUsers

        # Explicit opt-out is persisted separately so "never opted in" isn't treated as opt-out.
        has_explicitly_opted_out = is_web_user and self.has_explicit_web_opt_out(
            author_id
        )

        # Web users: treated as opted in UNLESS they've explicitly opted out
        if is_opted_in or (is_web_user and not has_explicitly_opted_out):
            # Format message for buffer
            adapter = self.platforms.get(platform_msg.platform)
            if adapter:
                formatted = adapter.format_message(author_id, platform_msg.content)
                if self._buffer_add(formatted, speaker_hint=author_id):
                    # Queue for training
                    if hasattr(self, "training_queue") and self.training_queue.qsize() < 20:
                        await self.training_queue.put(
                            {
                                "type": "chat",
                                "text": formatted,
                                "platform": platform_msg.platform,
                            }
                        )

    async def handle_platform_command(self, platform_ctx: PlatformContext):
        """Handle command from any platform"""
        command_name = platform_ctx.command
        platform = platform_ctx.message.platform

        # Get adapter
        adapter = self.platforms.get(platform)
        if not adapter:
            print(f"[PlatformIntegration] No adapter for platform: {platform}")
            return

        # Check if command is allowed on this platform
        if not adapter.is_command_allowed(command_name):
            await platform_ctx.reply(
                f"sorry! !{command_name} is too complex for {platform}. try it on discord! :)"
            )
            return

        if not self.cog:
            await platform_ctx.reply("oops! command system is not ready yet")
            return

        try:
            resolved_name, callback = self._resolve_platform_command(
                command_name, platform
            )
            if callback is None:
                await platform_ctx.reply(
                    f"sorry! !{command_name} isn't available right now"
                )
                return

            if platform == "twitch":
                source_command = (command_name or "").lower()
                message_content_for_ctx = platform_ctx.message.content
                if source_command == "bbyskunk":
                    legacy_arg_text = self._extract_command_arg_text(
                        platform_ctx.message.content
                    )
                    if legacy_arg_text:
                        message_content_for_ctx = (
                            f"!babyllm $skunkllm {legacy_arg_text}"
                        )
                    else:
                        message_content_for_ctx = "!babyllm $skunkllm"
                reply_budget = {"sent": 0, "notice_sent": False}

                async def twitch_reply_sink(content="", embed=None, **kwargs):
                    async def _send_twitch_text(text: str):
                        if not text:
                            return
                        text = to_british_english(text)
                        # Keep Twitch responses concise: long command dumps should move to Discord.
                        if reply_budget["sent"] >= 3:
                            if not reply_budget["notice_sent"]:
                                reply_budget["notice_sent"] = True
                                notice = "…this command is long; use Discord for the full output."
                                reply_method = getattr(
                                    platform_ctx.platform_ctx, "reply", None
                                )
                                if callable(reply_method):
                                    await reply_method(notice)
                                else:
                                    await platform_ctx.platform_ctx.send(notice)
                            return

                        text = text[:499]
                        reply_budget["sent"] += 1
                        reply_method = getattr(platform_ctx.platform_ctx, "reply", None)
                        if callable(reply_method):
                            await reply_method(text)
                        else:
                            await platform_ctx.platform_ctx.send(text)

                    # Twitch doesn't support Discord embeds directly; send plain text fallback.
                    if embed is not None:
                        fallback = embed_to_plain_text(embed)
                        if fallback:
                            await _send_twitch_text(fallback)
                    elif content:
                        await _send_twitch_text(str(content))

                raw_message = platform_ctx.message.raw_message
                message_id = (
                    str(
                        getattr(raw_message, "id", None)
                        or getattr(raw_message, "message_id", None)
                        or ""
                    )
                    or None
                )

                fake_ctx = create_platform_command_context(
                    bot=self,
                    platform="twitch",
                    author_id=platform_ctx.message.author_id,
                    author_name=platform_ctx.message.author_display_name,
                    channel_id=platform_ctx.message.channel_id,
                    message_content=message_content_for_ctx,
                    command_name=resolved_name,
                    message_id=message_id,
                    reply_sink=twitch_reply_sink,
                    send_sink=twitch_reply_sink,
                    is_mod=platform_ctx.message.is_mod,
                )

                arg_text = self._extract_command_arg_text(platform_ctx.message.content)
                positional_args, keyword_args = self._parse_command_arguments(
                    callback, arg_text
                )
                await self._invoke_platform_callback(
                    callback, fake_ctx, positional_args, keyword_args
                )
            else:
                # Keep Discord path straightforward.
                if inspect.ismethod(callback):
                    await callback(platform_ctx.platform_ctx)
                else:
                    await callback(self.cog, platform_ctx.platform_ctx)

        except Exception as e:
            print(f"[PlatformIntegration] Error executing command {command_name}: {e}")
            import traceback

            traceback.print_exc()
            await platform_ctx.reply(f"oops! something broke: {str(e)[:100]}")

    def _resolve_platform_command(self, command_name: str, platform: str):
        """Resolve a platform command to an executable callback."""
        normalized_name = command_name.lower()
        if platform == "twitch":
            normalized_name = self._TWITCH_COMMAND_ALIASES.get(
                normalized_name, normalized_name
            )

        # Prefer Discord's command registry (handles aliases).
        if hasattr(self, "get_command"):
            cmd = self.get_command(normalized_name)
            if cmd is not None:
                return normalized_name, cmd.callback

        # Fallback for direct method names.
        if hasattr(self.cog, normalized_name):
            return normalized_name, getattr(self.cog, normalized_name)

        suffixed = f"{normalized_name}_command"
        if hasattr(self.cog, suffixed):
            return normalized_name, getattr(self.cog, suffixed)

        return normalized_name, None

    def _extract_command_arg_text(self, content: str) -> str:
        if not content:
            return ""
        stripped = content.strip()
        if not stripped.startswith("!"):
            return ""
        parts = stripped.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""

    def _coerce_argument(self, raw_value: str, annotation: Any):
        if annotation in (inspect._empty, str, None):
            return raw_value
        if annotation is int:
            return int(raw_value)
        if annotation is float:
            return float(raw_value)
        if annotation is bool:
            lowered = raw_value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
            raise ValueError(f"could not parse boolean value: {raw_value}")
        return raw_value

    def _parse_command_arguments(self, callback, arg_text: str):
        try:
            tokens = shlex.split(arg_text) if arg_text else []
        except ValueError:
            tokens = arg_text.split() if arg_text else []

        signature = inspect.signature(callback)
        params = list(signature.parameters.values())
        if inspect.ismethod(callback):
            # Bound method: first param is ctx.
            command_params = params[1:]
        else:
            # Unbound callback: first two params are self, ctx.
            command_params = params[2:]

        positional_args = []
        keyword_args = {}

        for param in command_params:
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                if tokens:
                    raw_value = tokens.pop(0)
                    positional_args.append(
                        self._coerce_argument(raw_value, param.annotation)
                    )
                elif param.default is inspect._empty:
                    raise ValueError(f"missing required argument: {param.name}")
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                for raw_value in tokens:
                    positional_args.append(self._coerce_argument(raw_value, str))
                tokens = []
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                if tokens:
                    raw_rest = " ".join(tokens)
                    tokens = []
                    keyword_args[param.name] = self._coerce_argument(
                        raw_rest, param.annotation
                    )
                elif param.default is inspect._empty:
                    raise ValueError(f"missing required argument: {param.name}")

        return positional_args, keyword_args

    async def _invoke_platform_callback(
        self, callback, fake_ctx, positional_args, keyword_args
    ):
        if inspect.ismethod(callback):
            await callback(fake_ctx, *positional_args, **keyword_args)
            return
        await callback(self.cog, fake_ctx, *positional_args, **keyword_args)

    async def platform_send_message(self, platform: str, channel_id: str, content: str):
        """Send message to a specific platform"""
        adapter = self.platforms.get(platform)
        if adapter:
            await adapter.send_message(channel_id, content)

    async def platform_reply_message(
        self, platform: str, message: PlatformMessage, content: str
    ):
        """Reply to a message on a specific platform"""
        adapter = self.platforms.get(platform)
        if adapter:
            await adapter.reply_message(message, content)

