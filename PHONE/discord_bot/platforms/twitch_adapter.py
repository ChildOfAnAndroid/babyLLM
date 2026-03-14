# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM TWITCH ADAPTER // phone/discord_bot/platforms/twitch_adapter.py
# v1.0

import asyncio
import json
import os
import re
from datetime import datetime
from twitchio import eventsub
from twitchio.ext import commands as twitch_commands
from typing import Optional
from .base import PlatformAdapter, PlatformMessage, PlatformContext
from config import twitch_channel
from secret import SECRETtwitchTokenSECRET
import time
import aiohttp
from ..utils import to_british_english, normalise_embed_british_english

# Try to import new required Twitch credentials (TwitchIO v3+)
try:
    from secret import (
        SECRETtwitchClientIdSECRET,
        SECRETtwitchClientSecretSECRET,
        SECRETtwitchBotIdSECRET,
    )
except ImportError:
    # Fallback to None if not defined
    SECRETtwitchClientIdSECRET = None
    SECRETtwitchClientSecretSECRET = None
    SECRETtwitchBotIdSECRET = None
    print("[TwitchAdapter] WARNING: Missing Twitch credentials. Add to secret.py:")
    print("  SECRETtwitchClientIdSECRET = 'your_client_id'")
    print("  SECRETtwitchClientSecretSECRET = 'your_client_secret'")
    print("  SECRETtwitchBotIdSECRET = 'your_bot_user_id'")

try:
    from secret import SECRETtwitchRefreshTokenSECRET
except ImportError:
    SECRETtwitchRefreshTokenSECRET = ""

ANSI_RESET = "\033[0m"
ANSI_TWITCH_PURPLE = "\033[38;2;145;70;255m"

TWITCH_NAMED_CHAT_COLOURS = {
    "blue",
    "blue_violet",
    "cadet_blue",
    "chocolate",
    "coral",
    "dodger_blue",
    "firebrick",
    "golden_rod",
    "green",
    "hot_pink",
    "orange_red",
    "red",
    "sea_green",
    "spring_green",
    "yellow_green",
}

TWITCH_CHAT_COLOUR_ALIASES = {
    "purple": "blue_violet",
    "violet": "blue_violet",
    "pink": "hot_pink",
    "baby": "hot_pink",
    "teal": "cadet_blue",
    "cyan": "cadet_blue",
    "aqua": "cadet_blue",
    "orange": "orange_red",
    "yellow": "yellow_green",
    "gold": "golden_rod",
    "goldenrod": "golden_rod",
    "blue": "dodger_blue",
    "green": "green",
    "red": "red",
    "grey": "#808080",
    "gray": "#808080",
    "black": "#000000",
    "white": "#ffffff",
}


def _colourise_twitch_command_log(message: str) -> str:
    if os.getenv("NO_COLOR"):
        return message
    return f"{ANSI_TWITCH_PURPLE}{message}{ANSI_RESET}"


# Twitch command whitelist - only simple, short commands
TWITCH_ALLOWED_COMMANDS = {
    # Core commands
    "babyllm", "bby", "b", "ai", "bbylim", "babylim", "bbyskunk",

    # Opt-in/out
    "bbyoptin", "aioptin", "bbyoptout", "aioptout", "bbyoptcheck", "aioptcheck",

    # Simple social commands
    "bbyhug", "bbyfite", "bbygift", "bbyfeed", "bbytip",
    "bbybook_sign", "bbysig", "bsig", "bbysign", "bsign",

    # User info
    "bbybag", "bbynick", "bby", "bbydictionary", "bbywords", "bwords",
    "bbytimer", "bbysetzone",
    "bbybby", "bbyscore", "bbylove", "bbby",
    "bbyfaves",

    # Facts (short responses)
    "bbyteach", "bbyforget",

    # Help and status
    "bbyhelp", "bbywiki", "bbystatus", "bbythought",

    # Fun short commands
    "bbyrant", "bbyshoutout", "bbycolour", "bbycolor",
    "bbytranslate", "btranslate",
    "bbyship", "bship", "bcouple", "bbycouple",

    # Smink / timing bonuses
    "bbysminks", "bbysmink", "sminks", "bsmink", "bbycheers",
    "bbysminkboard", "sminkboard", "bsminkboard",

    # Save
    "bbysave",

    # NOTE: bbyjoin/bbyleave aliases are registered dynamically in _register_commands()
    # and handled in TwitchAdapter for Twitch-specific authorization logic.
}

# Commands NOT allowed on Twitch (too complex/long responses)
TWITCH_BLOCKED_COMMANDS = {
    "bbycraft",      # Complex crafting system
    "bbybook",       # Can be very long
    "bbyleaderboard", # Long output
    "bbyinventory",  # Can be very long
    "bbytokens",     # Technical, long
    "bbysentiment",  # Technical, long
    "bbyreact",      # Relies on Discord reactions
    "bbydeclarewar", # Discord-only spam/game behaviour
    "bbywtf",        # Discord-only word-definition flow
    "bbyspace",      # Discord-only profile page feature
    "bbyfriends",    # Discord-only list output
    "bbyrivals",     # Discord-only list output
    "bbytrain",      # Admin/queue workflow
    "babytrain",     # Admin alias
    "bbyschool",     # Admin alias
    "bbyqueue",      # Admin queue inspection
    "bqueue",        # Admin alias
}

# Twitch management commands that are handled directly in the adapter.
TWITCH_MANAGEMENT_COMMANDS = {
    "bbyjoin": "join",
    "join": "join",
    "bbyleave": "leave",
    "bbyfuckoff": "leave",
    "bbygtfo": "leave",
    "gtfo": "leave",
}


class TwitchBot(twitch_commands.Bot):
    """Internal Twitch bot that connects to the main BABYBOT_DISCORD"""

    def __init__(self, adapter, channels):
        # Support both string and list
        if isinstance(channels, str):
            channels = [channels]

        # CRITICAL: Validate bot_id is numeric (common gotcha!)
        print(f"[TwitchBot] Validating bot_id: {SECRETtwitchBotIdSECRET}")
        try:
            bot_id_int = int(SECRETtwitchBotIdSECRET)
            bot_id_str = str(bot_id_int)
            print(f"[TwitchBot] ✅ bot_id is numeric: {bot_id_int}")
        except (ValueError, TypeError):
            print(f"[TwitchBot] ❌ CRITICAL: bot_id must be numeric user ID, not username!")
            print(f"[TwitchBot] Get it from: https://api.twitch.tv/helix/users?login=babyllm")
            raise ValueError(f"bot_id must be numeric, got: {SECRETtwitchBotIdSECRET}")

        # Build kwargs for TwitchIO v3+ initialization
        runtime_login_token = adapter.get_runtime_login_token()
        init_kwargs = {
            'token': runtime_login_token,
            'prefix': '!',
            'initial_channels': channels,
            'client_id': SECRETtwitchClientIdSECRET,
            'client_secret': SECRETtwitchClientSecretSECRET,
            'bot_id': bot_id_str,  # Keep string for TwitchIO/EventSub condition payloads.
        }

        super().__init__(**init_kwargs)
        self.platform_adapter = adapter
        self.baby_name = self.platform_adapter.bot.babyName
        self.joined_channels = channels  # Don't override self.channels - TwitchIO owns it!

    @staticmethod
    def _is_broadcaster_user(user_obj) -> bool:
        return bool(
            getattr(user_obj, "broadcaster", False)
            or getattr(user_obj, "is_broadcaster", False)
        )

    @staticmethod
    def _is_moderator_user(user_obj) -> bool:
        return bool(
            getattr(user_obj, "moderator", False)
            or getattr(user_obj, "is_mod", False)
            or TwitchBot._is_broadcaster_user(user_obj)
        )

    async def setup_hook(self):
        """TwitchIO v3 requires explicit EventSub subscriptions for chat events."""
        await super().setup_hook()
        token = self.platform_adapter.get_runtime_access_token()
        refresh = self.platform_adapter.get_runtime_refresh_token()

        if not token:
            print("[TwitchBot] WARNING: No Twitch user access token available for EventSub auth.")
            await self._subscribe_initial_channels()
            return

        try:
            # Register the bot user token so `subscribe_websocket(..., as_bot=True)` can resolve token_for=bot_id.
            await self.add_token(token, refresh or "")
        except Exception as e:
            print(f"[TwitchBot] WARNING: Could not add Twitch user token to token store: {e}")
            refreshed_ok, refresh_note = await self.platform_adapter.refresh_runtime_token_if_possible(reason_hint=str(e))
            if refreshed_ok:
                token = self.platform_adapter.get_runtime_access_token()
                refresh = self.platform_adapter.get_runtime_refresh_token()
                try:
                    await self.add_token(token, refresh or "")
                    print("[TwitchBot] Refreshed Twitch token and registered it successfully.")
                except Exception as second_error:
                    print(f"[TwitchBot] WARNING: Refreshed token registration failed: {second_error}")
            try:
                validated = await self._http.validate_token(token)
                token_user_id = getattr(validated, "user_id", None)
                if token_user_id:
                    self._http._tokens[token_user_id] = {
                        "user_id": token_user_id,
                        "token": token,
                        "refresh": refresh or "",
                        "last_validated": datetime.now().isoformat(),
                    }
                    print(f"[TwitchBot] Registered fallback token for user_id={token_user_id} without refresh token.")
            except Exception as fallback_error:
                print(f"[TwitchBot] WARNING: Could not register fallback Twitch token: {fallback_error}")

        await self._subscribe_initial_channels()

    async def _subscribe_initial_channels(self):
        if not self.joined_channels:
            return

        normalized = [ch.strip().lstrip("#").lower() for ch in self.joined_channels if ch and ch.strip()]
        if not normalized:
            return

        for channel_login in normalized:
            ok, info = await self.platform_adapter.subscribe_channel_live(channel_login, startup=True)
            if not ok:
                print(f"[TwitchBot] WARNING: Startup subscription failed for #{channel_login}: {info}")

    async def event_ready(self):
        """Called when Twitch bot is ready"""
        # TwitchIO v3+ uses self.user.name instead of self.nick
        bot_name = self.user.name if (hasattr(self, 'user') and self.user) else 'babyllm'
        bot_id = (self.user.id if (hasattr(self, "user") and self.user and hasattr(self.user, "id")) else "unknown")
        print(f'[TwitchBot] Logged in as {bot_name} (ID: {bot_id})')
        print(f'[TwitchBot] TwitchIO v3+ ready! Channels: {self.joined_channels}')
        print(f'[TwitchBot] Bot should now be visible in chat and able to receive messages')

    async def event_raid(self, data):
        """Called when channel gets raided (TwitchIO v3)."""
        raider_name = (
            getattr(getattr(data, "from_broadcaster", None), "name", None)
            or getattr(data, "raider_name", None)
            or "unknown_raider"
        )
        viewer_count = int(getattr(data, "viewer_count", getattr(data, "raider_viewer_count", 0)) or 0)
        channel_name = (
            getattr(getattr(data, "to_broadcaster", None), "name", None)
            or getattr(data, "channel_name", None)
            or "unknown_channel"
        )

        print(f'[TwitchBot] RAID! {raider_name} brought {viewer_count} viewers to {channel_name}!')

        # Generate AI shoutout for the raider
        try:
            from ..context import create_platform_command_context

            # Craft prompt for baby to respond to the raid
            raid_prompt = f"say something nice to thank {raider_name} for raiding with {viewer_count} viewers!"

            async def raid_reply_sink(content="", embed=None, **kwargs):
                if embed is not None:
                    text = str(getattr(embed, "description", "") or getattr(embed, "title", "") or "").strip()
                    if text:
                        await self.platform_adapter.send_message(channel_name, text[:499])
                elif content:
                    await self.platform_adapter.send_message(channel_name, str(content)[:499])

            # Create platform context for AI generation
            fake_ctx = create_platform_command_context(
                bot=self.platform_adapter.bot,
                platform="twitch",
                author_id=raider_name.lower(),
                author_name=raider_name,
                channel_id=channel_name,
                message_content=raid_prompt,
                command_name="babyllm",
                reply_sink=raid_reply_sink,
                send_sink=raid_reply_sink,
                is_mod=False,
            )

            # Generate personalized AI response
            if hasattr(self.platform_adapter.bot, 'cog'):
                print(f"[TwitchBot] Generating AI shoutout for {raider_name}...")
                await self.platform_adapter.bot.cog.babyllm_command(fake_ctx)
            else:
                print(f"[TwitchBot] Warning: No cog available for raid response")

        except Exception as e:
            print(f"[TwitchBot] Error generating raid shoutout: {e}")
            # AI generation failed - raid notification still sent to Discord below

        # Notify Discord
        await self.platform_adapter.notify_raid(raider_name, viewer_count, channel_name)

        # Celebrate on web!
        if hasattr(self.platform_adapter.bot, '_web_jumping'):
            from ..web_effects import trigger_web_animation
            trigger_web_animation(self.platform_adapter.bot, 'celebrate', duration=10.0)

    async def event_channel_raid(self, data):
        """Backward-compatible alias."""
        await self.event_raid(data)

    async def event_message(self, message):
        """Handle incoming Twitch messages"""
        # CRITICAL DEBUG: Confirm this fires
        print("[Twitch] EVENT_MESSAGE FIRED")

        author_obj = getattr(message, "author", None) or getattr(message, "chatter", None)
        if author_obj is None:
            return

        if str(getattr(author_obj, "id", "")) == str(self.bot_id):
            return

        if getattr(message, "source_broadcaster", None) is not None:
            return

        tags = getattr(message, "tags", {}) or {}
        content = (getattr(message, "content", None) or getattr(message, "text", None) or "").strip()
        channel_obj = getattr(message, "channel", None)
        self.platform_adapter.remember_live_channel(channel_obj)

        author_name = (
            getattr(author_obj, "name", None)
            or getattr(author_obj, "login", None)
            or str(getattr(author_obj, "id", "unknown"))
        )
        author_id = author_name.lower()
        if hasattr(self.platform_adapter.bot, "normalise_user_identity"):
            author_id = self.platform_adapter.bot.normalise_user_identity(author_id)
        author_display = (
            getattr(author_obj, "display_name", None)
            or getattr(author_obj, "name", None)
            or author_name
        )
        channel_name = (
            getattr(channel_obj, "name", None)
            or getattr(getattr(message, "broadcaster", None), "name", None)
            or getattr(getattr(message, "broadcaster", None), "login", None)
            or "unknown"
        )

        # Debug: Log all incoming messages
        print(f"[Twitch] Message from {author_id}: {content}")

        # Extra guard: ignore any echo/self-identity messages even if Twitch ID checks miss.
        if hasattr(self.platform_adapter.bot, "is_bot_identity") and self.platform_adapter.bot.is_bot_identity(author_id):
            return

        # Track chat intensity for web emotions (OK for all messages - just counting)
        self.platform_adapter.track_message(content)

        # Check if message is a command or mention
        is_command = content.strip().startswith('!')
        # TwitchIO v3+ uses self.user.name instead of self.nick
        bot_name = self.user.name.lower() if hasattr(self, 'user') and self.user else 'babyllm'
        is_mention = '@babyllm' in content.lower() or bot_name in content.lower()

        # Translate game guesses should work in Twitch chat even for non-opted users.
        # We capture guess text for active sessions, but this does not add chat to training data.
        if not is_command and content:
            try:
                candidates = [
                    s for s in self.platform_adapter.bot.lex_sessions.values()
                    if s.get('mode') == 'translate' and str(s.get('channel_id')) == str(channel_name)
                ]
                if candidates:
                    latest = max(candidates, key=lambda s: s.get('created_at', 0.0))
                    extra = latest.setdefault('extra', {})
                    guesses = extra.setdefault('guesses', {})
                    if author_id not in guesses:
                        guesses[author_id] = {
                            'guess': content.strip().lower(),
                            'timestamp': time.time(),
                        }
            except Exception as guess_error:
                print(f"[Twitch] translate-guess capture error: {guess_error}")

        # PRIVACY: Only process messages that are commands/mentions AND user is opted in
        # OR just commands (commands are always handled, but buffer/training needs opt-in)

        # Update basic user memory (display name, last seen) - this is OK for everyone
        mem = self.platform_adapter.bot.userMemory.get(author_id, {})
        if author_id not in self.platform_adapter.bot.userMemory:
            # Initialize new user with opt_in = False
            from collections import defaultdict
            self.platform_adapter.bot.userMemory[author_id] = {
                "nickname": None,
                "display_name": author_display,
                "opt_in": False,  # CRITICAL: Default to NOT opted in
                "BBY": 0.0,
                "last_seen": time.time(),
            }
        else:
            mem['display_name'] = author_display
            mem['colour'] = tags.get("color", mem.get('colour', "#007bff"))
            mem['last_seen'] = time.time()

        # Check if user is opted in
        is_opted_in = author_id in self.platform_adapter.bot.AIoptInUsers

        # PRIVACY POLICY FOR TWITCH:
        # - Commands: ALWAYS RECORDED (opted in or not)
        # - @mentions from non-opted: "gotta opt in first!" response (NOT recorded)
        # - @mentions from opted: RESPONDS + RECORDED
        # - Regular chat from opted: RECORDED (no response)
        # - Regular chat from non-opted: IGNORED (not recorded)

        should_record = False
        should_generate_response = False

        if is_command:
            # Commands are ALWAYS recorded (even non-opted users)
            # Commands handled by TwitchIO decorators, so don't generate additional response
            should_record = True
            should_generate_response = False
        elif is_mention and not is_opted_in:
            # Non-opted user @mention: Tell them to opt in (NOT recorded)
            await self.platform_adapter.send_message(channel_name, f"@{author_id} hey! gotta opt in first with !bbyoptin if you want me to chat! commands still work tho ʕ·ᴥ·ʔ")
            return
        elif is_mention and is_opted_in:
            # Opted user @mention (not a command): RECORD + RESPOND
            should_record = True
            should_generate_response = True
        elif is_opted_in:
            # Regular chat from opted user: RECORDED (no response)
            should_record = True
            should_generate_response = False
        else:
            # Regular chat from non-opted user: IGNORED
            return

        if should_record:
            # Create platform message
            platform_msg = PlatformMessage(
                content=content,
                author_id=author_id,
                author_display_name=author_display,
                channel_id=channel_name,
                platform="twitch",
                timestamp=message.timestamp.timestamp() if hasattr(message, 'timestamp') else time.time(),
                raw_message=message,
                author_colour=tags.get("color"),
                is_bot=False,
                is_mod=self._is_moderator_user(author_obj),
            )

            # Add to buffer and training queue
            await self.platform_adapter.handle_message(platform_msg)

        # Generate conversational response if needed (for @mentions, not commands)
        if should_generate_response:
            print(f"[Twitch] Generating response for @mention from {author_id}")
            # Import platform context creator
            from ..context import create_platform_command_context

            # Create context for response generation
            reply_to_id = str(getattr(message, "id", None) or getattr(message, "message_id", None) or "")

            async def mention_reply_sink(content="", embed=None, **kwargs):
                if embed is not None:
                    text = str(getattr(embed, "description", "") or getattr(embed, "title", "") or "").strip()
                    if text:
                        if reply_to_id:
                            await self.platform_adapter._send_to_channel(
                                channel_name,
                                text[:499],
                                reply_to_message_id=reply_to_id,
                            )
                        else:
                            await self.platform_adapter.send_message(channel_name, text[:499])
                elif content:
                    if reply_to_id:
                        await self.platform_adapter._send_to_channel(
                            channel_name,
                            str(content)[:499],
                            reply_to_message_id=reply_to_id,
                        )
                    else:
                        await self.platform_adapter.send_message(channel_name, str(content)[:499])

            fake_ctx = create_platform_command_context(
                bot=self.platform_adapter.bot,
                platform="twitch",
                author_id=author_id,
                author_name=author_display,
                channel_id=channel_name,
                message_content=content,
                command_name="babyllm",
                message_id=(reply_to_id or None),
                reply_sink=mention_reply_sink,
                send_sink=mention_reply_sink,
                is_mod=self._is_moderator_user(author_obj),
            )

            # Generate response
            if hasattr(self.platform_adapter.bot, 'cog'):
                try:
                    # Use babyllm command to generate response
                    await self.platform_adapter.bot.cog.babyllm_command(fake_ctx)
                except Exception as e:
                    print(f"[Twitch] Error generating @mention response: {e}")
                    reply_to_id = str(getattr(message, "id", None) or getattr(message, "message_id", None) or "")
                    if reply_to_id:
                        await self.platform_adapter._send_to_channel(
                            channel_name,
                            f"@{author_id} (oops, my brain glitched!)",
                            reply_to_message_id=reply_to_id,
                        )
                    else:
                        await self.platform_adapter.send_message(channel_name, f"@{author_id} (oops, my brain glitched!)")

        # 🚨 REQUIRED for TwitchIO v3: Commands do NOT auto-dispatch without this!
        await self.process_commands(message)

    async def event_command_error(self, payload):
        """Handle command errors (TwitchIO v3 CommandErrorPayload)."""
        try:
            ctx = getattr(payload, "context", None)
            exc = getattr(payload, "exception", payload)
            cmd = getattr(ctx, "command", None) if ctx else None
            cmd_name = getattr(cmd, "name", str(cmd)) if cmd else "unknown"
            print(_colourise_twitch_command_log(f"[TwitchBot] Command error in '{cmd_name}': {exc}"))
        except Exception as e:
            print(_colourise_twitch_command_log(f"[TwitchBot] Command error handler failed: {e}"))

    @twitch_commands.command(name='bbyjoin')
    async def cmd_bbyjoin(self, ctx):
        """Allow streamers to add their channel"""
        user_name = ctx.author.name.lower()
        # Join command always targets the requester's own channel login.
        target_channel = user_name
        success, message = await self.platform_adapter.authorize_channel(target_channel, user_name)
        reply_method = getattr(ctx, "reply", None)
        text = to_british_english(f"@{user_name} {message}")
        if callable(reply_method):
            await reply_method(text)
        else:
            await ctx.send(text)

    @twitch_commands.command(name='bbyleave')
    async def cmd_bbyleave(self, ctx):
        """Allow streamers to remove their channel"""
        user_name = ctx.author.name.lower()
        # Leave command always targets the requester's own channel login.
        target_channel = user_name
        success, message = await self.platform_adapter.deauthorize_channel(target_channel)
        reply_method = getattr(ctx, "reply", None)
        text = to_british_english(f"@{user_name} {message}")
        if callable(reply_method):
            await reply_method(text)
        else:
            await ctx.send(text)

    @twitch_commands.command(name='bbyfuckoff')
    async def cmd_bbyfuckoff(self, ctx):
        """Rude but effective way to make baby leave"""
        # Just call bbyleave
        await self.cmd_bbyleave(ctx)

    @twitch_commands.command(name='bbygtfo')
    async def cmd_bbygtfo(self, ctx):
        """Another rude way to make baby leave"""
        # Just call bbyleave
        await self.cmd_bbyleave(ctx)

    # NOTE: TwitchIO v3 does not auto-register these decorators on Bot subclasses.
    # Command registration is centralized in TwitchAdapter._register_commands().


class TwitchAdapter(PlatformAdapter):
    """Twitch platform adapter"""

    def __init__(self, bot_instance, channels=None):
        super().__init__(bot_instance)
        self.platform_name = "twitch"

        # Support both single channel and list
        if channels is None:
            channels = twitch_channel
        if isinstance(channels, str):
            channels = [channels]

        # Load authorised channels from file and merge
        authorized = self.load_authorized_channels()
        authorized_names = [ch['name'] for ch in authorized]

        # Merge config channels with authorised channels (deduplicate)
        all_channels = list(set(channels + authorized_names))

        self.channels = all_channels
        self.channel = all_channels[0] if all_channels else "childofanandroid"  # Legacy compatibility

        print(f"[TwitchAdapter] Loaded {len(all_channels)} channels ({len(authorized_names)} authorised, {len(channels)} from config)")
        self.twitch_bot = None
        self.allowed_commands = set(TWITCH_ALLOWED_COMMANDS)
        self.blocked_commands = set(TWITCH_BLOCKED_COMMANDS)

        # Message queue for Twitch responses (rate limiting)
        self.message_queue = asyncio.Queue()
        self.queue_worker = None
        self.message_delay = 0.5  # seconds between messages (Twitch rate limit)
        self._channel_sender_cache = {}
        self._live_channel_cache = {}

        # Chat intensity tracking for web emotions
        self.recent_messages = []  # Timestamps of recent messages
        self.recent_hugs = []  # Track !bbyhug usage
        self.intensity_update_task = None

        # Runtime auth (supports refresh-token rotation and persistence)
        self._runtime_access_token = self._normalise_access_token(SECRETtwitchTokenSECRET)
        self._runtime_refresh_token = (SECRETtwitchRefreshTokenSECRET or "").strip()
        self._runtime_scopes = []
        self._runtime_token_expires_at = 0.0
        self._load_runtime_auth_state()

    @staticmethod
    def _normalise_access_token(token: str) -> str:
        value = (token or "").strip()
        if value.startswith("oauth:"):
            value = value.split(":", 1)[1].strip()
        return value

    def _get_auth_state_file(self):
        return os.path.join(os.path.dirname(__file__), '..', 'twitch_auth.json')

    def _load_runtime_auth_state(self):
        try:
            path = self._get_auth_state_file()
            if not os.path.exists(path):
                return
            with open(path, 'r') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            stored_access = self._normalise_access_token(data.get("access_token", ""))
            stored_refresh = (data.get("refresh_token", "") or "").strip()
            if stored_access:
                self._runtime_access_token = stored_access
            if stored_refresh:
                self._runtime_refresh_token = stored_refresh
            self._runtime_scopes = list(data.get("scopes", []) or [])
            self._runtime_token_expires_at = float(data.get("expires_at", 0.0) or 0.0)
        except Exception as e:
            print(f"[TwitchAdapter] Warning: couldn't load twitch auth cache: {e}")

    def _save_runtime_auth_state(self):
        try:
            payload = {
                "access_token": self._runtime_access_token,
                "refresh_token": self._runtime_refresh_token,
                "scopes": list(self._runtime_scopes or []),
                "expires_at": float(self._runtime_token_expires_at or 0.0),
                "updated_at": datetime.now().isoformat(),
            }
            path = self._get_auth_state_file()
            with open(path, 'w') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"[TwitchAdapter] Warning: couldn't save twitch auth cache: {e}")

    def get_runtime_access_token(self) -> str:
        return self._runtime_access_token

    def get_runtime_refresh_token(self) -> str:
        return self._runtime_refresh_token

    def get_runtime_login_token(self) -> str:
        token = self._runtime_access_token or self._normalise_access_token(SECRETtwitchTokenSECRET)
        if not token:
            return ""
        return f"oauth:{token}"

    @staticmethod
    def _normalise_twitch_chat_colour(colour_value: str):
        """Normalise user colour input to a Twitch API colour value."""
        raw = (colour_value or "").strip().lower()
        if not raw:
            return "", "missing colour value"

        # Accept RGB triplets e.g. "255 122 255" or "255,122,255".
        rgb_match = re.fullmatch(r"\s*(\d{1,3})\s*[, ]\s*(\d{1,3})\s*[, ]\s*(\d{1,3})\s*", raw)
        if rgb_match:
            r, g, b = [int(rgb_match.group(i)) for i in (1, 2, 3)]
            if any(v < 0 or v > 255 for v in (r, g, b)):
                return "", "rgb values must be between 0 and 255"
            return f"#{r:02x}{g:02x}{b:02x}", ""

        # Accept hex values with or without leading '#'.
        if re.fullmatch(r"#?[0-9a-f]{6}", raw):
            hex_value = raw if raw.startswith("#") else f"#{raw}"
            return hex_value, ""

        key = raw.replace("-", "_").replace(" ", "_")
        mapped = TWITCH_CHAT_COLOUR_ALIASES.get(key, key)
        if mapped in TWITCH_NAMED_CHAT_COLOURS:
            return mapped, ""
        if re.fullmatch(r"#?[0-9a-f]{6}", mapped):
            hex_value = mapped if mapped.startswith("#") else f"#{mapped}"
            return hex_value, ""

        return "", (
            "unsupported colour. use a twitch named colour "
            "(e.g. hot_pink) or a hex/rgb value like #ff7aff or 255 122 255"
        )

    async def set_bot_chat_colour(self, colour_value: str):
        """Set the bot account's Twitch chat colour via Helix API."""
        if not SECRETtwitchClientIdSECRET:
            return False, "missing twitch client id"

        bot_user_id = str(
            getattr(getattr(self, "twitch_bot", None), "bot_id", None)
            or SECRETtwitchBotIdSECRET
            or ""
        ).strip()
        if not bot_user_id:
            return False, "missing twitch bot user id"

        normalised_colour, colour_error = self._normalise_twitch_chat_colour(colour_value)
        if colour_error:
            return False, colour_error

        async def _attempt(token: str):
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {
                "Authorization": f"Bearer {token}",
                "Client-Id": SECRETtwitchClientIdSECRET,
            }
            params = {"user_id": bot_user_id, "color": normalised_colour}
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.put("https://api.twitch.tv/helix/chat/color", headers=headers, params=params) as resp:
                    body = await resp.text()
                    return resp.status, body

        token = (self._runtime_access_token or "").strip()
        if not token:
            refreshed, _ = await self.refresh_runtime_token_if_possible(reason_hint="chat colour set")
            token = (self._runtime_access_token or "").strip()
            if not refreshed or not token:
                return False, "no valid twitch user token available"

        status, body = await _attempt(token)
        if status == 401:
            refreshed, note = await self.refresh_runtime_token_if_possible(reason_hint="chat colour 401")
            if not refreshed:
                return False, f"token refresh failed: {note}"
            token = (self._runtime_access_token or "").strip()
            if not token:
                return False, "token refresh succeeded but access token is empty"
            status, body = await _attempt(token)

        if status in (200, 204):
            canonical_key = (
                self.bot.get_bot_identity_key()
                if hasattr(self.bot, "get_bot_identity_key")
                else "babyllm"
            )
            mem = self.bot.userMemory.setdefault(canonical_key, {})
            mem["colour"] = normalised_colour
            mem["color"] = normalised_colour
            mem["last_colour_update"] = time.time()
            print(f"[TwitchAdapter] Updated bot chat colour to {normalised_colour}")
            return True, f"updated to {normalised_colour}"

        body_excerpt = (body or "").strip().replace("\n", " ")[:180]
        if status == 400:
            return False, f"twitch rejected the colour ({body_excerpt or 'bad request'})"
        if status == 401:
            return False, "twitch auth failed; token likely missing user:manage:chat_color"
        if status == 403:
            return False, "twitch denied colour update; ensure token has user:manage:chat_color"
        return False, f"twitch colour update failed ({status}): {body_excerpt or 'unknown error'}"

    async def _validate_runtime_access_token(self):
        token = self._runtime_access_token
        if not token:
            return False, None

        try:
            timeout = aiohttp.ClientTimeout(total=8)
            headers = {"Authorization": f"OAuth {token}"}
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get("https://id.twitch.tv/oauth2/validate", headers=headers) as resp:
                    if resp.status != 200:
                        return False, None
                    data = await resp.json()
                    return True, data
        except Exception:
            return False, None

    async def _refresh_runtime_access_token(self, refresh_token: str):
        refresh_token = (refresh_token or "").strip()
        if not refresh_token:
            return False, "missing refresh token"
        if not SECRETtwitchClientIdSECRET or not SECRETtwitchClientSecretSECRET:
            return False, "missing twitch client id/secret"

        try:
            timeout = aiohttp.ClientTimeout(total=12)
            form = {
                "client_id": SECRETtwitchClientIdSECRET,
                "client_secret": SECRETtwitchClientSecretSECRET,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post("https://id.twitch.tv/oauth2/token", data=form) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status != 200:
                        message = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                        return False, f"refresh failed ({resp.status}): {message}"

            access_token = self._normalise_access_token(data.get("access_token", ""))
            new_refresh = (data.get("refresh_token", refresh_token) or "").strip()
            expires_in = int(data.get("expires_in", 0) or 0)
            scopes = list(data.get("scope", []) or [])

            if not access_token:
                return False, "refresh response missing access_token"

            self._runtime_access_token = access_token
            self._runtime_refresh_token = new_refresh
            self._runtime_scopes = scopes
            self._runtime_token_expires_at = time.time() + max(0, expires_in - 60)
            self._save_runtime_auth_state()
            return True, "token refreshed"
        except Exception as e:
            return False, f"refresh exception: {e}"

    async def refresh_runtime_token_if_possible(self, reason_hint: str = ""):
        refresh = self._runtime_refresh_token or (SECRETtwitchRefreshTokenSECRET or "").strip()
        if not refresh:
            return False, "no refresh token configured"
        ok, note = await self._refresh_runtime_access_token(refresh)
        if not ok and reason_hint:
            print(f"[TwitchAdapter] Refresh skipped after '{reason_hint}': {note}")
        return ok, note

    async def _ensure_valid_runtime_token(self):
        valid, _ = await self._validate_runtime_access_token()
        if valid:
            return True

        refreshed, note = await self.refresh_runtime_token_if_possible(reason_hint="startup validation")
        if refreshed:
            print("[TwitchAdapter] Refreshed Twitch user token from refresh token.")
            return True

        if note:
            print(f"[TwitchAdapter] WARNING: Could not refresh Twitch token: {note}")
        return False

    async def start(self):
        """Start the Twitch bot"""
        if self.twitch_bot is not None:
            print("[TwitchAdapter] Twitch bot already running")
            return

        await self._ensure_valid_runtime_token()
        await self._warn_if_twitch_credentials_mismatch()
        print(f"[TwitchAdapter] Starting Twitch bot for channels: {self.channels}")
        self.twitch_bot = TwitchBot(self, self.channels)
        await self._sync_linked_users_from_storage()

        # Register commands AFTER bot creation, BEFORE connection starts
        self._register_commands(self.twitch_bot)

        # Start message queue worker
        self.queue_worker = asyncio.create_task(self._message_queue_worker())

        # Start intensity tracker for web emotions
        self.intensity_update_task = asyncio.create_task(self._update_web_intensity())

        # Start Twitch bot (non-blocking)
        asyncio.create_task(self._run_twitch_bot())

    async def _run_twitch_bot(self):
        """Run Twitch bot in background"""
        try:
            print(f"[TwitchAdapter] Starting TwitchIO v3+ bot...")
            await self.twitch_bot.start()
        except Exception as e:
            import traceback
            print(f"[TwitchAdapter] CRITICAL ERROR running Twitch bot: {e}")
            if "address already in use" in str(e).lower() and "4343" in str(e):
                print("[TwitchAdapter] NOTE: TwitchIO adapter port 4343 is busy. Another bot process is probably running.")
            print(f"[TwitchAdapter] Full traceback:")
            traceback.print_exc()
            print(f"[TwitchAdapter] Bot will not function on Twitch!")

    async def stop(self):
        """Stop Twitch bot"""
        if self.twitch_bot:
            await self.twitch_bot.close()
            self.twitch_bot = None
        if self.queue_worker:
            self.queue_worker.cancel()
        if self.intensity_update_task:
            self.intensity_update_task.cancel()

    async def _warn_if_twitch_credentials_mismatch(self):
        """Best-effort validation to catch common Twitch auth mismatches early."""
        try:
            token = (self._runtime_access_token or "").strip()

            if not token:
                print("[TwitchAdapter] WARNING: Twitch token appears empty.")
                return

            timeout = aiohttp.ClientTimeout(total=8)
            headers = {"Authorization": f"OAuth {token}"}
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get("https://id.twitch.tv/oauth2/validate", headers=headers) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        print(f"[TwitchAdapter] WARNING: Twitch token validation failed ({resp.status}): {body[:120]}")
                        return
                    data = await resp.json()

            token_client_id = data.get("client_id")
            token_user_id = data.get("user_id")
            scopes = set(data.get("scopes") or [])

            if SECRETtwitchClientIdSECRET and token_client_id and token_client_id != SECRETtwitchClientIdSECRET:
                print("[TwitchAdapter] WARNING: SECRETtwitchClientIdSECRET does not match the OAuth token's client_id.")
                print("[TwitchAdapter] This can cause Twitch API auth failures and no chat events.")

            if SECRETtwitchBotIdSECRET and token_user_id and str(token_user_id) != str(SECRETtwitchBotIdSECRET):
                print("[TwitchAdapter] WARNING: SECRETtwitchBotIdSECRET does not match the OAuth token user_id.")
                print("[TwitchAdapter] This can cause the bot to appear connected but miss chat events.")

            missing_scopes = {"chat:read", "chat:edit"} - scopes
            if missing_scopes:
                print(f"[TwitchAdapter] WARNING: Twitch token is missing scopes: {sorted(missing_scopes)}")

            if "user:read:chat" not in scopes:
                print("[TwitchAdapter] WARNING: Twitch token is missing user:read:chat.")
                print("[TwitchAdapter] Incoming chat events (EVENT_MESSAGE) will fail without this scope.")

            if "user:write:chat" not in scopes:
                print("[TwitchAdapter] WARNING: Twitch token is missing user:write:chat.")
                print("[TwitchAdapter] Outbound API chat sends may fail on TwitchIO v3.")

            if "user:bot" not in scopes:
                print("[TwitchAdapter] WARNING: Twitch token is missing user:bot.")
                print("[TwitchAdapter] EventSub chat subscriptions can fail without user:bot.")

            if "user:manage:chat_color" not in scopes:
                print("[TwitchAdapter] NOTE: token lacks user:manage:chat_color.")
                print("[TwitchAdapter] !bbycolour can still update web/discord, but Twitch chat colour updates will fail.")

            if "channel:bot" not in scopes:
                print("[TwitchAdapter] NOTE: token lacks channel:bot.")
                print("[TwitchAdapter] For channels where babyllm is not broadcaster/moderator, broadcasters must authorise channel:bot for reliable chat events.")
        except Exception as e:
            # Never block startup on validation errors.
            print(f"[TwitchAdapter] Credential validation skipped: {e}")

    def track_message(self, content: str):
        """Track message for chat intensity calculation"""
        import time
        now = time.time()

        # Add to recent messages
        self.recent_messages.append(now)

        # Track !bbyhug commands specifically
        if 'bbyhug' in content.lower():
            self.recent_hugs.append(now)

        # Keep only last 60 seconds
        cutoff = now - 60
        self.recent_messages = [t for t in self.recent_messages if t > cutoff]
        self.recent_hugs = [t for t in self.recent_hugs if t > cutoff]

    async def _update_web_intensity(self):
        """Periodically update web state based on Twitch chat intensity"""
        import time
        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds

                if not hasattr(self.bot, '_web_jumping'):
                    continue  # Web adapter not enabled

                messages_per_minute = len(self.recent_messages)
                hugs_per_minute = len(self.recent_hugs)

                # Update baby emotions based on chat activity
                if messages_per_minute > 30:  # Very active chat
                    self.bot._web_jumping = True
                    self.bot._web_dream_intensity = min(20, messages_per_minute / 2)
                    print(f"[Twitch→Web] High intensity: {messages_per_minute} msg/min")

                elif messages_per_minute > 15:  # Moderate chat
                    self.bot._web_dream_intensity = 12
                    self.bot._web_jumping = False

                else:  # Quiet chat
                    self.bot._web_dream_intensity = 5
                    self.bot._web_jumping = False

                # Blush if lots of hugs!
                if hugs_per_minute > 3:
                    self.bot._web_cheeks = True
                    print(f"[Twitch→Web] Many hugs: baby blushing!")
                elif hugs_per_minute == 0:
                    self.bot._web_cheeks = False

            except Exception as e:
                print(f"[TwitchAdapter] Intensity update error: {e}")

    async def notify_raid(self, raider_name: str, viewer_count: int, channel_name: str):
        """Notify Discord about Twitch raid"""
        try:
            # Get Discord channel
            if not hasattr(self.bot, 'bby_spam_channel_id'):
                print("[TwitchAdapter] No Discord channel configured for raid notifications")
                return

            discord_channel = self.bot.get_channel(self.bot.bby_spam_channel_id)
            if not discord_channel:
                print("[TwitchAdapter] Discord channel not found")
                return

            # Try to import Discord Embed
            try:
                from discord import Embed
                embed = Embed(
                    title="🎉 Twitch Raid!",
                    description=f"**{raider_name}** raided **{channel_name}** with **{viewer_count}** viewers!",
                    color=0x9146FF  # Twitch purple
                )
                embed.set_footer(text="Twitch → Discord")
                embed = normalise_embed_british_english(embed)
                await discord_channel.send(embed=embed)
                print(f"[Twitch→Discord] Raid notification sent!")

            except ImportError:
                # Fallback to plain text
                message = to_british_english(
                    f"🎉 **Twitch Raid!** {raider_name} brought {viewer_count} viewers to {channel_name}!"
                )
                await discord_channel.send(message)

        except Exception as e:
            print(f"[TwitchAdapter] Error notifying Discord about raid: {e}")

    def get_channels_file(self):
        """Get path to authorised channels file"""
        import os
        return os.path.join(os.path.dirname(__file__), '..', 'twitch_channels.json')

    def load_authorized_channels(self):
        """Load authorised channels from JSON file"""
        try:
            file_path = self.get_channels_file()
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    return data.get('authorized_channels', [])
            return []
        except Exception as e:
            print(f"[TwitchAdapter] Error loading channels: {e}")
            return []

    def save_authorized_channels(self, channels):
        """Save authorised channels to JSON file"""
        try:
            file_path = self.get_channels_file()
            with open(file_path, 'w') as f:
                json.dump({'authorized_channels': channels}, f, indent=2)
            print(f"[TwitchAdapter] Saved {len(channels)} authorised channels")
            return True
        except Exception as e:
            print(f"[TwitchAdapter] Error saving channels: {e}")
            return False

    def _normalise_link_owner(self, owner: str) -> str:
        return (owner or "").strip().lower()

    async def _notify_link_debug(self, *, action: str, channel_name: str, owner: str = "", status: str = "ok", note: str = ""):
        if not hasattr(self.bot, "_discord_debug"):
            return

        owner_name = self._normalise_link_owner(owner) or "unknown"
        message = (
            f"[TWITCH_LINK] action={action} channel=#{self._normalize_channel_name(channel_name)} "
            f"owner={owner_name} status={status}"
        )
        if note:
            message += f" note={note}"

        try:
            await self.bot._discord_debug(to_british_english(message))
        except Exception as e:
            print(f"[TwitchAdapter] Link debug post failed: {e}")

    async def _set_owner_link_state(self, owner: str, channel_name: str, linked: bool):
        owner_key = self._normalise_link_owner(owner)
        channel_login = self._normalize_channel_name(channel_name)
        if not owner_key or not channel_login:
            return

        memory = self.bot.userMemory[owner_key]
        if not memory.get("display_name"):
            memory["display_name"] = owner_key

        linked_channels = memory.get("linked_twitch_channels", [])
        if not isinstance(linked_channels, list):
            linked_channels = []

        normalised = []
        seen = set()
        for item in linked_channels:
            login = self._normalize_channel_name(str(item))
            if not login or login in seen:
                continue
            seen.add(login)
            normalised.append(login)

        if linked:
            if channel_login not in seen:
                normalised.append(channel_login)
        else:
            normalised = [login for login in normalised if login != channel_login]

        memory["linked_twitch_channels"] = normalised
        memory["linked_channel_count"] = len(normalised)
        memory["is_twitch_linked"] = len(normalised) > 0
        memory["last_link_update"] = time.time()

        if hasattr(self.bot, "_save_user_data"):
            try:
                await self.bot._save_user_data()
            except Exception as e:
                print(f"[TwitchAdapter] Warning: couldn't persist linked-channel state: {e}")

    async def _sync_linked_users_from_storage(self):
        """Mirror persisted Twitch channel authorisations into user memory."""
        channels = self.load_authorized_channels()
        owner_map = {}
        for entry in channels:
            channel_name = self._normalize_channel_name(entry.get("name", ""))
            owner = self._normalise_link_owner(entry.get("authorized_by", ""))
            if not owner or not channel_name:
                continue
            owner_map.setdefault(owner, set()).add(channel_name)

        changed = False
        for owner, channel_set in owner_map.items():
            memory = self.bot.userMemory[owner]
            linked_channels = sorted(channel_set)
            if memory.get("linked_twitch_channels") != linked_channels:
                memory["linked_twitch_channels"] = linked_channels
                changed = True
            if memory.get("linked_channel_count") != len(linked_channels):
                memory["linked_channel_count"] = len(linked_channels)
                changed = True
            if memory.get("is_twitch_linked") != bool(linked_channels):
                memory["is_twitch_linked"] = bool(linked_channels)
                changed = True
            if not memory.get("display_name"):
                memory["display_name"] = owner
                changed = True

        for user_id, memory in self.bot.userMemory.items():
            if memory.get("is_twitch_linked") and user_id not in owner_map:
                memory["linked_twitch_channels"] = []
                memory["linked_channel_count"] = 0
                memory["is_twitch_linked"] = False
                changed = True

        if changed and hasattr(self.bot, "_save_user_data"):
            try:
                await self.bot._save_user_data()
            except Exception as e:
                print(f"[TwitchAdapter] Warning: couldn't save synced linked users: {e}")

    async def _resolve_channel_user(self, channel_name: str):
        if not self.twitch_bot:
            return None
        login = self._normalize_channel_name(channel_name)
        if not login:
            return None
        users = await self.twitch_bot.fetch_users(logins=[login])
        if not users:
            return None
        return users[0]

    def _find_live_subscription_ids_for_broadcaster(self, broadcaster_user_id: str):
        """Find active websocket EventSub subscription IDs for one broadcaster."""
        result = {"chat": None, "raid": None}
        if not self.twitch_bot:
            return result

        bot_id = str(getattr(self.twitch_bot, "bot_id", "") or "")
        for sub_id, sub_data in self.twitch_bot.websocket_subscriptions().items():
            condition = getattr(sub_data, "condition", {}) or {}
            sub_type = getattr(sub_data, "type", None)

            if (
                sub_type == eventsub.SubscriptionType.ChannelChatMessage
                and str(condition.get("broadcaster_user_id", "")) == str(broadcaster_user_id)
                and (not bot_id or str(condition.get("user_id", "")) == bot_id)
            ):
                result["chat"] = sub_id
            elif (
                sub_type == eventsub.SubscriptionType.ChannelRaid
                and str(condition.get("to_broadcaster_user_id", "")) == str(broadcaster_user_id)
            ):
                result["raid"] = sub_id

        return result

    async def subscribe_channel_live(self, channel_name: str, startup: bool = False):
        """Create live EventSub subscriptions for a channel without restart."""
        if not self.twitch_bot:
            return False, "twitch bot is not running"

        login = self._normalize_channel_name(channel_name)
        if not login:
            return False, "invalid channel name"

        try:
            user = await self._resolve_channel_user(login)
        except Exception as e:
            return False, f"couldn't resolve channel '{login}': {e}"

        if user is None:
            return False, f"couldn't find twitch channel '{login}'"

        broadcaster_user_id = str(user.id)

        try:
            existing = self._find_live_subscription_ids_for_broadcaster(broadcaster_user_id)
            raid_error = None

            if existing["chat"] is None:
                chat_payload = eventsub.ChatMessageSubscription(
                    broadcaster_user_id=user.id,
                    user_id=self.twitch_bot.bot_id,
                )
                try:
                    await self.twitch_bot.subscribe_websocket(payload=chat_payload, as_bot=True)
                except Exception as chat_error:
                    return False, f"chat subscription failed for #{login}: {chat_error}"

            if existing["raid"] is None:
                raid_payload = eventsub.ChannelRaidSubscription(to_broadcaster_user_id=user.id)
                try:
                    await self.twitch_bot.subscribe_websocket(payload=raid_payload, as_bot=True)
                except Exception as e:
                    raid_error = str(e)

            current = self._find_live_subscription_ids_for_broadcaster(broadcaster_user_id)
            if current["chat"] is None:
                return False, f"chat subscription missing for #{login} after subscribe attempt"

            if login not in self.channels:
                self.channels.append(login)
            if self.twitch_bot and login not in self.twitch_bot.joined_channels:
                self.twitch_bot.joined_channels.append(login)

            if current["raid"] is None and raid_error:
                print(f"[TwitchAdapter] NOTE: Raid subscription disabled for #{login}: {raid_error}")

            if not startup:
                print(
                    f"[TwitchAdapter] Live chat subscription ready for #{login} "
                    f"(chat={current['chat']}, raid={current['raid']})"
                )
            return True, "live chat subscription active"
        except Exception as e:
            return False, f"failed to subscribe live for #{login}: {e}"

    async def unsubscribe_channel_live(self, channel_name: str):
        """Remove live EventSub subscriptions for a channel without restart."""
        if not self.twitch_bot:
            return False, "twitch bot is not running"

        login = self._normalize_channel_name(channel_name)
        if not login:
            return False, "invalid channel name"

        try:
            user = await self._resolve_channel_user(login)
        except Exception as e:
            return False, f"couldn't resolve channel '{login}': {e}"

        if user is None:
            return False, f"couldn't find twitch channel '{login}'"

        broadcaster_user_id = str(user.id)
        subscription_ids = self._find_live_subscription_ids_for_broadcaster(broadcaster_user_id)
        removed = []
        errors = []

        for key in ("chat", "raid"):
            sub_id = subscription_ids.get(key)
            if not sub_id:
                continue
            try:
                await self.twitch_bot.delete_websocket_subscription(sub_id, force=True)
                removed.append(key)
            except Exception as e:
                errors.append(f"{key}:{e}")

        if errors:
            return False, f"failed to remove some live subscriptions ({'; '.join(errors)})"

        if login in self.channels:
            self.channels.remove(login)
        if self.twitch_bot and login in self.twitch_bot.joined_channels:
            self.twitch_bot.joined_channels.remove(login)

        self._channel_sender_cache.pop(login, None)
        self._live_channel_cache.pop(login, None)

        if removed:
            print(f"[TwitchAdapter] Removed live subscriptions for #{login}: {removed}")
            return True, "live subscriptions removed"
        return True, "no live subscriptions were active"

    async def authorize_channel(self, channel_name: str, authorized_by: str):
        """Add channel to authorised list and join"""
        channel_name = self._normalize_channel_name(channel_name)
        authorized_by = self._normalise_link_owner(authorized_by)
        channels = self.load_authorized_channels()

        # Check if already authorised
        for ch in channels:
            if ch['name'].lower() == channel_name.lower():
                await self._notify_link_debug(
                    action="link",
                    channel_name=channel_name,
                    owner=authorized_by,
                    status="ignored",
                    note="already authorised",
                )
                return False, "Channel already authorised!"

        # Add new channel
        channels.append({
            'name': channel_name,
            'authorized_at': datetime.now().isoformat(),
            'authorized_by': authorized_by,
            'permanent': False
        })

        if self.save_authorized_channels(channels):
            # Join the channel
            if self.twitch_bot:
                try:
                    live_ok, live_msg = await self.subscribe_channel_live(channel_name)
                    if not live_ok:
                        await self._notify_link_debug(
                            action="link",
                            channel_name=channel_name,
                            owner=authorized_by,
                            status="error",
                            note=f"live join failed: {live_msg}",
                        )
                        return False, f"live join failed: {live_msg}"

                    if hasattr(self.twitch_bot, "join_channels"):
                        try:
                            await self.twitch_bot.join_channels([channel_name])
                            print(f"[TwitchAdapter] Joined channel via join_channels: {channel_name}")
                        except Exception as join_err:
                            print(f"[TwitchAdapter] join_channels warning for #{channel_name}: {join_err}")

                    join_hello = "hello! i am awake ʕっʘ‿ʘʔっ 🫂"
                    try:
                        await self._send_to_channel(channel_name, join_hello)
                    except Exception as hello_err:
                        print(f"[TwitchAdapter] hello-on-join warning for #{channel_name}: {hello_err}")

                    await self._set_owner_link_state(authorized_by, channel_name, linked=True)
                    await self._notify_link_debug(
                        action="link",
                        channel_name=channel_name,
                        owner=authorized_by,
                        status="ok",
                        note="joined live",
                    )
                    return True, f"ʕっʘ‿ʘʔっ okay! i joined your channel! (you have to add me as a mod for me to answer commands though)"
                except Exception as e:
                    print(f"[TwitchAdapter] Error joining channel: {e}")
                    await self._notify_link_debug(
                        action="link",
                        channel_name=channel_name,
                        owner=authorized_by,
                        status="error",
                        note=f"join exception: {str(e)[:120]}",
                    )
                    return False, f"Error joining channel: {str(e)[:50]}"
            await self._set_owner_link_state(authorized_by, channel_name, linked=True)
            await self._notify_link_debug(
                action="link",
                channel_name=channel_name,
                owner=authorized_by,
                status="ok",
                note="queued (bot offline)",
            )
            return True, "Channel authorised (live join will happen once twitch bot is running)"
        await self._notify_link_debug(
            action="link",
            channel_name=channel_name,
            owner=authorized_by,
            status="error",
            note="save failed",
        )
        return False, "Error saving channel authorization"

    async def deauthorize_channel(self, channel_name: str):
        """Remove channel from authorised list and leave"""
        channel_name = self._normalize_channel_name(channel_name)
        channels = self.load_authorized_channels()

        # Find and remove channel
        found = False
        removed_owner = ""
        new_channels = []
        for ch in channels:
            if ch['name'].lower() == channel_name.lower():
                if ch.get('permanent', False):
                    return False, "This is a permanent channel (can't leave)"
                found = True
                removed_owner = self._normalise_link_owner(ch.get("authorized_by", ""))
            else:
                new_channels.append(ch)

        if not found:
            await self._notify_link_debug(
                action="unlink",
                channel_name=channel_name,
                owner=removed_owner,
                status="ignored",
                note="channel not authorised",
            )
            return False, "Channel not in authorised list"

        if self.save_authorized_channels(new_channels):
            # Leave the channel
            if self.twitch_bot:
                try:
                    live_ok, live_msg = await self.unsubscribe_channel_live(channel_name)
                    if not live_ok:
                        await self._notify_link_debug(
                            action="unlink",
                            channel_name=channel_name,
                            owner=removed_owner,
                            status="error",
                            note=f"live leave failed: {live_msg}",
                        )
                        return False, f"live leave failed: {live_msg}"

                    if hasattr(self.twitch_bot, "part_channels"):
                        try:
                            await self.twitch_bot.part_channels([channel_name])
                            print(f"[TwitchAdapter] Left channel via part_channels: {channel_name}")
                        except Exception as part_err:
                            print(f"[TwitchAdapter] part_channels warning for #{channel_name}: {part_err}")

                    if removed_owner:
                        await self._set_owner_link_state(removed_owner, channel_name, linked=False)
                    await self._notify_link_debug(
                        action="unlink",
                        channel_name=channel_name,
                        owner=removed_owner,
                        status="ok",
                        note="left live",
                    )
                    return True, "ʕっʘ︵ʘʔっ oh.. ok, uh, bye! thanks for having me! left live."
                except Exception as e:
                    print(f"[TwitchAdapter] Error leaving channel: {e}")
                    await self._notify_link_debug(
                        action="unlink",
                        channel_name=channel_name,
                        owner=removed_owner,
                        status="error",
                        note=f"leave exception: {str(e)[:120]}",
                    )
                    return False, f"Error leaving channel: {str(e)[:50]}"
            if removed_owner:
                await self._set_owner_link_state(removed_owner, channel_name, linked=False)
            await self._notify_link_debug(
                action="unlink",
                channel_name=channel_name,
                owner=removed_owner,
                status="ok",
                note="queued (bot offline)",
            )
            return True, "Channel removed (will leave when twitch bot is running)"
        await self._notify_link_debug(
            action="unlink",
            channel_name=channel_name,
            owner=removed_owner,
            status="error",
            note="save failed",
        )
        return False, "Error saving channel list"

    async def send_message(self, channel_id: str, content: str):
        """Send a message to Twitch channel (queued for rate limiting)"""
        await self.message_queue.put(("send", channel_id, content[:499]))

    async def reply_message(self, message: PlatformMessage, content: str):
        """Reply to a Twitch message (queued for rate limiting)"""
        await self.message_queue.put(("reply", message, content[:499]))

    def _normalize_channel_name(self, channel_name: str) -> str:
        return (channel_name or "").strip().lstrip("#").lower()

    def remember_live_channel(self, channel_obj):
        """Cache live channel objects from incoming events so replies can use channel.send()."""
        if channel_obj is None:
            return
        name = getattr(channel_obj, "name", None)
        if not name:
            return
        self._live_channel_cache[self._normalize_channel_name(name)] = channel_obj

    async def _get_channel_sender(self, channel_name: str):
        """Resolve channel login -> PartialUser for TwitchIO v3 send_message API."""
        if not self.twitch_bot:
            return None

        login = self._normalize_channel_name(channel_name)
        if not login:
            return None

        cached = self._channel_sender_cache.get(login)
        if cached is not None:
            return cached

        users = await self.twitch_bot.fetch_users(logins=[login])
        if not users:
            print(f"[TwitchAdapter] Could not resolve Twitch channel login: {login}")
            return None

        user = users[0]
        sender = self.twitch_bot.create_partialuser(user.id, user.login)
        self._channel_sender_cache[login] = sender
        return sender

    async def _send_to_channel(self, channel_name: str, content: str, reply_to_message_id: str | None = None):
        content = to_british_english(str(content or ""))
        login = self._normalize_channel_name(channel_name)
        sender = await self._get_channel_sender(channel_name)
        if sender is None:
            return

        if reply_to_message_id:
            await sender.send_message(
                content[:499],
                sender=str(SECRETtwitchBotIdSECRET),
                reply_to_message_id=str(reply_to_message_id),
            )
            return

        live_channel = self._live_channel_cache.get(login)
        if live_channel is not None and hasattr(live_channel, "send"):
            await live_channel.send(content[:499])
            return

        await sender.send_message(content[:499], sender=str(SECRETtwitchBotIdSECRET))

    async def _message_queue_worker(self):
        """Process queued messages with rate limiting"""
        while True:
            try:
                msg_type, *args = await self.message_queue.get()
                try:
                    if msg_type == "send":
                        channel_name, content = args
                        await self._send_to_channel(channel_name, content)

                    elif msg_type == "reply":
                        message, content = args
                        reply_to = (
                            getattr(getattr(message, "raw_message", None), "id", None)
                            or getattr(getattr(message, "raw_message", None), "message_id", None)
                        )
                        if reply_to:
                            await self._send_to_channel(
                                message.channel_id,
                                content,
                                reply_to_message_id=str(reply_to),
                            )
                        else:
                            await self._send_to_channel(message.channel_id, f"@{message.author_display_name} {content}")

                    # Rate limiting
                    await asyncio.sleep(self.message_delay)
                finally:
                    self.message_queue.task_done()

            except Exception as e:
                print(f"[TwitchAdapter] Queue worker error: {e}")
                await asyncio.sleep(1)

    def format_message(self, user: str, text: str) -> str:
        """Format a message for Twitch"""
        # Get nickname if available
        mem = self.bot.userMemory.get(user, {})
        display = mem.get("nickname") or mem.get("display_name") or user
        return f"{display}: {text}"

    def is_command_allowed(self, command_name: str) -> bool:
        """Check if command is allowed on Twitch"""
        # Remove 'bby' prefix variants for checking
        base_name = command_name.lower()

        if base_name in TWITCH_MANAGEMENT_COMMANDS:
            return True

        # Check if in blocked list
        if base_name in self.blocked_commands:
            return False

        cmd_obj = None
        if hasattr(self.bot, "get_command"):
            cmd_obj = self.bot.get_command(base_name)
            if cmd_obj is not None:
                canonical_name = (getattr(cmd_obj, "name", "") or "").strip().lower()
                if canonical_name in self.blocked_commands:
                    return False

        # Check if in allowed list
        if base_name in self.allowed_commands:
            return True

        if cmd_obj is not None:
            canonical_name = (getattr(cmd_obj, "name", "") or "").strip().lower()
            if canonical_name in self.allowed_commands:
                return True

        # Default deny for unknown commands
        return False

    def _register_commands(self, twitch_bot):
        """Register allowed commands from main bot to Twitch bot"""

        # Create a generic command handler
        async def twitch_command_handler(ctx):
            """Generic handler that routes to main bot's command system"""
            command_name = (getattr(getattr(ctx, "command", None), "name", "") or "").lower()
            author_obj = getattr(ctx, "author", None)
            author_name = (
                getattr(author_obj, "name", None)
                or getattr(author_obj, "login", None)
                or "unknown"
            )
            channel_obj = getattr(ctx, "channel", None)
            channel_name_for_log = getattr(channel_obj, "name", "unknown")
            print(
                _colourise_twitch_command_log(
                    f"[TwitchCmd] !{command_name} by {author_name} in #{channel_name_for_log}"
                )
            )
            self.remember_live_channel(getattr(ctx, "channel", None))

            async def _reply_or_send(text: str):
                reply_method = getattr(ctx, "reply", None)
                if callable(reply_method):
                    await reply_method(text)
                else:
                    await ctx.send(text)

            if command_name in TWITCH_MANAGEMENT_COMMANDS:
                user_obj = getattr(ctx, "author", None)
                user_name = (
                    getattr(user_obj, "name", None)
                    or getattr(user_obj, "login", None)
                    or "unknown"
                ).lower()
                target_channel = user_name
                if not target_channel or target_channel == "unknown":
                    await _reply_or_send(to_british_english("@unknown i couldn't work out your channel name from twitch auth."))
                    return

                action = TWITCH_MANAGEMENT_COMMANDS[command_name]
                if action == "join":
                    _, message = await self.authorize_channel(target_channel, user_name)
                else:
                    _, message = await self.deauthorize_channel(target_channel)

                await _reply_or_send(to_british_english(f"@{user_name} {message}"))
                return

            # Check if command is allowed
            if not self.is_command_allowed(command_name):
                # Prefer threaded reply when available; fall back to plain send.
                await _reply_or_send(
                    to_british_english(
                        f"sorry! {command_name} is too complex for twitch chat. try it on discord! :)"
                    )
                )
                return

            # Convert to platform context
            raw_text = getattr(ctx.message, "content", None) or getattr(ctx.message, "text", None) or ""
            platform_msg = PlatformMessage(
                content=raw_text,
                author_id=ctx.author.name.lower(),
                author_display_name=(getattr(ctx.author, "display_name", None) or ctx.author.name),
                channel_id=ctx.channel.name,
                platform="twitch",
                timestamp=time.time(),
                raw_message=ctx.message,
                is_mod=TwitchBot._is_moderator_user(ctx.author),
            )

            platform_ctx = PlatformContext(
                message=platform_msg,
                bot=self.bot,
                command=command_name,
                platform_ctx=ctx,
            )

            # Route to main bot's command handler
            await self.handle_command(platform_ctx)

        # Pull command names + aliases from Discord command registry so Twitch stays in sync.
        # We still honour the explicit blocklist for commands that are too heavy for chat.
        if hasattr(self.bot, "commands"):
            for cmd_obj in list(getattr(self.bot, "commands", []) or []):
                primary = (getattr(cmd_obj, "name", "") or "").strip().lower()
                if not primary or primary in self.blocked_commands:
                    continue
                self.allowed_commands.add(primary)
                for alias in getattr(cmd_obj, "aliases", []) or []:
                    alias_name = (alias or "").strip().lower()
                    if alias_name and alias_name not in self.blocked_commands:
                        self.allowed_commands.add(alias_name)

        # Register all allowed commands plus Twitch management aliases.
        command_names = set(self.allowed_commands) | set(TWITCH_MANAGEMENT_COMMANDS.keys())
        if hasattr(self.bot, "get_command"):
            expanded = set(command_names)
            for command_name in list(command_names):
                cmd_obj = self.bot.get_command(command_name)
                if cmd_obj is None:
                    continue
                for alias in getattr(cmd_obj, "aliases", []) or []:
                    alias_name = (alias or "").strip().lower()
                    if not alias_name or alias_name in self.blocked_commands:
                        continue
                    expanded.add(alias_name)
            command_names = expanded

        command_names = sorted(command_names)
        for cmd_name in command_names:
            # Create command (TwitchIO v3+ requires callback as first positional arg)
            cmd = twitch_commands.Command(
                twitch_command_handler,  # callback as first positional argument
                name=cmd_name,
            )
            twitch_bot.add_command(cmd)

        print(
            f"[TwitchAdapter] Registered {len(command_names)} commands "
            f"({len(TWITCH_MANAGEMENT_COMMANDS)} management commands included)"
        )
