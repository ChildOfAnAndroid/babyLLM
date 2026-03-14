# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM WEB ADAPTER // phone/discord_bot/platforms/web_adapter.py
# v1.0

"""
Web platform adapter that integrates bbyLocal Flask server into the unified bot.
Runs Flask in background threads while sharing bot state in memory.
"""

import asyncio
import concurrent.futures
import threading
import time
import logging
import inspect
from flask import Flask, jsonify, request
from flask_cors import CORS
from .base import PlatformAdapter, PlatformMessage
from ..context import create_platform_command_context
from ..utils import to_british_english
import os


class WebAdapter(PlatformAdapter):
    """Web API platform adapter - integrates bbyLocal functionality"""

    def __init__(self, bot_instance, port=4420):
        super().__init__(bot_instance)
        self.platform_name = "web"
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)

        self.flask_thread = None
        self.state_lock = threading.Lock()

        # Background animation loops
        self.animation_threads = []

        # Request counter for summary logging
        self.request_counts = {}
        self.request_lock = threading.Lock()

        # Activity tracking for mouth animation
        self.last_message_time = time.time()
        self.recent_activity = []  # Timestamps of recent messages

        # Set up Flask routes
        self._setup_routes()

    @staticmethod
    def _extract_generation_text(result) -> str:
        """Normalise mixed generation return values to response text."""
        if isinstance(result, tuple):
            # _generate_and_reply returns (message_obj, text)
            if len(result) >= 2 and result[1] is not None:
                return str(result[1])
            if result and result[0] is not None:
                return str(result[0])
            return "..."
        if result is None:
            return "..."
        return str(result)

    @staticmethod
    def _extract_command_name(text: str) -> str:
        stripped = (text or "").strip()
        if not stripped.startswith("!"):
            return ""
        return (stripped.split()[0][1:] if stripped.split() else "").strip().lower()

    @staticmethod
    def _extract_command_description(command_obj) -> str:
        """Return a short, human-readable description for a command."""
        candidates = [
            getattr(command_obj, "brief", None),
            getattr(command_obj, "help", None),
            getattr(command_obj, "description", None),
            inspect.getdoc(getattr(command_obj, "callback", None)) if getattr(command_obj, "callback", None) else None,
        ]

        for candidate in candidates:
            if not candidate:
                continue
            text = str(candidate).strip()
            if not text:
                continue
            first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
            if first_line:
                return to_british_english(first_line)
        return ""

    def _build_live_command_registry(self):
        """Build one canonical entry per Discord command with aliases."""
        entries = []
        seen_names = set()
        command_objects = list(getattr(self.bot, "commands", []) or [])

        twitch_adapter = None
        if hasattr(self.bot, "platforms"):
            twitch_adapter = self.bot.platforms.get("twitch")

        for command_obj in command_objects:
            command_name = str(getattr(command_obj, "name", "") or "").strip().lower()
            if not command_name or command_name in seen_names:
                continue
            seen_names.add(command_name)

            aliases = []
            for alias in list(getattr(command_obj, "aliases", []) or []):
                alias_name = str(alias or "").strip().lower()
                if alias_name and alias_name != command_name and alias_name not in aliases:
                    aliases.append(alias_name)

            signature = str(getattr(command_obj, "signature", "") or "").strip()
            usage = f"!{command_name}"
            if signature:
                usage = f"{usage} {signature}"

            description = self._extract_command_description(command_obj)

            entries.append({
                "name": command_name,
                "aliases": aliases,
                "signature": signature,
                "usage": usage,
                "description": description,
                "cog": str(getattr(getattr(command_obj, "cog", None), "qualified_name", "") or ""),
                "hidden": bool(getattr(command_obj, "hidden", False)),
                "twitch_allowed": (
                    bool(twitch_adapter.is_command_allowed(command_name))
                    if twitch_adapter is not None
                    else None
                ),
                "web_allowed": True,
            })

        entries.sort(key=lambda item: item["name"])
        return entries

    @staticmethod
    def _embed_to_plain_text(embed) -> str:
        if embed is None:
            return ""

        parts = []
        title = str(getattr(embed, "title", "") or "").strip()
        description = str(getattr(embed, "description", "") or "").strip()
        if title:
            parts.append(title)
        if description:
            parts.append(description)

        for field in list(getattr(embed, "fields", []) or [])[:5]:
            name = str(getattr(field, "name", "") or "").strip()
            value = str(getattr(field, "value", "") or "").strip()
            if name and value:
                parts.append(f"{name}: {value}")
            elif value:
                parts.append(value)

        footer = getattr(embed, "footer", None)
        footer_text = str(getattr(footer, "text", "") or "").strip() if footer is not None else ""
        if footer_text:
            parts.append(footer_text)

        return "\n".join([part for part in parts if part]).strip()

    def _setup_routes(self):
        """Set up Flask API endpoints"""

        @self.app.get("/api/ping")
        def ping():
            return jsonify(ok=True, msg="hello from unified bot web API")

        @self.app.get("/api/state")
        @self.app.get("/api/babystate")  # CORS-friendly alias
        def get_state():
            """Get baby's current state"""
            with self.state_lock:
                state = {
                    "eyes": getattr(self.bot, '_web_eyes', 5),
                    "mouth": getattr(self.bot, '_web_mouth', 1),
                    "cheeks_on": getattr(self.bot, '_web_cheeks', False),
                    "tears_on": getattr(self.bot, '_web_tears', False),
                    "jumping": getattr(self.bot, '_web_jumping', False),
                    "isSpeaking": getattr(self.bot, '_web_speaking', False),
                    "speechText": getattr(self.bot, '_web_speech_text', ""),
                    "R": getattr(self.bot, '_web_R', 133),
                    "G": getattr(self.bot, '_web_G', 239),
                    "B": getattr(self.bot, '_web_B', 238),
                    # Add stretch/squish states
                    "stretch_left": getattr(self.bot, '_web_stretch_left', False),
                    "stretch_right": getattr(self.bot, '_web_stretch_right', False),
                    "stretch_up": getattr(self.bot, '_web_stretch_up', False),
                    "stretch_down": getattr(self.bot, '_web_stretch_down', False),
                    "squish_left": getattr(self.bot, '_web_squish_left', False),
                    "squish_right": getattr(self.bot, '_web_squish_right', False),
                    "squish_up": getattr(self.bot, '_web_squish_up', False),
                    "squish_down": getattr(self.bot, '_web_squish_down', False),
                    # Emotional metrics from bot if available
                    "cerebralLoad": getattr(self.bot.tutor, 'totalAvgAbsDelta', 0.0) if hasattr(self.bot, 'tutor') else 0.0,
                    "dreamIntensity": getattr(self.bot, '_web_dream_intensity', 10.0),
                    "memoryFlux": getattr(self.bot, '_web_memory_flux', 0.0),
                    "learningStability": getattr(self.bot.tutor, 'perfectionistPassRate', 0.0) if hasattr(self.bot, 'tutor') else 0.0,
                    "metabolicRate": getattr(self.bot, '_web_metabolic_rate', 0.1),
                    "lastUpdated": time.time(),
                    "timestamp": time.time(),
                }

                # Add token events if available
                if hasattr(self.bot, '_web_token_events'):
                    state["token_events"] = self.bot._web_token_events[-10:]  # Last 10 events

                return jsonify(state)

        @self.app.post("/api/state")
        def set_state():
            """Update baby's state (for website controls)"""
            data = request.get_json(silent=True) or {}
            if not isinstance(data, dict):
                return jsonify(error="bad json"), 400

            with self.state_lock:
                # Map API field names to actual bot attributes.
                state_map = {
                    "eyes": "_web_eyes",
                    "mouth": "_web_mouth",
                    "cheeks_on": "_web_cheeks",
                    "tears_on": "_web_tears",
                    "jumping": "_web_jumping",
                    "isSpeaking": "_web_speaking",
                    "speechText": "_web_speech_text",
                    "R": "_web_R",
                    "G": "_web_G",
                    "B": "_web_B",
                    "stretch_left": "_web_stretch_left",
                    "stretch_right": "_web_stretch_right",
                    "stretch_up": "_web_stretch_up",
                    "stretch_down": "_web_stretch_down",
                    "squish_left": "_web_squish_left",
                    "squish_right": "_web_squish_right",
                    "squish_up": "_web_squish_up",
                    "squish_down": "_web_squish_down",
                    "dreamIntensity": "_web_dream_intensity",
                    "memoryFlux": "_web_memory_flux",
                }
                for key, attr_name in state_map.items():
                    if key in data:
                        setattr(self.bot, attr_name, data[key])

            return jsonify(ok=True, state=data)

        @self.app.post("/api/say")
        def say():
            """Handle chat from website (sync Flask route, async bot execution)."""
            data = request.get_json(silent=True) or {}
            text = (data.get("text") or "").strip()
            author = (data.get("author") or "web_user").strip().lower()

            if not text:
                return jsonify(status="error", reply="no text :("), 400

            print(f"[WebAdapter] Received chat from {author}: {text[:50]}...")

            # PRIVACY: Web users are treated as opted in by default unless explicitly opted out.
            command_name = self._extract_command_name(text)
            is_opt_in_command = command_name in {"bbyoptin", "aioptin", "boptin"}
            has_opted_out = (
                self.bot.has_explicit_web_opt_out(author)
                if hasattr(self.bot, "has_explicit_web_opt_out")
                else bool((self.bot.userMemory.get(author, {}) or {}).get("web_explicit_opt_out", False))
            )
            if has_opted_out and not is_opt_in_command:
                print(f"[WebAdapter] User {author} has opted out - rejecting message")
                return jsonify(
                    status="opted_out",
                    reply="You've opted out! Use !bbyoptin if you want to chat with me again ʕ·ᴥ·ʔ"
                ), 200

            try:
                reply = self._run_on_bot_loop(
                    self._handle_web_chat_async(author=author, text=text, raw_data=data),
                    timeout=185,
                )

                with self.state_lock:
                    self.bot._web_speaking = True
                    self.bot._web_speech_text = reply

                # Auto-stop speaking after 10s
                def stop_speaking():
                    time.sleep(10)
                    with self.state_lock:
                        self.bot._web_speaking = False
                        self.bot._web_speech_text = ""

                threading.Thread(target=stop_speaking, daemon=True).start()
                return jsonify(status="ok", reply=reply)

            except concurrent.futures.TimeoutError:
                return jsonify(status="error", reply="... (thinking too hard, sorry!)"), 504
            except Exception as e:
                print(f"[WebAdapter] Error generating response: {e}")
                import traceback
                traceback.print_exc()
                return jsonify(status="error", reply=f"oops! {str(e)[:100]}"), 500

        @self.app.get("/api/bbybook")
        def get_bbybook():
            """Get facts database"""
            try:
                # Return bot's bbyfacts
                if hasattr(self.bot, 'bbyfacts'):
                    return jsonify(self.bot.bbyfacts)
                else:
                    return jsonify({})
            except Exception as e:
                return jsonify(error=str(e)), 500

        @self.app.get("/api/leaderboard")
        def get_leaderboard():
            """Get unified leaderboard across all platforms"""
            try:
                users = []
                for user_id, data in self.bot.userMemory.items():
                    users.append({
                        "username": data.get("display_name") or user_id,
                        "user_id": user_id,
                        "bby": data.get("BBY", 0),
                        "wins": data.get("wins", 0),
                        "losses": data.get("losses", 0),
                        "message_count": data.get("message_count", 0),
                        "loyalty": data.get("loyalty", 1),
                    })

                # Sort by BBY descending
                users.sort(key=lambda x: x["bby"], reverse=True)

                # Calculate totals
                total_bby = sum(u["bby"] for u in users)
                total_messages = sum(u["message_count"] for u in users)

                return jsonify({
                    "leaderboard": users[:50],  # Top 50
                    "total_users": len(users),
                    "total_bby_in_circulation": total_bby,
                    "total_messages": total_messages,
                    "platforms": list(self.bot.platforms.keys()) if hasattr(self.bot, 'platforms') else ["discord"]
                })
            except Exception as e:
                return jsonify(error=str(e)), 500

        @self.app.get("/api/commands")
        def get_commands():
            """Get canonical command registry (one command with alias list)."""
            try:
                commands_payload = self._build_live_command_registry()
                return jsonify({
                    "source": "discord_registry",
                    "count": len(commands_payload),
                    "updated_at": time.time(),
                    "commands": commands_payload,
                })
            except Exception as e:
                return jsonify(error=str(e)), 500

    async def start(self):
        """Start Flask server and animation loops"""
        print(f"[WebAdapter] Starting web API on http://127.0.0.1:{self.port}")

        # Initialize bot web state
        self._init_web_state()

        # Disable Flask request logging (too noisy!)
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        # Add before_request hook to count requests
        @self.app.before_request
        def count_request():
            endpoint = request.path
            with self.request_lock:
                self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1

        # Start Flask in background thread
        def run_flask():
            self.app.run(host="127.0.0.1", port=self.port, debug=False, use_reloader=False)

        self.flask_thread = threading.Thread(target=run_flask, daemon=True)
        self.flask_thread.start()

        # Start animation loops
        self._start_animation_loops()

        print(f"[WebAdapter] Web API started successfully")

    async def stop(self):
        """Stop Flask server"""
        print("[WebAdapter] Stopping web API...")
        # Flask thread will stop when main process exits (daemon=True)

    def _init_web_state(self):
        """Initialize web-specific state on bot"""
        self.bot._web_eyes = 5
        self.bot._web_mouth = 1
        self.bot._web_cheeks = False
        self.bot._web_tears = False
        self.bot._web_jumping = False
        self.bot._web_speaking = False
        self.bot._web_speech_text = ""
        self.bot._web_R = 133
        self.bot._web_G = 239
        self.bot._web_B = 238
        self.bot._web_stretch_left = False
        self.bot._web_stretch_right = False
        self.bot._web_stretch_up = False
        self.bot._web_stretch_down = False
        self.bot._web_squish_left = False
        self.bot._web_squish_right = False
        self.bot._web_squish_up = False
        self.bot._web_squish_down = False
        self.bot._web_dream_intensity = 10.0
        self.bot._web_memory_flux = 0.0
        self.bot._web_metabolic_rate = 0.1
        self.bot._web_token_events = []

    def _start_animation_loops(self):
        """Start background animation loops (blink, pulse, etc.)"""
        import random

        def blink_loop():
            """Natural blinking animation"""
            while True:
                try:
                    with self.state_lock:
                        is_speaking = self.bot._web_speaking
                        original_eyes = self.bot._web_eyes
                        dream_intensity = self.bot._web_dream_intensity

                    if is_speaking:
                        time.sleep(0.2)
                        continue

                    wakefulness = max(1, round(dream_intensity))
                    time.sleep(2 + (random.random() * wakefulness))

                    # Blink
                    blink_direction = random.choice([0, 1])
                    with self.state_lock:
                        self.bot._web_eyes = blink_direction

                    time.sleep(self.bot._web_metabolic_rate * 0.5)

                    with self.state_lock:
                        self.bot._web_eyes = original_eyes

                except Exception as e:
                    print(f"[WebAdapter] Blink loop error: {e}")
                    time.sleep(1)

        def pulse_loop():
            """Emotional pulse animations"""
            while True:
                try:
                    with self.state_lock:
                        metabolic_rate = self.bot._web_metabolic_rate

                    stim_choice = random.choice(["random", "tense", "dreamy", "blushy"])

                    with self.state_lock:
                        if stim_choice == "random":
                            keys = ["_web_stretch_left", "_web_stretch_right",
                                   "_web_squish_up", "_web_squish_down"]
                            setattr(self.bot, random.choice(keys), random.choice([True, False]))
                        elif stim_choice == "blushy":
                            if self.bot._web_cheeks:
                                self.bot._web_cheeks = random.choice([True, False])

                    time.sleep(max(0.1, metabolic_rate))

                except Exception as e:
                    print(f"[WebAdapter] Pulse loop error: {e}")
                    time.sleep(1)

        def request_summary_loop():
            """Print request summaries every 100 total requests"""
            last_total = 0
            while True:
                time.sleep(10)  # Check every 10 seconds
                try:
                    with self.request_lock:
                        total = sum(self.request_counts.values())
                        if total >= last_total + 100:
                            print(f"[WebAdapter] Request summary (last {total - last_total} requests):")
                            for endpoint, count in sorted(self.request_counts.items()):
                                print(f"  {endpoint}: {count}")
                            last_total = total
                except Exception as e:
                    print(f"[WebAdapter] Summary loop error: {e}")

        def activity_monitor_loop():
            """Monitor overall bot activity and update mouth based on message frequency"""
            last_buffer_size = 0
            while True:
                time.sleep(3)  # Check every 3 seconds
                try:
                    # Track messages across ALL platforms
                    current_buffer_size = len(self.bot.buffer) if hasattr(self.bot, 'buffer') else 0
                    now = time.time()

                    # If buffer grew, messages were added!
                    if current_buffer_size > last_buffer_size:
                        # Activity detected!
                        messages_added = current_buffer_size - last_buffer_size
                        for _ in range(messages_added):
                            self.recent_activity.append(now)
                        self.last_message_time = now

                    # If buffer shrunk, messages were removed (gone quiet)
                    elif current_buffer_size < last_buffer_size:
                        # Messages removed - activity cooling down
                        pass

                    last_buffer_size = current_buffer_size

                    # Clean old activity (keep last 60 seconds)
                    cutoff = now - 60
                    self.recent_activity = [t for t in self.recent_activity if t > cutoff]

                    # Calculate activity level
                    messages_per_minute = len(self.recent_activity)
                    time_since_last = now - self.last_message_time

                    # Update mouth based on activity
                    if time_since_last > 120:  # 2 minutes of silence
                        new_mouth = 0  # Sad frown
                    elif time_since_last > 30:  # 30 seconds quiet
                        new_mouth = 1  # Normal small smile
                    elif messages_per_minute > 10:  # Very active!
                        new_mouth = 2  # Big happy smile!
                    else:  # Some activity
                        new_mouth = 1  # Normal smile

                    # Only update if changed
                    if self.bot._web_mouth != new_mouth:
                        self.bot._web_mouth = new_mouth
                        moods = {0: "sad (quiet)", 1: "normal", 2: "happy (active!)"}
                        print(f"[Activity→Web] Mouth: {moods.get(new_mouth, 'unknown')} ({messages_per_minute} msg/min)")

                except Exception as e:
                    print(f"[WebAdapter] Activity monitor error: {e}")

        def state_push_loop():
            """Reactively push meaningful state changes to the public website.

            Replaces the website's 10Hz polling loop with an event-driven push.
            Set BBY_PUBLIC_URL to the website's base URL to enable.
            Only pushes when state actually changes (checked every 200ms).
            """
            try:
                from config import bby_public_url as _cfg_url
            except ImportError:
                _cfg_url = ""
            push_url_base = os.environ.get("BBY_PUBLIC_URL", _cfg_url).rstrip("/")
            if not push_url_base:
                print("[WebAdapter] BBY_PUBLIC_URL not set — state push disabled (website will poll as fallback)")
                return

            push_url = f"{push_url_base}/api/brain_push"
            print(f"[WebAdapter] Reactive state push active → {push_url}")

            try:
                import requests as _req
            except ImportError:
                print("[WebAdapter] requests not available — state push disabled")
                return

            # Only watch attrs that represent meaningful emotional/visual state.
            # Micro-animation attrs (stretch/squish) are excluded intentionally.
            WATCHED = [
                '_web_R', '_web_G', '_web_B',
                '_web_eyes', '_web_mouth',
                '_web_cheeks', '_web_tears',
                '_web_jumping',
                '_web_speaking', '_web_speech_text',
                '_web_dream_intensity',
                '_web_memory_flux',
                '_web_metabolic_rate',
            ]

            last_snapshot = {}
            while True:
                time.sleep(0.2)
                try:
                    snapshot = {a: getattr(self.bot, a, None) for a in WATCHED}
                    if snapshot == last_snapshot:
                        continue

                    payload = {
                        "eyes":           snapshot.get('_web_eyes', 5),
                        "mouth":          snapshot.get('_web_mouth', 1),
                        "cheeks_on":      snapshot.get('_web_cheeks', False),
                        "tears_on":       snapshot.get('_web_tears', False),
                        "jumping":        snapshot.get('_web_jumping', False),
                        "isSpeaking":     snapshot.get('_web_speaking', False),
                        "speechText":     snapshot.get('_web_speech_text', ''),
                        "R":              snapshot.get('_web_R', 133),
                        "G":              snapshot.get('_web_G', 239),
                        "B":              snapshot.get('_web_B', 238),
                        "dreamIntensity": snapshot.get('_web_dream_intensity', 10.0),
                        "memoryFlux":     snapshot.get('_web_memory_flux', 0.0),
                        "metabolicRate":  snapshot.get('_web_metabolic_rate', 0.1),
                    }

                    _req.post(push_url, json=payload, timeout=2)
                    last_snapshot = snapshot

                except Exception:
                    pass  # best-effort; website polls as fallback

        # Start animation threads
        self.animation_threads = [
            threading.Thread(target=blink_loop, daemon=True),
            threading.Thread(target=pulse_loop, daemon=True),
            threading.Thread(target=request_summary_loop, daemon=True),
            threading.Thread(target=activity_monitor_loop, daemon=True),
            threading.Thread(target=state_push_loop, daemon=True),
        ]

        for thread in self.animation_threads:
            thread.start()

        print("[WebAdapter] Animation loops started")

    def _run_on_bot_loop(self, coro, timeout: float = 180.0):
        """Run a coroutine on the bot's event loop from Flask's thread."""
        loop = getattr(self.bot, "loop", None)
        if loop is None or not loop.is_running():
            raise RuntimeError("bot loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    async def _handle_web_chat_async(self, *, author: str, text: str, raw_data: dict) -> str:
        """Async work for /api/say executed on the Discord bot event loop."""
        author_key = str(author or "").strip().lower() or "web_user"
        command_name = self._extract_command_name(text)

        platform_msg = PlatformMessage(
            content=text,
            author_id=author_key,
            author_display_name=author_key,
            channel_id="web",
            platform="web",
            timestamp=time.time(),
            raw_message=raw_data,
        )
        await self.bot.handle_platform_message(platform_msg)

        if command_name:
            if not self.is_command_allowed(command_name):
                return to_british_english(
                    f"sorry! !{command_name} is too complex for web chat. try it on discord! :)"
                )
            if not getattr(self.bot, "cog", None):
                return "oops! command system is not ready yet"

            outputs: list[str] = []

            async def web_reply_sink(content="", embed=None, **kwargs):
                if embed is not None:
                    text_out = self._embed_to_plain_text(embed)
                elif content:
                    text_out = str(content)
                else:
                    text_out = ""

                text_out = to_british_english(text_out).strip()
                if text_out:
                    outputs.append(text_out)
                return None

            fake_ctx = create_platform_command_context(
                bot=self.bot,
                platform="web",
                author_id=author_key,
                author_name=author_key,
                channel_id="web",
                message_content=text,
                command_name=command_name,
                reply_sink=web_reply_sink,
                send_sink=web_reply_sink,
            )

            try:
                resolved_name, callback = self.bot._resolve_platform_command(command_name, "web")
                if callback is None:
                    return to_british_english(f"sorry! !{command_name} isn't available right now")

                arg_text = self.bot._extract_command_arg_text(text)
                positional_args, keyword_args = self.bot._parse_command_arguments(callback, arg_text)
                await self.bot._invoke_platform_callback(callback, fake_ctx, positional_args, keyword_args)
            except Exception as e:
                print(f"[WebAdapter] Command error in '{command_name}': {e}")
                return to_british_english(f"oops! something broke: {str(e)[:120]}")

            # Return compact plain text summary for web UI.
            if outputs:
                return "\n".join(outputs)[:1900]
            return "ok! command finished."

        fake_ctx = create_platform_command_context(
            bot=self.bot,
            platform="web",
            author_id=author_key,
            author_name=author_key,
            channel_id="web",
            message_content=text,
            command_name="babyllm",
        )

        if hasattr(self.bot, "generation_queue"):
            response_future = asyncio.get_running_loop().create_future()

            async def set_result(result):
                if not response_future.done():
                    response_future.set_result(self._extract_generation_text(result))

            await self.bot.generation_queue.put((
                fake_ctx,
                text,
                50,  # num tokens for web (medium length)
                set_result
            ))
            try:
                reply = await asyncio.wait_for(response_future, timeout=180)
            except asyncio.TimeoutError:
                reply = "... (thinking too hard, sorry!)"
        else:
            if hasattr(self.bot, "cog"):
                result = await self.bot.cog._generate_and_reply(fake_ctx, text, 50)
                reply = self._extract_generation_text(result)
            else:
                reply = "hi! :)"

        return to_british_english(str(reply))

    async def send_message(self, channel_id: str, content: str):
        """Web doesn't actively send messages"""
        pass

    async def reply_message(self, message: PlatformMessage, content: str):
        """Web replies are handled via /api/say response"""
        pass

    def format_message(self, user: str, text: str) -> str:
        """Format message for web"""
        return f"{user}: {text}"

    def is_command_allowed(self, command_name: str) -> bool:
        """All commands allowed on web"""
        return True
