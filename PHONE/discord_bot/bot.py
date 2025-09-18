# v2.1
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM // phone/discord_bot/bot.py
# v1.9

import os
import json
import time
import asyncio
import discord
from discord.ext import commands
import re
from collections import Counter, defaultdict
from config import *
from secret import *
from textCleaningTool import *
import traceback
import random
import random as pyrandom
import hashlib
import calendar
from typing import Optional
import pytz
from datetime import datetime, timedelta
from .logger import logger
from .safety import safety
from .data_manager import data_manager
from .performance import perf_monitor
import math
import aiohttp
from urllib.parse import urljoin

from helpers import save_json_if_changed

from .context import create_fake_context
from .utils import escape_markdown, is_similar, killExcessTags, getTimeRant
from .autonomy import AutonomyPlanner

bby_lounge = 1388782896084422788
bby_spam = 1156683242087387206
bby_debug = 1399818543125495970

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUEST_FILE_PATH = os.path.join(SCRIPT_DIR, "bby_request.json")
RESPONSE_DIR = os.path.join(SCRIPT_DIR, "bby_responses")

class BABYBOT_DISCORD(commands.Bot): 

    def __init__(self, babyLLM, tutor, librarian, scribe, calligraphist,  
                 discordToken = SECRETdiscordTokenSECRET, discordChannel = bby_spam,
                 rollingContextSize = rollingContextSize, idleTrainSeconds = 100, N = rollingContextSize - 1):

        self.babyLLM, self.tutor, self.librarian, self.scribe, self.calligraphist = babyLLM, tutor, librarian, scribe, calligraphist

        # lock protects user dictionaries/file saves
        self._user_data_save_lock = asyncio.Lock()
        self._fact_award_lock = asyncio.Lock()
        
        # --- Smink high score tracking ---
        self.smink_highscore_path = os.path.join(SCRIPT_DIR, "smink_highscore.json")
        if os.path.exists(self.smink_highscore_path):
            with open(self.smink_highscore_path, "r") as f:
                self.smink_highscore = json.load(f)
        else:
            self.smink_highscore = {"amount": 0, "user": ""}
        
        intents = discord.Intents.all()
        # Add heartbeat_timeout to prevent gateway issues
        super().__init__(
            command_prefix='!', 
            intents=intents,
            heartbeat_timeout=60.0,  # Increase from default 30s
        )
        self.cog = None

        self.faveEmotes = ("😭", "😤", "🔥", "✨", "❤️", "😡", "😠", "🤬", "💔", "💕", "🦊", "😊", "🎵", "🎶", "🤣", "🙌", "🥰", "🥨", "🥖", "😂", "🤞", "🍜", "🥯", "🌻", "🍞", "😀", 
                           "😃", "😄", "😁", "😅", "🥹", "😆", "🤣", "🥲", "☺️", "😊", "😉", "🙃", "🙂", "😇", "😌", "😍", "🥰", "😘", "🤨", "🧐", "🤓", "😎", "😏", "😔", "🙁", "😭", 
                           "😢", "🥺", "🤯", "😳", "😨", "😶‍🌫️", "🫣", "🤔", "😬", "🙄", "😑", "😐", "😵", "😵‍💫", "🤢", "😈", "👿", "💩", "👻", "👾", "🤖", "😸", "😹", "😻", "😼", "😾", 
                           "😺", "😿", "🙀", "😽", "🫶", "👍", "👎", "✌️", "🫵", "✍️", "👄", "🫦", "👶", "👧", "🧒", "👦", "👩", "🧑", "👨", "👩‍🦱", "🧑‍🦱", "👨‍🦱", "👩‍🦰", "🧑‍🦰", "👨‍🦰", "👱‍♀️", 
                           "👱", "👱‍♂️", "👩‍🦳", "🧑‍🦳", "👨‍🦳", "👩‍🦲", "🧑‍🦲", "👨‍🦲", "🧔‍♀️", "🧔", "🧔‍♂️", "👵", "🧓", "👴", "👲", "👳‍♀️", "👳", "👳‍♂️", "🧕🏻", "👮‍♀️", "👮", "👮‍♂️", "👷‍♀️", "👷", "👷‍♂️", "💂‍♀️", 
                           "💂", "💂‍♂️", "🕵️‍♀️", "🕵️", "🕵️‍♂️", "👩‍⚕️", "🧑‍⚕️", "👨‍⚕️", "👩‍🌾", "🧑‍🌾", "👨‍🌾", "👩‍🍳", "🧑‍🍳", "👨‍🍳", "🧑‍🎤", "👨‍🎤", "👩‍🏫", "🧑‍🏫", "👨‍🏫", "👩‍🏭", "🧑‍🏭", "👨‍🏭", "👩‍💻", "🧑‍💻", "👨‍💻", "👩‍💼", 
                           "👨‍💻", "🧑‍💼", "👨‍💼", "👩‍🔧", "🧑‍🔧", "👨‍🔧", "👩‍🔬", "🧑‍🔬", "👨‍🔬", "👩‍🎨", "🧑‍🎨", "👨‍🎨", "👩‍🚒", "🧑‍🚒", "👨‍🚒", "👩‍✈️", "🧑‍✈️", "👨‍✈️", "👩‍🚀", "🧑‍🚀", "👨‍🚀", "👩‍⚖️", "🧑‍⚖️", "👨‍⚖️", "👰‍♀️", "👰", 
                           "👰‍♂️", "🤵‍♀️", "🤵", "🤵‍♂️", "👸", "🫅", "🤴", "🥷", "🦸‍♀️", "🦸", "🦸‍♂️", "🦹‍♀️", "🦹", "🦹‍♂️", "🤶", "🍃", "🌚", "🌈", "🍌", "🍇", "🍆", "🧄", "🥦", "🍜", "🖥️", "💻", 
                           "🆒", "⚧", "🏳️‍⚧️", "🏳️‍🌈", "♀️", "♂️", "🫀", "🦤", "🦊", "🐺", "🐶", "🐕", "🐩", "🐾", "🐱", "🐈", "🐈‍⬛", "🐰", "🐇", "🐿️", "🧸", "🐻", "🐨", "🐼", "🐤", "🐥", 
                           "🐣", "🐦", "🕊️", "🐧", "🦜", "🐸", "🐢", "🦎", "🐍", "🦄", "🐉", "🐲", "👾", "👻", "🐷", "🐽", "🐮", "🐘", "🦔", "🦝", "🦦", "🦥", "🐧", "🎀", "🍓", "🍒", 
                           "🍉", "🍊", "🍋", "🍍", "🥭", "🍎", "🍏", "🍐", "🥝", "🍈", "🍞", "🥐", "🍰", "🎂", "🧁", "🍮", "🍩", "🍪", "🥞", "🍬", "🍭", "🍫", "🍯", "💌", "💟", "💜", 
                           "💙", "💚", "💛", "🧡", "🤍", "🧚", "🧜‍♀️", "🧜", "🧞‍♀️", "🧞", "🧙‍♀️", "🧙", "🧝‍♀️", "𝓯", "🐣", "🪿", "🦆", )       
        
        self.errorKeys = ["oops, error!", "missingno", "NaN", "the void"]
        self.errorValues = ["how did you manage to make this item!?"]
        self.errorAuthors = ["the void", "missingno", "error!", "NaN"]
        
        self.babyName, self.lastClockAnnounce = babyName, 0
        # Bots that are allowed to issue commands to babyllm. Messages from all bots
        # are processed, but only these bots are trusted to run commands.
        self.trusted_bot_names = {"buttsbot", "babyllm", "skunkllm", "tatsu", "tatsumaki"}
        self.temp_not_opt = ["chucklesw73", "rustypeugeot", "tomkenchmusic", "stereochromus", "noiseordinance", "kazumianzai", "wakelessnine", "hrh_ginsterbusch", "3roc", 
                             "shaka6331", "ave_maria33", "nequals", "3therealdescent", "merlinofthevoid"]
        self.discordToken, self.discordChannel, self.rollingContextSize = discordToken, discordChannel, rollingContextSize
        self.last_logged_author, self.idleTrainSeconds, self.N = None, idleTrainSeconds, N
        self.chatWindowMAX, self.dataStride = windowMAXSTART, round(windowMAXSTART * 0.1)
        self.idles, self.random, self.random2, self.random3, self.random4 = 0, 0.0, 0.0, 0.0, 0.0
        self._varied_rng_nonce = 0
        self.current_bestie, self.bestie_score = None, 0.0
        self.inventory = {}

        # --- unified lexicon game state ---
        # Multiple concurrent sessions keyed by the prompt message id.
        # Session schema:
        # { 'mode': 'wtf'|'translate', 'channel_id': int, 'message_id': int,
        #   'created_at': float, 'extra': {...}, optional 'task': asyncio.Task }
        self.lex_sessions = {}
        self.word_usage = Counter()            # trending unknowns for auto-wtf
        
        # --- Neural token sentiment tracking ---
        self.recent_positive_tokens = set()   # Token IDs that appeared in positive contexts
        self.recent_negative_tokens = set()   # Token IDs that appeared in negative contexts
        self.token_sentiment_decay = 100     # How many messages before token sentiment expires
        self.opt_in_token_usage = Counter()    # opted-in user token usage stats
        self.wtf_threshold = 30
        self.wtf_reacts = ["💡", "😳", "💀", "🤔", "😂", "🙀"]
        self.next_translate_time = time.time() + random.uniform(24 * 3600, 168 * 3600)

        # --- favourite token tracking ---
        self.babyFaveToken = ""
        self.baby_state_path = os.path.join(SCRIPT_DIR, "babyState.json")

        # Load chat buffer with proper file handling
        if os.path.exists(chatBufferFilepath):
            with open(chatBufferFilepath, "r") as f:
                self.buffer = json.load(f)
        else:
            self.buffer = []

        # --- Separate rolling training buffer (JSON file)
        try:
            self.training_buffer_path = bbyTrainingBufferFilepath
        except NameError:
            self.training_buffer_path = os.path.join(SCRIPT_DIR, "training_buffer.json")
        self.training_buffer: list[str] = []
        # Keep 2–4x chat buffer; default ~3x
        self.training_buffer_size = max(64, int(self.rollingContextSize * 3))

        self.user_data_path = bbyUserDataPath
        self.bbyfacts_path = bbybookPath
        self.bbycraft_recipes_path = os.path.join(SCRIPT_DIR, "bbycraft_recipes.json")

        def get_default_user_memory():
            return {"nickname": None, "display_name": None, "timezone": "Europe/London",
                    "BBY": 0.0, "spamMax": 0.3, "bbybook": [],
                    "wins": 0.0, "losses": 0.0, "draws": 0.0,
                    "last_seen": time.time(), "message_count": 0.0, "loyalty": 1,
                    "last_message_words": set(), "creative_combo": 1, "spammer": 1,
                    "inventory": {}, "favourites": [], "next_talk_milestone": 50,
                    "translate_wins": 0, "translate_losses": 0,
                    "fave_token_usage": 0, "command_usage": {}}

        # Global command statistics tracking
        self.command_stats_path = os.path.join(SCRIPT_DIR, "command_stats.json")
        self.command_stats = self._json_load(self.command_stats_path, default_type={})
        
        # Load crafting recipes
        self.bbycraft_recipes = self._json_load(self.bbycraft_recipes_path, default_type={})

        if os.path.exists(self.user_data_path):
            logger.info("INIT", f"{self.user_data_path} LOADING FROM PATH...")
            self.userMemory = defaultdict(get_default_user_memory)
            self._load_user_data()
        else:
            self.userMemory = defaultdict(get_default_user_memory)
        
        self.opt_in_path = optInUsersPath 
        if os.path.exists(self.opt_in_path):
            with open(self.opt_in_path, "r") as f: self.AIoptInUsers = json.load(f)
        else: self.AIoptInUsers = []

        self.bbyfacts = self._json_load(self.bbyfacts_path)
        logger.info("INIT", f"LOADED {len(self.bbyfacts)} FACTS")
        
        self.lastInteraction = time.time()
        self.idle_task = self.training_worker = None
        self.random_task = None
        self.web_task = None
        self.monthly_task = None
        self.decay_task = None
        self.training_queue = asyncio.Queue()
        self._refresh_brain_randoms()
        self._load_baby_state()
        # preload training buffer for early enrichment
        try:
            self._load_training_buffer()
        except Exception:
            pass

        # lightweight stats-guided autonomy planner for idle periods
        self.autonomy = AutonomyPlanner(self)

        # Setup centralized data manager for batched saves
        data_manager.set_bot_reference(self)
        data_manager.register_save_callback("user_data", self._save_user_data)
        data_manager.register_save_callback("bbyfacts", self.save_bbyfacts)
        data_manager.register_save_callback("bbycraft_recipes", self.save_bbycraft_recipes)
        data_manager.register_save_callback("command_stats", self._save_command_stats)
        logger.info("INIT", "Data manager initialised with batched save system")

        # Setup performance monitoring with health checks
        perf_monitor.add_health_check("neural_network", lambda: hasattr(self, 'babyLLM') and self.babyLLM is not None, critical=True)
        perf_monitor.add_health_check("user_memory", lambda: len(self.userMemory) > 0)
        perf_monitor.add_health_check("librarian", lambda: hasattr(self, 'librarian') and self.librarian is not None, critical=True)
        logger.info("INIT", "Performance monitoring system initialised")

    async def setup_bot(self):
        from .cog import babyBot_DISCORD_COG
        self.cog = babyBot_DISCORD_COG(self)
        await self.add_cog(self.cog)
        
    async def setup_hook(self):
        await super().setup_hook()
        self._ensure_random_task()

    def save_smink_highscore(self):
        with open(self.smink_highscore_path, "w") as f: json.dump(self.smink_highscore, f)

    def get_varied_random(self):
        """Brain-influenced random draw with per-call jitter to avoid identical streaks."""
        if not any((self.random, self.random2, self.random3, self.random4)):
            self._refresh_brain_randoms()

        slots = [self.random, self.random2, self.random3, self.random4]
        slot_index = pyrandom.randrange(len(slots))
        base = slots[slot_index] or pyrandom.random()

        influenced = self.get_brain_influence(base, influence_strength=0.35)
        jitter_primary = pyrandom.random()
        jitter_secondary = (time.perf_counter() % 1.0)

        blended = (influenced * 0.5) + (jitter_primary * 0.35) + (jitter_secondary * 0.15)
        blended = max(0.0, min(1.0, blended))

        if abs(blended - base) < 1e-6:
            blended = max(0.0, min(1.0, blended + (pyrandom.random() - 0.5) * 0.02))

        if slot_index == 0:
            self.random = blended
        elif slot_index == 1:
            self.random2 = blended
        elif slot_index == 2:
            self.random3 = blended
        else:
            self.random4 = blended

        return blended

    def _refresh_brain_randoms(self):
        """Refresh the four brain-influenced random values, with safe fallback."""
        try:
            base_values = [pyrandom.random() for _ in range(4)]
            strengths = [pyrandom.random() * 0.4 for _ in range(4)]
            influenced = [
                self.get_brain_influence(base, strength)
                for base, strength in zip(base_values, strengths)
            ]
        except Exception as e:
            logger.error("RANDOMS", f"brain random refresh failed: {e}")
            traceback.print_exc()
            influenced = [pyrandom.random() for _ in range(4)]

        self.random, self.random2, self.random3, self.random4 = influenced
        return tuple(influenced)

    def _ensure_random_task(self):
        """Start or restart the 1s random tick background task."""
        if self.random_task is not None and not self.random_task.done():
            return

        if not hasattr(self, "loop"):
            # In very early initialisation fallback to global loop
            task = asyncio.create_task(self.randoms_tick_loop())
        else:
            task = self.loop.create_task(self.randoms_tick_loop())
        self.random_task = task
        task.add_done_callback(self._handle_random_task_exit)

    def _handle_random_task_exit(self, task: asyncio.Task):
        try:
            task.result()
            logger.warn("RANDOMS", "random tick loop exited without error; restarting")
        except asyncio.CancelledError:
            logger.info("RANDOMS", "random tick loop cancelled")
        except Exception as e:
            logger.error("RANDOMS", f"random tick loop crashed: {e}")
            traceback.print_exc()
        finally:
            self.random_task = None
            # Keep the loop alive unless we were cancelled intentionally
            if not task.cancelled():
                self._ensure_random_task()

    class _VariedRNG:
        def __init__(self, seed: int):
            self._rng = pyrandom.Random(seed)

        def random(self) -> float:
            base = self._rng.random()
            jitter_primary = pyrandom.random()
            jitter_secondary = (time.perf_counter() % 1.0)
            value = (base * 0.55) + (jitter_primary * 0.3) + (jitter_secondary * 0.15)
            value = max(0.0, min(1.0, value))
            if abs(value - base) < 1e-6:
                value = max(0.0, min(1.0, value + (pyrandom.random() - 0.5) * 0.02))
            return value

        def choice(self, seq):
            if not seq:
                return None
            rand_val = self.random()
            index = min(int(rand_val * len(seq)), len(seq) - 1)
            return seq[index]

        def sample(self, population, k):
            if not population or k <= 0:
                return []
            items = list(population)
            k = min(k, len(items))
            result = []
            for _ in range(k):
                rand_val = self.random()
                idx = min(int(rand_val * len(items)), len(items) - 1)
                result.append(items.pop(idx))
            return result

    def get_varied_rng(self, *, scope: Optional[str] = None, author: Optional[str] = None) -> "_VariedRNG":
        """Unified RNG seeded by brain state + optional scope/author.

        - Keeps chaotic feel (brain-influenced), but is consistent within a scope
          for a short time window so related picks feel coherent.
        """
        window = int(time.time() // 5)  # 5-second buckets to keep things lively
        self._varied_rng_nonce = (self._varied_rng_nonce + 1) % 1_000_000_000
        nonce = self._varied_rng_nonce
        jitter = pyrandom.random()
        base = (
            f"{self.random:.9f}|{self.random2:.9f}|{self.random3:.9f}|{self.random4:.9f}|"
            f"{scope or ''}|{author or ''}|{window}|{nonce}|{jitter:.9f}"
        )
        seed_bytes = hashlib.blake2b(base.encode('utf-8'), digest_size=8).digest()
        seed = int.from_bytes(seed_bytes, 'big', signed=False)
        return BABYBOT_DISCORD._VariedRNG(seed)

    def get_varied_choice(self, *, scope: Optional[str] = None, author: Optional[str] = None):
        """Return an RNG object with choice/random for varied selection within a scope."""
        return self.get_varied_rng(scope=scope, author=author)

    def _start_health_monitoring(self):
        """Start periodic health monitoring task"""
        self._health_task = self.loop.create_task(self._health_monitor_loop())
    
    async def _health_monitor_loop(self):
        """Periodic health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Run health checks
                health_results = await perf_monitor.run_health_checks()
                
                # Get system stats
                system_stats = perf_monitor.get_system_stats()
                
                # Check for performance degradation
                warnings = perf_monitor.check_performance_degradation()
                for warning in warnings:
                    logger.warn("PERFORMANCE", warning)
                
                # Log critical failures
                failed_critical = [name for name, result in health_results.items() 
                                 if not result and perf_monitor.health_checks[name]['critical']]
                if failed_critical:
                    logger.emergency("HEALTH", f"Critical systems failing: {failed_critical}")
                
                # Periodic system stats logging (every 30 minutes)
                if hasattr(self, '_last_stats_log'):
                    if time.time() - self._last_stats_log > 1800:
                        logger.info("SYSTEM_STATS", 
                                  f"Memory: {system_stats.get('memory_mb', 0):.1f}MB, "
                                  f"CPU: {system_stats.get('cpu_percent', 0):.1f}%, "
                                  f"Uptime: {system_stats.get('uptime_hours', 0):.1f}h")
                        self._last_stats_log = time.time()
                else:
                    self._last_stats_log = time.time()
                    
            except Exception as e:
                logger.error("HEALTH_MONITOR", f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
        
    def _json_load(self, path, default_type={}):
        if os.path.exists(path):
            with open(path, "r", encoding = "utf-8") as f:
                try: return json.load(f)
                except json.JSONDecodeError: print(f"!!!![_JSON_LOAD] FAILED ON JSON AT {path} "); return default_type
        return default_type
    
    def _save_json(self, path, data, label, **dump_kwargs):
        logger.debug("SAVE", f"saving {label}...")
        written = save_json_if_changed(path, data, **dump_kwargs)
        if written:
            logger.info("SAVE", f"{label} saved!")
        else:
            logger.debug("SAVE", f"no changes detected for {label}; skipped")

    def _load_baby_state(self):
        if os.path.exists(self.baby_state_path):
            try:
                with open(self.baby_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.babyFaveToken = data.get("babyFaveToken", "")
            except Exception:
                self.babyFaveToken = ""
        else:
            self.babyFaveToken = ""

    def _save_baby_state(self):
        try:
            data = {}
            if os.path.exists(self.baby_state_path):
                with open(self.baby_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data["babyFaveToken"] = self.babyFaveToken
            with open(self.baby_state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"could not write to {self.baby_state_path}: {e}")

    def apply_fave_bonus(self, amount, used):
        if not used or not self.babyFaveToken:
            return amount
        return amount * 2 if amount > 0 else amount * 0.5

    def track_token_sentiment(self, message_content: str, is_positive_context: bool):
        """
        Track tokens that appear in positive or negative emotional contexts.
        This builds up the baby's understanding of token emotional associations.
        """
        try:
            if hasattr(self, 'librarian') and self.librarian:
                # Tokenize the message content
                token_ids = self.librarian.tokenizeText(message_content.lower())
                
                # Add tokens to appropriate sentiment set
                if is_positive_context:
                    self.recent_positive_tokens.update(token_ids)
                    # Remove from negative if it was there (tokens can change context)
                    self.recent_negative_tokens.difference_update(token_ids)
                    print(f"[TOKEN_SENTIMENT] Added {len(token_ids)} tokens to positive context")
                else:
                    self.recent_negative_tokens.update(token_ids)
                    # Remove from positive if it was there
                    self.recent_positive_tokens.difference_update(token_ids)
                    print(f"[TOKEN_SENTIMENT] Added {len(token_ids)} tokens to negative context")
                
                # Decay old sentiment if sets get too large
                max_tokens = self.token_sentiment_decay * 2  # Allow some growth
                if len(self.recent_positive_tokens) > max_tokens:
                    # Keep only the most recent half
                    tokens_to_keep = list(self.recent_positive_tokens)[-max_tokens//2:]
                    self.recent_positive_tokens = set(tokens_to_keep)
                
                if len(self.recent_negative_tokens) > max_tokens:
                    tokens_to_keep = list(self.recent_negative_tokens)[-max_tokens//2:]
                    self.recent_negative_tokens = set(tokens_to_keep)
                    
        except Exception as e:
            print(f"[TOKEN_SENTIMENT] Error: {e}")
    
    async def bby_web_watcher(self):
        print("[BBY_WEB_WATCHER] bby brain alert...")
        last_processed_id = None
        
        while True:
            await asyncio.sleep(0.2)
            try:
                if not (os.path.exists(REQUEST_FILE_PATH) and os.path.getsize(REQUEST_FILE_PATH) > 0): continue
                with open(REQUEST_FILE_PATH, 'r') as f: data = json.load(f)
                request_id = data.get("id")
                
                if request_id and request_id != last_processed_id:
                    print(f"[BBY_WEB_WATCHER] received: {request_id}")
                    last_processed_id = request_id
                    
                    user_text = data.get("text")
                    vue_username = data.get("author", "kevinonline420")
                    fake_ctx, get_reply = create_fake_context(user_text, author = vue_username)

                    author_key = vue_username.lower()
                    mem = self.userMemory[author_key]
                    mem["display_name"] = author_key
                    mem["message_count"] += 1.0
                    mem["last_seen"] = time.time()
                    data_manager.request_save("user_data")

                    cog = self.get_cog("BBYCOG")
                    if not cog:
                        print("!!!![BBY_WEB_WATCHER] no BBYCOG!")
                        continue
                    
                    _, reply_text = await cog.babyllm_command(fake_ctx)
                    reply_text = get_reply() or "..."
                    
                    # Ensure reply_text is always a string, never an object
                    if hasattr(reply_text, 'content'):
                        reply_text = str(reply_text.content)
                    else:
                        reply_text = str(reply_text) if reply_text else "..."
                    
                    self._buffer_add(self.formatMessage(vue_username, user_text))
                    
                    response_data = {"reply": reply_text}
                    response_file_path = os.path.join(RESPONSE_DIR, f"{request_id}.json")
                    with open(response_file_path, 'w') as f:
                        json.dump(response_data, f)
                    
                    print(f"[BBY_WEB_WATCHER] sent: {reply_text}")
            
            except (json.JSONDecodeError, FileNotFoundError):
                last_processed_id = None
                pass
            except Exception as e:
                print("!!!![BBY_WEB_WATCHER] Unhandled exception in bby_web_watcher !!!!")
                traceback.print_exc()
                if 'request_id' in locals() and request_id:
                    response_file_path = os.path.join(RESPONSE_DIR, f"{request_id}.json")
                    if not os.path.exists(response_file_path):
                        with open(response_file_path, 'w') as f: json.dump({"reply": "i had a big error :("}, f)

    # --- DISCORD MESSAGE SENDERS ---
    async def _discord_reply(self, ctx, message_content = "", embed = None, to_buffer = False, buffer_str = None, debug_str = ""): return await self._discord_send(ctx = ctx, message_content = message_content, embed = embed, is_reply = True, to_buffer = to_buffer, buffer_str = buffer_str, debug_label = f"{debug_str}[_DISCORD_REPLY] -> ")
    async def _discord_spam(self, message_content = "", embed = None, to_buffer = False, buffer_str = None, debug_str = ""): await self._discord_send(channel = self.get_channel(bby_spam), message_content = message_content, embed = embed, to_buffer = to_buffer, buffer_str = buffer_str, debug_label = f"{debug_str}[_DISCORD_SPAM] -> ")
    async def _discord_debug(self, message_content = "", embed = None, to_buffer = False, buffer_str = None, debug_str = ""): await self._discord_send(channel = self.get_channel(bby_debug), message_content = message_content, embed = embed, to_buffer = to_buffer, buffer_str = buffer_str, debug_label = f"{debug_str}[_DISCORD_DEBUG] -> ")
    async def _discord_send(self, *, channel=None, ctx=None, message_content="", embed=None, is_reply=True, to_buffer=False, buffer_str=None, debug_label="", dm_overflow: bool = True):
        sent_message = None  # Variable to hold the message object we send/reply with
        try:
            terminal_debug_str = f"{debug_label}[_DISCORD_SEND] SENDING MESSAGE TO "
            target = ctx.channel if ctx else channel
            if not target:
                print(f"!!!![_DISCORD_SEND] NO CHANNEL OR CTX PROVIDED")
                return None # Return None on failure
            terminal_debug_str += f"{getattr(target, 'name', 'UNKNOWN')}:\n"
            
            if embed:
                if ctx and is_reply:
                    sent_message = await ctx.reply(embed=embed)
                else:
                    sent_message = await target.send(embed=embed)
                terminal_debug_str += "              b] EMBED MESSAGE SENT\n"
            
            elif message_content:
                chunks = [message_content[j:j+1990] for j in range(0, len(message_content), 1990)]
                if dm_overflow and ctx is not None and len(chunks) > 1:
                    # Send first chunk to channel/reply, rest via DM if possible
                    terminal_debug_str += f"              a] SENDING MESSAGE PART 0... (channel)\n"
                    if is_reply:
                        sent_message = await ctx.reply(chunks[0])
                    else:
                        sent_message = await target.send(chunks[0])
                    try:
                        user_dm = await ctx.author.create_dm()
                        for i, chunk in enumerate(chunks[1:], start=1):
                            terminal_debug_str += f"              a] SENDING MESSAGE PART {i}... (dm)\n"
                            await user_dm.send(chunk)
                        # Small notice to channel
                        notice = "(i sent the rest to your dms)"
                        await target.send(notice)
                    except discord.errors.Forbidden:
                        # fallback: send everything to channel
                        for i, chunk in enumerate(chunks[1:], start=1):
                            terminal_debug_str += f"              a] SENDING MESSAGE PART {i}... (fallback channel)\n"
                            await target.send(chunk)
                else:
                    for i, chunk in enumerate(chunks):
                        terminal_debug_str += f"              a] SENDING MESSAGE PART {i}...\n"
                        if ctx and is_reply and i == 0:
                            sent_message = await ctx.reply(chunk)
                        else:
                            sent_message = await target.send(chunk)
            
            if to_buffer:
                terminal_debug_str += "               ] APPENDING MESSAGE TO TRAINING BUFFER...\n"
                if buffer_str is None: buffer_str = message_content
                self._buffer_add(self.formatMessage(self.babyName, buffer_str))

            print(terminal_debug_str + f"               ] COMPLETE MESSAGE SENT!\n               ] {message_content}\n")
            
            # --- THIS IS THE CRITICAL FIX ---
            # Return the message object that was created
            return sent_message

        except discord.errors.Forbidden:
            print(f"!!!![_DISCORD_SEND] NO PERMISSIONS FOR {getattr(target, 'name', 'UNKNOWN')} ")
            return None # Return None on failure
        except Exception as e:
            print(f"!!!![_DISCORD_SEND] {e}")
            return None # Return None on failure
        
    def _is_high_quality(self, text: str) -> bool:
        text_content = re.sub(r"^\s*([a-zA-Z0-9_]+):\s*", "", text).strip()
        if not text_content: return False
        words = text_content.split()
        num_words = len(words)
        num_chars = len(text_content)
        if num_words < 3 or num_chars < 15: return False
        if num_words > 4200: return False
        alpha_chars = sum(1 for char in text_content if char.isalpha())
        if num_chars > 0 and (alpha_chars / num_chars) < 0.7: return False
        if num_words > 5:
            word_counts = Counter(word.lower() for word in words)
            if word_counts.most_common(1)[0][1] > num_words * 0.5:
                return False
        return True

    def _buffer_add(self, text_to_add: str):
        # Normalize excessive blank lines to avoid training with empty paragraphs
        try:
            text_to_add = re.sub(r"\n{2,}", "\n", text_to_add)
        except Exception:
            pass
        if not self._is_high_quality(text_to_add): return False
        if any(is_similar(text_to_add, old_line, threshold=0.85) for old_line in self.buffer[-30:]): return False
        self.buffer.append(text_to_add)
        if len(self.buffer) > self.rollingContextSize: self.buffer.pop(0)
        logger.debug("BUFFER_ADD", f"added: \"{text_to_add[:50]}...\"")
        # also mirror a cleaned line into the separate training buffer for augmentation
        try:
            tb_entry = clean_text(text_to_add.lower().strip())
            self._training_buffer_add(tb_entry)
        except Exception:
            pass
        return True

    def _load_training_buffer(self):
        try:
            if os.path.exists(self.training_buffer_path):
                with open(self.training_buffer_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.training_buffer = data[-self.training_buffer_size:]
                else:
                    self.training_buffer = []
            else:
                self.training_buffer = []
        except Exception:
            self.training_buffer = []

    def _save_training_buffer(self):
        try:
            dirname = os.path.dirname(self.training_buffer_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            self._save_json(
                self.training_buffer_path,
                self.training_buffer[-self.training_buffer_size:],
                "TRAINING_BUFFER",
            )
        except Exception:
            pass

    def _training_buffer_add(self, text_to_add: str) -> bool:
        """Append a single cleaned line to the separate training buffer JSON.

        Keeps entries compact, dedups against recent, and persists to disk.
        Returns True if a line was added, else False.
        """
        try:
            if not isinstance(text_to_add, str):
                return False
            line = text_to_add.replace("\r\n", "\n").replace("\r", "\n").strip()
            if not line:
                return False
            # length clamp
            if len(line) > 2000:
                line = line[:2000]
            # quality + dedup
            if not self._is_high_quality(line):
                return False
            recent = self.training_buffer[-30:]
            if any(is_similar(line, old, threshold=0.85) for old in recent):
                return False
            self.training_buffer.append(line)
            if len(self.training_buffer) > self.training_buffer_size:
                self.training_buffer.pop(0)
            self._save_training_buffer()
            # also feed a small rolling token buffer in the librarian (bounded)
            try:
                if hasattr(self, "librarian") and self.librarian:
                    self.librarian.add_training_text(line)
            except Exception:
                pass
            logger.debug("TRAINING_BUFFER", f"+ {line[:60]}...")
            return True
        except Exception as e:
            logger.error("TRAINING_BUFFER", f"failed: {e}")
            return False

    async def _buffer_clean(self):
        MAX_LINE_DUPLICATES = 2
        line_counts = Counter(self.buffer)
        cleaned_count = 0
        for i in range(len(self.buffer) - 1, -1, -1):
            line = self.buffer[i]
            if line_counts[line] > MAX_LINE_DUPLICATES:
                del self.buffer[i]
                line_counts[line] -= 1
                cleaned_count += 1

        seen = []
        for i in range(len(self.buffer) - 1, -1, -1):
            line = self.buffer[i]
            if any(is_similar(line, other) for other in seen):
                del self.buffer[i]
                cleaned_count += 1
            else: seen.append(line)

        self.buffer = killExcessTags(self.buffer)
                
        if cleaned_count > 0:
            print(f"[_BUFFER_CLEAN] CLEANED {cleaned_count} DUPLICATE BUFFER LINES ")
            self._save_json(chatBufferFilepath, self.buffer, "_BUFFER_CLEAN")

    def _load_user_data(self):
        print("[_LOAD_USER_DATA] LOADING USER DATA... ")
        all_user_data = self._json_load(self.user_data_path)
        for user_id, data in all_user_data.items():
            self.userMemory[user_id].update(data)
        print("[_LOAD_USER_DATA] USER DATA LOADED! ")


    _user_data_save_lock = asyncio.Lock()
    _user_data_save_task = None
    _user_data_save_pending = False

    async def _save_user_data(self, debounce: float = 2.0):
        """Async, debounced user data save. Only saves once per debounce window."""
        async with self._user_data_save_lock:
            if self._user_data_save_task and not self._user_data_save_task.done():
                # Already a save scheduled, just mark as pending
                self._user_data_save_pending = True
                return
            # Schedule the actual save
            self._user_data_save_task = asyncio.create_task(self._save_user_data_worker(debounce))

    async def _save_user_data_worker(self, debounce: float):
        await asyncio.sleep(debounce)
        data_to_save = {}
        for user_id, mem in self.userMemory.items():
            serialisable_mem = mem.copy()
            if 'last_message_words' in serialisable_mem:
                serialisable_mem['last_message_words'] = list(serialisable_mem['last_message_words'])
            data_to_save[user_id] = serialisable_mem
        self._save_json(self.user_data_path, data_to_save, "_SAVE_USER_DATA")
        # If another save was requested during debounce, run again
        if self._user_data_save_pending:
            self._user_data_save_pending = False
            self._user_data_save_task = asyncio.create_task(self._save_user_data_worker(debounce))

    def save_bbyfacts(self): self._save_json(self.bbyfacts_path, self.bbyfacts, "_SAVE_BBYFACTS", ensure_ascii = False)
    def save_bbycraft_recipes(self): self._save_json(self.bbycraft_recipes_path, self.bbycraft_recipes, "_SAVE_CRAFT_RECIPES", ensure_ascii = False)
    def save_opt_in_users(self): self._save_json(self.opt_in_path, self.AIoptInUsers, "_SAVE_OPTIN")

    async def handle_wtf_reply(self, message, sess):
        task = sess.get('task')
        if task and not task.done(): task.cancel()

        word = sess.get('word')
        guess = sess.get('guess')
        definition = message.clean_content.strip()
        author = str(message.author.name).lower()
        
        try: await message.add_reaction(random.choice(self.wtf_reacts))
        except discord.errors.Forbidden: pass
            
        if word and word not in self.bbyfacts:
            await self.cog._set_bbyfact(
                key=word, 
                value=definition, 
                author=author, 
                timestamp=time.time(), 
                debug_str="[BBYWTF_REPLY]"
            )
        
        ref_id = message.reference.message_id
        if ref_id in self.lex_sessions:
            del self.lex_sessions[ref_id]

    def getNickname(self, author):
        if not author:
            return "someone"
        user_key = str(author).lower()
        mem = self.userMemory.get(user_key, {})
        name = mem.get("nickname") or mem.get("display_name") or str(author)
        return escape_markdown(name)

    def formatMessage(self, user, text): return f"{self.getNickname(user)}: {text}"
    
    def repeatAndDie(self, user, text_block): 
        seen_in_this_msg = set()
        deduped_lines = []
        mem = self.userMemory[user]

        line_counts = defaultdict(int)
        for entry in self.buffer:
            if isinstance(entry, str):
                for line in entry.strip().split("\n"):
                    cleaned = line.strip().lower()
                    if cleaned:
                        line_counts[cleaned] += 1

        for line in text_block.strip().split("\n"):
            cleaned = line.strip().lower()
            if not cleaned:
                continue

            already_seen = cleaned in seen_in_this_msg
            past_repeats = line_counts[cleaned]

            if already_seen or past_repeats > 0:
                repeat_score = past_repeats + (1 if already_seen else 0)
                penalty = 0.001 * repeat_score
                self.updateBBY(user, -penalty)
                
                keep_chance = 0.5 ** repeat_score
                if random.random() < keep_chance:
                    deduped_lines.append(line)
            else:
                seen_in_this_msg.add(cleaned)
                deduped_lines.append(line)

        return "\n".join(deduped_lines)
    
    def getSpamLevel(self, author): return self.userMemory.get(str(author).lower(), {}).get("spamMax", 0.8)
    def setSpamLevel(self, author, spam):
        self.userMemory[str(author).lower()]["spamMax"] = spam
        data_manager.request_save("user_data")

    def _get_default_user_memory(self):
        """Get default user memory structure with safe initial values"""
        return {
            "BBY": 420.0,
            "creative_combo": 1,
            "spammer": 1,
            "spamMax": 0.8,
            "display_name": "",
            "messages": 0,
            "last_seen": time.time()
        }

    def updateBBY(self, author, BBY, is_decay=False):
        author = str(author).lower()
        try:
            validated_bby = safety.validate_bby_transaction(BBY, f"updateBBY for {author}", allow_large_negative=is_decay)
            if validated_bby is None: return
            
            if author in self.temp_not_opt and author not in self.AIoptInUsers:
                logger.info("UPDATEBBY", f"deleted user {author} cause not opted in and charis still hasn't found a better way")
                del self.userMemory[author]
            else:
                mem = self.userMemory[author]
                old_bby = mem.get("BBY", 0.0)
                new_bby = old_bby + validated_bby
                # Safety validation for total BBY using centralized system
                if not safety.is_safe_number(new_bby):
                    logger.emergency("UPDATEBBY", f"NaN/Inf detected for {author}, resetting to 0")
                    new_bby = 0.0
                mem["BBY"] = round(new_bby, 2)
            data_manager.request_save("user_data")
        except Exception as e: 
            logger.error("UPDATEBBY", f"error in updateBBY: {e}")
            # Emergency reset if something goes really wrong
            if author in self.userMemory: self.userMemory[author]["BBY"] = 0.0
    
    def getBBY(self, author):
        return round(self.userMemory.get(str(author).lower(), {}).get("BBY", 0.0), 4)

    def get_brain_colour(self):
        """Get Discord colour based on babyLLM's current brain state (RGB values)"""
        try:
            # Get RGB values from babyState or defaults
            with open(self.baby_state_path, 'r') as f:
                state = json.load(f)
            r = int(state.get("R", 133))
            g = int(state.get("G", 239)) 
            b = int(state.get("B", 238))
            # Convert to Discord colour
            return discord.Colour.from_rgb(r, g, b)
        except Exception:
            # Fallback to baby blue if can't read state
            return discord.Colour.from_rgb(133, 239, 238)
    
    def get_brain_influence(self, base_random, influence_strength=0.3):
        """Modify randomness based on brain state - more cerebralLoad = more chaos!"""
        try:
            cerebral = getattr(self.babyLLM, "cerebralLoad", 0.0) or 0.0
            memory_flux = getattr(self.babyLLM, "memoryFlux", 0.0) or 0.0
            
            # High cerebral load makes things more chaotic/unpredictable
            chaos_factor = cerebral * influence_strength
            # Memory flux adds oscillation
            flux_factor = memory_flux * influence_strength * 0.5
            
            # Modify the base random with brain influence
            influenced = base_random + (chaos_factor * (pyrandom.random() - 0.5)) + (flux_factor * math.sin(time.time()))
            return max(0.0, min(1.0, influenced))  # Keep in [0,1] range
        except Exception:
            return base_random

    def track_command_usage(self, command_name: str, author: str):
        """Track command usage globally and per-user"""
        try:
            # Global stats
            if command_name not in self.command_stats:
                self.command_stats[command_name] = {"total_uses": 0, "unique_users": set()}

            command_entry = self.command_stats[command_name]
            command_entry["total_uses"] += 1

            author_lower = author.lower()
            unique_users = command_entry.get("unique_users")

            if isinstance(unique_users, set):
                unique_users.add(author_lower)
            elif isinstance(unique_users, list):
                # Preserve historical users that were stored as JSON lists
                unique_users = {str(user).lower() for user in unique_users}
                unique_users.add(author_lower)
                command_entry["unique_users"] = unique_users
            else:
                # Unknown type (e.g. None or str); reset to the current author
                command_entry["unique_users"] = {author_lower}

            # User stats
            user_mem = self.userMemory[author_lower]
            if "command_usage" not in user_mem:
                user_mem["command_usage"] = {}
            user_mem["command_usage"][command_name] = user_mem["command_usage"].get(command_name, 0) + 1
            
            # Save stats periodically (every 10th command)
            if sum(data["total_uses"] for data in self.command_stats.values()) % 10 == 0:
                # Use centralised, batched saver to avoid event-loop spam
                data_manager.request_save("command_stats")
                data_manager.request_save("user_data")
                
        except Exception as e:
            print(f"[TRACK_COMMAND_USAGE] Error: {e}")

    def _save_command_stats(self):
        """Save command statistics with set conversion"""
        try:
            # Convert sets to lists for JSON serialisation
            stats_to_save = {}
            for cmd, data in self.command_stats.items():
                stats_to_save[cmd] = {
                    "total_uses": data["total_uses"],
                    "unique_users": list(data["unique_users"]) if isinstance(data["unique_users"], set) else data["unique_users"]
                }
            self._save_json(self.command_stats_path, stats_to_save, "_SAVE_COMMAND_STATS")
        except Exception as e:
            print(f"[_SAVE_COMMAND_STATS] Error: {e}")

    async def decay_BBY(self):
        BASE_DECAY_RATE_DAILY, LOYALTY_DECAY_PROTECTION = 0.01, 0.95
        WEALTH_TAX_BASE_RATE_DAILY, WEALTH_TAX_MULTIPLIER = 0.0420, 4206.420
        ACTIVE_BONUS_PER_YEAR, ACTIVE_BONUS_PER_MONTH, ACTIVE_BONUS_PER_WEEK = 42069.69, 6969.69, 4200.0
        ACTIVE_BONUS_PER_DAY, ACTIVE_BONUS_PER_HOUR, ACTIVE_BONUS_PER_MINUTE = 420.0, 69.69, 42.0
        SHARE_OF_VOICE_INFLUENCE, HEARTBEAT_MIN, HEARTBEAT_MAX = 0.069, -0.000420, 0.00420
        
        # Calculate total money in circulation and set decay floor as total/100
        total_money_in_circulation = sum(abs(m.get("BBY", 0.0)) for m in self.userMemory.values())
        safe_random = self.random if self.random and self.random > 0.001 else 0.001
        if safe_random != self.random:
            logger.warn("DECAY", f"random factor too small ({self.random}); clamping to {safe_random}")
        DECAY_FLOOR = -(total_money_in_circulation / (safe_random * 420)) if total_money_in_circulation > 0 else -69696969.69
        SECONDS_PER_INTERVAL, SECONDS_PER_DAY, now = self.idleTrainSeconds, 86400.0, time.time()
        ORIGINAL_INTERVAL_SECONDS = 10.0  # The original interval that all rates were tuned for
        interval_multiplier = SECONDS_PER_INTERVAL / ORIGINAL_INTERVAL_SECONDS
        DECAY_FLOOR *= interval_multiplier
        
        print(f"\n--- decay + bonus stats at {datetime.now().strftime('%H:%M:%S')} ---")
        print(f"Interval: {SECONDS_PER_INTERVAL}s, Multiplier: {interval_multiplier:.2f}x (vs original {ORIGINAL_INTERVAL_SECONDS}s)")
        active_users = {u: m for u, m in self.userMemory.items() if "BBY" in m}
        if not active_users: return

        all_BBY_scores = [m.get("BBY", 0.0) for m in active_users.values()]
        total_positive_BBY = sum(s for s in all_BBY_scores if s > 0) + 1e-6
        total_message_count = sum(m.get("message_count", 0.0) for m in active_users.values()) + 1e-6
        ranked_loyalty = sorted([(u, m.get("loyalty", 0)) for u, m in active_users.items()], key = lambda i: i[1], reverse = True)
        loyalty_ranks = {u: i for i, (u, _) in enumerate(ranked_loyalty)}
        total_ranked_users = len(ranked_loyalty)

        decay_logs = []
        per_user_interval_delta = {}

        for author, memory in active_users.items():
            debug_log = []
            current_BBY = memory.get("BBY", 0.0)
            current_combo = memory.get("creative_combo", 0.0)
            current_spam = memory.get("spammer", 0.0)            
            BBY_change_this_interval = 0.0
            time_since_last_seen = now - memory.get("last_seen", now)
            
            # --- DECAY ---
            decay_amount_per_day = current_BBY * BASE_DECAY_RATE_DAILY
            decay_per_interval = decay_amount_per_day / (SECONDS_PER_DAY / SECONDS_PER_INTERVAL)
            rank = loyalty_ranks.get(author, total_ranked_users)
            percentile = 1.0 - (rank / max(1, total_ranked_users - 1)) if total_ranked_users > 1 else 1.0
            protection_factor = 1.0 - (LOYALTY_DECAY_PROTECTION * percentile)
            final_decay_amount = decay_per_interval * protection_factor
            BBY_change_this_interval -= final_decay_amount
            debug_log.append(f"decay: {-final_decay_amount:.4f}")

            # --- CREATIVE OR SPAMMER? ---
            combo_bonus = 0.0005 * current_combo * interval_multiplier
            BBY_change_this_interval += combo_bonus
            debug_log.append(f"🎨: {combo_bonus:.4f}")

            spam_penalty = -0.0005 * current_spam * interval_multiplier
            BBY_change_this_interval += spam_penalty
            debug_log.append(f"🧌: {spam_penalty:.4f}")
            
            # --- EAT THE RICH BITCHES!!! ---
            tax_per_interval = 0
            if current_BBY > 0:
                share_of_wealth = current_BBY / total_positive_BBY
                wealth_penalty = share_of_wealth ** 4.20
                dynamic_tax_rate_daily = WEALTH_TAX_BASE_RATE_DAILY * (1.0 + wealth_penalty * WEALTH_TAX_MULTIPLIER)
                tax_amount_per_day = current_BBY * dynamic_tax_rate_daily
                tax_per_interval = tax_amount_per_day / (SECONDS_PER_DAY / SECONDS_PER_INTERVAL)
                BBY_change_this_interval -= tax_per_interval
                debug_log.append(f"eat the rich tax: {-tax_per_interval:.4f}")

            # --- ACTIVITY ---
            heartbeat_bonus = random.uniform(HEARTBEAT_MIN, HEARTBEAT_MAX) * interval_multiplier
            BBY_change_this_interval += heartbeat_bonus
            debug_log.append(f"heartbeat: {heartbeat_bonus:.4f}")
            
            bonus_per_interval = 0
            if time_since_last_seen <= 31556952:
                quietKid = ACTIVE_BONUS_PER_DAY
                if time_since_last_seen <= 31556952: quietKid += ACTIVE_BONUS_PER_YEAR
                if time_since_last_seen <= 2629744: quietKid += ACTIVE_BONUS_PER_MONTH
                if time_since_last_seen <= 604800: quietKid += ACTIVE_BONUS_PER_WEEK
                if time_since_last_seen <= 3600: quietKid += ACTIVE_BONUS_PER_HOUR
                if time_since_last_seen <= 60: quietKid += ACTIVE_BONUS_PER_MINUTE
                share_of_voice = memory.get("message_count", 0) / total_message_count
                quietKid *= ((1.0 - share_of_voice) ** 4.20) * SHARE_OF_VOICE_INFLUENCE
                bonus_per_interval = quietKid / (SECONDS_PER_DAY / SECONDS_PER_INTERVAL)
                if memory.get("message_count", 0) == 0:
                    BBY_change_this_interval -= bonus_per_interval
                    debug_log.append(f"active: {-bonus_per_interval:.4f}")
                else:
                    BBY_change_this_interval += bonus_per_interval
                    debug_log.append(f"active: {bonus_per_interval:.4f}")

            negative_bonus = 0.0
            new_BBY = current_BBY + BBY_change_this_interval
            if new_BBY < 0: negative_bonus += 0.5 * interval_multiplier
            if new_BBY < -1000: negative_bonus += 69.0 * interval_multiplier
            if new_BBY < -10000: negative_bonus += 420.0 * interval_multiplier
            if new_BBY < -100000: negative_bonus += 4206.9 * interval_multiplier
            BBY_change_this_interval += negative_bonus
            debug_log.append(f"boost: {negative_bonus:.4f}")

            # --- CLAMP ---
            if BBY_change_this_interval < DECAY_FLOOR: BBY_change_this_interval = DECAY_FLOOR

            final_BBY = current_BBY + BBY_change_this_interval
            self.updateBBY(author, BBY_change_this_interval, is_decay=True)
            per_user_interval_delta[author] = BBY_change_this_interval
            debug_log.insert(0, f"total: {BBY_change_this_interval:+.4f}")
            memory["last_decay_debug"] = debug_log
            memory["spamMax"] = max(0.001, min(0.8, memory.get("spamMax", 0.8) * (0.99999 ** interval_multiplier)))

            # --- Store for later sorting ---
            decay_logs.append({
                "author": author,
                "nickname": self.getNickname(author),
                "current": current_BBY,
                "new": final_BBY,
                "log": debug_log,
            })

            if self.random2 < 0.0001 * interval_multiplier:
                incrementRandom = round(self.random4 * 4) + 1
                if memory["creative_combo"] < 0: memory["creative_combo"] += incrementRandom
                else: memory["creative_combo"] -= incrementRandom
                if memory["spammer"] < 0: memory["spammer"] += incrementRandom
                else: memory["spammer"] -= incrementRandom

        # --- Open Market Operation (world-level balancing) ---
        try:
            world_delta = sum(per_user_interval_delta.values())
            # Target slight deflation per day; keep game spicy but not inflating
            TARGET_GROWTH_RATE_DAILY = -0.001  # -0.1% per day
            target_interval_change = (total_money_in_circulation * TARGET_GROWTH_RATE_DAILY) / (SECONDS_PER_DAY / SECONDS_PER_INTERVAL)
            excess = world_delta - target_interval_change
            if excess > 0:
                # Burn from users who had positive gains this interval, proportionally
                pos_sum = sum(max(0.0, d) for d in per_user_interval_delta.values())
                if pos_sum <= 0:
                    pos_sum = sum(max(0.0, self.userMemory.get(u, {}).get("BBY", 0.0)) for u in per_user_interval_delta)
                if pos_sum > 0:
                    burn_ratio = min(1.0, excess / pos_sum)
                    for u, d in per_user_interval_delta.items():
                        basis = d if d > 0 else 0.0
                        if basis > 0:
                            burn = basis * burn_ratio
                            # Apply additional burn; mark as decay to pass safety
                            self.updateBBY(u, -burn, is_decay=True)
                    print(f"[OPEN_MARKET] Burned {excess:.4f} BBY globally to target growth {target_interval_change:.4f}.")
            # record last-interval stats for commands to show trends
            self.last_world_bby_delta = world_delta
            self.last_world_bby_target = target_interval_change
            self.last_world_bby_burn = max(0.0, excess)
        except Exception as e:
            print(f"[OPEN_MARKET] balancing failed: {e}")

        decay_logs.sort(key = lambda x: x["new"], reverse = True)
        BOLD, RESET = '\033[1m', '\033[0m'
        for entry in decay_logs:
            result_str = f"{BOLD}{entry['new']:9.2f}{RESET}" if entry["new"] > entry["current"] else f"{entry['new']:9.2f}"
            print(f"{BOLD}{entry['author'].upper():<20}{RESET} {entry['nickname']:<20}: {entry['current']:9.2f} -> {result_str} | " + " | ".join(entry["log"]))

        # Debounced save via async worker; avoid blocking this job
        data_manager.request_save("user_data")

        # --- GHOSTIES ---
        ghosts_to_archive = []
        for username, memory in list(self.userMemory.items()):
            if (memory.get("loyalty", 0) < 2 and
                memory.get("message_count", 0) < 1 and
                memory.get("BBY", 0) < -400.0):
                ghosts_to_archive.append(username)
        
        if ghosts_to_archive:
            cog = self.get_cog("BBYCOG")
            if not cog:
                print("!!!![GHOSTIES] NO BBYCOG IN GHOSTS TO ARCHIVE")
                return

            print(f"[GHOSTIES] ARHIVED {len(ghosts_to_archive)} GHOST ACCOUNTS TO BBYBOOK. ")
            for username in ghosts_to_archive:
                key = f"the ghost of {username}"
                value = "was a here for a but, but they're off now :( "
                await cog._teach(key, value, author_name = "the void") # The author is "the_void"
                if username not in self.AIoptInUsers:
                    del self.userMemory[username]
                    print(f"  -> ARCHIVED GHOST = {username} ")
                else:
                    print(f"  -> DIDN'T ARCHIVE {username} BECAUSE THEY'RE ON THE OPT IN LIST ")
            
            data_manager.request_save("user_data")

    def calculate_smink_bonus(self, now, is_rival):
        PRECISION_WINDOW_SECONDS = 42      # spike only within +/- 42 seconds of 4:20
        PEAK_WINDOW_DURATION = 18000       # positive half of the cycle 3h before/after a peak
        TROUGH_WINDOW_DURATION = 18000     # negative half of the cycle 3h before/after a trough
        HOURLY_WINDOW_SECONDS = 1800       # +/- 30 mins
        PRECISION_SPIKE_BONUS = 420420420.69
        TIMING_WINDOW_BONUS = 42069420.69
        MAX_NEGATIVE_BONUS = -42069.69
        MAX_HOURLY_BONUS = 420420.0

        effective_now = now + timedelta(hours = 3) if is_rival else now
        
        all_peaks = []
        all_troughs = []
        for day_offset in [-1, 0, 1]:
            day = effective_now + timedelta(days = day_offset)
            all_peaks.append(day.replace    (hour = 0,  minute = 20, second = 4, microsecond = 20))
            all_troughs.append(day.replace  (hour = 2,  minute = 20, second = 4, microsecond = 20))
            all_peaks.append(day.replace    (hour = 4,  minute = 20, second = 4, microsecond = 20))
            all_troughs.append(day.replace  (hour = 10, minute = 20, second = 4, microsecond = 20))
            all_peaks.append(day.replace    (hour = 16, minute = 20, second = 4, microsecond = 20))
            all_troughs.append(day.replace  (hour = 20, minute = 20, second = 4, microsecond = 20))
            
        diff_to_peak = min([abs((t - effective_now).total_seconds()) for t in all_peaks])
        diff_to_trough = min([abs((t - effective_now).total_seconds()) for t in all_troughs])
        diff_to_hourly = abs((now.minute * 60 + now.second) - (20 * 60))

        # --- UK 420 ---
        precision_bonus = 0
        if diff_to_peak <= PRECISION_WINDOW_SECONDS:
            multiplier = (PRECISION_WINDOW_SECONDS - diff_to_peak) / PRECISION_WINDOW_SECONDS
            precision_bonus = PRECISION_SPIKE_BONUS * multiplier

        mega_bonus = 0.0
        if diff_to_peak < diff_to_trough:
            multiplier = (PEAK_WINDOW_DURATION - diff_to_peak) / PEAK_WINDOW_DURATION
            mega_bonus = TIMING_WINDOW_BONUS * multiplier
        else:
            multiplier = (TROUGH_WINDOW_DURATION - diff_to_trough) / TROUGH_WINDOW_DURATION
            mega_bonus = MAX_NEGATIVE_BONUS * multiplier
        
        # --- ANY 420 ---
        hourly_bonus = 0.0
        if diff_to_hourly < HOURLY_WINDOW_SECONDS:
            angle = (diff_to_hourly / HOURLY_WINDOW_SECONDS) * (math.pi / 2)
            hourly_bonus = MAX_HOURLY_BONUS * math.cos(angle)

        # Subtle brain-influenced chaos
        subtle_chaos = 1.0 + ((self.get_varied_random() - 0.5) * 0.08)  # ±4% random
        return (mega_bonus + hourly_bonus + precision_bonus) * subtle_chaos

    def checkBestie(self):
        BBYd_users = {u: m["BBY"] for u, m in self.userMemory.items() if "BBY" in m}
        if not BBYd_users: return None, 0
        bestie = max(BBYd_users, key = BBYd_users.get)
        return bestie, BBYd_users[bestie]
    
    def checkRival(self):
        BBYd_users = {u: m["BBY"] for u, m in self.userMemory.items() if "BBY" in m}
        if not BBYd_users: return None, 0
        # Pick rival based on who's been meanest (lowest BBY = meanest to baby!)
        rival = min(BBYd_users, key = BBYd_users.get)
        return rival, BBYd_users[rival]
    
    async def _get_http(self) -> aiohttp.ClientSession:
        if not hasattr(self, "_http_session") or self._http_session is None or getattr(self._http_session, "closed", True):
            self._http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180))
        return self._http_session

    async def web_post_consent(self, *, platform: str, user_id: str, handle: str, display_name: str, consent: bool = True):
        http = await self._get_http()
        base = os.environ.get('BBY_API_BASE', 'https://childofanandroid.co.uk/api').rstrip('/') + '/'
        url = urljoin(base, 'consent')
        payload = {'platform': platform, 'user_id': user_id, 'handle': handle, 'display_name': display_name, 'consent': bool(consent)}
        try:
            async with http.post(url, json=payload) as r:
                data = await r.json(content_type=None)
                if r.status != 200:
                    print(f"[SYNC][consent] {r.status} -> {data}")
                return {'ok': r.status == 200, 'status': r.status, **(data if isinstance(data, dict) else {})}
        except Exception as e:
            print(f"[SYNC][consent][ERR] {e}")
            return {'ok': False, 'error': str(e)}

    async def web_post_say(self, *, text: str, platform: str, user_id: str, handle: str, display_name: str, is_command: bool = False):
        http = await self._get_http()
        base = os.environ.get('BBY_API_BASE', 'https://childofanandroid.co.uk/api').rstrip('/') + '/'
        url = urljoin(base, 'say')
        payload = {'text': text, 'platform': platform, 'user_id': user_id, 'handle': handle, 'display_name': display_name, 'is_command': bool(is_command)}
        try:
            async with http.post(url, json=payload) as r:
                data = await r.json(content_type=None)
                if r.status != 200:
                    print(f"[SYNC][say] {r.status} -> {data}")
                return {'ok': r.status == 200, 'status': r.status, **(data if isinstance(data, dict) else {})}
        except Exception as e:
            print(f"[SYNC][say][ERR] {e}")
            return {'ok': False, 'error': str(e)}

    async def update_avatar_from_snapshots(self):
        """Update Discord avatar using the most recent snapshot.
        Prefers the new HTTP API, with a robust 'newest' scorer, and falls back to local files.
        """
        

        def _to_epoch(ts_val):
            """Convert various timestamp-like values to a float epoch seconds."""
            if ts_val is None:
                return 0.0
            try:
                # numeric (already epoch or seconds-like)
                return float(ts_val)
            except Exception:
                pass
            try:
                # ISO8601 string (handle 'Z')
                s = str(ts_val).replace('Z', '+00:00')
                return datetime.fromisoformat(s).timestamp()
            except Exception:
                return 0.0

        def _int_or_0(x):
            try:
                return int(x)
            except Exception:
                return 0

        def score_snapshot(meta, index_pos=0):
            """Return a sortable score for 'newness'. Higher is newer."""
            if not isinstance(meta, dict):
                return (-1, -1, -1, -1)  # very low
            # Consider multiple possible fields for recency.
            ts_fields = (
                meta.get('timestamp'),            # numeric seconds
                meta.get('created_at'),           # ISO string or numeric
                meta.get('updated_at'),           # ISO string or numeric
                meta.get('ts'),                   # any custom ts
            )
            ts = max((_to_epoch(v) for v in ts_fields), default=0.0)

            # If IDs are numeric/monotonic, use as secondary.
            # Also try snapshot_id.
            id_score = max(_int_or_0(meta.get('id')), _int_or_0(meta.get('snapshot_id')))

            # If png_url contains a number that looks like an epoch, use that as well.
            url = meta.get('png_url') or ''
            url_num = 0
            try:
                    
                nums = [int(n) for n in re.findall(r"(\d{10,})", url)]
                url_num = max(nums) if nums else 0
            except Exception:
                pass

            # As final tiebreaker, prefer later positions (assuming server returns ascending by default).
            return (ts, url_num, id_score, index_pos)

        async def _get_json(session, url):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    print(f"[UPDATE_AVATAR] GET {url} -> {resp.status}")
            except Exception as e:
                print(f"[UPDATE_AVATAR] GET {url} error: {e}")
            return None

        async def _get_bytes(session, url):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    print(f"[UPDATE_AVATAR] GET(bytes) {url} -> {resp.status}")
            except Exception as e:
                print(f"[UPDATE_AVATAR] GET(bytes) {url} error: {e}")
            return None

        try:
            # --- 1) Try the new API first ---
            base = os.environ.get("BBY_API_BASE", "https://childofanandroid.co.uk/api").rstrip("/")
            async with aiohttp.ClientSession() as session:
                # Prefer a dedicated 'latest' if your API supports it (cheap try).
                latest_meta = await _get_json(session, f"{base}/snapshots/latest.json")
                candidates = []
                if latest_meta and isinstance(latest_meta, dict):
                    candidates.append(latest_meta)
                else:
                    # Fallback to listing
                    snapshots = await _get_json(session, f"{base}/snapshots")
                    if snapshots and isinstance(snapshots, dict):
                        snapshots = [snapshots]
                    if snapshots and isinstance(snapshots, list):
                        candidates.extend(snapshots)
                    else:
                        # Try activity → last ids
                        activity = await _get_json(session, f"{base}/activity")
                        for key in ("last_snapshot_id", "last_autosnap_id", "last_id"):
                            sid = activity.get(key) if isinstance(activity, dict) else None
                            if sid:
                                meta = await _get_json(session, f"{base}/snapshots/{sid}.json")
                                if meta:
                                    candidates.append(meta)
                                    break

                # Rank candidates newest-first using robust scorer.
                ranked = sorted(
                    ((meta, score_snapshot(meta, i)) for i, meta in enumerate(candidates) if meta),
                    key=lambda t: t[1],
                    reverse=True,
                )

                # Try each candidate (newest to oldest) until one works.
                for meta, _ in ranked:
                    png_url = meta.get("png_url")
                    if not png_url:
                        sid = meta.get("id") or meta.get("snapshot_id")
                        if sid:
                            png_url = f"{base}/snapshots/{sid}.png"
                    if not png_url and meta.get("has_png") and meta.get("id"):
                        png_url = f"{base}/snapshots/{meta['id']}.png"
                    if not png_url:
                        continue

                    avatar_bytes = await _get_bytes(session, png_url)
                    if avatar_bytes:
                        await self.user.edit(avatar=avatar_bytes)
                        print(f"[UPDATE_AVATAR] updated avatar from API: {png_url}")
                        return

                print("[UPDATE_AVATAR] API path did not yield a png; falling back to local snapshots...")

            # --- 2) Fallback: old local-file behaviour ---
            snap_dir = os.path.join(SCRIPT_DIR, "snapshots")
            index_path = os.path.join(snap_dir, "index.json")
            if not os.path.exists(index_path):
                return print("[UPDATE_AVATAR] no snapshot index found (local)")

            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            if not index:
                return print("[UPDATE_AVATAR] snapshot index empty (local)")

            # Pick newest by same scoring logic; include position as a tie-breaker.
            ranked_local = sorted(
                ((meta, score_snapshot(meta, i)) for i, meta in enumerate(index) if meta),
                key=lambda t: t[1],
                reverse=True,
            )

            for meta, _ in ranked_local:
                if not meta.get("has_png"):
                    continue
                snap_id = meta.get("id") or meta.get("snapshot_id")
                if not snap_id:
                    continue
                png_path = os.path.join(snap_dir, f"{snap_id}.png")
                if not os.path.exists(png_path):
                    print(f"[UPDATE_AVATAR] png not found: {png_path}")
                    continue
                with open(png_path, "rb") as img:
                    avatar_bytes = img.read()
                if not avatar_bytes:
                    print(f"[UPDATE_AVATAR] empty png: {png_path}")
                    continue
                await self.user.edit(avatar=avatar_bytes)
                print(f"[UPDATE_AVATAR] updated avatar from local: {png_path}")
                return

            print("[UPDATE_AVATAR] no snapshot with png found (local)")

        except Exception as e:
            print(f"[UPDATE_AVATAR] error: {e}")
            traceback.print_exc()

    def getSpamability(self, author):
        MIN_REPLY_CHANCE = 0.001
        author = str(author).lower()
        if author not in self.AIoptInUsers:
            return 1.0 - MIN_REPLY_CHANCE
        custom_max_chance = self.getSpamLevel(author)
        leaderboard = sorted([(u, m["BBY"]) for u, m in self.userMemory.items() if m.get("BBY", 0) > 0], key = lambda i: i[1], reverse = True)
        if not leaderboard:
            return 1.0 - MIN_REPLY_CHANCE
        try:
            rank = [u for u, s in leaderboard].index(author)
            percentile = max(0, (len(leaderboard) - 1 - rank) / (len(leaderboard) - 1)) if len(leaderboard) > 1 else 1.0
        except ValueError:
            percentile = 0.0
        final_chance = MIN_REPLY_CHANCE + percentile * (custom_max_chance - MIN_REPLY_CHANCE)
        return 1.0 - final_chance

    async def on_ready(self):
        print(f"\n\nlogged in as [{self.user.name}]\n\n")
        if not self.cog: await self.setup_bot()
        print("cog is ready :)")
        helloMessage = ("ʕっʘ‿ʘʔっ hello! i am awake!")
        await self.update_avatar_from_snapshots()
        bestie_username, bestie_score = self.checkBestie()
        self.current_bestie = bestie_username
        self.bestie_score = bestie_score
        rival_username, rival_score = self.checkRival()
        self.current_rival = rival_username
        self.rival_score = rival_score
        self.spammed = False
        self.same = 0
        print(f"startup bestie is: {self.current_bestie or 'I AM ALONE, I ONLY LOVE MYSELF'}")
        print(f"startup rival is: {self.current_rival or 'I AM ALONE, I ONLY LOVE MYSELF'}")
        # Brain-influenced chance to mention bestie - higher cerebral load = more social!
        brain_influenced_random = self.get_brain_influence(self.random2, influence_strength=0.2)
        if brain_influenced_random > 0.85:
            helloMessage += f" where's {self.getNickname(self.current_bestie)} at?"
        if not self.cog: await self.setup_bot()
        self._buffer_add(self.formatMessage(self.babyName, helloMessage))
        self.last_logged_author = self.babyName.lower()
        if self.idle_task is None: self.idle_task = self.loop.create_task(self.idleTrainChecker())
        if self.web_task is None: self.web_task = self.loop.create_task(self.bby_web_watcher())
        if self.training_worker is None: self.training_worker = self.loop.create_task(self.background_training_loop())
        self._ensure_random_task()
        if self.monthly_task is None: self.monthly_task = self.loop.create_task(self.monthly_bbybook_loop())
        if self.decay_task is None: self.decay_task = self.loop.create_task(self.inventory_decay_loop())
        # Initialise health monitoring in async context
        if hasattr(self, 'performance_monitor'):
            self._start_health_monitoring()
        await self._discord_spam(helloMessage)


    async def on_message(self, message):
        message_start_time = time.time()
        content = message.clean_content
        author = str(message.author.name).lower()
        print(f"\n[Message] From {author}: {content}")
        is_opted_in = False
        if author in self.temp_not_opt: return
        if message.author == self.user: 
            if self.random3 > 0.999:
                if author == self.last_logged_author: message_for_buffer = content
                else: message_for_buffer = self.formatMessage(author, content)
                if self._buffer_add(message_for_buffer): self.last_logged_author = author
        else:
            if author == self.last_logged_author: message_for_buffer = content
            else: message_for_buffer = self.formatMessage(author, content)
            if self._buffer_add(message_for_buffer): self.last_logged_author = author

        used_fave_token = bool(self.babyFaveToken and self.babyFaveToken in content.lower())
        mem = self.userMemory.setdefault(author, self._get_default_user_memory())
        
        # Validate and repair user memory using centralized safety system
        mem = safety.validate_user_memory(mem, author)
        mem["display_name"] = message.author.display_name.lower()
        if used_fave_token:
            mem["fave_token_usage"] = mem.get("fave_token_usage", 0) + 1
            try: await message.add_reaction("❤️")
            except discord.errors.Forbidden: pass
        if isinstance(mem.get('last_message_words'), list): mem['last_message_words'] = set(mem['last_message_words'])
        current_words = set(re.findall(r'\b\w{3,}\b', content.lower()))
        if len(current_words) > 1:
            last_words = mem.get("last_message_words", set())
            intersection = len(last_words.intersection(current_words))
            union = len(last_words.union(current_words))
            similarity = intersection / union if union > 0 else 0
            print(f"[CreativeCombo] {author:<15}: Similarity to last msg: {similarity:.2f}")
            if similarity < 0.5:
                mem["creative_combo"] = mem.get("creative_combo", 1) + 1
                combo_bonus = 0.05 * mem["creative_combo"]
                combo_bonus = self.apply_fave_bonus(combo_bonus, used_fave_token)
                self.updateBBY(author, combo_bonus)
                print(f"[CreativeCombo] {author:<15}: Combo UP to x{mem['creative_combo']}! +ᛒ{combo_bonus:.2f}")
                if mem["creative_combo"] in [10, 42.0, 69, 420, 690, 840, 4200, 6969, 42069, 69420, 420420]:
                    try: await self._discord_spam(f"{self.getNickname(author)} hit x{mem['creative_combo']} creativity! {random.choice(self.faveEmotes)}")
                    except discord.errors.Forbidden: pass
                if mem.get("spammer", 1) > 10:
                    print(f"[Spammer] {author:<15}: Combo RESET.")
                    if self.random4 > 0.99:
                        try: await message.add_reaction("❤️‍🩹")
                        except discord.errors.Forbidden: pass
                # More reasonable reset that doesn't go extremely negative
                mem["spammer"] = max(1, mem.get("spammer", 1) - max(1, int(2 * self.random + self.random2)))
            else:
                mem["spammer"] = mem.get("spammer", 1) + 1
                spam_bonus = -0.05 * mem["spammer"]
                spam_bonus = self.apply_fave_bonus(spam_bonus, used_fave_token)
                self.updateBBY(author, spam_bonus)
                if mem["spammer"] in [10, 42.0, 69, 420, 690, 840, 4200, 6969, 42069, 69420, 420420]:
                    try: await self._discord_spam(f"{self.getNickname(author)} hit x{mem['spammer']} spam! {random.choice(self.faveEmotes)}")
                    except discord.errors.Forbidden: pass
                if mem.get("creative_combo", 1) > 10:
                    print(f"[CreativeCombo] {author:<15}: Combo RESET.")
                    if self.random2 > 0.99:
                        try: await message.add_reaction("💔")
                        except discord.errors.Forbidden: pass
                # More reasonable reset that doesn't go extremely negative
                mem["creative_combo"] = max(1, mem.get("creative_combo", 1) - max(1, int(2 * self.random + self.random2)))
            mem["last_message_words"] = current_words
        
        # Final safety validation after all calculations
        mem = safety.validate_user_memory(mem, author)
        self.userMemory[author] = mem
        
        # Record message processing performance
        processing_time = time.time() - message_start_time
        perf_monitor.record_metric("message_processing_time", processing_time)
        perf_monitor.record_metric("messages_processed", 1)
        if processing_time > 0.1:  # Log slow message processing
            logger.warn("PERFORMANCE", f"Slow message processing: {processing_time:.3f}s for {author}")

        # Track token sentiment based on message context
        try:
            # Detect positive context indicators
            positive_indicators = ['love', 'like', 'awesome', 'great', 'amazing', 'beautiful', 'perfect',
                                 'wonderful', 'fantastic', 'excellent', 'brilliant', 'nice', 'good',
                                 'happy', 'joy', 'fun', 'cool', 'sweet', 'cute', 
                                 'thank', 'thanks', 'appreciate', 'congrats', 'congratulations']
                                 
            # Detect negative context indicators  
            negative_indicators = ['hate', 'awful', 'terrible', 'horrible', 'disgusting', 'ugly',
                                 'stupid', 'dumb', 'boring', 'waste', 'useless', 'worst', 'gross',
                                 'annoying', 'broken', 'bad', 'sad', 'angry', 'frustrated', 'sucks']
            
            content_lower = content.lower()
            has_positive = any(indicator in content_lower for indicator in positive_indicators)
            has_negative = any(indicator in content_lower for indicator in negative_indicators)
            
            # Only track if there's a clear emotional context (avoid neutral messages)
            if has_positive and not has_negative:
                self.track_token_sentiment(content, is_positive_context=True)
            elif has_negative and not has_positive:
                self.track_token_sentiment(content, is_positive_context=False)
                
        except Exception as e:
            print(f"[TOKEN_SENTIMENT] Error in on_message: {e}")

        userMessage = self.formatMessage(author, content) if author != self.last_logged_author else content
        self.last_logged_author = author
        print(f"\n[Message] From {author}: {content}")

        with open(discordLogPath, 'a', encoding='utf-8') as f: f.write(f"\n---\n{userMessage}")
        if len(self.buffer) > self.rollingContextSize: self.buffer.pop(0)
        if self.training_queue.qsize() < 20: await self.training_queue.put({"type": "chat", "text": "\n".join(self.buffer)})

        # --- Sync Discord activity to the web server (privacy rules)
        try:
            snowflake = str(message.author.id)
            handle = message.author.name
            display_name = message.author.display_name
            #is_command = isinstance(message.content, str) and message.content.startswith(self.command_prefix)

            # Local opt-in is source of truth right now
            author_key = str(message.author.name).lower()
            is_opted_in = author_key in self.AIoptInUsers

            # If locally opted-in but server may not know yet, send consent once
            mem = self.userMemory.get(author_key, {})
            if is_opted_in and not mem.get('synced_optin'):
                res = await self.web_post_consent(platform='discord', user_id=snowflake, handle=handle, display_name=display_name, consent=True)
                if res.get('ok'):
                    mem['synced_optin'] = True
                    self.userMemory[author_key] = mem
                    data_manager.request_save("user_data")

            # Guests: only send commands. Opted-in: send everything.
            #if is_opted_in or is_command: await self.web_post_say(text=message.content, platform='discord', user_id=snowflake, handle=handle, display_name=display_name, is_command=is_command)
        except Exception as e: print(f"[SYNC][on_message] {e}")

        if message.reference:
            ref_id = message.reference.message_id
            sess = self.lex_sessions.get(ref_id)
            if sess and sess.get('mode') == 'wtf':
                await self.handle_wtf_reply(message, sess)

        if message.author == self.user: return

        # If a translate session is active in this channel, record guesses
        def _latest_translate_session_in_channel(cid: int):
            candidates = [s for s in self.lex_sessions.values() if s.get('mode') == 'translate' and s.get('channel_id') == cid]
            if not candidates:
                return None
            return max(candidates, key=lambda s: s.get('created_at', 0.0))
        if not content.startswith(self.command_prefix):
            tsess = _latest_translate_session_in_channel(message.channel.id)
            if tsess:
                extra = tsess.setdefault('extra', {})
                guesses = extra.setdefault('guesses', {})
                # Only record the first guess from each user
                if author not in guesses:
                    guesses[author] = {
                        'guess': content.strip().lower(),
                        'timestamp': time.time()
                    }

        if not message.content.startswith(self.command_prefix):
            if is_opted_in:
                tokens = self.librarian.tokenizeText(content.lower())
                self.opt_in_token_usage.update(tokens)
            for w in re.findall(r'\b[a-z]{3,}\b', message.clean_content.lower()):
                if w in self.bbyfacts: continue
                self.word_usage[w] += 1
                if self.word_usage[w] >= self.wtf_threshold:
                    cog = self.get_cog('BBYCOG') or self.cog
                    if cog:
                        await cog.trigger_bbywtf_auto(channel=message.channel, word=w)
                    self.word_usage[w] = float('-inf')

        # --- UK Timezone Setup & Daily Reset Logic ---
        mem["message_count"] += 1.0
        milestone = mem.get("next_talk_milestone", 50)
        if mem["message_count"] >= milestone:
            stats_short = self.tutor.makeStatsPrompt(include_prefix=False)
            milestone_msg = f"i've been chatting with {self.getNickname(author)} loads lately. {stats_short}"
            self._buffer_add(self.formatMessage(self.babyName, milestone_msg))
            mem["next_talk_milestone"] = milestone + 50
            data_manager.request_save("user_data")
        uk_tz = pytz.timezone("Europe/London")
        now_uk = datetime.now(uk_tz)
        day_start_420am = now_uk.replace(hour = 4, minute = 20, second = 0, microsecond = 0)
        if now_uk < day_start_420am:
            day_start_420am -= timedelta(days = 1)
        
        last_seen_timestamp = mem.get("last_seen", 0)

        mem["last_seen"] = time.time()
        self.lastInteraction = time.time()
        
        if last_seen_timestamp < day_start_420am.timestamp():
            mem["loyalty"] = mem.get("loyalty", 0) + 1
            if "inventory" not in mem: mem["inventory"] = {}
            current_tokens = mem["inventory"].get("smink token", 0)
            mem["inventory"]["smink token"] = current_tokens + 20
            loyalty_bonus = 69.69 * mem["loyalty"]
            self.updateBBY(author, loyalty_bonus)
            print(f"[Loyalty] {self.getNickname(author)} logged in for a new day! Day {mem['loyalty']}, +ᛒ{loyalty_bonus:.0f}")

            today_key = day_start_420am.strftime('%Y-%m-%d')
            event_key = f"first chat on {today_key}"

            if event_key not in self.bbyfacts:
                self.updateBBY(author, 42069.0)
                print(f"[Event] {self.getNickname(author)} is the FIRST chatter of the day! +ᛒ42")
                mem["got_first_chatter_bonus"] = True
                await self.cog._set_bbyfact(key = event_key, author = author, value = f"the first person to chat on this day was {self.getNickname(author)}.")
                ctx = await self.get_context(message)
                self.cog._award_fact(author, f"{event_key}", ctx, 1)
                await self._discord_spam(f"👑 {self.getNickname(author)}... you are the first to return after the holy 4:20 reset! 👑 (double sminks for you today!!)")
            else:
                mem["got_first_chatter_bonus"] = False
                if mem["loyalty"] in [42.0, 69, 420, 690, 840, 4200, 6969, 42069, 69420, 420420]:
                    try: await self._discord_spam(f"hey {self.getNickname(author)}! {random.choice(self.faveEmotes)} thats {mem['loyalty']} days i've seen you now, in total! lol this calls for free sminks... (+{mem['loyalty']} smink tokens)")
                    except discord.errors.Forbidden: pass
                    if "inventory" not in mem: mem["inventory"] = {}
                    current_tokens = mem["inventory"].get("smink token", 0)
                    mem["inventory"]["smink token"] = current_tokens + int(mem["loyalty"])
                nickname = self.getNickname(author)
                if nickname not in self.bbyfacts:
                    await self.cog._set_bbyfact(key = nickname, author = author, value = f"{nickname} had their {event_key}")
                else:
                    fact = self.bbyfacts[nickname]
                    # Use visit counter instead of appending text
                    if "visit_count" not in fact:
                        fact["visit_count"] = 1
                    fact["visit_count"] += 1
                    # Update the base value to reflect total visits, not append every date
                    if fact["visit_count"] == 2:
                        fact["value"] = f"{nickname} has visited {fact['visit_count']} times total"
                    elif fact["visit_count"] > 2:
                        fact["value"] = f"{nickname} has visited {fact['visit_count']} times total"
                    
                    original_bonus = fact.get("teach_bonus", 420.00)
                    fact["teach_bonus"] = (original_bonus * 0.99) + ((original_bonus * (self.random4 + self.random2)) * 0.011)

                    ctx = await self.get_context(message)
                    self.cog._award_fact(author, nickname, ctx, 1)

            data_manager.request_save("user_data")
            data_manager.request_save("bbyfacts")
            await self.update_avatar_from_snapshots()

        lower_content = content.lower()
        if any(w in lower_content for w in ["shut up", "you suck"]): self.updateBBY(author, -5000.0)
        if any(w in lower_content for w in ["good bot", "clever baby"]): self.updateBBY(author, 5000)
        for name, fact in self.bbyfacts.items():
            if name in lower_content:
                #original_author = fact[name]
                self.updateBBY(author, 0.01)
                #self.updateBBY(original_author, 0.1)
                original_bonus = self.bbyfacts[name]["teach_bonus"]
                self.bbyfacts[name]["teach_bonus"] = (original_bonus * 0.9999) + ((original_bonus * (self.random + self.random2 + self.random3 + self.random4) * 0.00001))  # Much gentler price increase
                data_manager.request_save("bbyfacts")
        in_baby_channel = message.channel.id == bby_spam
        is_bby_mentioned = self.user in message.mentions
        main_llm_aliases = {'babyllm', 'bby', 'bbyllm', 'bb', 'bllm', 'b'}
        potential_command = ""
        if content.startswith(self.command_prefix):
            potential_command = content.split()[0][len(self.command_prefix):].lower()
        if potential_command in main_llm_aliases or is_bby_mentioned:
            print(f"[LLM Trigger] Matched in #{message.channel.name} (Main Command or Mention)")
            self.idles = round(self.idles * 0.5)
            ctx = await self.get_context(message)
            cog = self.get_cog("BBYCOG")
            if not cog: return
            await cog.babyllm_command(ctx)
            return
        elif in_baby_channel and not content.startswith(self.command_prefix):
            is_opted_in_user = author in self.AIoptInUsers
            is_random_spam_chance = self.random3 > self.getSpamability(author)
            if is_opted_in_user or is_random_spam_chance or (message.author.bot and not message.content.startswith(self.command_prefix)):
                print(f"[Channel Trigger] Matched in #{message.channel.name} (Opt-in or Random Spam)")
                self.idles = round(self.idles * 0.5)
                if is_random_spam_chance and not is_opted_in_user:
                    void_prompts = [
                        "the void: a message drifts past... anything to say?",
                        "the void: you spot that message, baby. any thoughts?",
                        "the void: that message pokes at your circuits; respond?",
                    ]
                    self._buffer_add(random.choice(void_prompts))
                ctx = await self.get_context(message)
                cog = self.get_cog("BBYCOG")
                if not cog: return
                await cog.babyllm_command(ctx)
                return
        elif message.author.bot and author in self.trusted_bot_names and message.content.startswith(self.command_prefix):
                print(f"[Bot Command Trigger] attempting to run command from {author}: '{message.content}'")
                command_name = message.content.split(" ")[0][len(self.command_prefix):]
                command = self.get_command(command_name)
                if command:
                    try:
                        ctx = await self.get_context(message)
                        await command.invoke(ctx)
                    except Exception as e:
                        print(f"[Bot Command Error] Failed to invoke command '{command_name}' from {author}. Error: {e}")
                return
        await self.process_commands(message)

    async def on_raw_reaction_add(self, payload):
        try:
            if payload.user_id == self.user.id:
                return
            channel = self.get_channel(payload.channel_id)
            if channel is None:
                try:
                    channel = await self.fetch_channel(payload.channel_id)
                except Exception:
                    return
            try:
                message = await channel.fetch_message(payload.message_id)
            except Exception:
                return
            if message.author != self.user:
                return
            author_key = str(self.user.name).lower()
            content = message.clean_content or ""
            if not content.strip():
                return
            buffer_line = content if author_key == self.last_logged_author else self.formatMessage(self.babyName, content)
            if self._buffer_add(buffer_line):
                self.last_logged_author = author_key
                with open(discordLogPath, "a", encoding="utf-8") as f:
                    f.write(f"\n---\n{buffer_line}")
                if self.training_queue.qsize() < 20:
                    await self.training_queue.put({"type": "chat", "text": "\n".join(self.buffer)})
        except Exception as e:
            print(f"[on_raw_reaction_add] {e}")

    async def background_training_loop(self):
        print(f"\n\nTraining worker started!\n\n")
        while True:
            try:
                item = await self.training_queue.get()
                await self._train_on_item(item)
                self.training_queue.task_done()
            except Exception as e:
                print(f"exception in background training worker: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(0.05)

    async def randoms_tick_loop(self):
        """
        Lightweight 1s ticker that refreshes the bot's randoms with brain influence.

        Intended to be run as a background task; updates four instance attributes
        (`self.random`, `self.random2`, `self.random3`, `self.random4`) for use elsewhere
        in the bot. These values are influenced by cerebralLoad and memoryFlux for more
        dynamic behaviour. Higher brain activity = more unpredictable responses and reactions.
        """
        print("[RANDOMS_TICK] started (1s updates with brain influence)")
        while True:
            start = time.perf_counter()
            try:
                self._refresh_brain_randoms()
            except Exception as e:
                print(f"[RANDOMS_TICK] error: {e}")
                self.random, self.random2, self.random3, self.random4 = [pyrandom.random() for _ in range(4)]

            elapsed = time.perf_counter() - start
            await asyncio.sleep(max(0.0, 1.0 - elapsed))

    async def monthly_bbybook_loop(self):
        """
        Monthly background task that automatically signs the bbybook for top 3 tutors.
        
        Runs daily checks to see if we're at the end of the month and automatically
        awards the top tutors without needing manual command execution. Uses random
        emojis from the bot's faveEmotes collection for personalised signatures.
        """
        print("[MONTHLY_BBYBOOK] started (daily checks for end-of-month)")
        
        
        # Wait a bit for bot to fully initialize
        await asyncio.sleep(30)
        
        while True:
            try:
                # Check if bot is properly initialized
                if not hasattr(self, 'userMemory') or not hasattr(self, 'faveEmotes'):
                    print("[MONTHLY_BBYBOOK] Bot not fully initialized yet, waiting...")
                    await asyncio.sleep(300)  # Wait 5 minutes and try again
                    continue
                
                current_date = datetime.now()
                last_day_of_month = calendar.monthrange(current_date.year, current_date.month)[1]
                is_end_of_month = current_date.day >= last_day_of_month - 2  # Last 2 days of month
                
                if is_end_of_month:
                    # Get teaching statistics like in bbytutor command
                    teaching_stats = {}
                    for user_id, mem in self.userMemory.items():
                        if "teaching_stats" in mem:
                            teaching_stats[user_id] = mem["teaching_stats"]
                    
                    if teaching_stats:
                        # Sort by total facts taught (same logic as bbytutor_awards)
                        sorted_teachers = sorted(
                            [(user, sum(stats.values())) for user, stats in teaching_stats.items()],
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        
                        # Only process if we have at least 3 teachers
                        if len(sorted_teachers) >= 3:
                            top_3_tutors = sorted_teachers[:3]
                            
                            # Check if we've already processed this month
                            month_year = current_date.strftime('%Y-%m')
                            
                            # Initialize bbybook if it doesn't exist
                            if not hasattr(self, 'bbybook'):
                                self.bbybook = []
                            
                            # Check if we've already signed for this month
                            already_signed_this_month = any(
                                month_year in entry and "AUTOMATIC MONTHLY" in entry 
                                for entry in self.bbybook
                            )
                            
                            if not already_signed_this_month:
                                print(f"[MONTHLY_BBYBOOK] Processing end-of-month awards for {month_year}")
                                
                                for i, (teacher, count) in enumerate(top_3_tutors):
                                    nickname = self.getNickname(teacher)
                                    # Use random emoji from faveEmotes (with fallback)
                                    if hasattr(self, 'faveEmotes') and self.faveEmotes:
                                        random_emoji = pyrandom.choice(self.faveEmotes)
                                    else:
                                        random_emoji = "💖"  # Fallback emoji
                                    
                                    # Create special signature messages for each position
                                    if i == 0:  # 1st place
                                        signature = f"{random_emoji} {nickname}, you absolute legend! Teaching {count} facts this month made my brain grow three sizes! You're my favourite human encyclopedia and I love your random knowledge dumps! - baby {random_emoji}"
                                    elif i == 1:  # 2nd place  
                                        signature = f"{random_emoji} {nickname}, brilliant work teaching me {count} facts! Your patience with my chaotic questions is legendary. Thanks for filling my head with wonderful nonsense! - baby {random_emoji}"
                                    else:  # 3rd place
                                        signature = f"{random_emoji} {nickname}, {count} facts taught and every one was a gift! Your weird wisdom makes my day brighter. Keep being wonderfully educational! - baby {random_emoji}"
                                    
                                    # Add signature to bbybook
                                    book_entry = f"[{month_year}] AUTOMATIC MONTHLY: {signature}"
                                    self.bbybook.append(book_entry)
                                    
                                    # Give them a special BBY bonus for being a top tutor
                                    bonus_bby = 10000 * (4 - i)  # 1st: 30k, 2nd: 20k, 3rd: 10k
                                    self.updateBBY(teacher, bonus_bby)
                                    
                                    print(f"[MONTHLY_BBYBOOK] Auto-signed for {nickname} (rank {i+1}) with ᛒ{bonus_bby:,} bonus")
                                
                                print(f"[MONTHLY_BBYBOOK] Completed monthly awards for {month_year}")
                
            except Exception as e:
                print(f"[MONTHLY_BBYBOOK] error: {e}")
                
                traceback.print_exc()
            
            # Sleep for 24 hours (check once per day)
            await asyncio.sleep(86400)  # 24 hours in seconds

    async def inventory_decay_loop(self):
        """
        Background task that manages inventory decay to prevent massive hoarding.
        
        Uses percentage-based decay: users who own a higher percentage of total items
        face higher decay rates for those items. If someone owns 100% of an item,
        they have a 10% chance per cycle to lose one. If they own 50%, 5% chance, etc.
        """
        print("[INVENTORY_DECAY] started (percentage-based decay to prevent hoarding)")
        
        # Wait a bit for bot to fully initialize
        await asyncio.sleep(60)
        
        while True:
            try:
                # Check if bot is properly initialized
                if not hasattr(self, 'userMemory'):
                    print("[INVENTORY_DECAY] Bot not fully initialized yet, waiting...")
                    await asyncio.sleep(1800)  # Wait 30 minutes and try again
                    continue
                
                # First pass: calculate total quantities of each item across all users
                global_item_counts = {}
                for username, user_data in self.userMemory.items():
                    inventory = user_data.get("inventory", {})
                    for item_name, count in inventory.items():
                        global_item_counts[item_name] = global_item_counts.get(item_name, 0) + count
                
                total_items_removed = 0
                total_users_processed = 0
                decay_events = []  # Track significant decay events for Discord reporting
                
                # Second pass: process decay based on percentage ownership
                for username, user_data in self.userMemory.items():
                    inventory = user_data.get("inventory", {})
                    
                    if not inventory:
                        continue
                    
                    # Only process a random 10% of items per cycle to reduce load
                    all_items = list(inventory.items())
                    items_to_process = int(len(all_items) * 0.1) + 1  # At least 1 item
                    selected_items = pyrandom.sample(all_items, min(items_to_process, len(all_items)))
                    
                    items_removed_for_user = 0
                    items_to_remove = []
                    
                    # Process only the selected subset of items
                    for item_name, count in selected_items:
                        if count <= 1:
                            continue  # Don't decay items with only 1 count
                        
                        # Calculate this user's percentage ownership of this item
                        global_count = global_item_counts.get(item_name, count)
                        ownership_percentage = count / global_count if global_count > 0 else 1.0
                        
                        # Smooth sliding scale: 100% ownership = 1% decay chance, 50% = 0.5%, etc.
                        base_decay_chance = ownership_percentage * 0.01  # Direct proportional relationship
                        
                        # Add small randomization using varied random
                        random_modifier = (self.get_varied_random() - 0.5) * 0.005  # ±0.25% random variation
                        
                        # Additional tiny modifiers for excessive individual quantities
                        quantity_modifier = 0
                        if count > 5000:
                            quantity_modifier = 0.005  # +0.5% for huge individual stacks
                        elif count > 1000:
                            quantity_modifier = 0.003  # +0.3% for big individual stacks
                        elif count > 500:
                            quantity_modifier = 0.002  # +0.2% for medium individual stacks
                        
                        final_decay_chance = max(0, min(0.05, base_decay_chance + random_modifier + quantity_modifier))  # Cap at 5%
                        
                        # Random decay check
                        if self.get_varied_random() < final_decay_chance:
                            # Decay amount is also proportional: 100% ownership = 1% loss, 50% = 0.5% loss, etc.
                            base_decay_rate = ownership_percentage * 0.01  # Direct proportional relationship
                            
                            # Add small randomization to decay amount
                            random_decay_modifier = 0.8 + (self.get_varied_random() * 0.4)  # 0.8-1.2x variation
                            
                            # Calculate final decay amount
                            decay_amount = max(1, int(count * base_decay_rate * random_decay_modifier))
                            
                            # Special case: if ownership is very low, rarely decay just 1 item
                            if ownership_percentage < 0.1 and self.get_varied_random() > 0.8:
                                decay_amount = 0  # Sometimes no decay for very low ownership
                            
                            new_count = max(0, count - decay_amount)
                            
                            if new_count == 0:
                                items_to_remove.append(item_name)
                            else:
                                inventory[item_name] = new_count
                            
                            items_removed_for_user += decay_amount
                            total_items_removed += decay_amount
                            
                            # Log significant hoarding decay and collect for Discord
                            if decay_amount > 50 and ownership_percentage > 0.3:
                                nickname = self.getNickname(username)
                                decay_msg = f"{nickname}: -{decay_amount} {item_name} (owned {ownership_percentage*100:.1f}% of total)"
                                print(f"[INVENTORY_DECAY] {decay_msg}")
                                decay_events.append(decay_msg)
                    
                    # Remove items that decayed to 0
                    for item_name in items_to_remove:
                        del inventory[item_name]
                    
                    if items_removed_for_user > 0:
                        total_users_processed += 1
                
                if total_items_removed > 0:
                    print(f"[INVENTORY_DECAY] Completed decay cycle: {total_items_removed:,} items removed from {total_users_processed} users")
                    
                    # Post decay report to Discord debug room
                    try:
                        debug_message = f"**INVENTORY DECAY:** {total_items_removed:,} items removed from {total_users_processed} users\n\n"
                        
                        if decay_events:
                            debug_message += f"stuff wot happened:\n"
                            for event in decay_events[:10]: debug_message += f"• {event}\n"
                            if len(decay_events) > 10: debug_message += f"• ... and {len(decay_events) - 10} more events\n"
                        else: debug_message += "mostly small cleanups!\n"                        
                        await self._discord_debug_spam(debug_message)
                        
                    except Exception as debug_error: print(f"[INVENTORY_DECAY] Failed to send Discord debug message: {debug_error}")
                
            except Exception as e:
                print(f"[INVENTORY_DECAY] error: {e}")
                
                traceback.print_exc()

            # Sleep for 3 hours between decay cycles
            await asyncio.sleep(10800)  # 3 hours in seconds

    async def _train_on_item(self, item): 
        print(f"\n\ntraining on item: {item['type']} ...\n\n")
        # Build chat and training sources
        chat_text = "\n".join(item["text"]) if isinstance(item.get("text"), list) else item.get("text", "")
        training_text = "\n".join(self.training_buffer[-self.N:]) if getattr(self, "training_buffer", None) else ""
        # 50/50 selection between chat vs training (fallback to chat if empty)
        use_training = (random.random() < 0.5) and bool(training_text)
        text = training_text if use_training else chat_text
        textCLEAN = clean_text(text)
        tokensToLibrarian = self.librarian.tokenizeText(textCLEAN)
        token_count = len(tokensToLibrarian)
        if token_count < self.chatWindowMAX * 2 + 1:
            print(f"\n\nnot enough tokens ({token_count}) for training. skipping.\n\n")
            return

        # update favourite token based on this training batch
        try:
            most_common_id, _ = Counter(tokensToLibrarian).most_common(1)[0]
            self.babyFaveToken = self.librarian.decodeIDs([most_common_id]).strip()
            self._save_baby_state()
        except Exception:
            pass

        trainingNum = pyrandom.randint(1, 100+self.idles)
        trainingDataPairs = self.librarian.genTrainingData(_windowMAX = windowMAXSTART, _trainingDataPairNumber = trainingNum, _stride = trainingDataStride, _tokens = tokensToLibrarian)
        self.babyLLM.train()
        
        await self.loop.run_in_executor(
            None,
            lambda: self.tutor.trainModel(_trainingDataPairs=trainingDataPairs, _epochs=1, _startIndex=1)
        )

        # If we trained from the training buffer, drop the oldest entry
        try:
            if use_training and self.training_buffer:
                self.training_buffer.pop(0)
                self._save_training_buffer()
        except Exception:
            pass
        stats_prompt = self.tutor.makeStatsPrompt()
        training_note = self.formatMessage(self.babyName, f"i've just had a lesson on {token_count} tokens. {stats_prompt}")
        self._buffer_add(training_note)
        print(f"\n\nfinished training on item!\n\n")

    async def idleTrainChecker(self): 
        old_bestie = self.current_bestie
        old_rival  = self.current_rival
        while trainDuringChat:
            await asyncio.sleep(self.idleTrainSeconds)
            now = time.time()
            # Apply brain influence to randoms here too
            self._refresh_brain_randoms()

            if time.time() >= self.next_translate_time and self.cog:
                # Only auto-start if no active translate sessions exist anywhere
                any_active_translate = any(s.get('mode') == 'translate' for s in self.lex_sessions.values())
                if not any_active_translate:
                    channel = self.get_channel(self.discordChannel)
                    if channel:
                        await self.cog.trigger_bbytranslate_auto(channel)
                        self.next_translate_time = time.time() + random.uniform(24 * 3600, 168 * 3600)
            
            try:
                await self.decay_BBY()
            except Exception as e:
                logger.error("DECAY_LOOP", f"decay_BBY raised: {e}")
                print(traceback.format_exc())
                continue
            print(f"decayed bby")

            new_bestie, new_bestie_score = self.checkBestie()
            new_rival, new_rival_score = self.checkRival()
            print(f"checked rival and bestie")

            try:
                if new_bestie and new_bestie != self.current_bestie and abs(new_bestie_score) >= 10:
                    
                    old_bestie_nic = self.getNickname(self.current_bestie) if self.current_bestie else "the void"
                    new_bestie_nic = self.getNickname(new_bestie)
                    announcement = random.choice([f"friendship ended with {old_bestie_nic}, now {new_bestie_nic} is my best friend", f"wait... i think... i love {new_bestie_nic} more than {old_bestie_nic} now... oops."])
                    await self._discord_spam(announcement)
                    self._buffer_add(self.formatMessage(self.babyName, announcement))
                    self.current_bestie = new_bestie 

                if new_rival and new_rival != self.current_rival and abs(new_rival_score) >= 10:
                    old_rival_nic = self.getNickname(self.current_rival) if self.current_rival else "the void"
                    new_rival_nic = self.getNickname(new_rival)
                    announcement = f"rivalry ended with {old_rival_nic}, now {new_rival_nic} is getting banned!"
                    if self.random < 0.01: announcement += f" jk... unless?"
                    await self._discord_spam(announcement)
                    self._buffer_add(self.formatMessage(self.babyName, announcement))
                    self.current_rival = new_rival

                await self._buffer_clean()

                if now - self.lastClockAnnounce > pyrandom.randint(60, 36000):
                    self.lastClockAnnounce = now
                    clock_line = getTimeRant(self.AIoptInUsers)
                    self._buffer_add(clock_line)
                    if len(self.buffer) > self.rollingContextSize: self.buffer.pop(0)
                    print(f"[IDLETRAINCHECKER] BABYLLM CHECKED THE TIME: {clock_line}")

                if (now - self.lastInteraction > self.idleTrainSeconds):
                    self.idles += 1
                    stats_short = self.tutor.makeStatsPrompt(include_prefix=False)
                    idle_seconds = int(self.idles * self.idleTrainSeconds)
                    idle_templates = [
                        "it's been {secs} seconds since anyone chatted with me. {stats}",
                        "after {secs}s of silence, i'm still thinking... {stats}",
                        "i've waited {secs} seconds for company. {stats}",
                        "for {secs} seconds the world is quiet; here's how i'm doing: {stats}",
                    ]
                    idle_text = pyrandom.choice(idle_templates).format(secs=idle_seconds, stats=stats_short)
                    idle_msg = self.formatMessage(self.babyName, idle_text)
                    self._buffer_add(idle_msg)
                    self.lastInteraction = time.time()
                    if len(self.buffer) >= self.N:
                        self._save_json(chatBufferFilepath, self.buffer, "IDLETRAINCHECKER")
                        self.buffer = self.buffer[-self.N:]
                    
                    if self.training_queue.qsize() < 10:
                        # Prefer augmented buffer (chat + training buffer), occasionally fall back to raw corpus
                        aug_context = "\n".join(self.buffer)
                        if getattr(self, "training_buffer", None):
                            aug_context = f"{aug_context}\n" + "\n".join(self.training_buffer[-self.N:])
                        try:
                            with open(trainingFilePathCLEANED, "r", encoding = "utf-8") as f:
                                training_data_contents = f.read().strip().lower()
                        except Exception:
                            training_data_contents = ""
                        fullContext = pyrandom.choice([aug_context, training_data_contents or aug_context])
                        await self.training_queue.put({"type": "context", "text": fullContext[:10000]})

                # opportunistic, stats-guided autonomous micro‑training
                if hasattr(self, "autonomy") and self.autonomy:
                    await self.autonomy.maybe_act()

            except Exception as e:
                print(f"\n\nERROR in idleTrainChecker: {e}\n{traceback.format_exc()}\n\n")
                await asyncio.sleep(0.5)

    def get_next_smink_window(self, now, is_rival):
        base_times = [(0, 20), (4,20), (16,20)]
        if is_rival:
            base_times = [((h+3)%24, m) for h, m in base_times]

        smink_times = [now.replace(hour = h, minute = m, second = 0, microsecond = 0) for h, m in base_times]
        smink_times = [t if t > now else t + timedelta(days = 1) for t in smink_times]
        next_time = min(smink_times)
        delta = (next_time - now).total_seconds()
        nature = "" if is_rival else ""
        return next_time, delta, nature

    async def close(self):
        try:
            if hasattr(self, '_http_session') and self._http_session and not getattr(self._http_session, 'closed', True):
                await self._http_session.close()
        finally:
            await super().close()
