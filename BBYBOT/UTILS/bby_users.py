# BBYBOT/UTILS/bby_users.py
import os
import json
import time
from collections import defaultdict
import pytz
from datetime import datetime, timedelta
import random
import re
import numpy as np
import traceback

from config import bbyUserDataPath, optInUsersPath
from .bby_book import BBYBook
from .bby_utils import howLongAgo

class BBYUsers:
    def __init__(self, bby_book_manager: BBYBook):
        self.user_data_path = bbyUserDataPath
        self.opt_in_path = optInUsersPath
        self.bby_book = bby_book_manager

        self.user_memory = defaultdict(self._get_default_user_memory)
        self.ai_opt_in_users = []
        
        self._load_opt_in_users()
        self._load_user_data()

        self.current_bestie = None
        self.current_rival = None
        self.update_bestie_rival()

    def _get_default_user_memory(self):
        return {
            "nickname": None, "display_name": None, "timezone": "Europe/London", "colour": None,
            "BBY": 0.0, "spamMax": 0.1, "bbybook": [],
            "wins": 0.0, "losses": 0.0, "draws": 0.0,
            "last_seen": time.time(), "message_count": 0.0, "loyalty": 1,
            "last_message_words": [], "creative_combo": 1, "spammer": 1,
            "inventory": {}, "favourites": []
        }
        
    def update_user_on_message(self, author_id, display_name, content, timestamp):
        """Updates user stats based on a new message. Ported from old on_message logic."""
        mem = self.get_user_memory(author_id)
        mem['display_name'] = display_name
        mem['last_seen'] = timestamp
        mem['message_count'] = mem.get('message_count', 0) + 1

        # Handle creative combo / spammer logic
        current_words = set(re.findall(r'\b\w{3,}\b', content.lower()))
        if len(current_words) > 1:
            # Ensure last_message_words is a set for comparison
            last_words = set(mem.get("last_message_words", []))
            intersection = len(last_words.intersection(current_words))
            union = len(last_words.union(current_words))
            similarity = intersection / union if union > 0 else 0
            
            if similarity < 0.5:
                mem["creative_combo"] = mem.get("creative_combo", 1) + 1
                self.update_bby(author_id, 0.05 * mem["creative_combo"])
                mem["spammer"] = max(1, mem.get("spammer", 1) - 1)
            else:
                mem["spammer"] = mem.get("spammer", 1) + 1
                self.update_bby(author_id, -0.05 * mem["spammer"])
                mem["creative_combo"] = 1
            # Store as list for JSON compatibility
            mem["last_message_words"] = list(current_words)
        
        # Daily login bonus logic
        try:
            uk_tz = pytz.timezone("Europe/London")
            now_uk = datetime.now(uk_tz)
            day_start_420am = now_uk.replace(hour=4, minute=20, second=0, microsecond=0)
            if now_uk < day_start_420am:
                day_start_420am -= timedelta(days=1)
            
            last_seen_timestamp = mem.get("last_seen_login_check", 0)

            if timestamp > day_start_420am.timestamp() and last_seen_timestamp < day_start_420am.timestamp():
                mem["last_seen_login_check"] = timestamp
                mem["loyalty"] = mem.get("loyalty", 0) + 1
                
                # --- THIS IS THE CRITICAL FIX ---
                # CORRECTLY access the inventory within the user's specific memory `mem`
                inventory = mem.setdefault("inventory", {})
                inventory["smink token"] = inventory.get("smink token", 0) + 20
                # -----------------------------
                
                self.update_bby(author_id, 69.69 * mem["loyalty"])
                print(f"[Daily Bonus] {self.get_nickname(author_id)} logged in for a new day! Day {mem['loyalty']}, +ᛒ{69.69 * mem['loyalty']:.0f}")

        except Exception as e:
            print(f"!!!![ERROR in update_user_on_message daily bonus]: {e}")
            traceback.print_exc()
    
    def _load_user_data(self):
        print(f"[UserDataManager] Loading user data from {self.user_data_path}...")
        if os.path.exists(self.user_data_path):
            all_user_data = self._json_load(self.user_data_path)
            for user_id, data in all_user_data.items():
                self.user_memory[user_id.lower()].update(data)
        print("[UserDataManager] User data loaded!")

    def save_user_data(self):
        print("[UserDataManager] Saving user data...")
        data_to_save = {}
        for user_id, mem in self.user_memory.items():
            serializable_mem = mem.copy()
            if 'last_message_words' in serializable_mem and isinstance(serializable_mem['last_message_words'], set):
                serializable_mem['last_message_words'] = list(serializable_mem['last_message_words'])
            data_to_save[user_id] = serializable_mem

        with open(self.user_data_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)
        print("[UserDataManager] User data saved!")

    def _load_opt_in_users(self):
        if os.path.exists(self.opt_in_path):
            with open(self.opt_in_path, "r") as f:
                self.ai_opt_in_users = [user.lower() for user in json.load(f)]
        print(f"[UserDataManager] Loaded {len(self.ai_opt_in_users)} opt-in users.")

    def save_opt_in_users(self):
        with open(self.opt_in_path, 'w', encoding='utf-8') as f:
            json.dump(self.ai_opt_in_users, f, indent = 2)
        print("[UserDataManager] Opt-in users saved!")

    def _json_load(self, path, default_type={}):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try: return json.load(f)
                except json.JSONDecodeError: print(f"!!!![UserDataManager] FAILED ON JSON AT {path}"); return default_type
        return default_type

    def get_user_memory(self, user_id):
        return self.user_memory[user_id.lower()]

    def get_nickname(self, user_id):
        if not user_id: return "someone"
        user_id = user_id.lower()
        mem = self.get_user_memory(user_id)
        return mem.get("nickname") or mem.get("display_name") or user_id

    def set_nickname(self, user_id, nickname):
        self.get_user_memory(user_id)["nickname"] = nickname
        self.save_user_data()

    def format_message(self, user_id, text):
        return f"{self.get_nickname(user_id)}: {text}"

    def update_bby(self, user_id, amount):
        mem = self.get_user_memory(user_id)
        mem["BBY"] = round(mem.get("BBY", 0.0) + amount, 4)

    def get_bby(self, user_id):
        return round(self.get_user_memory(user_id).get("BBY", 0.0), 4)

    def get_spam_level(self, user_id):
        return self.get_user_memory(user_id).get("spamMax", 0.1)

    def set_spam_level(self, user_id, level):
        self.get_user_memory(user_id)["spamMax"] = level
        self.save_user_data()
        
    def check_bestie(self):
        bby_users = {u: m["BBY"] for u, m in self.user_memory.items() if "BBY" in m}
        if not bby_users: return None, 0
        return max(bby_users.items(), key=lambda item: item[1])

    def check_rival(self):
        bby_users = {u: m["BBY"] for u, m in self.user_memory.items() if "BBY" in m}
        if not bby_users: return None, 0
        return min(bby_users.items(), key=lambda item: item[1])

    def update_bestie_rival(self):
        bestie, _ = self.check_bestie()
        rival, _ = self.check_rival()
        self.current_bestie = bestie
        self.current_rival = rival
        
    def get_world_total_for_item(self, item_name):
        return sum(user_mem.get("inventory", {}).get(item_name, 0) for user_mem in self.user_memory.values())

    def get_effective_item_value(self, item_name):
        base_value = self.bby_book.get_fact_value_base(item_name)
        world_total = self.get_world_total_for_item(item_name)
        return base_value / max(1, world_total)
    
    def get_war_ammo(self, author_id):
        full_board = sorted([(u, m["BBY"]) for u, m in self.user_memory.items()], key=lambda x: x[1], reverse=True)
        try: rank = [u for u, _ in full_board].index(author_id)
        except ValueError: rank = len(full_board)
        total_bby = sum(abs(score) for _, score in full_board)
        author_bby = abs(self.get_bby(author_id))
        share = author_bby / total_bby if total_bby > 0 else 0
        ammo = min(len(full_board), max(1, share * (1 + (len(full_board) - rank))))
        return ammo
    
    async def decay_bby(self):
        print(f"[UserDataManager] Running BBY decay cycle...")
        for user_id, mem in self.user_memory.items():
            if time.time() - mem.get("last_seen", 0) > 60 * 60 * 24: # 24 hours
                decay_rate = 0.005
                mem["BBY"] *= (1 - decay_rate)
        self.save_user_data()

    async def handle_ghost_archiving(self):
        ghosts = [u for u, m in self.user_memory.items() if m.get("loyalty", 0) < 2 and m.get("BBY", 0) < -400]
        if ghosts:
            print(f"[UserDataManager] Archiving {len(ghosts)} ghost accounts...")
            for username in ghosts:
                self.bby_book.archive_as_fact(username)
                if username not in self.ai_opt_in_users:
                    del self.user_memory[username]
            self.save_user_data()

    def opt_in(self, user_id):
        user_id = user_id.lower()
        if user_id not in self.ai_opt_in_users:
            self.ai_opt_in_users.append(user_id)
            self.save_opt_in_users()
            self.update_bby(user_id, 1000.0)
            return f"hey {user_id}, thanks for opting in! i can now use your messages to learn!"
        return f"uhhh, {user_id}... you're already opted in!"

    def opt_out(self, user_id):
        user_id = user_id.lower()
        if user_id in self.ai_opt_in_users:
            self.update_bby(user_id, -1000.0)
            self.ai_opt_in_users.remove(user_id)
            self.save_opt_in_users()
            return f"hey {user_id}, your messages will no longer be used for my learning."
        return f"lol you're not even in the list, {user_id}!"

    def opt_check(self, user_id):
        user_id = user_id.lower()
        return f"hey, {user_id}, you are currently {'opted in' if user_id in self.ai_opt_in_users else 'not opted in'}."