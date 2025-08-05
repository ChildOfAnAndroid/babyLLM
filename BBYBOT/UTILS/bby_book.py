# BBYBOT/UTILS/bby_book.py
import json
import os
import random
import time
from config import bbybookPath

class BBYBook:
    def __init__(self):
        self.path = bbybookPath
        self.facts = self._load()
        self.errorKeys = ["oops, error!", "missingno", "NaN", "the void"]
        self.errorValues = ["how did you manage to make this item!?"]
        self.errorAuthors = ["the void", "missingno", "error!", "NaN"]
        print(f"[BBYBookManager] Loaded {len(self.facts)} facts from {self.path}")

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"!!!![BBYBookManager] FAILED ON JSON AT {self.path}")
                    return {}
        return {}

    def save(self):
        print("[BBYBookManager] Saving bbyfacts...")
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=2)
        print("[BBYBookManager] Bbyfacts saved!")

    def get_fact(self, key):
        return self.facts.get(key.lower(), {})

    def get_random_fact(self):
        if not self.facts:
            return None, {}
        key = random.choice(list(self.facts.keys()))
        return key, self.get_fact(key)

    def set_fact(self, key, value, author, timestamp=None, teach_bonus=420.0, num_produced=2.0):
        key = key.lower()
        if not key: key = random.choice(self.errorKeys)
        if not value: value = random.choice(self.errorValues)
        if not author: author = random.choice(self.errorAuthors)
        if timestamp is None: timestamp = time.time()
        
        self.facts[key] = {
            "value": value,
            "author": author,
            "timestamp": timestamp,
            "teach_bonus": teach_bonus,
            "num_produced": num_produced
        }
        self.save()
        print(f"[BBYBookManager] Set fact '{key}'")
        return self.facts[key]

    def discover_fact(self, key, author_id, author_nickname):
        return self.set_fact(
            key=key,
            value=f"first discovered by {author_nickname}.",
            author=author_id,
            teach_bonus=random.uniform(50, 25000),
            num_produced=random.randint(10, 10000)
        )

    def archive_as_fact(self, user_id):
        return self.set_fact(
            key=f"the ghost of {user_id}",
            value="was here for a bit, but something happened...",
            author="the void"
        )

    def fact_exists(self, key):
        return key.lower() in self.facts

    def delete_fact(self, key):
        key = key.lower()
        if self.fact_exists(key):
            del self.facts[key]
            self.save()
            return True
        return False
        
    def get_fact_value_base(self, key):
        return self.get_fact(key).get("teach_bonus", 420.0)

    def get_fact_num_produced(self, key):
        return self.get_fact(key).get("num_produced", 2.0)

    def get_fact_author(self, key):
        return self.get_fact(key).get("author")
