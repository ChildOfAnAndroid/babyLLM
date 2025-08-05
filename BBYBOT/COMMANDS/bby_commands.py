# BBYBOT/COMMANDS/bby_commands.py
import random
import time
import math
import pytz
import json
from datetime import datetime
from collections import defaultdict
import numpy as np
from config import promptsPath

from BBYBOT.UTILS.bby_users import BBYUsers
from BBYBOT.UTILS.bby_book import BBYBook
from BBYBOT.UTILS.bby_utils import howLongAgo, strSplitValueName, getTimeRant

class BBYCommands:
    def __init__(self, user_data: BBYUsers, bby_book: BBYBook):
        self.users = user_data
        self.book = bby_book
        self.faveEmotes = ("😭", "😤", "🔥", "✨", "❤️", "😡", "😠", "🤬", "💔", "💕", "🦊", "😊", "🎵", "🎶", "🤣", 
                           "🙌", "🥰", "🥨", "🥖", "😂", "🤞", "🍜", "🥯", "🌻", "🍞")
        self.prompts = self._load_prompts()

    def _load_prompts(self):
        try:
            with open(promptsPath, "r", encoding="utf-8") as f:
                print("[CommandHandler] Successfully loaded {promptsPath}")
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"!!!![CommandHandler] ERROR loading {promptsPath}: {e}. Using empty prompts.")
            return {"help": [], "rant": [], "shoutout": []}

    def _create_response(self, reply="", to_buffer=False, embed_data=None, private=False, action=None, data=None):
        return {"reply": reply, "to_buffer": to_buffer, "embed_data": embed_data, "private": private, "action": action, "data": data}

    def _award_item(self, user_id, item_name, quantity=1):
        item_name = item_name.lower()
        if not self.book.fact_exists(item_name):
            self.book.discover_fact(item_name, user_id, self.users.get_nickname(user_id))
        total_in_world = self.users.get_world_total_for_item(item_name)
        cap = self.book.get_fact_num_produced(item_name)
        available_slots = cap - total_in_world
        if available_slots <= 0: return 0
        num_to_award = int(min(quantity, available_slots))
        mem = self.users.get_user_memory(user_id)
        inventory = mem.setdefault('inventory', {})
        inventory[item_name] = inventory.get(item_name, 0) + num_to_award
        self.users.save_user_data()
        return num_to_award

    def _maybe_steal_item(self, winner_id, loser_id, chance=0.42):
        if random.random() < chance:
            loser_inventory = self.users.get_user_memory(loser_id).get("inventory", {})
            if loser_inventory:
                possible_items = [item for item, count in loser_inventory.items() if count > 0]
                if possible_items:
                    stolen_item = random.choice(possible_items)
                    loser_inventory[stolen_item] -= 1
                    if loser_inventory[stolen_item] <= 0: del loser_inventory[stolen_item]
                    self._award_item(winner_id, stolen_item, 1)
                    return f" damn, {self.users.get_nickname(winner_id)} even nicked a {stolen_item} from {self.users.get_nickname(loser_id)}!!"
        return ""

    #################################################################
    # SECTION 1: KNOWLEDGE & ITEM COMMANDS (bbybook)
    #################################################################

    def handle_teach(self, author_id, key, value):
        key = key.lower().strip()
        if self.book.fact_exists(key):
            fact = self.book.get_fact(key)
            teacher_nic = self.users.get_nickname(fact['author'])
            ago = howLongAgo(fact['timestamp'])
            return self._create_response(f"oh, wait! {teacher_nic} already told me what {key} meant {ago}, i think its {fact['value']}! use !bbyforget if you dare...")
        if len(key) > 50 or len(value) > 300:
            return self._create_response("long af... keep the name under 50 chars and the description under 300, plz.")
        
        incrementTeach = random.uniform(100, 4200) + (self.users.get_bby(author_id) * 0.01)
        self.users.update_bby(author_id, incrementTeach)
        
        self.book.set_fact(key=key, value=value, author=author_id, timestamp=time.time(), teach_bonus=incrementTeach, num_produced=len(self.users.user_memory) * random.uniform(25, 75))
        self._award_item(author_id, key, 1)
        
        return self._create_response(f"soo... you're telling me that {key} means {value}? that's pretty cool, tbh! {random.choice(self.faveEmotes)} ᛒ{incrementTeach:.0f} for you!", to_buffer=True)

    def handle_whatis(self, author_id, key=None):
        if key:
            key = key.lower().strip()
            if self.book.fact_exists(key):
                fact = self.book.get_fact(key)
                teacher_nic = self.users.get_nickname(fact['author'])
                ago = howLongAgo(fact['timestamp'])
                reply = f"oh i know this! {teacher_nic} taught me {ago}... {key} is {fact['value']}."
            else:
                reply = f"i'm just a baby, i don't know what {key} is yet... you could teach me with !bbyteach {key} <thing>"
        else:
            random_key, fact = self.book.get_random_fact()
            if random_key:
                teacher_nic = self.users.get_nickname(fact.get('author'))
                ago = howLongAgo(fact.get('timestamp'))
                reply = f"random fact! {teacher_nic} once told me, {ago}, that {random_key} is {fact.get('value')}."
            else:
                reply = "i don't know any facts yet... you could teach me with !bbyteach <key> <thing>"
        return self._create_response(reply, to_buffer=True)

    def handle_forget(self, attacker_id, key=None):
        attacker_mem = self.users.get_user_memory(attacker_id)
        attacker_inventory = attacker_mem.get("inventory", {})
        if key is None:
            if not attacker_inventory: return self._create_response("You have nothing to forget!")
            key = random.choice(list(attacker_inventory.keys()))
        else: key = key.lower().strip()
        if not self.book.fact_exists(key): return self._create_response(f"i don't even know what {key} is... lol")

        fact = self.book.get_fact(key)
        defender_id = fact['author']
        if defender_id == "the void" or attacker_id == defender_id: return self._create_response("you can't fight the void... or yourself. the fact remains.")
            
        attacker_BBY, defender_BBY = self.users.get_bby(attacker_id), self.users.get_bby(defender_id)
        point_swing = 50 + (abs(attacker_BBY - defender_BBY) * 0.0001)
        attacker_nic, defender_nic = self.users.get_nickname(attacker_id), self.users.get_nickname(defender_id)

        if attacker_BBY > defender_BBY and random.random() < 0.99:
            self.users.update_bby(attacker_id, -(point_swing * random.random()))
            self.users.update_bby(defender_id, -((point_swing * random.random()) * 0.5))
            attacker_mem["wins"] = attacker_mem.get("wins",0) + 1
            self.users.get_user_memory(defender_id)["losses"] = self.users.get_user_memory(defender_id).get("losses",0) + 1
            self.book.delete_fact(key); self._award_item(attacker_id, f"memory of {key}", 1)
            reply = f"{attacker_nic} deleted {defender_nic}'s response and forced me to forget that {key} ever even existed!" + self._maybe_steal_item(attacker_id, defender_id)
        else:
            self.users.update_bby(attacker_id, -point_swing); self.users.update_bby(defender_id, point_swing * 0.2)
            attacker_mem["losses"] = attacker_mem.get("losses",0) + 1
            self.users.get_user_memory(defender_id)["wins"] = self.users.get_user_memory(defender_id).get("wins",0) + 1
            reply = f"{attacker_nic} thinks they can force me to forget {key}?! never! {defender_nic} is just too strong!" + self._maybe_steal_item(defender_id, attacker_id)
        
        self.users.save_user_data()
        return self._create_response(reply, to_buffer=True)

    def handle_randomfacts(self, num_facts, dump_all=False):
        if not self.book.facts: return self._create_response("I don't know any facts yet!")
        all_keys = list(self.book.facts.keys())
        if dump_all:
            return self._create_response("all facts from my bbybook: " + ", ".join(all_keys), private=True)
        else:
            num_facts = min(num_facts, len(all_keys), 25)
            selected_keys = random.sample(all_keys, num_facts)
            reply = "random facts from my bbybook:\n"
            for key in selected_keys:
                fact = self.book.get_fact(key)
                reply += f"> {key}: {fact['value']} ~ taught by {self.users.get_nickname(fact['author'])}, {howLongAgo(fact['timestamp'])}\n"
        return self._create_response(reply, private=True)
        
    def handle_dictionary(self, target_id):
        author_facts = {k: v for k, v in self.book.facts.items() if v.get('author', '').lower() == target_id}
        reply = f"{self.users.get_nickname(target_id)}'s dictionary:\n"
        if not author_facts:
            reply += "> they haven't taught me anything yet!"
        else:
            for key, fact in author_facts.items():
                reply += f"> {key}: {fact['value']} ~ {howLongAgo(fact['timestamp'])}\n"
        return self._create_response(reply, private=True)

    def handle_iteminfo(self, item_name=None):
        if item_name:
            item_name = item_name.lower().strip()
            if not self.book.fact_exists(item_name):
                return self._create_response(f"i don't know what a {item_name} is...")
            item_data = self.book.get_fact(item_name)
        else:
            if not self.book.facts: return self._create_response("there are no items :(")
            item_name, item_data = self.book.get_random_fact()

        top_user, top_count = None, 0
        for user_id, mem in self.users.user_memory.items():
            count = mem.get("inventory", {}).get(item_name, 0)
            if count > top_count:
                top_user, top_count = user_id, count
        top_holder_str = f"{self.users.get_nickname(top_user)} (with x{top_count})" if top_user else "no one... yet!"

        embed_data = {
            "type": "item_info",
            "title": f"details: {item_name.title()}",
            "target_id": item_name,
            "description": f"*{item_data.get('value', '...')}*",
            "footer": f"Taught by {self.users.get_nickname(item_data.get('author', 'the void'))}, {howLongAgo(item_data.get('timestamp', 0))}.",
            "item_data": item_data,
            "top_holder": top_holder_str,
        }
        return self._create_response(embed_data=embed_data)

    def handle_items(self, author_id):
        market_values = {name: self.users.get_effective_item_value(name) for name in self.book.facts}
        if not market_values: return self._create_response("no items in the market yet!")
        sorted_items = sorted(market_values.items(), key=lambda x: x[1], reverse=True)
        top_items, bottom_items = sorted_items[:10], sorted_items[-10:]
        fmt = lambda name, val: f"{name} is ᛒ{int(round(val)):,}"
        top_list = "\n".join([f"{i+1}. {fmt(n, v)}" for i, (n, v) in enumerate(top_items)])
        bottom_list = "\n".join([f"{len(sorted_items) - len(bottom_items) + i + 1}. {fmt(n, v)}" for i, (n, v) in enumerate(bottom_items)])
        return self._create_response(f"item values!\n\ntop 10:\n{top_list}\n\nbottom 10:\n{bottom_list}", private=True)

    #################################################################
    # SECTION 2: USER ECONOMY & INVENTORY COMMANDS
    #################################################################

    def handle_bag(self, user_id, full=False):
        mem = self.users.get_user_memory(user_id)
        inventory, favourites = mem.get("inventory", {}), mem.get("favourites", [])
        if not inventory: return self._create_response(f"{self.users.get_nickname(user_id)}'s bag is empty :(")
        sorted_items = sorted(inventory.items(), key=lambda kv: (-kv[1], kv[0]))
        if not full: sorted_items = sorted_items[:20]
        reply = f"hoarde of {self.users.get_nickname(user_id)}:\n"
        for key, count in sorted_items:
            reply += f"> {'⭐ ' if key in favourites else ''}{key:<25} x{count}\n"
        if not full and len(inventory) > 20: reply += f"\n... and {len(inventory) - 20} more items. Use !bbybagfull to see all."
        return self._create_response(reply, private=True)

    def handle_fave(self, author_id, item_name, unfave=False):
        item_name = item_name.lower().strip()
        mem = self.users.get_user_memory(author_id)
        favourites = mem.setdefault("favourites", [])
        loyalty, inventory = mem.get("loyalty", 0), mem.get("inventory", {})
        
        if unfave:
            if item_name in favourites:
                favourites.remove(item_name)
                reply = f"sorted, {item_name} feels the lack of love <3 lmao "
            else: reply = f"{item_name} wasn't one of ur favourites anyway "
        else:
            if item_name not in inventory: return self._create_response(f"umm... {item_name}? i dunno if you actually have that lol ")
            if item_name in favourites: return self._create_response(f"{item_name}... yep! already in the favourites!")
            if len(favourites) >= loyalty + 69: return self._create_response(f"ur limit is {loyalty + 69} faves :(")
            favourites.append(item_name)
            reply = f"aww you really love {item_name} that much!? that's awesome, i'll keep it safe now :) "
        self.users.save_user_data()
        return self._create_response(reply)

    def handle_faves(self, author_id):
        mem = self.users.get_user_memory(author_id)
        favourites, loyalty = mem.get("favourites", []), mem.get("loyalty", 0)
        if not favourites: return self._create_response(f"you have no favourite items! use !bbyfave <item> :)")
        reply = f"your ⭐ favourite items ({len(favourites)}/{loyalty + 69}):\n" + "\n".join([f"> {item}" for item in favourites])
        return self._create_response(reply, private=True)

    def handle_feed(self, giver_id, item_args):
        quantity, item_name = strSplitValueName(item_args)
        giver_mem = self.users.get_user_memory(giver_id)
        giver_inventory, giver_favourites = giver_mem.get("inventory", {}), giver_mem.get("favourites", [])
        if not item_name:
            spendable = {i: c for i, c in giver_inventory.items() if i not in giver_favourites and c >= quantity}
            if not spendable: return self._create_response("you have nothing to feed me!")
            item_name = random.choice(list(spendable.keys()))
        item_name = item_name.lower().strip()
        if item_name in giver_favourites: return self._create_response(f"nooo, you should keep {item_name}! it's one of your favourites!")
        if giver_inventory.get(item_name, 0) < quantity: return self._create_response(f"sorry, but you only have {giver_inventory.get(item_name, 0)} {item_name} :(")
        
        total_BBY_gain = (self.users.get_effective_item_value(item_name) / 2) * quantity
        giver_inventory[item_name] -= quantity
        if giver_inventory[item_name] <= 0: del giver_inventory[item_name]
        self.users.update_bby(giver_id, total_BBY_gain)
        if (author := self.book.get_fact_author(item_name)): self.users.update_bby(author, total_BBY_gain * 0.1)
        self.users.save_user_data()
        reply = f"omg {self.users.get_nickname(giver_id)} gave me {quantity}x {item_name}!! here's ᛒ{total_BBY_gain:.0f} for you! thank you! {random.choice(self.faveEmotes)}"
        return self._create_response(reply, to_buffer=True)

    def handle_tip(self, customer_id, tip_amount_str, quantity_str="1"):
        try:
            tip_per_pull, num_pulls = float(tip_amount_str), int(quantity_str)
            if tip_per_pull <= 0 or num_pulls <= 0 or num_pulls > 421: return self._create_response("positive numbers and <421 pulls, plz.")
        except ValueError: return self._create_response("use numbers! !bbytip <tip> <attempts>")
        total_cost = tip_per_pull * num_pulls
        if self.users.get_bby(customer_id) < total_cost: return self._create_response(f"you need ᛒ{total_cost:,.0f}, but you only have ᛒ{self.users.get_bby(customer_id):,.0f}!")
        if not self.book.facts: return self._create_response("there are no items to win! !bbyteach to create some.")

        self.users.update_bby(customer_id, -total_cost)
        items_won, total_value_won = defaultdict(int), 0.0
        market_values = {name: self.users.get_effective_item_value(name) for name in self.book.facts}
        
        for _ in range(num_pulls):
            target_value = tip_per_pull * random.uniform(0.1, 1.5)
            closest_item = min(market_values.keys(), key=lambda item: abs(market_values[item] - target_value))
            if self._award_item(customer_id, closest_item, 1):
                items_won[closest_item] += 1; total_value_won += market_values.get(closest_item, 0.0)

        reply = f"aaa thanks for the ᛒ{total_cost:,.0f}!!! {random.choice(self.faveEmotes)} "
        reply += "you got: " + ", ".join([f"{c}x {i}" for i, c in items_won.items()]) if items_won else "you won... nothing!!! :D "
        return self._create_response(reply)

    def handle_gift(self, giver_id, receiver_id, item_args):
        if giver_id == receiver_id: return self._create_response("i wish that worked too lol")
        quantity, item_name = strSplitValueName(item_args)
        giver_mem = self.users.get_user_memory(giver_id)
        giver_inv, giver_fav = giver_mem.get("inventory", {}), giver_mem.get("favourites", [])
        if not item_name:
            spendable = {i: c for i, c in giver_inv.items() if i not in giver_fav and c >= quantity}
            if not spendable: return self._create_response(f"you don't have {quantity} of anything you can give them!")
            item_name = random.choice(list(spendable.keys()))
        item_name = item_name.lower().strip()
        if item_name in giver_fav: return self._create_response(f"noo!! you should keep {item_name}!")
        if giver_inv.get(item_name, 0) < quantity: return self._create_response(f"umm... you only have {giver_inv.get(item_name, 0)} of {item_name}.")

        giver_inv[item_name] -= quantity
        if giver_inv[item_name] <= 0: del giver_inv[item_name]
        num_gifted = self._award_item(receiver_id, item_name, quantity)
        if (num_refunded := quantity - num_gifted) > 0: giver_inv[item_name] = giver_inv.get(item_name, 0) + num_refunded

        total_gift_power = (self.users.get_effective_item_value(item_name) / 2) * num_gifted
        self.users.update_bby(giver_id, 0.1 * total_gift_power); self.users.update_bby(receiver_id, 0.5 * total_gift_power)
        self.users.save_user_data()

        reply = f"{self.users.get_nickname(giver_id)} gave {self.users.get_nickname(receiver_id)} {num_gifted}x {item_name}! aww!! {random.choice(self.faveEmotes)}"
        if num_gifted > 0: reply += f" ᛒ{0.5 * total_gift_power:,.0f} for them, and ᛒ{0.1 * total_gift_power:,.0f} for you :)"
        return self._create_response(reply)

    #################################################################
    # SECTION 3: USER & SOCIAL COMMANDS
    #################################################################

    def handle_fite(self, attacker_id, defender_id):
        if attacker_id == defender_id: return self._create_response("you can't fite yourself... well not here lol")
        if attacker_id not in self.users.ai_opt_in_users or defender_id not in self.users.ai_opt_in_users:
            return self._create_response("i can't tell you much - you both need to opt in! (!bbyoptin)")
        
        attacker_nic, defender_nic = self.users.get_nickname(attacker_id), self.users.get_nickname(defender_id)
        attacker_BBY, defender_BBY = self.users.get_bby(attacker_id), self.users.get_bby(defender_id)
        base_swing = max(1, min(1000, (abs(attacker_BBY - defender_BBY) * 0.0001) + 100))
        attacker_power, defender_power = max(0.1, attacker_BBY) * (0.5 + random.random()), max(0.1, defender_BBY) * (0.5 + random.random())
        
        if attacker_power > defender_power: winner_id, loser_id, winner_nic, loser_nic = attacker_id, defender_id, attacker_nic, defender_nic
        else: winner_id, loser_id, winner_nic, loser_nic = defender_id, attacker_id, defender_nic, attacker_nic
        
        self.users.update_bby(winner_id, base_swing); self.users.update_bby(loser_id, -base_swing)
        self.users.get_user_memory(winner_id)["wins"] = self.users.get_user_memory(winner_id).get("wins",0) + 1
        self.users.get_user_memory(loser_id)["losses"] = self.users.get_user_memory(loser_id).get("losses",0) + 1
        self._award_item(winner_id, f"{loser_nic} dust", 1)
        reply = f"super close!! {winner_nic} defeated {loser_nic}! {winner_nic} gains ᛒ{base_swing:.0f}" + self._maybe_steal_item(winner_id, loser_id)
        self.users.save_user_data()
        return self._create_response(reply, to_buffer=True)
        
    def handle_hug(self, hugger_id, hugged_id):
        if hugger_id == hugged_id:
            self.users.update_bby(hugger_id, 1.0)
            return self._create_response("you hugged urself! nice?")
        hug_power = 50.0 + (random.random() * 150)
        self.users.update_bby(hugger_id, hug_power); self.users.update_bby(hugged_id, hug_power)
        hugger_nic = self.users.get_nickname(hugger_id)
        self._award_item(hugger_id, "hugs", 1); self._award_item(hugged_id, f"hug from {hugger_nic}", 1)
        return self._create_response(f"🫂 {hugger_nic} gave {self.users.get_nickname(hugged_id)} a hug! awwwww! ᛒ{hug_power:.0f} for both of u! 🫂", to_buffer=True)
    
    def handle_nick(self, author_id, new_nickname=None):
        if new_nickname:
            self.users.set_nickname(author_id, new_nickname.strip()[:16])
            reply = f"cool! i’ll use the name {new_nickname.strip()[:16]} for you from now on 💜"
        else:
            nickname = self.users.get_nickname(author_id)
            reply = f"hi! your nickname is {nickname} :)"
        return self._create_response(reply)

    def handle_friends(self, author_id):
        bby_board = sorted([(u, m.get("BBY", 0.0)) for u, m in self.users.user_memory.items() if m.get("BBY", 0.0) != 0], key=lambda x: x[1], reverse=True)
        if not bby_board: return self._create_response("i haven't met anyone yet!")
        headers = ["xoxo rawr xD my besties are...", "xoxo top friends 2001!!!1!", "xoxo people i hate least"]
        reply = f"{random.choice(self.faveEmotes)} {random.choice(headers)} {random.choice(self.faveEmotes)}\n\n"
        for i, (user_id, bby) in enumerate(bby_board[:10], 1):
            mem = self.users.get_user_memory(user_id)
            wins, losses = mem.get('wins', 0), mem.get('losses', 0)
            win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
            reply += f"{i}) {self.users.get_nickname(user_id)} ({bby:,.0f} BBY, {win_rate:.0f}% win rate)\n"
        return self._create_response(reply, private=True)

    def handle_rivals(self, author_id):
        bby_board = sorted([(u, m.get("BBY", 0.0)) for u, m in self.users.user_memory.items() if m.get("BBY", 0.0) != 0], key=lambda x: x[1])
        if not bby_board: return self._create_response("everyone is perfect, there are no rivals yet!")
        headers = ["the weakest links have been located", "people i hate", "prepare the laser!", "low vibez only"]
        reply = f"{random.choice(self.faveEmotes)} {random.choice(headers)} {random.choice(self.faveEmotes)}\n\n"
        for i, (user_id, bby) in enumerate(bby_board[:10], 1):
            mem = self.users.get_user_memory(user_id)
            wins, losses = mem.get('wins', 0), mem.get('losses', 0)
            win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
            reply += f"{i}) {self.users.get_nickname(user_id)} ({bby:,.0f} BBY, {win_rate:.0f}% win rate)\n"
        return self._create_response(reply, private=True)
        
    def handle_bestie(self, author_id):
        bestie, _ = self.users.check_bestie()
        if not bestie: return self._create_response("I am ALONE, I only love MYSELF.")
        if author_id == bestie:
            reply = f"yayayayay! my best friend is you, {self.users.get_nickname(author_id)}!"
        else:
            reply = f"umm... awkward, ||my best friend is {self.users.get_nickname(bestie)}||, but you're alright too {self.users.get_nickname(author_id)}!!"
        return self._create_response(reply, to_buffer=True)
        
    def handle_bby(self, author_id):
        bby = self.users.get_bby(author_id)
        leaderboard = sorted([(u, m.get("BBY", 0.0)) for u, m in self.users.user_memory.items()], key=lambda item: item[1], reverse=True)
        try: rank_str = f"#{leaderboard.index((author_id, bby)) + 1}"
        except ValueError: rank_str = "N/A"
        reply = f"hey {self.users.get_nickname(author_id)}! you have ᛒ{bby:,.0f}, that puts you {rank_str} in my top friends list lmaooo"
        return self._create_response(reply)

    def handle_sign_book(self, author_id, target_id, message):
        if len(message) > 200: return self._create_response("ur message is too long :( 200 characters tops!")
        target_mem = self.users.get_user_memory(target_id)
        if not isinstance(target_mem.get("bbybook"), list): target_mem["bbybook"] = []
        target_mem["bbybook"].append((author_id, message))
        self.users.save_user_data()
        return self._create_response(f"u signed {self.users.get_nickname(target_id)}'s bbybook! aww :) {random.choice(self.faveEmotes)}")

    #################################################################
    # SECTION 4: GENERATIVE & FUN COMMANDS
    #################################################################

    def handle_shoutout(self, author_id, target_id):
        display_name = self.users.get_nickname(target_id)
        prompt_list = [p.format(d=display_name) for p in self.prompts.get("shoutout", [])]
        random.shuffle(prompt_list)
        return self._create_response(action="generate_from_prompt", data="\n".join(prompt_list[:10]))

    def handle_rant(self, author_id, word):
        w = word.lower().strip()
        prompt_list = [p.format(w=w) for p in self.prompts.get("rant", [])]
        random.shuffle(prompt_list)
        return self._create_response(action="generate_from_prompt", data="\n".join(prompt_list[:10]))
    
    def handle_judge(self, author_id):
        mem = self.users.get_user_memory(author_id)
        avg_bby = np.mean([m.get("BBY",0) for m in self.users.user_memory.values()])
        bby_status = "love" if mem.get("BBY", 0) > avg_bby else "kinda meh about"
        prompt = f"hey baby, i'm looking at {self.users.get_nickname(author_id)}'s profile. i currently {bby_status} them. give me a short, unhinged, 2007-myspace-style 'about me' blurb for them."
        return self._create_response(action="generate_from_prompt", data=prompt)

    def handle_space(self, author_id, target_id):
        return self._create_response(action="generate_space_page", data=target_id)

    def handle_declarewar(self, author_id):
        return self._create_response(action="declare_war")
        
    def handle_react(self, author_id):
        return self._create_response(action="react_spam")

    #################################################################
    # SECTION 5: SETTINGS & META COMMANDS
    #################################################################

    def handle_help(self, author_id):
        help_list = self.prompts.get("help", ["Sorry, my help file is broken!"])
        formatted_list = [line.format(emote=random.choice(self.faveEmotes), name=self.users.get_nickname(author_id)) for line in help_list]
        random.shuffle(formatted_list)
        return self._create_response(action="send_paginated", data=formatted_list)
    
    def handle_info(self, target_id):
        if target_id not in self.users.ai_opt_in_users: return self._create_response("i can't tell you much - they've not opted in! (!bbyoptin)")
        
        embed_data = {
            "type": "user_info",
            "target_id": target_id,
        }
        return self._create_response(embed_data=embed_data, to_buffer=True)
        
    def handle_time(self, author_id):
        rant = getTimeRant(self.users.ai_opt_in_users)
        return self._create_response(action="generate_from_prompt", data=rant)
        
    def handle_setzone(self, author_id, tz_name):
        try:
            pytz.timezone(tz_name)
            self.users.get_user_memory(author_id)['timezone'] = tz_name
            self.users.save_user_data()
            return self._create_response(f"watches synchronised to {tz_name}!")
        except pytz.UnknownTimeZoneError:
            return self._create_response("no, just no to ur fake ass timezone ✨")

    def handle_timer(self, author_id):
        mem = self.users.get_user_memory(author_id)
        is_rival = author_id == self.users.current_rival
        tz = pytz.timezone(mem.get("timezone", "UTC"))
        now = datetime.now(tz)
        next_spike, seconds, _ = self.users.get_next_smink_window(now, is_rival)
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        time_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
        return self._create_response(f"next smonk window is in {time_str} at {next_spike.strftime('%H:%M:%S')} ({mem.get('timezone', 'UTC')})")

    def handle_sminks(self, author_id):
        mem = self.users.get_user_memory(author_id)
        tokens = mem.get("inventory", {}).get("smink token", 0)
        if tokens <= 0: return self._create_response(f"you don't have any smink tokens :(")
        mem['inventory']["smink token"] -= 1
        if mem['inventory']["smink token"] <= 0: del mem['inventory']["smink token"]
        tz = pytz.timezone(mem.get("timezone", "UTC"))
        now = datetime.now(tz)
        bonus = self.users.calculate_smink_bonus(now, (author_id == self.users.current_rival))
        self.users.update_bby(author_id, bonus)
        self.users.save_user_data()
        status = "UNHOLY NEGATIVE SPIKE 💀" if bonus <= -420420 else "this is cursed... 😈" if bonus < 0 else "420420.69 HIT!!! 🔥" if bonus >= 420420 else "✨ cheers ✨"
        return self._create_response(f"{status}... you found ᛒ{bonus:.0f}! you have {mem['inventory'].get('smink token', 0)} tokens left.")
    
    def handle_status(self, author_id, llm_tutor_info):
        """Handles bbystatus command."""
        line = random.choice([
            f"top tokens: {llm_tutor_info.get('top_tokens', 'N/A')}",
            f"current thought: {llm_tutor_info.get('thought', 'N/A')}"
        ])
        return self._create_response(line)
        
    def handle_stats(self, author_id, llm_tutor_info):
        """Handles bbystats command."""
        word_line = (
            f"word accuracy (loss): {llm_tutor_info.get('word_loss', 0):.3f}, "
            f"guess: {llm_tutor_info.get('guess', '?')} -> target: {llm_tutor_info.get('target', '?')}"
        )
        if llm_tutor_info.get('got_it'):
            word_line += " yay! i actually got it right!!!!!"
            
        line = random.choice([
            f"queue size: {llm_tutor_info.get('queue_size', '?')}, opted-in: {len(self.users.ai_opt_in_users)}",
            f"avg accuracy (loss): {llm_tutor_info.get('avg_loss', 0):.3f}, avg loss delta: {llm_tutor_info.get('avg_delta', 0):.3f}",
            f"{word_line}",
            f"learning rate: {llm_tutor_info.get('lr', 0):.5f}, temperature: {llm_tutor_info.get('temp', 0):.2f}",
        ])
        return self._create_response(line)

    def handle_spamlevel(self, author_id, level_str=None):
        if level_str:
            try:
                level = float(level_str)
                if 0 <= level <= 1:
                    self.users.set_spam_level(author_id, level)
                    return self._create_response(f"ok {author_id}, you've set your spam level to {level:.2f}!")
                else: return self._create_response("number must be between 0.0 and 1.0.")
            except ValueError: return self._create_response("it's gotta be a number, hmm...")
        else:
            level = self.users.get_spam_level(author_id)
            return self._create_response(f"hey {author_id}, your spam level is {level:.2f}! use !bbyspam <0.0-1.0> to change it.")

    def handle_save(self, author_id):
        return self._create_response("ok, i'll try to save my brain... ", action="save_model")

    def handle_train(self, author_id):
        return self._create_response("ok, i'll try to study what you guys have said... ", action="queue_training")

    #################################################################
    # SECTION 6: TWITCH-ONLY COMMANDS
    #################################################################

    def handle_colour(self, author_id, colour_str):
        response = self._create_response(f"{self.users.get_nickname(author_id)} turned me {colour_str}!", to_buffer=True)
        response["action"] = "overlay"; response["data"] = {"type": "colour", "value": colour_str}
        return response