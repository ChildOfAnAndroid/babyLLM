# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM // phone/discord_bot/command_helpers.py
# v1.0 - Command abstraction helpers for cleaner code

"""
Command Helper Classes for Discord Bot Refactoring

This module provides abstraction classes to eliminate repetitive patterns
in command implementations. See DISCORD_BOT_REFACTORING_ANALYSIS.md for details.

Classes:
    - CommandContext: Handles author extraction, target resolution, cooldown management
    - UserData: Safe user memory access with defaults
    - InventoryManager: Atomic inventory operations
    - RandomGenerator: Simplified RNG interface
"""

from typing import Any, Dict, List, Optional

import discord
from discord.ext import commands

from .utils import normalise_embed_british_english, to_british_english


class CommandContext:
    """
    Abstraction for common command setup patterns.

    Eliminates the repetitive author extraction and cooldown reset pattern
    that appears in 38+ commands.

    Usage:
        @commands.command()
        async def mycommand(self, ctx, target=None):
            cmd = CommandContext(ctx, self, target)
            if not cmd.author:
                return await cmd.error("User not found")

            # Your command logic here
            await cmd.reply("Success!")
    """

    def __init__(self, ctx: commands.Context, cog, target: Optional[str] = None):
        """
        Initialize command context with author and optional target resolution.

        Args:
            ctx: Discord command context
            cog: The cog instance (for accessing bot and methods)
            target: Optional target user mention or name
        """
        self.ctx = ctx
        self.cog = cog
        self.bot = cog.bot
        self.varied_random = cog.varied_random

        # Extract author
        self.author = self._extract_author(ctx)
        self.author_id = str(ctx.author.id) if ctx.author else None
        self.author_name = ctx.author.name if ctx.author else None
        self.author_display_name = ctx.author.display_name if ctx.author else None

        # Extract target if provided
        self.target = None
        self.target_id = None
        if target:
            self.target = self._extract_target(target)
            if self.target:
                self.target_id = self.target

    def _extract_author(self, ctx: commands.Context) -> Optional[str]:
        """Extract author ID from context."""
        if not ctx or not ctx.author:
            return None
        return str(ctx.author.id)

    def _extract_target(self, target: str) -> Optional[str]:
        """
        Extract target user ID from mention or name.

        Supports:
            - Discord mentions: <@123456789>
            - User IDs: 123456789
            - Usernames: charis (looks up in bot.baby_users)
        """
        import re

        # Handle Discord mention format
        mention_match = re.match(r"<@!?(\d+)>", target)
        if mention_match:
            return mention_match.group(1)

        # Handle direct ID
        if target.isdigit():
            return target

        # Handle username lookup
        target_lower = target.lower()
        for uid, user_data in self.bot.baby_users.items():
            username = user_data.get("username", "").lower()
            display_name = user_data.get("display_name", "").lower()
            if username == target_lower or display_name == target_lower:
                return uid

        return None

    async def error(self, message: str) -> None:
        """
        Send error message and automatically reset command cooldown.

        This eliminates the need for manual cooldown reset on every error return.
        """
        await self.reply(message)
        self._reset_cooldown()

    async def reply(
        self, content: str = "", embed: discord.Embed = None
    ) -> discord.Message:
        """Send a reply to the command."""
        content = to_british_english(str(content or ""))
        embed = normalise_embed_british_english(embed)
        return await self.ctx.send(content=content, embed=embed)

    def _reset_cooldown(self):
        """Reset command cooldown for this context."""
        try:
            if hasattr(self.cog, "reset_command_cooldown"):
                self.cog.reset_command_cooldown(self.ctx)
        except Exception as e:
            print(f"[CommandContext] Failed to reset cooldown: {e}")

    def get_user_data(self) -> "UserData":
        """Get a UserData wrapper for the command author."""
        return UserData(self.author, self.cog)

    def get_target_user_data(self) -> Optional["UserData"]:
        """Get a UserData wrapper for the target user (if any)."""
        if not self.target:
            return None
        return UserData(self.target, self.cog)


class UserData:
    """
    Safe wrapper for user memory access with defaults.

    Eliminates the repetitive null-checks and nested dictionary access
    that appears 197+ times in the codebase.

    Usage:
        user_data = UserData(author_id, self)
        bby_amount = user_data.get('bby', 0)  # Safe with default
        user_data.set('bby', new_amount)
        user_data.increment('message_count', 1)
    """

    def __init__(self, user_id: str, cog):
        """
        Initialize user data wrapper.

        Args:
            user_id: Discord user ID
            cog: The cog instance (for accessing bot.baby_users)
        """
        self.user_id = user_id
        self.cog = cog
        self.bot = cog.bot
        self._data = None
        self._load_data()

    def _load_data(self):
        """Load user data from bot.baby_users."""
        if not hasattr(self.bot, "baby_users"):
            self._data = {}
            return

        self._data = self.bot.baby_users.get(self.user_id, {})

        # If user doesn't exist, create minimal entry
        if not self._data:
            self._data = {
                "username": "",
                "display_name": "",
                "bby": 0,
                "inventory": {},
                "facts_awarded": 0,
                "last_daily": 0,
                "message_count": 0,
            }
            self.bot.baby_users[self.user_id] = self._data

    def exists(self) -> bool:
        """Check if user data exists."""
        return bool(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from user data with default fallback.

        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The value or default
        """
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        """
        Set a value in user data.

        Args:
            key: The key to set
            value: The value to store
        """
        self._data[key] = value
        self._mark_dirty()

    def increment(self, key: str, amount: float = 1.0) -> float:
        """
        Increment a numeric value (creates if doesn't exist).

        Args:
            key: The key to increment
            amount: Amount to add (can be negative)

        Returns:
            The new value
        """
        current = self.get(key, 0)
        new_value = float(current) + float(amount)
        self.set(key, new_value)
        return new_value

    def get_inventory(self, item: str, default: int = 0) -> int:
        """
        Get item count from inventory.

        Args:
            item: Item name
            default: Default count if item doesn't exist

        Returns:
            Item count
        """
        inventory = self.get("inventory", {})
        return inventory.get(item, default)

    def has_item(self, item: str, amount: int = 1) -> bool:
        """
        Check if user has at least amount of item.

        Args:
            item: Item name
            amount: Required amount

        Returns:
            True if user has enough
        """
        return self.get_inventory(item, 0) >= amount

    def _mark_dirty(self):
        """Mark user data as needing to be saved."""
        # This will integrate with the existing save system
        pass

    @property
    def username(self) -> str:
        """Get username."""
        return self.get("username", "")

    @property
    def display_name(self) -> str:
        """Get display name."""
        return self.get("display_name", "")

    @property
    def bby(self) -> float:
        """Get BBY currency amount."""
        return self.get("bby", 0)

    @bby.setter
    def bby(self, value: float):
        """Set BBY currency amount."""
        self.set("bby", value)


class InventoryManager:
    """
    Atomic inventory operations to prevent corruption.

    Centralizes inventory modifications that appear 50+ times,
    reducing bugs from scattered state updates.

    Usage:
        inventory = InventoryManager(user_data, cog)
        success = inventory.add_item('pixel', 10)
        success = inventory.remove_item('pixel', 5)
        success = inventory.transfer_item('pixel', 5, from_user, to_user)
    """

    def __init__(self, user_data: UserData, cog):
        """
        Initialize inventory manager.

        Args:
            user_data: UserData instance
            cog: The cog instance
        """
        self.user_data = user_data
        self.cog = cog
        self.bot = cog.bot

    def get_inventory(self) -> Dict[str, int]:
        """Get full inventory dict."""
        return self.user_data.get("inventory", {})

    def get_item_count(self, item: str) -> int:
        """Get count of specific item."""
        return self.get_inventory().get(item, 0)

    def has_item(self, item: str, amount: int = 1) -> bool:
        """Check if user has at least amount of item."""
        return self.get_item_count(item) >= amount

    def add_item(self, item: str, amount: int = 1) -> bool:
        """
        Add item to inventory (atomic operation).

        Args:
            item: Item name
            amount: Amount to add (must be positive)

        Returns:
            True if successful
        """
        if amount <= 0:
            return False

        inventory = self.get_inventory()
        current = inventory.get(item, 0)
        inventory[item] = current + amount
        self.user_data.set("inventory", inventory)
        return True

    def remove_item(self, item: str, amount: int = 1) -> bool:
        """
        Remove item from inventory (atomic operation).

        Args:
            item: Item name
            amount: Amount to remove (must be positive)

        Returns:
            True if successful, False if insufficient quantity
        """
        if amount <= 0:
            return False

        inventory = self.get_inventory()
        current = inventory.get(item, 0)

        if current < amount:
            return False

        new_amount = current - amount
        if new_amount == 0:
            # Remove item entirely if count reaches 0
            inventory.pop(item, None)
        else:
            inventory[item] = new_amount

        self.user_data.set("inventory", inventory)
        return True

    def set_item(self, item: str, amount: int) -> bool:
        """
        Set item count directly (use with caution).

        Args:
            item: Item name
            amount: New amount

        Returns:
            True if successful
        """
        if amount < 0:
            return False

        inventory = self.get_inventory()
        if amount == 0:
            inventory.pop(item, None)
        else:
            inventory[item] = amount

        self.user_data.set("inventory", inventory)
        return True

    @staticmethod
    def transfer_item(
        item: str, amount: int, from_user: UserData, to_user: UserData, cog
    ) -> bool:
        """
        Transfer item between users (atomic operation).

        Args:
            item: Item name
            amount: Amount to transfer
            from_user: Source user data
            to_user: Destination user data
            cog: The cog instance

        Returns:
            True if successful, False if sender has insufficient quantity
        """
        from_inv = InventoryManager(from_user, cog)
        to_inv = InventoryManager(to_user, cog)

        # Check if sender has enough
        if not from_inv.has_item(item, amount):
            return False

        # Perform atomic transfer
        if not from_inv.remove_item(item, amount):
            return False

        if not to_inv.add_item(item, amount):
            # Rollback if somehow add fails
            from_inv.add_item(item, amount)
            return False

        return True


class RandomGenerator:
    """
    Simplified interface for varied random selection.

    Eliminates the verbose .get_varied_choice().choice(...) pattern
    that appears 298+ times in the codebase.

    Usage:
        rng = RandomGenerator(cog.varied_random)
        message = rng.choice(["option1", "option2", "option3"])
        number = rng.randint(1, 100)
        items = rng.sample(list, 3)
    """

    def __init__(self, varied_random):
        """
        Initialize random generator.

        Args:
            varied_random: The bot's _VariedRNG instance
        """
        self._rng = varied_random

    def choice(self, seq: List[Any]) -> Any:
        """
        Choose a random element from sequence.

        Args:
            seq: Sequence to choose from

        Returns:
            Random element
        """
        return self._rng.get_varied_choice().choice(seq)

    def randint(self, a: int, b: int) -> int:
        """
        Return random integer in range [a, b].

        Args:
            a: Lower bound (inclusive)
            b: Upper bound (inclusive)

        Returns:
            Random integer
        """
        return self._rng.get_varied_choice().randint(a, b)

    def random(self) -> float:
        """
        Return random float in range [0.0, 1.0).

        Returns:
            Random float
        """
        return self._rng.get_varied_choice().random()

    def sample(self, population: List[Any], k: int) -> List[Any]:
        """
        Choose k unique random elements from population.

        Args:
            population: List to sample from
            k: Number of elements to choose

        Returns:
            List of k random unique elements
        """
        return self._rng.get_varied_choice().sample(population, k)

    def shuffle(self, x: List[Any]) -> None:
        """
        Shuffle list in place.

        Args:
            x: List to shuffle
        """
        self._rng.get_varied_choice().shuffle(x)

    def choices(
        self, population: List[Any], weights: List[float] = None, k: int = 1
    ) -> List[Any]:
        """
        Choose k elements with replacement (with optional weights).

        Args:
            population: List to choose from
            weights: Optional weights for weighted choice
            k: Number of elements to choose

        Returns:
            List of k random elements (possibly with duplicates)
        """
        return self._rng.get_varied_choice().choices(population, weights=weights, k=k)


# Convenience functions for backward compatibility
def create_command_context(
    ctx: commands.Context, cog, target: Optional[str] = None
) -> CommandContext:
    """Create a CommandContext instance."""
    return CommandContext(ctx, cog, target)


def create_user_data(user_id: str, cog) -> UserData:
    """Create a UserData instance."""
    return UserData(user_id, cog)


def create_inventory_manager(user_data: UserData, cog) -> InventoryManager:
    """Create an InventoryManager instance."""
    return InventoryManager(user_data, cog)


def create_random_generator(varied_random) -> RandomGenerator:
    """Create a RandomGenerator instance."""
    return RandomGenerator(varied_random)


__all__ = [
    "CommandContext",
    "UserData",
    "InventoryManager",
    "RandomGenerator",
    "create_command_context",
    "create_user_data",
    "create_inventory_manager",
    "create_random_generator",
]
