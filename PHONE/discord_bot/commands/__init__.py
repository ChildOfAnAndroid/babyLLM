# CHARIS CAT 2025
# BABYLLM Command Modules
# Organized by hidden stat categories

from .bbybook_cmds import BBYBookCog
from .cooking_cmds import CookingCog
from .curse_cmds import CurseCog
from .timing_cmds import TimingCog
from .train_cmds import TrainCog

COMMAND_EXTENSION_MODULES = (
    "phone.discord_bot.commands.curse_cmds",
    "phone.discord_bot.commands.cooking_cmds",
    "phone.discord_bot.commands.bbybook_cmds",
    "phone.discord_bot.commands.timing_cmds",
    "phone.discord_bot.commands.train_cmds",
)

# Export all cogs for easy import
__all__ = [
    "BBYBookCog",
    "COMMAND_EXTENSION_MODULES",
    "CookingCog",
    "CurseCog",
    "TimingCog",
    "TrainCog",
]
