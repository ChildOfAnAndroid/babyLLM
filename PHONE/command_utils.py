# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM // phone/command_utils.py
# v3.7

import re

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def strip_ansi(text: str) -> str:
    return ansi_escape.sub('', text)


def get_status_line(bot) -> str:
    """Return the bot's current top tokens."""
    bot.tutor.update_top_tokens()
    return f"top tokens: {strip_ansi(bot.tutor.topTokens_forBot)}"


def get_thought_line(bot) -> str:
    """Return the bot's current decoded thought."""
    return f"current thought: {bot.tutor.decodedTokenIndices}"


__all__ = ["strip_ansi", "get_status_line", "get_thought_line"]
