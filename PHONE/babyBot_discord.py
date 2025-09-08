# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM DISCORD BOT FACADE // phone/babyBot_discord.py
# v13.4

from phone.discord_bot.bot import BABYBOT_DISCORD
from phone.discord_bot.cog import babyBot_DISCORD_COG
from config import modelDevice

def run_discord_bot(babyLLM, tutor, librarian, scribe, calligraphist, token):
    """Start the Discord bot using the provided LLM components and token."""
    bot = BABYBOT_DISCORD(babyLLM, tutor, librarian, scribe, calligraphist)
    babyLLM.loadModel()
    babyLLM.to(modelDevice)
    bot.run(token)

__all__ = ["BABYBOT_DISCORD", "babyBot_DISCORD_COG", "run_discord_bot"]
