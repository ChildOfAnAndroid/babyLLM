# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM // phone/discord_bot/__init__.py
# v1.1

"""bbys on discord!"""

import asyncio
import logging

from config import modelDevice
from phone.discord_bot import utils
from phone.discord_bot.bot import BABYBOT_DISCORD
from phone.discord_bot.cog import babyBot_DISCORD_COG
from phone.discord_bot.context import create_fake_context


def run_discord_bot(babyLLM, tutor, librarian, scribe, calligraphist, token):
    """Start the canonical Discord bot with model setup and bounded retries."""
    bot = BABYBOT_DISCORD(babyLLM, tutor, librarian, scribe, calligraphist)
    babyLLM.loadModel()
    babyLLM.to(modelDevice)
    babyLLM.activate_mps_owner()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("discord").setLevel(logging.INFO)

    async def run_bot_with_retry():
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                await bot.start(token)
                return
            except Exception as error:
                print(
                    f"[DISCORD_BOT] Connection failed "
                    f"(attempt {attempt}/{max_retries}): {error}"
                )
                if attempt == max_retries:
                    print("[DISCORD_BOT] Max retries reached. Exiting.")
                    raise
                await asyncio.sleep(5)
                print("[DISCORD_BOT] Retrying connection...")

    try:
        asyncio.run(run_bot_with_retry())
    except KeyboardInterrupt:
        print("[DISCORD_BOT] Bot stopped by user.")
    except Exception as error:
        print(f"[DISCORD_BOT] Fatal error: {error}")
        raise

__all__ = [
    "BABYBOT_DISCORD",
    "babyBot_DISCORD_COG",
    "create_fake_context",
    "run_discord_bot",
    "utils",
]
