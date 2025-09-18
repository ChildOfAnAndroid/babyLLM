# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM DISCORD BOT FACADE // phone/babyBot_discord.py
# v2.2

from phone.discord_bot.bot import BABYBOT_DISCORD
from phone.discord_bot.cog import babyBot_DISCORD_COG
from config import modelDevice

def run_discord_bot(babyLLM, tutor, librarian, scribe, calligraphist, token):
    """Start the Discord bot using the provided LLM components and token."""
    bot = BABYBOT_DISCORD(babyLLM, tutor, librarian, scribe, calligraphist)
    babyLLM.loadModel()
    babyLLM.to(modelDevice)
    
    # Add better connection handling and error recovery
    import asyncio
    import logging
    
    # Set up logging to help debug connection issues
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('discord')
    logger.setLevel(logging.INFO)
    
    async def run_bot_with_retry():
        """Run bot with automatic reconnection on failures"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                await bot.start(token)
            except Exception as e:
                retry_count += 1
                print(f"[DISCORD_BOT] Connection failed (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(5)  # Wait before retrying
                    print("[DISCORD_BOT] Retrying connection...")
                else:
                    print("[DISCORD_BOT] Max retries reached. Exiting.")
                    raise
    
    # Run the bot with retry logic
    try:
        asyncio.run(run_bot_with_retry())
    except KeyboardInterrupt:
        print("[DISCORD_BOT] Bot stopped by user.")
    except Exception as e:
        print(f"[DISCORD_BOT] Fatal error: {e}")
        raise

__all__ = ["BABYBOT_DISCORD", "babyBot_DISCORD_COG", "run_discord_bot"]
