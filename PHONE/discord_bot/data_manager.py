# v1.9
"""
Centralized data persistence manager for babyLLM
Handles all save/load operations with batching and error recovery
"""
import json
import asyncio
import time
import inspect
from pathlib import Path
from typing import Any, Dict, Optional
from .logger import logger

class DataManager:
    """Centralized data persistence with batching and safety"""
    
    def __init__(self, batch_interval: float = 5.0):
        self.batch_interval = batch_interval
        self.pending_saves = set()
        self.last_save_times = {}
        self.save_task = None
        self._save_callbacks = {}  # Store actual save functions
        self._bot_ref = None
    
    def set_bot_reference(self, bot):
        """Set reference to the bot for accessing save methods"""
        self._bot_ref = bot
    
    def register_save_callback(self, data_type: str, callback_func):
        """Register a save callback function for a data type"""
        self._save_callbacks[data_type] = callback_func
    
    def request_save(self, data_type: str, urgent: bool = False):
        """Request a save operation (batched unless urgent)"""
        if urgent:
            return self._immediate_save(data_type)
        
        self.pending_saves.add(data_type)
        if self.save_task is None or self.save_task.done():
            self.save_task = asyncio.create_task(self._batched_save())
    
    async def _batched_save(self):
        """Perform batched saves after a delay"""
        await asyncio.sleep(self.batch_interval)
        
        if self.pending_saves:
            save_count = len(self.pending_saves)
            for data_type in list(self.pending_saves):
                self._immediate_save(data_type)
            
            logger.info("DATA_BATCH", f"Completed batched save of {save_count} data types")
        
        self.pending_saves.clear()
    
    def _immediate_save(self, data_type: str) -> bool:
        """Perform immediate save operation"""
        if data_type not in self._save_callbacks:
            logger.error("DATA_SAVE", f"Unknown data type: {data_type}")
            return False
        
        try:
            # Call the registered save callback (supports sync or async)
            callback = self._save_callbacks[data_type]
            if inspect.iscoroutinefunction(callback):
                asyncio.create_task(callback())
            else:
                result = callback()
                # If a coroutine was returned (bound async method), schedule it
                if inspect.iscoroutine(result):
                    asyncio.create_task(result)
            self.last_save_times[data_type] = time.time()
            logger.debug("DATA_SAVE", f"Saved {data_type}")
            return True
        except Exception as e:
            logger.error("DATA_SAVE", f"Failed to save {data_type}: {e}")
            return False
    
    def get_save_frequency(self, data_type: str) -> Optional[float]:
        """Get how often this data type is being saved"""
        if data_type not in self.last_save_times:
            return None
        return time.time() - self.last_save_times[data_type]
    
    async def emergency_save_all(self):
        """Emergency save of all registered data types"""
        logger.emergency("DATA_SAVE", "Performing emergency save of all data")
        for data_type in self._save_callbacks:
            self._immediate_save(data_type)

# Global data manager instance
data_manager = DataManager()
