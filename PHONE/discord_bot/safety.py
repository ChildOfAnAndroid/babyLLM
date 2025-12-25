# v1.1
"""
Centralized safety validation system for babyLLM
Prevents code duplication of NaN/Inf checks and value validation
"""
import math
import asyncio
from typing import Union, Any, Optional, Dict
from .logger import logger
from .data_manager import data_manager

def _emit_safety_debug(message: str) -> None:
    bot = getattr(data_manager, "_bot_ref", None)
    if not bot or not hasattr(bot, "_discord_debug"):
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(bot._discord_debug(message))
        return
    except RuntimeError:
        pass
    try:
        loop = getattr(bot, "loop", None)
        if loop and loop.is_running():
            loop.create_task(bot._discord_debug(message))
    except Exception:
        pass

class SafetyValidator:
    """Centralized validation for numerical safety and data integrity"""
    
    @staticmethod
    def is_safe_number(value: Any, allow_negative: bool = True) -> bool:
        """Check if a value is a safe, finite number"""
        return (
            isinstance(value, (int, float)) and
            not math.isnan(value) and
            not math.isinf(value) and
            (allow_negative or value >= 0)
        )
    
    @staticmethod
    def clamp_value(value: Union[int, float], min_val: float, max_val: float, 
                   fallback: float = 0.0, context: str = "") -> float:
        """Safely clamp a value to a range with fallback for invalid values"""
        if not SafetyValidator.is_safe_number(value):
            if context:
                logger.warn("SAFETY_CLAMP", f"Invalid value {value} in {context}, using fallback {fallback}")
            return fallback
        
        clamped = max(min_val, min(max_val, float(value)))
        if clamped != value and context:
            logger.debug("SAFETY_CLAMP", f"Clamped {value} to {clamped} in {context}")
        
        return clamped
    
    @staticmethod
    def validate_user_memory(mem: Dict[str, Any], user_id: str = "unknown") -> Dict[str, Any]:
        """Validate and repair user memory data structure"""
        defaults = {
            "BBY": 420.0,
            "creative_combo": 1,
            "spammer": 1,
            "spamMax": 0.8,
            "display_name": "",
            "messages": 0,
            "last_seen": 0.0
        }
        
        # Define which fields should be numbers vs strings
        numeric_fields = {"BBY", "creative_combo", "spammer", "spamMax", "messages", "last_seen"}
        string_fields = {"display_name"}
        
        repaired = False
        repair_notes = []
        for key, default_value in defaults.items():
            if key not in mem:
                mem[key] = default_value
                repaired = True
            elif key in numeric_fields and not SafetyValidator.is_safe_number(mem[key]):
                logger.emergency("MEMORY_REPAIR", f"Corrupted {key}={mem[key]} for user {user_id}, resetting to {default_value}")
                mem[key] = default_value
                repaired = True
                repair_notes.append(f"{key}=invalid")
            elif key in string_fields and not isinstance(mem[key], str):
                logger.emergency("MEMORY_REPAIR", f"Corrupted {key}={mem[key]} for user {user_id}, resetting to '{default_value}'")
                mem[key] = default_value
                repaired = True
                repair_notes.append(f"{key}=invalid_type")
        
        # Apply specific safety ranges only for critical corruption, not normal high values
        if not SafetyValidator.is_safe_number(mem["creative_combo"]):
            logger.emergency("MEMORY_REPAIR", f"Corrupted creative_combo={mem['creative_combo']} for user {user_id}, resetting to 1")
            mem["creative_combo"] = 1
            repaired = True
            repair_notes.append("creative_combo=invalid")
        elif mem["creative_combo"] < -50:  # Only clamp extremely negative values
            mem["creative_combo"] = 1
            repaired = True
            repair_notes.append("creative_combo=clamped")
            
        if not SafetyValidator.is_safe_number(mem["spammer"]):
            logger.emergency("MEMORY_REPAIR", f"Corrupted spammer={mem['spammer']} for user {user_id}, resetting to 1")
            mem["spammer"] = 1
            repaired = True
            repair_notes.append("spammer=invalid")
        elif mem["spammer"] < -50:  # Only clamp extremely negative values
            mem["spammer"] = 1
            repaired = True
            repair_notes.append("spammer=clamped")
            
        mem["spamMax"] = SafetyValidator.clamp_value(
            mem["spamMax"], 0.1, 2.0, 0.8, f"spamMax for {user_id}"
        )
        
        # Validate and fix inventory item counts - ensure dict with integer counts
        if "inventory" in mem and not isinstance(mem["inventory"], dict):
            logger.emergency("MEMORY_REPAIR", f"Corrupted inventory type for user {user_id}, resetting inventory to empty dict")
            mem["inventory"] = {}
            repaired = True
            repair_notes.append("inventory=invalid_type")

        if "inventory" in mem and isinstance(mem["inventory"], dict):
            inventory_repaired = False
            inventory_change_count = 0
            for item_name, count in list(mem["inventory"].items()):
                if not SafetyValidator.is_safe_number(count):
                    logger.emergency("MEMORY_REPAIR", f"Corrupted inventory count {item_name}={count} for user {user_id}, removing item")
                    del mem["inventory"][item_name]
                    inventory_repaired = True
                    repaired = True
                    inventory_change_count += 1
                elif not isinstance(count, int):
                    # Round fractional item counts to integers
                    fixed_count = int(round(count)) if count > 0 else 0
                    if fixed_count <= 0:
                        # Remove items with 0 or negative count
                        del mem["inventory"][item_name]
                        logger.info("MEMORY_REPAIR", f"Removed item {item_name} with count {count} for user {user_id}")
                    else:
                        mem["inventory"][item_name] = fixed_count
                        if abs(count - fixed_count) > 0.001:  # Only log if there was a significant change
                            logger.info("MEMORY_REPAIR", f"Rounded inventory count {item_name}: {count} -> {fixed_count} for user {user_id}")
                    inventory_repaired = True
                    repaired = True
                    inventory_change_count += 1
                elif count <= 0:
                    # Remove items with non-positive counts
                    del mem["inventory"][item_name]
                    logger.info("MEMORY_REPAIR", f"Removed item {item_name} with non-positive count {count} for user {user_id}")
                    inventory_repaired = True
                    repaired = True
                    inventory_change_count += 1
            
            if inventory_repaired:
                logger.info("MEMORY_REPAIR", f"Fixed inventory items for user {user_id}")
                repair_notes.append(f"inventory=items_fixed({inventory_change_count})")
        
        if repaired:
            logger.info("MEMORY_REPAIR", f"Repaired user memory for {user_id}")
            if repair_notes:
                _emit_safety_debug(f"[SAFETY_REPAIR] {user_id}: {', '.join(repair_notes)}")
        
        return mem
    
    @staticmethod
    def validate_bby_transaction(amount: Any, context: str = "", allow_large_negative: bool = False) -> Optional[float]:
        """Validate BBY transaction amount"""
        if not SafetyValidator.is_safe_number(amount):
            logger.warn("BBY_SAFETY", f"Invalid BBY amount {amount} in {context}")
            return None
        
        # Different limits based on context - allow larger negative amounts for decay system
        if allow_large_negative:
            # For decay operations, allow very large magnitudes without clamping.
            # Still require finite numbers via is_safe_number at the top.
            return float(amount)
        else:
            # For normal transactions, allow meme values (no clamp, just NaN/Inf check)
            return float(amount)
    
    @staticmethod
    def validate_brain_sentiment(sentiment: Any) -> float:
        """Safely validate brain sentiment value"""
        if not SafetyValidator.is_safe_number(sentiment):
            logger.warn("BRAIN_SENTIMENT_SAFETY", f"Corrupted brain sentiment: {sentiment}, using 0.0")
            return 0.0
        
        # Clamp to safe sentiment range
        return SafetyValidator.clamp_value(sentiment, -1.0, 1.0, 0.0, "brain sentiment")

# Global validator instance
safety = SafetyValidator()
