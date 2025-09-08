#!/usr/bin/env python3
# v1.1

# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM VERSION DEMO // demo_versioning.py

"""
Demonstration of how the clean versioning system works:

Day 1: v1.1, v1.2, v1.3 (first day, multiple edits)
Day 2: v2.1, v2.2 (second day, edit counter resets)
Day 3: v3.1 (third day)
Day 5: v4.1 (skipped day 4, so this is still the 4th unique day)
"""

import json
from pathlib import Path
from datetime import date, timedelta

def simulate_version_progression():
    """Show how versions would progress over time"""
    
    # Simulate editing on different days
    test_dates = [
        "2025-09-08",  # Day 1: v1.x
        "2025-09-09",  # Day 2: v2.x  
        "2025-09-10",  # Day 3: v3.x
        "2025-09-12",  # Day 4: v4.x (skipped day 11)
    ]
    
    file_history = {
        "days_edited": [],
        "current_day": None,
        "edits_today": 0
    }
    
    print("Version progression simulation:")
    print("=" * 40)
    
    for day in test_dates:
        print(f"\nEditing on {day}:")
        
        # Simulate multiple edits on same day
        edits_on_this_day = [1, 2, 3] if day == test_dates[0] else [1, 2]
        
        for edit_num in edits_on_this_day:
            # Check if this is a new day
            if file_history["current_day"] != day:
                # New day
                if day not in file_history["days_edited"]:
                    file_history["days_edited"].append(day)
                file_history["current_day"] = day
                file_history["edits_today"] = 1
            else:
                # Same day - increment edit count
                file_history["edits_today"] += 1
            
            days_count = len(file_history["days_edited"])
            edits_today = file_history["edits_today"]
            version = f"v{days_count}.{edits_today}"
            
            print(f"  Edit #{edit_num}: {version}")
    
    print(f"\nTotal unique days edited: {len(file_history['days_edited'])}")
    print(f"Days: {file_history['days_edited']}")

if __name__ == "__main__":
    simulate_version_progression()
