#!/usr/bin/env python3
# v4.2

# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM CLEAN VERSION BUMP RUNNER // clean_version_bump.py

"""
Clean version bumping script that uses v<DAYS_EDITED>.<EDITS_TODAY> format.
- DAYS_EDITED: Number of unique days this file has been edited
- EDITS_TODAY: Number of edits on the current day
"""

import sys
import json
import re
from pathlib import Path
from datetime import date
from typing import Dict, Tuple, List

# Add the tools directory to the path
tools_dir = Path(__file__).parent / "tools"
sys.path.insert(0, str(tools_dir))

import header_version

# Version tracking file
VERSION_HISTORY_PATH = Path("VERSION_HISTORY.json")
RE_CLEAN_VERSION = re.compile(r"^#\s*v(\d+)\.(\d+)\s*$")

def load_version_history() -> Dict:
    """Load version history tracking unique days and edit counts"""
    if VERSION_HISTORY_PATH.exists():
        try:
            return json.loads(VERSION_HISTORY_PATH.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}

def save_version_history(history: Dict) -> None:
    """Save version history"""
    VERSION_HISTORY_PATH.write_text(
        json.dumps(history, indent=2, sort_keys=True), 
        encoding='utf-8'
    )

def get_today_str() -> str:
    """Get today's date as string"""
    return date.today().isoformat()

def bump_clean_version(file_path: Path, rel_path: str) -> Tuple[List[str], bool, str]:
    """Bump version using clean format v<DAYS_EDITED>.<EDITS_TODAY>"""
    lines = header_version.read_text(file_path)
    if not lines:
        return lines, False, ""
    
    # Load version history
    history = load_version_history()
    today = get_today_str()
    
    # Initialize file history if not exists
    if rel_path not in history:
        history[rel_path] = {
            "days_edited": [],
            "current_day": None,
            "edits_today": 0
        }
    
    file_history = history[rel_path]
    
    # Check if this is a new day
    if file_history["current_day"] != today:
        # New day - add to days_edited if not already there
        if today not in file_history["days_edited"]:
            file_history["days_edited"].append(today)
        file_history["current_day"] = today
        file_history["edits_today"] = 1
    else:
        # Same day - increment edit count
        file_history["edits_today"] += 1
    
    # Calculate version numbers
    days_edited_count = len(file_history["days_edited"])
    edits_today = file_history["edits_today"]
    new_version = f"v{days_edited_count}.{edits_today}"
    
    # Find and update version line
    insert_at = header_version.find_insert_index(lines)
    block_end = header_version.get_header_block_end(lines, insert_at)
    
    version_updated = False
    for i in range(insert_at, block_end):
        if RE_CLEAN_VERSION.match(lines[i] or "") or header_version.RE_VERSION.match(lines[i] or ""):
            lines[i] = f"# {new_version}"
            version_updated = True
            break
    
    # If no version line found, add one
    if not version_updated:
        lines = lines[:block_end] + [f"# {new_version}"] + lines[block_end:]
    
    # Save history
    history[rel_path] = file_history
    save_version_history(history)
    
    return lines, True, new_version

def main(argv: List[str]) -> int:
    """Main function for clean version bumping"""
    import argparse
    
    ap = argparse.ArgumentParser(description="Clean version bumping with v<DAYS>.<EDITS> format")
    ap.add_argument("--clean-bump", action="store_true", help="Bump versions using clean format")
    ap.add_argument("--root", default=".", help="Project root directory")
    ap.add_argument("--dry-run", action="store_true", help="Don't write changes")
    
    args = ap.parse_args(argv)
    
    if args.clean_bump:
        root = Path(args.root)
        files = header_version.iter_py_files(root)
        versions = header_version.load_versions()
        changed_any = False
        
        for file_path in files:
            rel_path = str(file_path.relative_to(root))
            new_lines, changed, new_version = bump_clean_version(file_path, rel_path)
            
            if changed and not args.dry_run:
                header_version.write_text(file_path, new_lines)
                changed_any = True
            
            if changed:
                print(f"[CLEAN-BUMP] {rel_path} -> {new_version}")
                versions[rel_path] = new_version
        
        if changed_any and not args.dry_run:
            header_version.save_versions(versions)
        
        return 0
    
    print("Use --clean-bump to bump versions")
    return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
