#!/usr/bin/env python3
# v1.5

# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM AUTO VERSION BUMP // auto_version_bump.py
# v0.0.1

"""
Auto version bump wrapper script that runs on file save.
This script automatically increments version numbers using the clean version system:
- vX.Y where X = number of unique days edited, Y = edits on current day
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the clean version bump command"""
    script_dir = Path(__file__).parent
    clean_version_script = script_dir / "clean_version_bump.py"
    
    if not clean_version_script.exists():
        print(f"Error: {clean_version_script} not found!")
        return 1
    
    try:
        # Run the clean version bump command
        result = subprocess.run([
            sys.executable,
            str(clean_version_script),
            "--clean-bump",
            "--root",
            str(script_dir)
        ], capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            
        return result.returncode
    except Exception as e:
        print(f"Error running version bump: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
