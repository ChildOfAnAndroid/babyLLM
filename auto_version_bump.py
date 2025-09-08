#!/usr/bin/env python3
# v1.1

# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM AUTO VERSION BUMP // auto_version_bump.py
# v0.0.1

"""
Auto version bump wrapper script that runs on file save.
This script automatically increments version numbers using the daycount system:
- vYYYYMMDD.N.0 where YYYYMMDD is the date and N is the edit count for that day
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the daycount bump command"""
    script_dir = Path(__file__).parent
    header_version_script = script_dir / "tools" / "header_version.py"
    
    if not header_version_script.exists():
        print(f"Error: {header_version_script} not found!")
        return 1
    
    try:
        # Run the daycount bump command
        result = subprocess.run([
            sys.executable,
            str(header_version_script),
            "--daycount-bump",
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
