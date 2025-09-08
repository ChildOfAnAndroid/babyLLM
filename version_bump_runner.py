#!/usr/bin/env python3
# v1.1

# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM VERSION BUMP RUNNER // version_bump_runner.py

"""
Fixed version bumping script that properly handles the daycount bump functionality.
This script patches the argument parsing issue in header_version.py
"""

import sys
from pathlib import Path

# Add the tools directory to the path
tools_dir = Path(__file__).parent / "tools"
sys.path.insert(0, str(tools_dir))

# Import the original module
import header_version

# Monkey patch the main function to handle --daycount-bump without requiring a command
original_main = header_version.main

def patched_main(argv):
    import argparse
    from pathlib import Path
    
    # Check if this is a daycount bump call
    if '--daycount-bump' in argv:
        # Create a minimal parser for daycount bump
        ap = argparse.ArgumentParser()
        ap.add_argument("--daycount-bump", action="store_true")
        ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
        ap.add_argument("--dry-run", action="store_true")
        args = ap.parse_args(argv)
        
        # Execute daycount bump directly
        root = Path(args.root)
        files = header_version.iter_py_files(root)
        versions = header_version.load_versions()
        changed_any = False
        
        for path in files:
            rel = str(path.relative_to(root))
            lines = header_version.read_text(path)
            if not lines:
                continue
            new_lines, changed, stamp = header_version.bump_daycount(lines)
            if changed and not args.dry_run:
                header_version.write_text(path, new_lines)
                changed_any = True
            if changed:
                print(f"[BUMP:daycount] {rel} -> {stamp}")
            versions[rel] = stamp
            
        if changed_any and not args.dry_run:
            header_version.save_versions(versions)
        return 0
    
    # Otherwise use original function
    return original_main(argv)

# Replace the main function
header_version.main = patched_main

if __name__ == "__main__":
    sys.exit(header_version.main(sys.argv[1:]))
