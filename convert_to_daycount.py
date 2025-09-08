#!/usr/bin/env python3
# v1.1

# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM DAYCOUNT CONVERTER // convert_to_daycount.py
# v20250908.1.0

"""
Convert all Python files from semantic versioning (v0.0.1) to daycount format (vYYYYMMDD.1.0)
"""

import sys
import re
from pathlib import Path
from datetime import date

# Add the tools directory to the path
tools_dir = Path(__file__).parent / "tools"
sys.path.insert(0, str(tools_dir))

import header_version

def convert_file_to_daycount(file_path: Path):
    """Convert a single file from semantic versioning to daycount format"""
    lines = header_version.read_text(file_path)
    if not lines:
        return False, "Empty file"
    
    # Find the version line
    changed = False
    today_str = date.today().strftime("%Y%m%d")
    
    for i, line in enumerate(lines):
        # Match semantic version pattern v0.0.1
        if re.match(r'^#\s*v\d+\.\d+\.\d+\s*$', line):
            # Replace with daycount format
            lines[i] = f"# v{today_str}.1.0"
            changed = True
            break
    
    if changed:
        header_version.write_text(file_path, lines)
        return True, f"v{today_str}.1.0"
    
    return False, "No semantic version found"

def main():
    root = Path(".")
    files = header_version.iter_py_files(root)
    converted_count = 0
    
    print("Converting files to daycount format...")
    
    for file_path in files:
        rel_path = str(file_path.relative_to(root))
        success, message = convert_file_to_daycount(file_path)
        
        if success:
            print(f"[CONVERTED] {rel_path} -> {message}")
            converted_count += 1
    
    print(f"\nConverted {converted_count} files to daycount format.")
    
    # Update VERSIONS.json
    if converted_count > 0:
        print("Updating VERSIONS.json...")
        import subprocess
        result = subprocess.run([
            sys.executable, 
            "version_bump_runner.py",
            "--daycount-bump",
            "--root", 
            "."
        ], capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
