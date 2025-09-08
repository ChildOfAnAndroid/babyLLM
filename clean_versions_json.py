#!/usr/bin/env python3
# v1.1

# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM VERSIONS CLEANER // clean_versions_json.py

"""
Clean up VERSIONS.json to remove entries for non-existent files
"""

import json
from pathlib import Path

def clean_versions_json():
    """Remove entries for files that no longer exist"""
    print("🧹 CLEANING VERSIONS.JSON...")
    
    versions_file = Path("VERSIONS.json")
    if not versions_file.exists():
        print("❌ VERSIONS.json not found!")
        return
    
    versions = json.loads(versions_file.read_text())
    original_count = len(versions)
    
    # Check each file
    cleaned_versions = {}
    removed_files = []
    
    for filepath, version in versions.items():
        full_path = Path(filepath)
        
        # Skip if file doesn't exist
        if not full_path.exists():
            removed_files.append(filepath)
            continue
            
        cleaned_versions[filepath] = version
    
    # Save cleaned versions
    versions_file.write_text(
        json.dumps(cleaned_versions, indent=2, sort_keys=True)
    )
    
    print(f"✅ Cleaned VERSIONS.json:")
    print(f"   Original entries: {original_count}")
    print(f"   Cleaned entries: {len(cleaned_versions)}")
    print(f"   Removed entries: {len(removed_files)}")
    
    if removed_files:
        print("\n🗑️  Removed entries for non-existent files:")
        for f in removed_files[:10]:  # Show first 10
            print(f"   - {f}")
        if len(removed_files) > 10:
            print(f"   ... and {len(removed_files) - 10} more")

def main():
    """Main function"""
    print("🚀 VERSIONS.JSON CLEANER")
    print("=" * 40)
    
    clean_versions_json()
    
    print("\n🎉 VERSIONS.json cleaned successfully!")
    
    # Self-destruct
    Path(__file__).unlink()
    print("🧹 Cleaned up cleaner script")

if __name__ == "__main__":
    main()
