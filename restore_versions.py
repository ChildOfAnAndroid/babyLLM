#!/usr/bin/env python3
# v1.1

# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM HISTORICAL VERSION RESTORER // restore_versions.py

"""
EMERGENCY version restore script! 
Restores the historically accurate versions that got reset.
"""

import sys
from pathlib import Path

# Add tools directory
tools_dir = Path(__file__).parent / "tools"
sys.path.insert(0, str(tools_dir))
import header_version

# The historically accurate versions we calculated earlier
HISTORICAL_VERSIONS = {
    'babyLLM.py': 'v67.2',
    'config.py': 'v80.2', 
    'school/staffroom/tutor.py': 'v53.2',
    'phone/discord_bot/cog.py': 'v11.5',
    'school/staffroom/calligraphist.py': 'v36.2',
    'brain/LAYERS/interneuronNetwork.py': 'v31.2',
    'wakeup.py': 'v32.3',
    'brain/LAYERS/memory.py': 'v26.2',
    'textCleaningTool.py': 'v28.1',
    'phone/discord_bot/bot.py': 'v9.4',
    'school/staffroom/librarian.py': 'v20.2',
    'brain/LAYERS/embed.py': 'v19.2',
    'helpers.py': 'v6.4',
    'brain/LAYERS/logits.py': 'v18.2',
    'school/staffroom/HE_IS_SCRIBE.py': 'v16.2',
    'phone/babyBot.py': 'v11.3',
    'brain/LAYERS/attention.py': 'v2.2',
    'phone/discord_bot/bbyLocal.py': 'v3.1',
    'phone/command_utils.py': 'v3.2',
    'CONFIG_trainingData.py': 'v4.1',
    'phone/babyBot_discord.py': 'v12.1',
    'phone/discord_bot/__init__.py': 'v3.1',
    'school/staffroom/counsellor.py': 'v6.1',
    'phone/discord_bot/context.py': 'v3.1',
    'wakeupUtils.py': 'v1.2',
    'phone/discord_bot/shoutouts.py': 'v1.1',
    'phone/discord_bot/utils.py': 'v5.2',
    'secret.py': 'v1.1',
    'school/staffroom/painter.py': 'v1.1',
    # Add more as needed
}

def restore_file_version(filepath: Path, version: str) -> bool:
    """Restore a single file's version"""
    if not filepath.exists():
        return False
        
    try:
        lines = header_version.read_text(filepath)
        if not lines:
            return False
            
        # Find and update version line
        insert_at = header_version.find_insert_index(lines)
        block_end = header_version.get_header_block_end(lines, insert_at)
        
        version_updated = False
        for i in range(insert_at, block_end):
            if lines[i].strip().startswith('# v'):
                old_version = lines[i].strip()
                lines[i] = f"# {version}"
                version_updated = True
                print(f"  {filepath.relative_to(Path('.'))}: {old_version} → # {version}")
                break
        
        if version_updated:
            header_version.write_text(filepath, lines)
            return True
            
    except Exception as e:
        print(f"  ⚠️  Error restoring {filepath}: {e}")
    
    return False

def main():
    """Restore all historical versions"""
    print("🚨 EMERGENCY HISTORICAL VERSION RESTORE")
    print("=" * 50)
    
    restored_count = 0
    
    for rel_path, version in HISTORICAL_VERSIONS.items():
        filepath = Path(rel_path)
        if restore_file_version(filepath, version):
            restored_count += 1
    
    # Update VERSIONS.json
    versions_json = header_version.load_versions()
    for rel_path, version in HISTORICAL_VERSIONS.items():
        versions_json[rel_path] = version
    header_version.save_versions(versions_json)
    
    print(f"\n🎉 Restored {restored_count} files to their historical versions!")
    print("✅ Your version numbers are now historically accurate again!")
    
    # Clean up this script
    Path(__file__).unlink()
    print("🧹 Cleaned up restore script")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
