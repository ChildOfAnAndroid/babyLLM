# BabyLLM Automatic Versioning System

## How It Works

Your versioning system now automatically increments version numbers using the **clean format**:

### Version Format: `v<DAYS_EDITED>.<EDITS_TODAY>`

- **DAYS_EDITED**: Number of unique days this file has been edited on
- **EDITS_TODAY**: Number of times edited on the current day

### Examples:
- `v1.1` - First edit on the first day this file was ever edited
- `v1.2` - Second edit on the same day  
- `v1.3` - Third edit on the same day
- `v2.1` - First edit on the second day this file was edited (resets daily counter)
- `v3.1` - First edit on the third day (even if you skipped some days)

## What Happens Automatically

1. **On File Save**: Every time you save a Python file in VS Code, the version number automatically increments
2. **Daily Reset**: The edit counter (second number) resets to 1 when you edit a file on a new day
3. **Day Counter**: The first number increments only when you edit a file on a day you haven't edited it before
4. **History Tracking**: All editing history is tracked in VERSION_HISTORY.json

## Version Progression Example

```
Day 1 (Sept 8): v1.1 → v1.2 → v1.3
Day 2 (Sept 9): v2.1 → v2.2  
Day 3 (Sept 10): v3.1
Day 5 (Sept 12): v4.1  # Note: Still v4.x even though we skipped day 4
```

## Files Created

- `.vscode/settings.json` - Configures VS Code to run versioning on save
- `.vscode/tasks.json` - VS Code tasks for manual version operations
- `clean_version_bump.py` - Clean version bumping script (no timestamps)
- `VERSION_HISTORY.json` - Tracks editing history per file
- `demo_versioning.py` - Demonstrates version progression

## Manual Commands

If you need to manually run versioning commands:

```bash
# Bump versions for all files (clean format)
python3 clean_version_bump.py --clean-bump --root .

# Run demonstration
python3 demo_versioning.py
```

## Extension Dependency

The system requires the "Run on Save" extension by emeraldwalk, which has been installed automatically.

## Status

✅ **Clean versioning system is now fully operational!** All files now use the clean `v<DAYS>.<EDITS>` format and will automatically version on save without visible timestamps.
