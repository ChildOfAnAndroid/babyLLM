#!/usr/bin/env python3
# v1.1

# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM COMPREHENSIVE GIT ANALYZER // comprehensive_git_analyzer.py

"""
ULTIMATE git history analyzer that gets EVERY SINGLE COMMIT from ALL branches!
This will make sure we have the complete 338+ commit history!
"""

import subprocess
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

def get_all_commit_count():
    """Get comprehensive commit count from all possible sources"""
    print("🔍 Getting comprehensive commit count...")
    
    try:
        # Method 1: All commits on all branches
        result1 = subprocess.run(['git', 'rev-list', '--all', '--count'], 
                                capture_output=True, text=True, cwd='.')
        all_commits = int(result1.stdout.strip()) if result1.returncode == 0 else 0
        
        # Method 2: Current branch only
        result2 = subprocess.run(['git', 'rev-list', '--count', 'HEAD'], 
                                capture_output=True, text=True, cwd='.')
        main_commits = int(result2.stdout.strip()) if result2.returncode == 0 else 0
        
        # Method 3: Including unreachable commits
        result3 = subprocess.run(['git', 'rev-list', '--all', '--reflog', '--count'], 
                                capture_output=True, text=True, cwd='.')
        reflog_commits = int(result3.stdout.strip()) if result3.returncode == 0 else 0
        
        print(f"📊 Commit Count Analysis:")
        print(f"  All branches: {all_commits}")
        print(f"  Main branch: {main_commits}")
        print(f"  Including reflog: {reflog_commits}")
        
        return max(all_commits, main_commits, reflog_commits)
        
    except Exception as e:
        print(f"Error counting commits: {e}")
        return 0

def get_comprehensive_file_history() -> Dict[str, List[Tuple[str, str, str]]]:
    """Get COMPLETE git history for ALL Python files from ALL branches"""
    print("🚀 Analyzing COMPLETE git history from ALL branches...")
    
    try:
        # Get history from ALL branches and reflog
        result = subprocess.run([
            'git', 'log', '--all', '--reflog',
            '--pretty=format:%H|%ad|%an', '--date=short', '--name-only'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode != 0:
            print(f"Git error: {result.stderr}")
            return {}
        
        lines = result.stdout.strip().split('\n')
        file_history = defaultdict(list)
        current_commit = None
        current_date = None
        current_author = None
        
        for line in lines:
            if '|' in line and len(line.split('|')) == 3:  # Commit line
                commit_hash, date_str, author = line.split('|', 2)
                current_commit = commit_hash[:7]
                current_date = date_str
                current_author = author
            elif line.strip() and line.endswith('.py'):  # Python file
                if current_commit and current_date:
                    # Normalize all possible path variations
                    normalized_path = line.strip()
                    
                    # Handle ALL possible old path structures
                    if normalized_path.startswith('BRAIN/'):
                        normalized_path = 'brain/' + normalized_path[6:]
                    elif normalized_path.startswith('PHONE/'):
                        normalized_path = 'phone/' + normalized_path[6:]
                    elif normalized_path.startswith('SCHOOL/'):
                        normalized_path = 'school/' + normalized_path[7:]
                    elif normalized_path.startswith('VER1/'):
                        # Skip VER1 files as they're archived versions
                        continue
                    
                    file_history[normalized_path].append((current_date, current_commit, current_author))
        
        print(f"📈 Found history for {len(file_history)} Python files!")
        return dict(file_history)
    
    except Exception as e:
        print(f"Error analyzing comprehensive git history: {e}")
        return {}

def show_top_edited_files(file_history: Dict, top_n: int = 20):
    """Show the most edited files with their commit counts"""
    print(f"\n🏆 TOP {top_n} MOST EDITED FILES:")
    print("=" * 60)
    
    # Sort files by total commit count
    sorted_files = sorted(file_history.items(), 
                         key=lambda x: len(x[1]), reverse=True)
    
    for i, (file_path, commits) in enumerate(sorted_files[:top_n], 1):
        unique_days = len(set(commit[0] for commit in commits))
        latest_day = max(commits, key=lambda x: x[0])[0]
        edits_on_latest = sum(1 for c in commits if c[0] == latest_day)
        
        version = f"v{unique_days}.{edits_on_latest}"
        print(f"{i:2d}. {file_path:<50} {version:<8} ({len(commits)} commits, {unique_days} days)")

def analyze_commit_patterns(file_history: Dict):
    """Analyze interesting patterns in the commit history"""
    print(f"\n📊 COMMIT PATTERN ANALYSIS:")
    print("=" * 40)
    
    total_commits = sum(len(commits) for commits in file_history.values())
    total_files = len(file_history)
    
    # Find files with most days
    max_days = max((len(set(c[0] for c in commits)) for commits in file_history.values()), default=0)
    max_commits = max((len(commits) for commits in file_history.values()), default=0)
    
    # Find most active days
    all_dates = []
    for commits in file_history.values():
        all_dates.extend(commit[0] for commit in commits)
    
    from collections import Counter
    date_counts = Counter(all_dates)
    most_active_day = max(date_counts.items(), key=lambda x: x[1])
    
    print(f"Total commits to Python files: {total_commits}")
    print(f"Total Python files in history: {total_files}")
    print(f"Most commits on single file: {max_commits}")
    print(f"Most days editing single file: {max_days}")
    print(f"Most active day: {most_active_day[0]} ({most_active_day[1]} commits)")
    
    return {
        'total_commits': total_commits,
        'total_files': total_files,
        'max_commits': max_commits,
        'max_days': max_days
    }

def main():
    """Comprehensive analysis of ALL git history"""
    print("🎊 COMPREHENSIVE GIT HISTORY ANALYZER")
    print("=" * 50)
    
    # Get total commit count
    total_commits = get_all_commit_count()
    print(f"\n🎯 MAXIMUM COMMITS FOUND: {total_commits}")
    
    # Get comprehensive file history
    file_history = get_comprehensive_file_history()
    
    if not file_history:
        print("❌ Could not analyze git history")
        return 1
    
    # Show analysis
    stats = analyze_commit_patterns(file_history)
    show_top_edited_files(file_history)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"We found {stats['total_commits']} commits across {stats['total_files']} Python files!")
    print(f"Your most edited file has {stats['max_commits']} commits across {stats['max_days']} days! 🔥")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
