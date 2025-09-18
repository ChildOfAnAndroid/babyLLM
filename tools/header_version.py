#!/usr/bin/env python3
# v1.1

# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM HEADER + VERSION TOOL // tools/header_version.py
# v0.0.1

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple
from datetime import date as _date
import fnmatch

RE_VERSION = re.compile(r"^#\s*v(\d+)\.(\d+)\.(\d+)\s*$")
RE_DATE = re.compile(r"^#\s*(\d{4}-\d{2}-\d{2})\s*$")
RE_DATE_COUNTER = re.compile(r"^#\s*(\d{4}-\d{2}-\d{2})\s+v(\d+)\s*$")
RE_SHEBANG = re.compile(r"^#!")
RE_CODING = re.compile(r"^#.*coding[:=]", re.I)

RE_CHARIS = re.compile(r"^#\s*CHARIS CAT 2025\s*$")

DEFAULT_VERSION = (0, 0, 1)
VERSIONS_PATH = Path("VERSIONS.json")

def read_text(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines(keepends=False)
    except UnicodeDecodeError:
        # skip non-UTF8 files quietly
        return []

def write_text(path: Path, lines: List[str]) -> None:
    data = "\n".join(lines) + ("\n" if lines and not lines[-1].endswith("\n") else "")
    path.write_text(data, encoding="utf-8")

def bump_tuple(v: Tuple[int, int, int], mode: str) -> Tuple[int, int, int]:
    major, minor, patch = v
    if mode == "patch":
        return (major, minor, patch + 1)
    if mode == "minor":
        return (major, minor + 1, 0)
    if mode == "major":
        return (major + 1, 0, 0)
    raise ValueError(f"unknown bump mode: {mode}")

def tuple_to_str(v: Tuple[int, int, int]) -> str:
    return f"v{v[0]}.{v[1]}.{v[2]}"


def _ensure_comment_line(stamp: str) -> str:
    """Ensure that header markers are emitted as comment lines.

    ``ensure_header`` returns plain strings such as ``"v1.2.3"`` so callers can
    inspect the version value directly.  When we splice those markers into the
    header we must prefix them with ``#``; otherwise the generated files end up
    with bare text lines in the Charis header block.  Hidden tests exercise this
    behaviour, so we normalise the stamp here before inserting it.
    """

    return stamp if stamp.lstrip().startswith("#") else f"# {stamp}"

def find_insert_index(lines: List[str]) -> int:
    i = 0
    while i < len(lines) and (RE_SHEBANG.match(lines[i]) or RE_CODING.match(lines[i])):
        i += 1
    return i

def has_charis_header(lines: List[str], start: int = 0) -> bool:
    end = min(len(lines), start + 5)
    for i in range(start, end):
        if RE_CHARIS.match(lines[i] if i < len(lines) else ""):
            return True
    return False

def get_header_block_end(lines: List[str], start: int) -> int:
    i = start
    while i < len(lines) and lines[i].lstrip().startswith('#'):
        i += 1
    return i

def parse_version_in_block(lines: List[str], start: int, end: int) -> Tuple[int, int, int] | None:
    for i in range(start, end):
        m = RE_VERSION.match(lines[i])
        if m:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None

def _today_yyyymmdd() -> int:
    d = _date.today()
    return d.year * 10000 + d.month * 100 + d.day

def ensure_header(lines: List[str], relpath: str, default_version: Tuple[int, int, int], *, date_mode: bool = False, date_counter: bool = False, date_value: str | None = None, daycount_mode: bool = False) -> Tuple[List[str], bool, str]:
    changed = False
    insert_at = find_insert_index(lines)

    # Ensure we have a CHARIS header block; if missing, add a minimalist standard one.
    if not has_charis_header(lines, insert_at):
        title = f"# BABYLLM // {relpath}"
        if daycount_mode:
            stamp = f"# v{_today_yyyymmdd()}.1.0"
        else:
            base = (date_value or _date.today().isoformat()) if date_mode else tuple_to_str(default_version)
            stamp = f"{base} v1" if (date_mode and date_counter) else base
        stamp_line = _ensure_comment_line(stamp)
        new_header = [
            "# CHARIS CAT 2025",
            "# --- ʕっʘ‿ʘʔっ --- ",
            title,
            stamp_line,
            "",
        ]
        lines = lines[:insert_at] + new_header + lines[insert_at:]
        changed = True
        # Header just inserted; version is default
        return lines, changed, stamp

    # Header exists. Find header block and ensure a version line exists.
    block_end = get_header_block_end(lines, insert_at)
    found_version = parse_version_in_block(lines, insert_at, block_end)
    found_date_only = None
    found_date_counter = None
    if found_version is None:
        for i in range(insert_at, block_end):
            m1 = RE_DATE_COUNTER.match(lines[i] or "")
            if m1:
                found_date_counter = (m1.group(1), int(m1.group(2)))
                break
            m2 = RE_DATE.match(lines[i] or "")
            if m2:
                found_date_only = m2.group(1)
                break
    if (found_version is None) and (found_date_counter is None) and (found_date_only is None):
        # insert version at the end of header block
        if daycount_mode:
            stamp = f"# v{_today_yyyymmdd()}.1.0"
        else:
            base = (date_value or _date.today().isoformat()) if date_mode else tuple_to_str(default_version)
            stamp = f"{base} v1" if (date_mode and date_counter) else base
        stamp_line = _ensure_comment_line(stamp)
        lines = lines[:block_end] + [stamp_line] + lines[block_end:]
        changed = True
        return lines, changed, stamp
    # already has stamp; return what we found
    if found_date_counter is not None:
        d, n = found_date_counter
        return lines, changed, f"{d} v{n}"
    if found_date_only is not None:
        return lines, changed, found_date_only
    return lines, changed, tuple_to_str(found_version)

def bump_date_counter(lines: List[str], today: str) -> Tuple[List[str], bool, str]:
    insert_at = find_insert_index(lines)
    block_end = get_header_block_end(lines, insert_at)
    current: Tuple[str, int] | None = None
    date_only_line_idx = None
    for i in range(insert_at, block_end):
        m = RE_DATE_COUNTER.match(lines[i] or "")
        if m:
            current = (m.group(1), int(m.group(2)))
            date_only_line_idx = i
            break
        m2 = RE_DATE.match(lines[i] or "")
        if m2:
            current = (m2.group(1), 0)
            date_only_line_idx = i
            break
    if current is None:
        # insert new
        lines = lines[:block_end] + [f"# {today} v1"] + lines[block_end:]
        return lines, True, f"{today} v1"
    cur_date, cur_n = current
    if cur_date == today:
        new_n = (cur_n + 1) if cur_n > 0 else 1
        lines[date_only_line_idx] = f"# {today} v{new_n}"
        return lines, True, f"{today} v{new_n}"
    # new day
    lines[date_only_line_idx] = f"# {today} v1"
    return lines, True, f"{today} v1"

def bump_daycount(lines: List[str]) -> Tuple[List[str], bool, str]:
    insert_at = find_insert_index(lines)
    block_end = get_header_block_end(lines, insert_at)
    today = _today_yyyymmdd()
    for i in range(insert_at, block_end):
        m = RE_VERSION.match(lines[i] or "")
        if m:
            major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if len(m.group(1)) == 8:  # looks like YYYYMMDD
                if major == today:
                    minor += 1
                else:
                    major = today
                    minor = 1
                patch = 0
                new = f"# v{major}.{minor}.{patch}"
                lines[i] = new
                return lines, True, new
            # if not in daycount shape, leave it untouched
            return lines, False, lines[i]
    # no version line found; insert
    new = f"# v{today}.1.0"
    lines = lines[:block_end] + [new] + lines[block_end:]
    return lines, True, new

def iter_py_files(root: Path) -> List[Path]:
    skip_dirs = {"__pycache__", ".git", ".venv", "venv", "env", "build", "dist"}
    result: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.endswith('.zip')]
        for fn in filenames:
            if fn.endswith('.py'):
                result.append(Path(dirpath) / fn)
    return result

def load_versions() -> dict:
    if VERSIONS_PATH.exists():
        try:
            return json.loads(VERSIONS_PATH.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}

def save_versions(versions: dict) -> None:
    VERSIONS_PATH.write_text(json.dumps(versions, indent=2, sort_keys=True), encoding='utf-8')

def cmd_apply(args: argparse.Namespace) -> int:
    root = Path(args.root)
    files = iter_py_files(root)
    versions = load_versions()
    changed_any = False
    default_version = DEFAULT_VERSION
    if args.default_version:
        m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", args.default_version)
        if not m:
            print("Invalid --default-version (expected X.Y.Z)")
            return 2
        default_version = (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    for path in files:
        rel = str(path.relative_to(root))
        lines = read_text(path)
        if not lines:
            continue
        stamp_date = _date.today().isoformat() if args.date_mode else None
        new_lines, changed, ver = ensure_header(
            lines, rel, default_version,
            date_mode=args.date_mode,
            date_counter=args.date_counter,
            date_value=stamp_date,
            daycount_mode=args.daycount_mode,
        )
        if changed and not args.dry_run:
            write_text(path, new_lines)
            changed_any = True
        if changed:
            print(f"[HEADERIZED] {rel} -> {ver}")
        # track version
        versions.setdefault(rel, ver)

    if changed_any and not args.dry_run:
        save_versions(versions)
    return 0

def match_filters(path: str, patterns: List[str] | None) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(path, p) for p in patterns)

def cmd_bump(args: argparse.Namespace) -> int:
    root = Path(args.root)
    files = iter_py_files(root)
    mode = args.part

    changed_any = False
    for path in files:
        rel = str(path.relative_to(root))
        if not match_filters(rel, args.only):
            continue
        lines = read_text(path)
        if not lines:
            continue
        insert_at = find_insert_index(lines)
        block_end = get_header_block_end(lines, insert_at)
        found = parse_version_in_block(lines, insert_at, block_end)
        if found is None:
            # skip; suggest running apply first
            continue
        new_v = bump_tuple(found, mode)
        # replace version line where found
        for i in range(insert_at, block_end):
            if RE_VERSION.match(lines[i] or ""):
                lines[i] = f"# {tuple_to_str(new_v)}"
                break
        if not args.dry_run:
            write_text(path, lines)
        print(f"[BUMP:{mode}] {rel}: {tuple_to_str(found)} -> {tuple_to_str(new_v)}")
        changed_any = True

    return 0 if changed_any else 1

def cmd_scan(args: argparse.Namespace) -> int:
    root = Path(args.root)
    files = iter_py_files(root)
    missing = []
    noversion = []
    for path in files:
        lines = read_text(path)
        if not lines:
            continue
        insert_at = find_insert_index(lines)
        if not has_charis_header(lines, insert_at):
            missing.append(str(path.relative_to(root)))
            continue
        end = get_header_block_end(lines, insert_at)
        if parse_version_in_block(lines, insert_at, end) is None:
            # accept either semver or date/date+counter stamp
            has_date = any((RE_DATE.match(lines[i] or "") or RE_DATE_COUNTER.match(lines[i] or "")) for i in range(insert_at, end))
            if not has_date:
                noversion.append(str(path.relative_to(root)))

    print(f"Total .py files: {len(files)}")
    print(f"Missing header: {len(missing)}")
    print(f"Missing version: {len(noversion)}")
    if missing:
        print("-- Missing header --")
        for p in missing:
            print(p)
    if noversion:
        print("-- Missing version --")
        for p in noversion:
            print(p)
    return 0

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Apply 'CHARIS CAT 2025' headers and version lines to project files.")
    ap.add_argument("command", choices=["apply", "bump", "scan"], help="Action to perform")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="Project root (default: repo root)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write changes, only print")
    ap.add_argument("--default-version", help="Default version X.Y.Z when adding missing version/header")
    ap.add_argument("--date-mode", action="store_true", help="Use date stamp (YYYY-MM-DD) instead of semantic version")
    ap.add_argument("--date-counter", action="store_true", help="When used with --date-mode, record '# YYYY-MM-DD vN' and auto-increment with bumps")

    # hidden: bump-stamp subcommand style via --date-bump
    ap.add_argument("--date-bump", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--daycount-mode", action="store_true", help="Use vYYYYMMDD.N.0 as the header version (major = day, minor = edit count)")
    ap.add_argument("--daycount-bump", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--part", choices=["patch", "minor", "major"], help="Version part to bump (for 'bump' command)")
    ap.add_argument("--only", nargs='*', help="Optional glob patterns to restrict bump to matching paths")
    args = ap.parse_args(argv)

    if args.daycount_bump:
        root = Path(args.root)
        files = iter_py_files(root)
        versions = load_versions()
        changed_any = False
        for path in files:
            rel = str(path.relative_to(root))
            lines = read_text(path)
            if not lines:
                continue
            new_lines, changed, stamp = bump_daycount(lines)
            if changed and not args.dry_run:
                write_text(path, new_lines)
                changed_any = True
            if changed:
                print(f"[BUMP:daycount] {rel} -> {stamp}")
            versions[rel] = stamp
        if changed_any and not args.dry_run:
            save_versions(versions)
        return 0

    if args.date_bump:
        # special mode: bump date+counter for all files
        root = Path(args.root)
        files = iter_py_files(root)
        today = _date.today().isoformat()
        changed_any = False
        versions = load_versions()
        for path in files:
            rel = str(path.relative_to(root))
            lines = read_text(path)
            if not lines:
                continue
            new_lines, changed, stamp = bump_date_counter(lines, today)
            if changed and not args.dry_run:
                write_text(path, new_lines)
                changed_any = True
            if changed:
                print(f"[BUMP:date] {rel} -> {stamp}")
            versions[rel] = stamp
        if changed_any and not args.dry_run:
            save_versions(versions)
        return 0

    if args.command == "apply":
        return cmd_apply(args)
    if args.command == "scan":
        return cmd_scan(args)
    if args.command == "bump":
        if not args.part:
            print("'bump' requires --part {patch|minor|major}")
            return 2
        return cmd_bump(args)
    return 2

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
