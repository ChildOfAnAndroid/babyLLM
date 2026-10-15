import os
import glob
import time
import errno
import tempfile
import atexit

def cleanup_log(msg: str):
    if os.environ.get("BABYLLM_MPSGRAPH_TRACE") == "1" or os.environ.get("BABYLLM_MPSGRAPH_CLEANUP_LOG") == "1":
        print(f"[MPS_CLEANUP] {msg}", flush=True)

def mps_trace(phase: str, details: str = ""):
    if os.environ.get("BABYLLM_MPSGRAPH_TRACE") == "1":
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # macOS temp directories are under /var/folders/ (or symlinked /private/var/folders/)
        path = os.path.join(tempfile.gettempdir(), 'com.apple.MetalPerformanceShadersGraph')
        count = -1
        if os.path.exists(path):
            try:
                count = len([name for name in os.listdir(path) if name.startswith("mpsgraph-")])
            except Exception:
                pass

        detail_str = f" | {details}" if details else ""
        print(f"\n[MPS_TRACE] {timestamp} | Phase: {phase:<30}{detail_str} | Dirs: {count if count >= 0 else 'N/A'}\n", flush=True)

def cleanup_mps_temp_dirs(dry_run: bool = False, min_age_seconds: float = 600.0, base_paths: list = None) -> int:
    """
    Cleans up stale empty macOS MPSGraph temp directories left by previous interrupted runs.
    Ensures it does not interfere with concurrent live processes.
    """
    current_pid = os.getpid()
    cleanup_log(f"Starting cleanup (PID={current_pid}, dry_run={dry_run}, min_age={min_age_seconds}s)...")

    # Locate all com.apple.MetalPerformanceShadersGraph directories if not custom provided
    if base_paths is not None:
        paths = base_paths
    else:
        paths = []
        curr_temp = os.path.join(tempfile.gettempdir(), 'com.apple.MetalPerformanceShadersGraph')
        if os.path.isdir(curr_temp):
            paths.append(curr_temp)

        for pattern in ['/var/folders/*/*/T/com.apple.MetalPerformanceShadersGraph', '/private/var/folders/*/*/T/com.apple.MetalPerformanceShadersGraph']:
            try:
                for p in glob.glob(pattern):
                    if os.path.isdir(p) and p not in paths:
                        paths.append(p)
            except Exception:
                pass

    deleted_count = 0
    now = time.time()

    for base_path in paths:
        try:
            if not os.path.exists(base_path):
                continue

            # List children
            try:
                children = os.listdir(base_path)
            except Exception as e:
                cleanup_log(f"Failed to list directory {base_path}: {e}")
                continue

            for child in children:
                # Must start with mpsgraph-
                if not child.startswith("mpsgraph-"):
                    continue

                child_path = os.path.join(base_path, child)
                if not os.path.isdir(child_path):
                    continue

                # Parse PID
                parts = child.split("-")
                if len(parts) < 3:
                    continue
                try:
                    pid = int(parts[1])
                except ValueError:
                    continue

                # Rule 1: Skip if it matches current PID
                if pid == current_pid:
                    continue

                # Rule 2: Skip if PID is currently alive
                try:
                    if pid > 0:
                        os.kill(pid, 0)
                        # Process is alive, skip
                        continue
                except OSError as err:
                    if err.errno != errno.ESRCH:
                        # EPERM or other error, assume process is alive, skip
                        continue

                # Rule 3: Only delete if directory is empty
                try:
                    is_empty = (len(os.listdir(child_path)) == 0)
                except Exception:
                    is_empty = False

                if not is_empty:
                    continue

                # Rule 4: Skip if modified/created within min_age_seconds
                try:
                    mtime = os.path.getmtime(child_path)
                    age = now - mtime
                except Exception:
                    age = 0

                if age < min_age_seconds:
                    cleanup_log(f"Skipping recently modified directory: {child} (age: {age:.1f}s)")
                    continue

                # If we passed all checks, we can delete it!
                if dry_run:
                    cleanup_log(f"[DRY-RUN] Would delete empty stale dir: {child_path}")
                    deleted_count += 1
                else:
                    try:
                        os.rmdir(child_path)
                        cleanup_log(f"Deleted empty stale dir: {child_path}")
                        deleted_count += 1
                    except Exception as e:
                        cleanup_log(f"Failed to delete {child_path}: {e}")

        except Exception as e:
            cleanup_log(f"Error scanning path {base_path}: {e}")

    cleanup_log(f"Cleanup finished. Stale directories cleaned: {deleted_count}")
    return deleted_count

def _cleanup_own_dirs():
    current_pid = os.getpid()
    curr_temp = os.path.join(tempfile.gettempdir(), 'com.apple.MetalPerformanceShadersGraph')
    if os.path.isdir(curr_temp):
        try:
            for child in os.listdir(curr_temp):
                if child.startswith(f"mpsgraph-{current_pid}-"):
                    child_path = os.path.join(curr_temp, child)
                    if os.path.isdir(child_path) and len(os.listdir(child_path)) == 0:
                        try:
                            os.rmdir(child_path)
                            cleanup_log(f"Exit cleanup: deleted own empty dir: {child_path}")
                        except Exception:
                            pass
        except Exception:
            pass

# Register atexit handler
atexit.register(_cleanup_own_dirs)
