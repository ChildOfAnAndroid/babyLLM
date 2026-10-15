import os
import sys
import tempfile
import time
import shutil

# Make sure we can import from utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mps_trace import cleanup_mps_temp_dirs

def test_cleanup():
    print("=== STARTING MPS GRAPH CLEANUP TEST ===")

    # Enable logging
    os.environ["BABYLLM_MPSGRAPH_CLEANUP_LOG"] = "1"

    # 1. Verify startup cleanup does not import torch
    print("\n1. Verifying 'torch' is not imported by utils.mps_trace...")
    if 'torch' in sys.modules:
        print("   ⚠ Warning: 'torch' already in sys.modules (was imported elsewhere).")
        import subprocess
        cmd = [sys.executable, "-c", "import sys; from utils.mps_trace import cleanup_mps_temp_dirs; print('torch' in sys.modules)"]
        out = subprocess.check_output(cmd).decode().strip()
        print(f"   Subprocess import check ('torch' in sys.modules): {out} (expect False)")
        assert out == "False", "Error: 'torch' was imported by mps_trace!"
    else:
        print("   ✓ 'torch' is not in sys.modules")

    # Create an isolated sandbox directory layout for the test
    with tempfile.TemporaryDirectory() as sandbox:
        mps_base = os.path.join(sandbox, 'com.apple.MetalPerformanceShadersGraph')
        os.makedirs(mps_base, exist_ok=True)

        # Helper to create folder
        def make_folder(name):
            path = os.path.join(mps_base, name)
            os.makedirs(path, exist_ok=True)
            return path

        current_pid = os.getpid()
        dead_pid_1 = 99999
        dead_pid_2 = 88888

        # Verify PIDs are dead
        for dead_pid in [dead_pid_1, dead_pid_2]:
            try:
                os.kill(dead_pid, 0)
                print(f"   ⚠ Warning: PID {dead_pid} is alive! Test might fail.")
            except OSError:
                pass

        print("\n2. Creating sandbox directory structure...")
        # A. Stale empty dead PID dirs (older than 10 mins)
        dir_dead_1 = make_folder(f"mpsgraph-{dead_pid_1}-2026-06-13_22_00_00-abc")
        dir_dead_2 = make_folder(f"mpsgraph-{dead_pid_2}-2026-06-13_22_00_00-def")
        # Backdate modification time of these dirs to exceed 10 minutes (650 seconds ago)
        old_time = time.time() - 650
        os.utime(dir_dead_1, (old_time, old_time))
        os.utime(dir_dead_2, (old_time, old_time))

        # B. Fake dir using current Python PID
        dir_curr = make_folder(f"mpsgraph-{current_pid}-2026-06-13_22_00_00-ghi")
        os.utime(dir_curr, (old_time, old_time))

        # C. Non-empty fake dead PID dir
        dir_nonempty = make_folder(f"mpsgraph-{dead_pid_1}-2026-06-13_22_00_00-nonempty")
        with open(os.path.join(dir_nonempty, "dummy.txt"), "w") as f:
            f.write("content")
        os.utime(dir_nonempty, (old_time, old_time))

        # D. Stale empty live process PID (use parent process PID os.getppid())
        live_pid = os.getppid()
        dir_live = make_folder(f"mpsgraph-{live_pid}-2026-06-13_22_00_00-live")
        os.utime(dir_live, (old_time, old_time))

        # E. Stale empty but too recent directory (age < 10 mins)
        dir_recent = make_folder(f"mpsgraph-{dead_pid_2}-2026-06-13_22_00_00-recent")

        print(f"   Created stale dead PIDs: {os.path.basename(dir_dead_1)}, {os.path.basename(dir_dead_2)}")
        print(f"   Created current process PID: {os.path.basename(dir_curr)}")
        print(f"   Created non-empty: {os.path.basename(dir_nonempty)}")
        print(f"   Created live parent PID: {os.path.basename(dir_live)}")
        print(f"   Created recent folder: {os.path.basename(dir_recent)}")

        # 3. Test dry-run mode
        print("\n3. Testing dry-run mode in sandbox...")
        dry_deleted = cleanup_mps_temp_dirs(dry_run=True, min_age_seconds=600, base_paths=[mps_base])
        print(f"   Dry run reported count: {dry_deleted}")
        assert dry_deleted == 2, f"Expected 2 dry-run deletions, got {dry_deleted}"

        # Confirm they were NOT deleted
        assert os.path.exists(dir_dead_1), "Error: dir_dead_1 deleted during dry-run!"
        assert os.path.exists(dir_dead_2), "Error: dir_dead_2 deleted during dry-run!"

        # 4. Test actual cleanup mode
        print("\n4. Testing actual cleanup in sandbox...")
        actual_deleted = cleanup_mps_temp_dirs(dry_run=False, min_age_seconds=600, base_paths=[mps_base])
        print(f"   Actual cleanup reported count: {actual_deleted}")
        assert actual_deleted == 2, f"Expected 2 actual deletions, got {actual_deleted}"

        # Verify what was deleted and what was kept
        print("\n5. Verifying directory presence/absence...")
        assert not os.path.exists(dir_dead_1), "Error: dir_dead_1 was not deleted!"
        assert not os.path.exists(dir_dead_2), "Error: dir_dead_2 was not deleted!"
        print("   ✓ Stale empty folders successfully deleted.")

        assert os.path.exists(dir_curr), "Error: current PID folder was deleted!"
        print("   ✓ Current process PID folder was kept.")

        assert os.path.exists(dir_nonempty), "Error: non-empty folder was deleted!"
        print("   ✓ Non-empty folder was kept.")

        assert os.path.exists(dir_live), "Error: live parent PID folder was deleted!"
        print("   ✓ Live process PID folder was kept.")

        assert os.path.exists(dir_recent), "Error: recent folder was deleted!"
        print("   ✓ Recently modified folder was kept.")

    print("\n=== ALL CLEANUP TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_cleanup()
