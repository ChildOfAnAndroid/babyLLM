import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


_MODULE_PATH = Path(__file__).with_name("live_pressure.py")
_SPEC = importlib.util.spec_from_file_location("_babyllm_live_pressure_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
live_pressure = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = live_pressure
_SPEC.loader.exec_module(live_pressure)


class TestLivePressureMemoryAccounting(unittest.TestCase):
    def test_pressure_uses_footprint_when_swap_hides_rss(self):
        with (
            patch.object(live_pressure, "current_rss_gb", return_value=0.022),
            patch.object(live_pressure, "current_phys_footprint_gb", return_value=110.5),
        ):
            self.assertEqual(live_pressure.current_pressure_gb(), 110.5)

    def test_guard_rejects_large_footprint_despite_small_rss(self):
        guard = live_pressure.LivePressureGuard(hard_rss_gb=72.0)
        with patch.object(live_pressure, "current_pressure_gb", return_value=110.5):
            rejected, reason = guard.should_hard_reject()

        self.assertTrue(rejected)
        self.assertIn("memory footprint 110.5 GiB", reason)

    def test_darwin_footprint_reads_kernel_rusage(self):
        fake_proc_pid_rusage = MagicMock(return_value=0)

        def populate(_pid, _flavor, info_pointer):
            info_pointer._obj.ri_phys_footprint = 3 * 1024 ** 3
            return 0

        fake_proc_pid_rusage.side_effect = populate
        fake_libproc = MagicMock(proc_pid_rusage=fake_proc_pid_rusage)

        with (
            patch.object(live_pressure.sys, "platform", "darwin"),
            patch.object(live_pressure.ctypes, "CDLL", return_value=fake_libproc),
        ):
            self.assertEqual(live_pressure.current_phys_footprint_gb(), 3.0)

        fake_proc_pid_rusage.assert_called_once()
        self.assertEqual(fake_proc_pid_rusage.call_args.args[1], 0)

    def test_non_darwin_footprint_falls_back_to_rss(self):
        with (
            patch.object(live_pressure.sys, "platform", "linux"),
            patch.object(live_pressure, "current_rss_gb", return_value=1.25),
        ):
            self.assertEqual(live_pressure.current_phys_footprint_gb(), 1.25)


if __name__ == "__main__":
    unittest.main()
