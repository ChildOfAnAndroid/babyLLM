import ast
import asyncio
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

source = Path(__file__).resolve().parents[1] / "wakeup.py"
module = ast.parse(source.read_text())
functions = [
    n
    for n in module.body
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    and n.name
    in {
        "check_web_tunnel",
        "supervise_web_tunnel",
        "adopt_web_tunnel",
        "AdoptedWebTunnel",
    }
]
namespace = {}
exec(
    compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace
)


real_adopt = namespace["adopt_web_tunnel"]


class TunnelTests(unittest.TestCase):
    def setUp(self):
        self.adopt = patch.dict(
            namespace, adopt_web_tunnel=MagicMock(return_value=None)
        )
        self.adopt.start()
        self.addCleanup(self.adopt.stop)
        self.local = patch("urllib.request.urlopen")
        self.urlopen = self.local.start()
        self.addCleanup(self.local.stop)
        self.urlopen.return_value.__enter__.return_value.status = 200
        self.json = patch("json.load", return_value={})
        self.json.start()
        self.addCleanup(self.json.stop)
        self.remote = patch("subprocess.run")
        self.run = self.remote.start()
        self.addCleanup(self.remote.stop)
        self.run.return_value.returncode = 0
        self.run.return_value.stdout = "{}"
        self.spawn = patch("subprocess.Popen")
        self.popen = self.spawn.start()
        self.addCleanup(self.spawn.stop)

    def test_healthy_forwarding_reused(self):
        self.assertEqual(namespace["check_web_tunnel"](), (True, None))
        self.popen.assert_not_called()

    def test_local_outage_does_not_spawn(self):
        self.urlopen.side_effect = OSError("offline")
        self.assertEqual(namespace["check_web_tunnel"](), (False, None))
        self.run.assert_not_called()
        self.popen.assert_not_called()

    def test_dead_forwarding_starts_child_without_false_success(self):
        self.run.return_value.returncode = 7
        healthy, child = namespace["check_web_tunnel"]()
        self.assertFalse(healthy)
        self.assertIs(child, self.popen.return_value)
        self.assertIn("ExitOnForwardFailure=yes", self.popen.call_args.args[0])

    def test_malformed_remote_json_not_healthy(self):
        self.run.return_value.stdout = "[]"
        self.assertFalse(namespace["check_web_tunnel"]()[0])

    def test_replaces_only_owned_stale_child(self):
        self.run.return_value.returncode = 7
        owned = MagicMock()
        owned.poll.return_value = None
        self.assertFalse(namespace["check_web_tunnel"](owned)[0])
        owned.terminate.assert_called_once()
        owned.wait.assert_called_once()

    def test_adopts_standalone_existing_tunnel(self):
        command = "ssh -f -N -o ExitOnForwardFailure=yes -R 1420:127.0.0.1:4420 midgard.loveangel1337.ovh"
        self.run.side_effect = [
            MagicMock(returncode=0, stdout="123 " + command),
            MagicMock(returncode=0, stdout="Wed Oct 7 14:31:20 2026 " + command),
        ]
        adopted = real_adopt()
        self.assertEqual(adopted.pid, 123)
        self.popen.assert_not_called()

    def test_does_not_adopt_shared_or_other_sessions(self):
        for command in [
            "ssh -N -R 1420:127.0.0.1:4420 -R 18765:127.0.0.1:8765 midgard.loveangel1337.ovh",
            "ssh -N -R 1420:127.0.0.1:4420 other-host",
            "ssh -R 1420:127.0.0.1:4420 midgard.loveangel1337.ovh",
            "ssh -N -L 1234:localhost:22 -R 1420:127.0.0.1:4420 midgard.loveangel1337.ovh",
        ]:
            with self.subTest(command=command):
                self.run.return_value.stdout = "123 " + command
                self.assertIsNone(real_adopt())

    def test_pid_reuse_during_adoption_is_rejected(self):
        command = "ssh -N -R 1420:127.0.0.1:4420 midgard.loveangel1337.ovh"
        self.run.side_effect = [
            MagicMock(returncode=0, stdout="123 " + command),
            MagicMock(returncode=0, stdout="Wed Oct 7 14:31:20 2026 unrelated-process"),
        ]
        self.assertIsNone(real_adopt())

    def test_adopted_handle_never_signals_changed_identity(self):
        handle = namespace["AdoptedWebTunnel"](123, "original identity")
        self.run.return_value.stdout = "different identity"
        with patch("os.kill") as kill:
            handle.terminate()
            handle.kill()
            kill.assert_not_called()

    def test_adopted_broken_tunnel_is_replaced(self):
        adopted = MagicMock()
        adopted.poll.return_value = None
        namespace["adopt_web_tunnel"].return_value = adopted
        self.run.return_value.returncode = 7
        healthy, replacement = namespace["check_web_tunnel"]()
        self.assertFalse(healthy)
        self.assertIs(replacement, self.popen.return_value)
        adopted.terminate.assert_called_once()
        adopted.wait.assert_called_once()

    def test_recovery_backoff_and_shutdown(self):
        async def exercise():
            stop = asyncio.Event()
            delays = []
            checks = []
            child = MagicMock()
            child.poll.return_value = None

            def check(owned):
                checks.append(owned)
                return (len(checks) > 2, child)

            async def wait(awaitable, timeout):
                awaitable.close()
                delays.append(timeout)
                if len(delays) == 4:
                    stop.set()
                    return True
                raise asyncio.TimeoutError

            with (
                patch.dict(namespace, check_web_tunnel=check),
                patch("asyncio.wait_for", side_effect=wait),
            ):
                await namespace["supervise_web_tunnel"](stop)
            self.assertEqual(delays, [2, 4, 30, 30])
            child.terminate.assert_called_once()

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
