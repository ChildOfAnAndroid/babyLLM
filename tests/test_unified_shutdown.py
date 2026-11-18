import asyncio
import signal
import unittest
from unittest.mock import Mock

from phone.discord_bot.bot import BABYBOT_DISCORD
from wakeup import run_with_shutdown_signals, serve_unified_bot


class TrainingBot:
    background_training_loop = BABYBOT_DISCORD.background_training_loop
    finish_training = BABYBOT_DISCORD.finish_training

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()
        self.training_queue = asyncio.Queue()
        self.training_worker = None
        self.idle_task = None
        self._training_in_progress = self._training_stopping = False
        self.lessons = []

    async def start(self, token):
        self.training_worker = asyncio.create_task(self.background_training_loop())
        await self.closed.wait()

    async def _train_on_item(self, item):
        self.lessons.append(item)
        self.started.set()
        await self.release.wait()

    async def close(self):
        self.closed.set()

    def is_closed(self):
        return self.closed.is_set()


class UnifiedShutdownTest(unittest.IsolatedAsyncioTestCase):
    async def test_restart_waits_for_active_lesson_and_saves_before_return(self):
        loop = asyncio.get_running_loop()
        bot = TrainingBot()
        bot.training_queue.put_nowait("active")
        bot.training_queue.put_nowait("queued")
        shutdown = loop.create_future()
        save = Mock(return_value=True)
        server = asyncio.create_task(serve_unified_bot(bot, None, shutdown, save))
        await asyncio.wait_for(bot.started.wait(), 1)
        shutdown.set_result(signal.SIGHUP)
        while not bot._training_stopping:
            await asyncio.sleep(0)
        self.assertFalse(server.done())
        save.assert_not_called()
        bot.release.set()
        self.assertTrue(await asyncio.wait_for(server, 1))
        self.assertEqual(bot.lessons, ["active"])
        self.assertTrue(bot.closed.is_set())
        save.assert_called_once()

    async def test_idle_shutdown_exits_without_restarting(self):
        bot = TrainingBot()
        shutdown = asyncio.get_running_loop().create_future()
        shutdown.set_result(signal.SIGINT)
        self.assertFalse(await asyncio.wait_for(
            serve_unified_bot(bot, None, shutdown, lambda: True), 1,
        ))
        self.assertEqual(bot.lessons, [])

    async def test_refused_checkpoint_prevents_restart(self):
        bot = TrainingBot()
        shutdown = asyncio.get_running_loop().create_future()
        shutdown.set_result(signal.SIGHUP)
        with self.assertRaisesRegex(RuntimeError, "checkpoint was refused"):
            await asyncio.wait_for(serve_unified_bot(bot, None, shutdown, lambda: False), 1)

    async def test_first_signal_owns_the_shutdown(self):
        async def run(shutdown):
            signal.raise_signal(signal.SIGHUP)
            self.assertEqual(await asyncio.wait_for(asyncio.shield(shutdown), 1), signal.SIGHUP)
            signal.raise_signal(signal.SIGINT)
            await asyncio.sleep(0.01)
            return shutdown.result()

        self.assertEqual(await run_with_shutdown_signals(run), signal.SIGHUP)
