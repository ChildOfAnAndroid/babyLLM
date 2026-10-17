The issue is that `WebAdapter` triggers animations by spawning new synchronous daemon threads that just sit and block in `time.sleep` (e.g. in `trigger_animation` and inside a route setting `stop_speaking`). This is a waste of OS resources (thread creation and context switching overhead).
I have verified that an async equivalent using `asyncio.sleep` takes a fraction of the time and resources.

My proposed plan is:
1. Identify all places in `phone/discord_bot/platforms/web_adapter.py` where a temporary animation or state reset is scheduled via a background thread + `time.sleep()`.
   - `trigger_animation` -> `jump`
   - `trigger_animation` -> `blush`
   - `trigger_animation` -> `cry`
   - `trigger_animation` -> `celebrate` (this one has a small sleep loop, which is also well suited for `asyncio.sleep`)
   - `_setup_routes` -> `stop_speaking` timer thread inside `/api/say`

2. Replace the thread creation with scheduling an async coroutine on the main bot loop using `asyncio.run_coroutine_threadsafe`.
   - To do this safely since these might be called from outside the event loop thread (like from the Flask thread), we can define a helper method `_schedule_on_bot_loop` that uses `asyncio.run_coroutine_threadsafe(coro, self.bot.loop)`.
   - Update `stop_jumping`, `stop_blushing`, `stop_crying` to be `async def` and use `await asyncio.sleep(duration)`.
   - Update `celebrate` to `async def` and use `await asyncio.sleep(0.3)`.
   - Update `stop_speaking` to `async def` and use `await asyncio.sleep(10)`.

3. Benchmark the before-and-after behavior of creating threads vs scheduling async tasks. (I have already established this in a standalone `test_perf.py` benchmark: ~12.8x improvement for 1000 concurrent sleeps).

4. Complete pre-commit step using `pre_commit_instructions` tool to run checks.
5. Create a git branch and submit PR with measured improvements.
