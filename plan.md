1. **Understand**:
   - The method `trigger_animation` inside `WebAdapter` class uses `time.sleep` in nested thread functions (`stop_jumping`, `stop_blushing`, `stop_crying`, `celebrate`, and `stop_speaking` in the `_setup_routes` route).
   - Threads use `time.sleep`, which is blocking. Although it happens on a background daemon thread (`threading.Thread`), creating a thread specifically just to sleep is very unoptimized.
   - `trigger_animation` is synchronous, but we can execute async functions on the bot loop using `asyncio.run_coroutine_threadsafe`.

2. **Refactor plan**:
   - For `trigger_animation` method: `jump`, `blush`, `cry`, `celebrate` - replace `threading.Thread` and `time.sleep` with `asyncio.sleep` running on `self.bot.loop` via `asyncio.run_coroutine_threadsafe`.
   - In `_setup_routes`, for the `stop_speaking` thread: change the thread creation to an `asyncio.create_task` or run a coroutine on `self.bot.loop` using `asyncio.run_coroutine_threadsafe()`.

Let's test if `self.bot.loop` is generally available. It should be, based on `self._run_on_bot_loop` checking it.
