# BabyLLM live pressure guard v2

This is the "do not amputate live training" fix.

It keeps the live training + generation overlap, but adds pressure valves:

- bounded generation queue
- active generation semaphore
- pressure replies instead of infinite queued Futures
- `/api/pressure` web telemetry
- GC / torch cache cleanup after generations
- grad-mode restoration after generation calls

## Important no_grad point

This patch deliberately does **not** do:

```python
with torch.no_grad():
    await generate(...)
```

That pattern is dodgy in an async bot because PyTorch grad mode is thread-local.
If the generation coroutine awaits while grad mode is off, another training task
running on the same event-loop thread can inherit the wrong grad state.

So v2 instead restores grad mode after generation and keeps the no-grad boundary
out of the async wrapper.

## Apply

From the BabyLLM repo root:

```bash
unzip babyllm_live_pressure_v2.zip
python3 apply_babyllm_live_pressure_v2.py .
python3 -m py_compile PHONE/discord_bot/live_pressure.py PHONE/discord_bot/bot.py PHONE/discord_bot/platforms/web_adapter.py babyLLM.py
```

Then inspect:

```bash
git diff -- PHONE/discord_bot/live_pressure.py PHONE/discord_bot/bot.py PHONE/discord_bot/platforms/web_adapter.py babyLLM.py
```

## Runtime knobs

```bash
export BBY_MAX_GENERATION_QUEUE=2
export BBY_MAX_ACTIVE_GENERATIONS=1
export BBY_SOFT_RSS_GB=48
export BBY_HARD_RSS_GB=72
export BBY_GC_AFTER_GENERATION=1
```
