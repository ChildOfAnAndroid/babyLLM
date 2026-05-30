# BabyLLM live pressure guard v3

This is the corrected installer. v2 accidentally had no executable `main()` path, so it could unzip and exit without writing `live_pressure.py`.

v3 keeps Baby's foundational live training + generation overlap. It adds pressure valves instead of serialising the whole brain:

- creates `PHONE/discord_bot/live_pressure.py`
- bounds the global generation queue
- limits simultaneous generation calls only, not training/generation overlap
- wraps direct `_generate_and_reply` calls
- replaces queued `generation_queue.put(...)` with a pressure-aware helper
- adds `/api/pressure` when the expected ping block is present
- reverts `with torch.inference_mode():` back to `with torch.no_grad():` in `babyLLM.py`

Important no-grad distinction:

- v3 **does** revert the synchronous stateful Baby model generation context away from `torch.inference_mode()`.
- v3 **does not** wrap an awaiting async generation coroutine in `torch.no_grad()`, because that can leak grad-off state into concurrent live training on the same event-loop thread.

Apply from repo root:

```bash
unzip -o babyllm_live_pressure_v3.zip
python3 apply_babyllm_live_pressure_v3.py .
python3 -m py_compile PHONE/discord_bot/live_pressure.py PHONE/discord_bot/bot.py PHONE/discord_bot/platforms/web_adapter.py babyLLM.py
git diff -- PHONE/discord_bot/live_pressure.py PHONE/discord_bot/bot.py PHONE/discord_bot/platforms/web_adapter.py babyLLM.py
```

Runtime knobs:

```bash
export BBY_MAX_GENERATION_QUEUE=2
export BBY_MAX_ACTIVE_GENERATIONS=1
export BBY_SOFT_RSS_GB=48
export BBY_HARD_RSS_GB=72
export BBY_GC_AFTER_GENERATION=1
```
