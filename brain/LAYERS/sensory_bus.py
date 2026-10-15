import json
import math
import os
import random
import re
import struct
import subprocess
import time
from datetime import datetime
from pathlib import Path
from shutil import which

import torch

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


_PROBE_CACHE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "SHKAIRA"
    / "soul"
    / ".vision_probe_cache.json"
)
_PROBE_CACHE_TTL_SECONDS = 3600.0


def _read_probe_cache() -> "dict | None":
    if os.environ.get("BBY_VISION_CACHE_DISABLE", "").strip() in {
        "1",
        "true",
        "yes",
    }:
        return None
    try:
        if not _PROBE_CACHE_PATH.exists():
            return None
        payload = json.loads(_PROBE_CACHE_PATH.read_text())
        timestamp = float(payload.get("timestamp") or 0)
        if (time.time() - timestamp) > _PROBE_CACHE_TTL_SECONDS:
            return None
        return payload
    except Exception:
        return None


def _write_probe_cache(payload: dict) -> None:
    try:
        _PROBE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PROBE_CACHE_PATH.write_text(
            json.dumps({**payload, "timestamp": time.time()})
        )
    except Exception:
        pass


class VisionBus:
    """Internal camera adapter for the model-facing sensory bus."""

    def __init__(
        self,
        *,
        downsample: int = 32,
        use_dream_noise: bool = False,
        dream_noise_scale: float = 0.02,
        device_index=None,
        backend=None,
        probe_indices=None,
        skip_indices=None,
    ):
        self.downsample = max(8, int(downsample))
        self.use_dream_noise = use_dream_noise
        self.dream_noise_scale = float(dream_noise_scale)
        self.device_index = device_index
        self.backend = backend
        self.probe_indices = probe_indices
        self.skip_indices = skip_indices
        self.has_camera = False
        self._cv2 = None
        self.camera = None
        self.last_frame = None
        self._missed_reads = 0
        self._last_reopen = 0.0
        self._reopen_cooldown = 2.0

        if os.environ.get("BBY_VISION_DISABLE", "").strip() in {
            "1",
            "true",
            "yes",
        }:
            print("[VisionBus] Camera disabled via BBY_VISION_DISABLE.")
            return

        cached = _read_probe_cache()
        if cached and not cached.get("has_camera", False):
            print(
                "[VisionBus] Camera sensor unavailable (cached; clear "
                f"{_PROBE_CACHE_PATH.name} or unset BBY_VISION_CACHE_DISABLE to re-probe)."
            )
            return

        try:
            import cv2 as _cv2

            self._cv2 = _cv2
            self._open_camera()
        except Exception:
            print("[VisionBus] Camera sensor unavailable.")
            _write_probe_cache({"has_camera": False, "reason": "import_error"})

        _write_probe_cache(
            {
                "has_camera": bool(self.has_camera),
                "device_index": self.device_index if self.has_camera else None,
            }
        )

    def step(self):
        frame = self._read_frame()
        if frame is None:
            self._on_frame_fail()
            return self._fallback_values()

        height, width = frame.shape
        if height == 0 or width == 0:
            return self._fallback_values()

        global_light = self._safe_float(frame.mean(), 0.0)
        if self.last_frame is None or self.last_frame.shape != frame.shape:
            global_motion = 0.0
        else:
            global_motion = self._safe_float(
                np.abs(frame - self.last_frame).mean(), 0.0
            )

        left = frame[:, : width // 2]
        right = frame[:, width // 2 :]
        top = frame[: height // 2, :]
        bottom = frame[height // 2 :, :]
        left_right_bias = 0.0
        top_bottom_bias = 0.0
        if left.size and right.size:
            left_right_bias = self._safe_float(left.mean() - right.mean(), 0.0)
        if top.size and bottom.size:
            top_bottom_bias = self._safe_float(top.mean() - bottom.mean(), 0.0)

        contrast = self._safe_float(frame.max() - frame.min(), 0.0)
        contrast_intrusion = contrast * global_motion
        self.last_frame = frame.copy()
        self._missed_reads = 0

        return {
            "global_light": self._clamp(global_light, 0.0, 1.0),
            "global_motion": self._clamp(global_motion, 0.0, 1.0),
            "left_right_bias": self._clamp(left_right_bias, -1.0, 1.0),
            "top_bottom_bias": self._clamp(top_bottom_bias, -1.0, 1.0),
            "contrast_intrusion": self._clamp(contrast_intrusion, 0.0, 1.0),
        }

    def cleanup(self):
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None
            self.has_camera = False

    def _open_camera(self):
        if self._cv2 is None:
            return
        backend_id = self._resolve_backend(self.backend)
        indices = self.probe_indices
        if indices is None:
            indices = [0 if self.device_index is None else self.device_index]
        else:
            indices = list(indices)
            if self.device_index is not None and self.device_index not in indices:
                indices = [self.device_index] + indices
        skip = {int(index) for index in (self.skip_indices or []) if index is not None}
        if skip:
            indices = [index for index in indices if int(index) not in skip]
        if not indices:
            print("[VisionBus] Camera sensor unavailable (all indices skipped).")
            return

        fallback_camera = None
        fallback_index = None
        for index in indices:
            try:
                if backend_id is None:
                    camera = self._cv2.VideoCapture(int(index))
                else:
                    camera = self._cv2.VideoCapture(int(index), backend_id)
            except Exception:
                camera = None
            if camera is not None and camera.isOpened():
                try:
                    camera.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if fallback_camera is None:
                    fallback_camera = camera
                    fallback_index = int(index)
                frame = self._read_frame_from(camera)
                if frame is None or not self._frame_has_signal(frame):
                    try:
                        camera.release()
                    except Exception:
                        pass
                    if fallback_camera is camera:
                        fallback_camera = None
                        fallback_index = None
                    continue
                self.camera = camera
                self.has_camera = True
                self.device_index = int(index)
                print(f"[VisionBus] Camera sensor active (index {self.device_index}).")
                return
            if camera is not None:
                try:
                    camera.release()
                except Exception:
                    pass
        if fallback_camera is not None:
            self.camera = fallback_camera
            self.has_camera = True
            self.device_index = fallback_index
            try:
                fallback_camera.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            print(
                f"[VisionBus] Camera sensor active (index {self.device_index}, probe fallback)."
            )
            return
        print("[VisionBus] Camera sensor unavailable.")

    def _on_frame_fail(self):
        self._missed_reads += 1
        now = time.time()
        if self._missed_reads < 5 or (now - self._last_reopen) < self._reopen_cooldown:
            return
        self._last_reopen = now
        self.cleanup()
        self._open_camera()
        self._missed_reads = 0

    def _resolve_backend(self, backend):
        if backend is None or self._cv2 is None:
            return None
        if isinstance(backend, int):
            return backend
        try:
            name = str(backend).upper()
            if not name.startswith("CAP_"):
                name = f"CAP_{name}"
            return getattr(self._cv2, name, None)
        except Exception:
            return None

    def _read_frame(self):
        if (
            not self.has_camera
            or self.camera is None
            or self._cv2 is None
            or np is None
        ):
            return None
        return self._read_frame_from(self.camera)

    def _read_frame_from(self, camera):
        if camera is None or self._cv2 is None or np is None:
            return None
        try:
            ok, frame = camera.read()
            if not ok or frame is None:
                return None
            gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
            small = self._cv2.resize(
                gray,
                (self.downsample, self.downsample),
                interpolation=self._cv2.INTER_AREA,
            )
            return small.astype("float32") / 255.0
        except Exception:
            return None

    def _frame_has_signal(self, frame):
        try:
            mean = float(frame.mean())
            std = float(frame.std())
            dynamic = float(frame.max() - frame.min())
        except Exception:
            return False
        if not all(math.isfinite(value) for value in (mean, std, dynamic)):
            return False
        return not (dynamic < 0.02 and std < 0.01)

    def _fallback_values(self):
        return {
            "global_light": self._dream_value(0.5, 0.0, 1.0),
            "global_motion": self._dream_value(0.0, 0.0, 1.0),
            "left_right_bias": self._dream_value(0.0, -1.0, 1.0),
            "top_bottom_bias": self._dream_value(0.0, -1.0, 1.0),
            "contrast_intrusion": self._dream_value(0.0, 0.0, 1.0),
        }

    def _dream_value(self, base: float, min_val: float, max_val: float) -> float:
        if not self.use_dream_noise:
            return float(base)
        value = float(base + random.gauss(0.0, self.dream_noise_scale))
        return self._clamp(value, min_val, max_val)

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, float(value)))

    def _safe_float(self, value, fallback: float) -> float:
        try:
            if value is None or not math.isfinite(value):
                return float(fallback)
            return float(value)
        except Exception:
            return float(fallback)


class SensoryBus:
    """Samples environment and maintains slow-moving sensory state."""

    def __init__(
        self,
        *,
        use_dream_noise: bool = False,
        dream_noise_scale: float = 0.05,
        vision_device_index=None,
        vision_backend=None,
        vision_probe_indices=None,
        vision_skip_indices=None,
        vision_downsample: int = 32,
        vision_step_interval: int = 1,
        audio_device_index=None,
        audio_sample_rate: int = 44100,
        audio_frames_per_buffer: int = 1024,
        audio_rms_scale: float = 10000.0,
        audio_step_interval: int = 1,
        temp_step_interval: int = 1,
    ):
        self.t = 0
        self.use_dream_noise = use_dream_noise
        self.dream_noise_scale = dream_noise_scale
        self.audio_device_index = audio_device_index
        self.audio_sample_rate = int(audio_sample_rate)
        self.audio_frames_per_buffer = int(audio_frames_per_buffer)
        self.audio_rms_scale = float(audio_rms_scale)
        self.vision_step_interval = max(1, int(vision_step_interval))
        self.audio_step_interval = max(1, int(audio_step_interval))
        self.temp_step_interval = max(1, int(temp_step_interval))

        self.ema = {
            "global_light": 0.5,
            "global_motion": 0.0,
            "left_right_bias": 0.0,
            "top_bottom_bias": 0.0,
            "contrast_intrusion": 0.0,
            "noise": 0.3,
            "time_of_day": 0.0,
            "interaction_recency": 0.0,
            "training_age": 0.0,
            "device_temp_c": 40.0,
        }

        neutral = 0.5
        self.state = {
            "global_light_delta": neutral,
            "global_motion_delta": neutral,
            "left_right_bias_delta": neutral,
            "top_bottom_bias_delta": neutral,
            "contrast_intrusion_delta": neutral,
            "noise_delta": neutral,
            "time_of_day_delta": neutral,
            "interaction_recency_delta": neutral,
            "training_age_delta": neutral,
            "device_temp_c_delta": neutral,
        }

        self.last_interaction_time = time.time()
        self._ema_initialized = False
        self._last_vision_raw = None
        self._last_noise = None
        self._last_temp_c = None
        self._last_audio_reopen = 0.0
        self._audio_failures = 0

        self.vision_bus = VisionBus(
            use_dream_noise=self.use_dream_noise,
            dream_noise_scale=self.dream_noise_scale,
            device_index=vision_device_index,
            backend=vision_backend,
            probe_indices=vision_probe_indices,
            skip_indices=vision_skip_indices,
            downsample=vision_downsample,
        )

        self.has_audio = False
        self._pyaudio = None
        self.audio = None
        self.audio_stream = None
        self._audio_device_name = None
        self._osx_cpu_temp_available = None

        try:
            import pyaudio as _pyaudio

            self._pyaudio = _pyaudio
            self.audio = _pyaudio.PyAudio()
            self.has_audio = True
            self._print_audio_devices()
            self._audio_device_name = self._resolve_audio_device_name()
            self._open_audio_stream()
            print("[SensoryBus] Audio sensor active.")
        except Exception:
            print("[SensoryBus] Audio sensor unavailable.")

    def mark_interaction(self):
        self.last_interaction_time = time.time()

    def step(self, total_turns: int):
        self.t += 1

        now = datetime.now()
        time_of_day = (
            now.hour
            + (now.minute / 60.0)
            + (now.second / 3600.0)
            + (now.microsecond / 3_600_000_000.0)
        ) / 24.0

        dt = time.time() - self.last_interaction_time
        interaction_recency = math.exp(-dt / 120.0)

        if total_turns < 1:
            training_age = 0.0
        else:
            training_age = math.log10(total_turns + 1) / 5.0

        vision_raw = self.vision_bus.step()
        if (self.t % self.vision_step_interval == 0) or self._last_vision_raw is None:
            self._last_vision_raw = vision_raw
        else:
            vision_raw = self._last_vision_raw or {}

        raw_noise = None
        if (self.t % self.audio_step_interval == 0) or self._last_noise is None:
            self._last_noise = self._sample_noise()
        raw_noise = self._last_noise

        raw_temp_c = None
        if (self.t % self.temp_step_interval == 0) or self._last_temp_c is None:
            self._last_temp_c = self._sample_temperature_c()
        raw_temp_c = self._last_temp_c
        if raw_noise is None:
            raw_noise = self._maybe_dream(0.3)
        if raw_temp_c is None:
            raw_temp_c = self._maybe_dream_temp(40.0)

        raw_values = {
            "global_light": self._safe_value(
                vision_raw.get("global_light"),
                self.ema["global_light"],
                min_val=0.0,
                max_val=1.0,
            ),
            "global_motion": self._safe_value(
                vision_raw.get("global_motion"),
                self.ema["global_motion"],
                min_val=0.0,
                max_val=1.0,
            ),
            "left_right_bias": self._safe_value(
                vision_raw.get("left_right_bias"),
                self.ema["left_right_bias"],
                min_val=-1.0,
                max_val=1.0,
            ),
            "top_bottom_bias": self._safe_value(
                vision_raw.get("top_bottom_bias"),
                self.ema["top_bottom_bias"],
                min_val=-1.0,
                max_val=1.0,
            ),
            "contrast_intrusion": self._safe_value(
                vision_raw.get("contrast_intrusion"),
                self.ema["contrast_intrusion"],
                min_val=0.0,
                max_val=1.0,
            ),
            "noise": self._safe_value(
                raw_noise, self.ema["noise"], min_val=0.0, max_val=1.0
            ),
            "time_of_day": self._safe_value(
                time_of_day, self.ema["time_of_day"], min_val=0.0, max_val=1.0
            ),
            "interaction_recency": self._safe_value(
                interaction_recency,
                self.ema["interaction_recency"],
                min_val=0.0,
                max_val=1.0,
            ),
            "training_age": self._safe_value(
                training_age, self.ema["training_age"], min_val=0.0, max_val=1.0
            ),
            "device_temp_c": self._safe_value(
                raw_temp_c, self.ema["device_temp_c"], min_val=-10.0, max_val=120.0
            ),
        }

        if not self._ema_initialized:
            for key, raw in raw_values.items():
                self.ema[key] = raw
            neutral = 0.5
            self.state["global_light_delta"] = neutral
            self.state["global_motion_delta"] = neutral
            self.state["left_right_bias_delta"] = neutral
            self.state["top_bottom_bias_delta"] = neutral
            self.state["contrast_intrusion_delta"] = neutral
            self.state["noise_delta"] = neutral
            self.state["time_of_day_delta"] = neutral
            self.state["interaction_recency_delta"] = neutral
            self.state["training_age_delta"] = neutral
            self.state["device_temp_c_delta"] = neutral
            self._ema_initialized = True
            return

        K = {
            "global_light": 3.0,
            "global_motion": 4.0,
            "left_right_bias": 3.0,
            "top_bottom_bias": 3.0,
            "contrast_intrusion": 3.0,
            "noise": 3.0,
            "time_of_day": 2.0,
            "interaction_recency": 4.0,
            "training_age": 2.0,
            "device_temp_c": 1.5,
        }
        ema_alpha = {
            "global_motion": 0.9,
            "left_right_bias": 0.9,
            "top_bottom_bias": 0.9,
            "contrast_intrusion": 0.9,
            "noise": 0.9,
            "time_of_day": 0.999,  # Keep slow (it IS slow)
            "interaction_recency": 0.9,
        }

        delta_keys = {
            "global_light": "global_light_delta",
            "global_motion": "global_motion_delta",
            "left_right_bias": "left_right_bias_delta",
            "top_bottom_bias": "top_bottom_bias_delta",
            "contrast_intrusion": "contrast_intrusion_delta",
            "noise": "noise_delta",
            "time_of_day": "time_of_day_delta",
            "interaction_recency": "interaction_recency_delta",
            "training_age": "training_age_delta",
            "device_temp_c": "device_temp_c_delta",
        }

        for key, raw in raw_values.items():
            prev = self.ema[key]
            alpha = ema_alpha.get(key, 0.99)
            self.ema[key] = (alpha * prev) + ((1.0 - alpha) * raw)

            k_base = K.get(key, 2.0)
            prior = self._safe_value(
                self.state.get(delta_keys.get(key, ""), 0.5),
                0.5,
                min_val=0.0,
                max_val=1.0,
            )
            delta_hint = abs(prior - 0.5) * 2.0
            k = (k_base * 0.9) + (delta_hint * 0.1)
            delta_raw = raw - self.ema[key]
            if key == "time_of_day":
                delta_raw = (delta_raw + 0.5) % 1.0 - 0.5
            delta = math.tanh(delta_raw * k)
            delta_pos = (delta + 1.0) * 0.5

            delta_key = delta_keys.get(key)
            if delta_key is not None:
                self.state[delta_key] = delta_pos

    def get_tensor(self, device):
        return torch.tensor(
            [
                self.state["global_light_delta"],
                self.state["noise_delta"],
                self.state["time_of_day_delta"],
                self.state["interaction_recency_delta"],
                self.state["training_age_delta"],
                self.state["global_motion_delta"],
                self.state["left_right_bias_delta"],
                self.state["top_bottom_bias_delta"],
                self.state["contrast_intrusion_delta"],
            ],
            device=device,
            dtype=torch.float32,
        )

    def get_device_temp_c_tensor(self, device):
        return torch.tensor(
            self.state["device_temp_c_delta"], device=device, dtype=torch.float32
        )



    def cleanup(self):
        if self.vision_bus is not None:
            self.vision_bus.cleanup()
        self._close_audio_stream()
        if self.has_audio and self.audio is not None:
            try:
                self.audio.terminate()
            except Exception:
                pass
            self.audio = None

    def _safe_value(self, value, fallback, *, min_val=None, max_val=None) -> float:
        try:
            if value is None or not math.isfinite(value):
                value = fallback
            value = float(value)
        except Exception:
            value = float(fallback)
        if min_val is not None:
            value = max(min_val, value)
        if max_val is not None:
            value = min(max_val, value)
        return value

    def _maybe_dream_temp(self, fallback_c: float) -> float:
        if not self.use_dream_noise:
            return fallback_c
        return float(fallback_c + random.gauss(0.0, 1.5))

    def _maybe_dream(self, fallback: float) -> float:
        if not self.use_dream_noise:
            return fallback
        value = fallback + random.gauss(0.0, self.dream_noise_scale)
        return max(0.0, min(1.0, float(value)))

    def _sample_noise(self):
        if not self.has_audio or self.audio is None or self._pyaudio is None:
            return None
        if self.audio_stream is None:
            self._open_audio_stream()
        if self.audio_stream is None:
            return None
        try:
            data = self.audio_stream.read(
                self.audio_frames_per_buffer, exception_on_overflow=False
            )

            count = len(data) / 2
            if count <= 0:
                return None
            shorts = struct.unpack(f"{int(count)}h", data)
            rms = math.sqrt(sum(s * s for s in shorts) / count)
            self._audio_failures = 0
            return max(0.0, min(1.0, rms / self.audio_rms_scale))
        except Exception:
            self._audio_failures += 1
            self._close_audio_stream()
            return None

    def _sample_temperature_c(self):
        try:
            import psutil

            temps = psutil.sensors_temperatures()
            if not temps:
                return self._sample_temperature_osx()
            readings = []
            for entries in temps.values():
                for entry in entries:
                    if entry.current is not None:
                        readings.append(entry.current)
            if not readings:
                return self._sample_temperature_osx()
            avg_c = sum(readings) / len(readings)
            if avg_c < -10.0 or avg_c > 120.0:
                return self._sample_temperature_osx()
            return float(avg_c)
        except Exception:
            return self._sample_temperature_osx()

    def _sample_temperature_osx(self):
        if self._osx_cpu_temp_available is None:
            self._osx_cpu_temp_available = which("osx-cpu-temp") is not None
        if not self._osx_cpu_temp_available:
            return self._sample_temperature_ioreg()
        try:
            result = subprocess.run(
                ["osx-cpu-temp"],
                capture_output=True,
                text=True,
                timeout=0.5,
                check=False,
            )
            text = result.stdout.strip() or result.stderr.strip()
            match = re.search(r"([0-9]+(?:\\.[0-9]+)?)", text)
            if match:
                temp_c = float(match.group(1))
                if 0.0 < temp_c <= 120.0:
                    return temp_c
        except Exception:
            return self._sample_temperature_ioreg()
        return self._sample_temperature_ioreg()

    def _sample_temperature_ioreg(self):
        try:
            result = subprocess.run(
                ["ioreg", "-r", "-n", "AppleSmartBattery", "-k", "Temperature"],
                capture_output=True,
                text=True,
                timeout=0.5,
                check=False,
            )
            text = result.stdout + result.stderr
            match = re.search(r"\"Temperature\"\\s*=\\s*(\\d+)", text)
            if match:
                raw = int(match.group(1))
                temp_c = raw / 100.0
                if 0.0 < temp_c <= 120.0:
                    return float(temp_c)
        except Exception:
            return None
        return None

    def _resolve_audio_device_name(self):
        if not self.has_audio or self.audio is None:
            return None
        try:
            if self.audio_device_index is None:
                info = self.audio.get_default_input_device_info()
            else:
                info = self.audio.get_device_info_by_index(int(self.audio_device_index))
            name = info.get("name")
            if name:
                print(f"[SensoryBus] Audio input device: {name}")
            return name
        except Exception:
            return None

    def _open_audio_stream(self):
        if not self.has_audio or self.audio is None or self._pyaudio is None:
            return
        now = time.time()
        if (now - self._last_audio_reopen) < 2.0:
            return
        self._last_audio_reopen = now
        try:
            stream_kwargs = {
                "format": self._pyaudio.paInt16,
                "channels": 1,
                "rate": self.audio_sample_rate,
                "input": True,
                "frames_per_buffer": self.audio_frames_per_buffer,
            }
            if self.audio_device_index is not None:
                stream_kwargs["input_device_index"] = int(self.audio_device_index)
            self.audio_stream = self.audio.open(**stream_kwargs)
        except Exception:
            self.audio_stream = None

    def _close_audio_stream(self):
        if self.audio_stream is None:
            return
        try:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        except Exception:
            pass
        self.audio_stream = None

    def _print_audio_devices(self):
        if not self.has_audio or self.audio is None:
            return
        try:
            count = self.audio.get_device_count()
            print("[SensoryBus] Audio input devices:")
            for idx in range(count):
                info = self.audio.get_device_info_by_index(idx)
                if info.get("maxInputChannels", 0) > 0:
                    name = info.get("name", "?")
                    rate = info.get("defaultSampleRate", "?")
                    print(f"  - index {idx}: {name} (default rate {rate})")
        except Exception:
            pass
