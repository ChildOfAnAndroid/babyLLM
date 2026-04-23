import math
import random
import time

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


class VisionBus:
    """Samples webcam frames and emits derived, bounded signals."""

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

        try:
            import cv2 as _cv2

            self._cv2 = _cv2
            self._open_camera()
        except Exception:
            print("[VisionBus] Camera sensor unavailable.")

    def step(self):
        frame = self._read_frame()
        if frame is None:
            self._on_frame_fail()
            return self._fallback_values()

        h, w = frame.shape
        if h == 0 or w == 0:
            return self._fallback_values()

        global_light = self._safe_float(frame.mean(), 0.0)

        if self.last_frame is None or self.last_frame.shape != frame.shape:
            global_motion = 0.0
        else:
            diff = np.abs(frame - self.last_frame)
            global_motion = self._safe_float(diff.mean(), 0.0)

        left = frame[:, : w // 2]
        right = frame[:, w // 2 :]
        top = frame[: h // 2, :]
        bottom = frame[h // 2 :, :]

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
        skip = {int(i) for i in (self.skip_indices or []) if i is not None}
        if skip:
            indices = [idx for idx in indices if int(idx) not in skip]
        if not indices:
            print("[VisionBus] Camera sensor unavailable (all indices skipped).")
            return

        fallback_cam = None
        fallback_idx = None
        for idx in indices:
            try:
                if backend_id is None:
                    cam = self._cv2.VideoCapture(int(idx))
                else:
                    cam = self._cv2.VideoCapture(int(idx), backend_id)
            except Exception:
                cam = None
            if cam is not None and cam.isOpened():
                try:
                    cam.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if fallback_cam is None:
                    fallback_cam = cam
                    fallback_idx = int(idx)
                frame = self._read_frame_from(cam)
                if frame is None or not self._frame_has_signal(frame):
                    try:
                        cam.release()
                    except Exception:
                        pass
                    if fallback_cam is cam:
                        fallback_cam = None
                        fallback_idx = None
                    continue
                self.camera = cam
                self.has_camera = True
                self.device_index = int(idx)
                print(f"[VisionBus] Camera sensor active (index {self.device_index}).")
                return
            if cam is not None:
                try:
                    cam.release()
                except Exception:
                    pass
        if fallback_cam is not None:
            self.camera = fallback_cam
            self.has_camera = True
            self.device_index = fallback_idx
            try:
                fallback_cam.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
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
        if self._missed_reads < 5:
            return
        if (now - self._last_reopen) < self._reopen_cooldown:
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

    def _read_frame_from(self, cam):
        if cam is None or self._cv2 is None or np is None:
            return None
        try:
            ret, frame = cam.read()
            if not ret or frame is None:
                return None
            gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
            small = self._cv2.resize(
                gray,
                (self.downsample, self.downsample),
                interpolation=self._cv2.INTER_AREA,
            )
            small = small.astype("float32") / 255.0
            return small
        except Exception:
            return None

    def _frame_has_signal(self, frame):
        try:
            mean = float(frame.mean())
            std = float(frame.std())
            dynamic = float(frame.max() - frame.min())
        except Exception:
            return False
        if (
            not math.isfinite(mean)
            or not math.isfinite(std)
            or not math.isfinite(dynamic)
        ):
            return False
        if dynamic < 0.02 and std < 0.01:
            return False
        return True

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
