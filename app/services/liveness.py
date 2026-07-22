"""
Passive liveness / anti-spoofing (deteksi wajah asli vs foto cetak / replay layar).

Mode (via LIVENESS_ENGINE):
  - "stub"             : deterministik, tanpa model (default).
  - "silent-face-onnx" : model Silent-Face-Anti-Spoofing (MiniFASNet) format ONNX.

Import berat (onnxruntime/opencv/numpy) LAZY di dalam fungsi.
"""
from __future__ import annotations

import hashlib

from app.config import settings
from app.services.errors import EngineUnavailableError, InferenceError
from app.services.image_utils import inspect_image

_SESSION = None


def check_liveness(image_bytes: bytes) -> dict:
    inspect_image(image_bytes)

    if settings.LIVENESS_ENGINE == "silent-face-onnx":
        try:
            return _liveness_onnx(image_bytes)
        except (EngineUnavailableError, InferenceError):
            if not settings.ALLOW_STUB_FALLBACK:
                raise
    return _liveness_stub(image_bytes)


def _liveness_stub(image_bytes: bytes) -> dict:
    seed = int(hashlib.sha1(image_bytes).hexdigest(), 16)
    score = 0.95 if len(image_bytes) > 30_000 else 0.45
    return {
        "passed": score >= settings.LIVENESS_THRESHOLD,
        "score": int(score * 100),
        "is_printed_photo": False,
        "is_replay": False,
        "engine": "stub",
        "seed": seed % 100000,
    }


def _onnx_session():
    global _SESSION
    if _SESSION is None:
        if not settings.LIVENESS_MODEL_DIR:
            raise EngineUnavailableError("LIVENESS_MODEL_DIR belum diset.")
        model_path = settings.LIVENESS_MODEL_DIR
        if model_path.is_dir():
            candidates = sorted(model_path.glob("*.onnx"))
            if not candidates:
                raise EngineUnavailableError(f"Tidak ada file .onnx di {model_path}.")
            model_path = candidates[0]
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise EngineUnavailableError("onnxruntime belum terpasang.") from exc
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if settings.USE_GPU else ["CPUExecutionProvider"]
        _SESSION = ort.InferenceSession(str(model_path), providers=providers)
    return _SESSION


def _liveness_onnx(image_bytes: bytes) -> dict:
    from app.services.image_utils import decode_image

    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise EngineUnavailableError("opencv/numpy diperlukan engine liveness.") from exc

    image = decode_image(image_bytes)
    w, h = settings.LIVENESS_INPUT_WIDTH, settings.LIVENESS_INPUT_HEIGHT
    resized = cv2.resize(image, (w, h))
    blob = resized.astype("float32").transpose(2, 0, 1)[None, ...]  # NCHW

    try:
        session = _onnx_session()
        logits = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
    except Exception as exc:  # pragma: no cover
        raise InferenceError(f"Liveness gagal: {exc}") from exc

    # softmax → probabilitas per kelas (live / print / replay)
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    live_p = float(probs[settings.LIVENESS_LIVE_CLASS_INDEX]) if settings.LIVENESS_LIVE_CLASS_INDEX < len(probs) else 0.0
    print_p = float(probs[settings.LIVENESS_PRINT_CLASS_INDEX]) if settings.LIVENESS_PRINT_CLASS_INDEX < len(probs) else 0.0
    replay_p = float(probs[settings.LIVENESS_REPLAY_CLASS_INDEX]) if settings.LIVENESS_REPLAY_CLASS_INDEX < len(probs) else 0.0

    return {
        "passed": live_p >= settings.LIVENESS_THRESHOLD,
        "score": int(round(live_p * 100)),
        "is_printed_photo": print_p > live_p,
        "is_replay": replay_p > live_p,
        "engine": "silent-face-onnx",
    }
