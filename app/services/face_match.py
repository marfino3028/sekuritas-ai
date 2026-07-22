"""
Face match: selfie vs foto wajah pada KTP.

Mode (via FACE_MATCH_ENGINE):
  - "stub"        : deterministik, tanpa model (default).
  - "insightface" : InsightFace ArcFace (embedding + cosine similarity).

Import berat (insightface/opencv/numpy) LAZY di dalam fungsi.
"""
from __future__ import annotations

import hashlib
import zlib

from app.config import settings
from app.services.errors import (
    EngineUnavailableError,
    FaceNotFoundError,
    InferenceError,
    MultipleFacesError,
)
from app.services.image_utils import inspect_image

_APP = None


def match_faces(selfie_bytes: bytes, ktp_bytes: bytes) -> dict:
    inspect_image(selfie_bytes)
    inspect_image(ktp_bytes)

    if settings.FACE_MATCH_ENGINE == "insightface":
        try:
            return _match_insightface(selfie_bytes, ktp_bytes)
        except (EngineUnavailableError, InferenceError):
            if not settings.ALLOW_STUB_FALLBACK:
                raise
    return _match_stub(selfie_bytes, ktp_bytes)


def _match_stub(selfie_bytes: bytes, ktp_bytes: bytes) -> dict:
    a = zlib.crc32(hashlib.sha1(selfie_bytes).digest())
    b = zlib.crc32(hashlib.sha1(ktp_bytes).digest())
    score = 0.85 + ((a ^ b) % 14) / 100.0  # 0.85..0.98
    return {
        "matched": score >= settings.FACE_MATCH_THRESHOLD,
        "score": int(score * 100),
        "embedding": [],
        "engine": "stub",
    }


def _model():
    global _APP
    if _APP is None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover
            raise EngineUnavailableError(
                "insightface belum terpasang. `pip install insightface onnxruntime`."
            ) from exc
        app = FaceAnalysis(
            name=settings.INSIGHTFACE_MODEL_NAME,
            root=str(settings.INSIGHTFACE_MODEL_ROOT),
            allowed_modules=["detection", "recognition"],
        )
        det = settings.INSIGHTFACE_DET_SIZE
        app.prepare(ctx_id=0 if settings.USE_GPU else -1, det_size=(det, det))
        _APP = app
    return _APP


def _largest_embedding(image):
    """Ambil embedding wajah terbesar; validasi jumlah wajah bila diminta."""
    faces = _model().get(image)
    if not faces:
        raise FaceNotFoundError("Wajah tidak terdeteksi.")
    if settings.FACE_REQUIRE_SINGLE_FACE and len(faces) > 1:
        raise MultipleFacesError("Terdeteksi lebih dari satu wajah.")
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].normed_embedding


def _match_insightface(selfie_bytes: bytes, ktp_bytes: bytes) -> dict:
    from app.services.image_utils import decode_image

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise EngineUnavailableError("numpy diperlukan engine insightface.") from exc

    selfie = decode_image(selfie_bytes)
    ktp = decode_image(ktp_bytes)

    try:
        e1 = _largest_embedding(selfie)
        e2 = _largest_embedding(ktp)
    except (FaceNotFoundError, MultipleFacesError):
        raise
    except Exception as exc:  # pragma: no cover
        raise InferenceError(f"Face match gagal: {exc}") from exc

    cosine = float(np.dot(e1, e2))          # embedding sudah dinormalisasi → cosine
    similarity = max(0.0, min(1.0, (cosine + 1) / 2))  # -1..1 → 0..1

    return {
        "matched": similarity >= settings.FACE_MATCH_THRESHOLD,
        "score": int(round(similarity * 100)),
        "embedding": e1.tolist() if settings.RETURN_FACE_EMBEDDING else [],
        "engine": "insightface",
    }
