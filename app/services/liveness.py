"""
Passive liveness detection.

Implementasi produksi: Silent-Face-Anti-Spoofing (MiniFASNet) atau ekosistem
InsightFace. Deteksi wajah palsu (foto cetak, replay layar). STUB untuk sekarang.
"""
from __future__ import annotations

import hashlib

from app.config import settings


def check_liveness(image_bytes: bytes) -> dict:
    """
    Ganti dengan model nyata:
        score = minifasnet.predict(face_crop)  # 0..1
    """
    seed = int(hashlib.sha1(image_bytes).hexdigest(), 16)
    score = 0.95 if len(image_bytes) > 30_000 else 0.45
    passed = score >= settings.LIVENESS_THRESHOLD
    return {
        "passed": passed,
        "score": int(score * 100),
        "is_printed_photo": False,
        "is_replay": False,
        "engine": "stub",
        "seed": seed % 100000,
    }
