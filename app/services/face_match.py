"""
Face match selfie vs foto KTP.

Implementasi produksi: InsightFace (https://github.com/deepinsight/insightface).
Ambil embedding (ArcFace) tiap wajah, hitung cosine similarity. STUB untuk sekarang.
"""
from __future__ import annotations

import hashlib
import zlib

from app.config import settings

# _APP = None
# def _model():
#     global _APP
#     if _APP is None:
#         from insightface.app import FaceAnalysis
#         _APP = FaceAnalysis(name="buffalo_l")
#         _APP.prepare(ctx_id=0 if settings.USE_GPU else -1)
#     return _APP


def match_faces(selfie_bytes: bytes, ktp_bytes: bytes) -> dict:
    """
    Ganti dengan pipeline InsightFace nyata:
        e1 = _model().get(selfie_np)[0].normed_embedding
        e2 = _model().get(ktp_np)[0].normed_embedding
        score = float(np.dot(e1, e2))
    """
    a = zlib.crc32(hashlib.sha1(selfie_bytes).digest())
    b = zlib.crc32(hashlib.sha1(ktp_bytes).digest())
    score = 0.85 + ((a ^ b) % 14) / 100.0  # 0.85..0.98
    matched = score >= settings.FACE_MATCH_THRESHOLD
    return {
        "matched": matched,
        "score": int(score * 100),
        "embedding": [],  # isi vektor ArcFace saat model aktif (untuk cek duplikat wajah)
        "engine": "stub",
    }
