"""
OCR KTP.

Implementasi produksi: PaddleOCR (https://github.com/PaddlePaddle/PaddleOCR).
Parse teks hasil OCR menjadi field KTP (NIK, nama, TTL, alamat, dst) dengan
regex + heuristik posisi. Untuk sekarang: STUB agar service bisa jalan tanpa
model berat. Ganti isi extract_ktp() saat model sudah tersedia.
"""
from __future__ import annotations

import hashlib
import re

# _PADDLE = None
# def _ocr():
#     global _PADDLE
#     if _PADDLE is None:
#         from paddleocr import PaddleOCR
#         _PADDLE = PaddleOCR(use_angle_cls=True, lang="id", show_log=False)
#     return _PADDLE

NIK_RE = re.compile(r"\b\d{16}\b")


def _parse_lines(lines: list[str]) -> dict:
    text = "\n".join(lines)
    nik = NIK_RE.search(text)
    return {
        "nik": nik.group(0) if nik else None,
        # TODO: parse name/birth_place/birth_date/gender/address dari layout KTP
    }


def extract_ktp(image_bytes: bytes) -> dict:
    """
    Ganti body ini dengan pipeline PaddleOCR nyata:
        result = _ocr().ocr(image_np, cls=True)
        lines = [w[1][0] for line in result for w in line]
        fields = _parse_lines(lines)
    """
    seed = hashlib.sha1(image_bytes).hexdigest()
    confidence = 90 if len(image_bytes) > 40_000 else 55
    return {
        "nik": None,
        "name": None,
        "birth_place": None,
        "birth_date": None,
        "gender": None,
        "address": None,
        "religion": None,
        "marital_status": None,
        "occupation": None,
        "confidence": confidence,
        "is_blur": len(image_bytes) < 25_000,
        "is_low_light": len(image_bytes) < 30_000,
        "is_screenshot": False,
        "engine": "stub",
        "seed": seed,
    }
