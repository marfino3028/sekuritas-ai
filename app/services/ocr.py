"""
OCR KTP.

Dua mode (dipilih via OCR_ENGINE di config):
  - "stub"     : deterministik, tanpa model (default; validasi header + skor dummy).
  - "nanonets" : model vision-language Nanonets-OCR-s (GGUF via llama-cpp-python),
                 di-load IN-PROCESS lewat app/services/nanonets_engine.py — bukan
                 lagi memanggil service Flask terpisah lewat HTTP.

PaddleOCR sudah dihapus total dari service ini (per keputusan project: diganti
model Nanonets lokal karena hasil PaddleOCR tidak memadai).
"""
from __future__ import annotations

import hashlib
import logging

from app.config import settings
from app.services.errors import EngineUnavailableError, InferenceError
from app.services.image_utils import inspect_image

logger = logging.getLogger("ekyc.ocr")

# Field "inti" yang dipakai untuk menghitung confidence heuristik hasil Nanonets
# (model vision ini tidak memberi skor per-field seperti PaddleOCR).
_CORE_FIELDS = ("nik", "name", "birth_date", "gender", "address")


def extract_ktp(image_bytes: bytes) -> dict:
    """Ekstrak field KTP dari bytes gambar. Selalu validasi header dulu."""
    info = inspect_image(image_bytes)  # raises InvalidImageError bila bukan gambar valid

    if settings.OCR_ENGINE == "nanonets":
        try:
            return _extract_nanonets(image_bytes)
        except (EngineUnavailableError, InferenceError) as exc:
            if not settings.ALLOW_STUB_FALLBACK:
                raise
            # fallback ke stub bila model Nanonets belum siap; tetap log
            # alasan aslinya supaya kelihatan di terminal uvicorn.
            logger.warning("Nanonets OCR gagal, fallback ke stub: %s", exc)
    return _extract_stub(image_bytes, info)


# --------------------------------------------------------------------------- #
# STUB
# --------------------------------------------------------------------------- #
def _extract_stub(image_bytes: bytes, info) -> dict:
    size = len(image_bytes)
    confidence = 90 if size > 40_000 else 55
    small_side = min(info.width, info.height)
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
        "is_blur": small_side < 480,
        "is_low_light": size < 30_000,
        "is_screenshot": False,
        "engine": "stub",
        "seed": hashlib.sha1(image_bytes).hexdigest(),
    }


# --------------------------------------------------------------------------- #
# Nanonets-OCR-s (in-process, lihat app/services/nanonets_engine.py)
# --------------------------------------------------------------------------- #
def _extract_nanonets(image_bytes: bytes) -> dict:
    from app.services.nanonets_engine import extract as nanonets_extract

    fields = nanonets_extract(image_bytes, settings.NANONETS_OCR_TEMPLATE)

    filled_core = sum(1 for key in _CORE_FIELDS if fields.get(key))
    confidence = int(round(100 * filled_core / len(_CORE_FIELDS))) if _CORE_FIELDS else 0

    is_blur, is_low_light = _quality_metrics_safe(image_bytes)

    return {
        # --- kontrak lama (dipakai FastApiProvider.php di sekuritas-api) ---
        "nik": fields.get("nik"),
        "name": fields.get("name"),
        "birth_place": fields.get("birth_place"),
        "birth_date": fields.get("birth_date"),
        "gender": fields.get("gender"),
        "address": fields.get("address"),
        "religion": fields.get("religion"),
        "marital_status": fields.get("marital_status"),
        "occupation": fields.get("occupation"),
        "confidence": confidence,
        "is_blur": is_blur,
        "is_low_light": is_low_light,
        "is_screenshot": False,
        "engine": "nanonets",
        # --- field tambahan khas template `ktp` ---
        # tidak dipakai FastApiProvider.php saat ini, tapi tersimpan di kolom
        # `raw` (lihat OcrResult::$raw) sehingga tetap bisa dipakai FE/CMS
        # kalau nanti dibutuhkan, tanpa perlu ubah kontrak lagi.
        "province": fields.get("province"),
        "city": fields.get("city"),
        "blood_type": fields.get("blood_type"),
        "rt_rw": fields.get("rt_rw"),
        "kelurahan_desa": fields.get("kelurahan_desa"),
        "kecamatan": fields.get("kecamatan"),
        "nationality": fields.get("nationality"),
        "valid_until": fields.get("valid_until"),
    }


def _quality_metrics_safe(image_bytes: bytes) -> tuple[bool, bool]:
    """Cek blur/low-light pakai OpenCV bila tersedia; fallback heuristik ukuran file."""
    try:
        from app.services.image_utils import decode_image, quality_metrics

        image = decode_image(image_bytes)
        return quality_metrics(image)
    except Exception:  # pragma: no cover - opencv opsional
        return len(image_bytes) < 30_000, len(image_bytes) < 20_000
