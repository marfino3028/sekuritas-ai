"""
OCR KTP.

Dua mode (dipilih via OCR_ENGINE di config):
  - "stub"   : deterministik, tanpa model (default; validasi header + skor dummy).
  - "paddle" : PaddleOCR (https://github.com/PaddlePaddle/PaddleOCR) — model asli.

Import berat (paddleocr/opencv) dilakukan LAZY di dalam fungsi agar service tetap
bisa di-import & di-compile tanpa dependency model.
"""
from __future__ import annotations

import hashlib
import re

from app.config import settings
from app.services.errors import EngineUnavailableError, InferenceError
from app.services.image_utils import inspect_image

NIK_RE = re.compile(r"\b\d{16}\b")
DATE_RE = re.compile(r"(\d{2})[-/. ](\d{2})[-/. ](\d{4})")

_PADDLE = None


def extract_ktp(image_bytes: bytes) -> dict:
    """Ekstrak field KTP dari bytes gambar. Selalu validasi header dulu."""
    info = inspect_image(image_bytes)  # raises InvalidImageError bila bukan gambar valid

    if settings.OCR_ENGINE == "paddle":
        try:
            return _extract_paddle(image_bytes)
        except (EngineUnavailableError, InferenceError):
            if not settings.ALLOW_STUB_FALLBACK:
                raise
            # fallback ke stub bila model belum tersedia
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
# PADDLEOCR
# --------------------------------------------------------------------------- #
def _ocr_engine():
    global _PADDLE
    if _PADDLE is None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - butuh extras model
            raise EngineUnavailableError(
                "paddleocr belum terpasang. `pip install paddleocr paddlepaddle`."
            ) from exc
        _PADDLE = PaddleOCR(
            use_angle_cls=settings.PADDLE_USE_ANGLE_CLS,
            lang=settings.PADDLE_LANG,
            det_model_dir=str(settings.PADDLE_DET_MODEL_DIR) if settings.PADDLE_DET_MODEL_DIR else None,
            rec_model_dir=str(settings.PADDLE_REC_MODEL_DIR) if settings.PADDLE_REC_MODEL_DIR else None,
            cls_model_dir=str(settings.PADDLE_CLS_MODEL_DIR) if settings.PADDLE_CLS_MODEL_DIR else None,
            use_gpu=settings.USE_GPU,
            show_log=False,
        )
    return _PADDLE


def _extract_paddle(image_bytes: bytes) -> dict:
    from app.services.image_utils import decode_image, quality_metrics

    image = decode_image(image_bytes)  # BGR np.ndarray (raise EngineUnavailableError bila cv2 absen)
    try:
        result = _ocr_engine().ocr(image, cls=True)
    except Exception as exc:  # pragma: no cover - runtime model
        raise InferenceError(f"OCR gagal: {exc}") from exc

    # PaddleOCR mengembalikan [[ [box, (text, score)], ... ]]
    items: list[tuple[str, float]] = []
    for page in result or []:
        for line in page or []:
            try:
                text, score = line[1][0], float(line[1][1])
            except (IndexError, TypeError, ValueError):
                continue
            if text and text.strip():
                items.append((text.strip(), score))

    lines = [t for t, _ in items]
    fields = _parse_ktp(lines)
    is_blur, is_low_light = quality_metrics(image)
    avg_conf = int(round(100 * (sum(s for _, s in items) / len(items)))) if items else 0

    return {
        **fields,
        "confidence": avg_conf,
        "is_blur": is_blur,
        "is_low_light": is_low_light,
        "is_screenshot": False,
        "engine": "paddle",
        "lines": lines,
    }


# --------------------------------------------------------------------------- #
# Parser field KTP (label-based heuristik)
# --------------------------------------------------------------------------- #
def _after_label(line: str) -> str:
    # ambil teks setelah ':' bila ada, kalau tidak buang kata label pertama
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return line.strip()


def _parse_ktp(lines: list[str]) -> dict:
    text = "\n".join(lines)
    upper = text.upper()

    nik_match = NIK_RE.search(re.sub(r"\s", "", text))
    fields: dict = {
        "nik": nik_match.group(0) if nik_match else None,
        "name": None,
        "birth_place": None,
        "birth_date": None,
        "gender": None,
        "address": None,
        "religion": None,
        "marital_status": None,
        "occupation": None,
    }

    for line in lines:
        u = line.upper()
        if fields["name"] is None and "NAMA" in u:
            val = _after_label(line)
            if val and len(re.sub(r"[^A-Za-z ]", "", val).strip()) > 2:
                fields["name"] = re.sub(r"[^A-Za-z .'-]", "", val).strip().upper()
        elif ("TEMPAT" in u and "LAHIR" in u) or "TGL LAHIR" in u:
            val = _after_label(line)
            d = DATE_RE.search(val)
            if d:
                fields["birth_date"] = f"{d.group(3)}-{d.group(2)}-{d.group(1)}"
                place = val[: d.start()].strip(" ,")
                if place:
                    fields["birth_place"] = re.sub(r"[^A-Za-z ]", "", place).strip().upper()
        elif "JENIS KELAMIN" in u or "KELAMIN" in u:
            if "PEREMPUAN" in u:
                fields["gender"] = "PEREMPUAN"
            elif "LAKI" in u:
                fields["gender"] = "LAKI-LAKI"
        elif "ALAMAT" in u and fields["address"] is None:
            fields["address"] = _after_label(line)
        elif "AGAMA" in u:
            fields["religion"] = _after_label(line).upper() or None
        elif "PERKAWINAN" in u or "STATUS" in u:
            fields["marital_status"] = _after_label(line).upper() or None
        elif "PEKERJAAN" in u:
            fields["occupation"] = _after_label(line).upper() or None

    # gender fallback dari seluruh teks
    if fields["gender"] is None:
        if "PEREMPUAN" in upper:
            fields["gender"] = "PEREMPUAN"
        elif "LAKI-LAKI" in upper or "LAKI LAKI" in upper:
            fields["gender"] = "LAKI-LAKI"

    return fields
