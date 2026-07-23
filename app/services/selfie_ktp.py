"""
Deteksi KTP di dalam foto selfie ("selfie sambil pegang KTP").

Dipakai sebagai pelengkap OPSIONAL pada endpoint /liveness: kalau di foto
selfie itu ada juga KTP yang ikut kefoto, service ini memvalidasi:

  1. NIK yang kebaca di foto tsb cocok dengan NIK yang sudah tersimpan dari
     hasil OCR KTP sebelumnya (dikirim sebagai `expected_nik`).
  2. Wajah orangnya (wajah terbesar di foto) dicocokkan — pakai embedding
     InsightFace — terhadap wajah foto yang tercetak di kartu KTP (wajah
     terbesar KEDUA) di FOTO YANG SAMA. Ini beda dari /face-match yang
     membandingkan file selfie terpisah vs file KTP terpisah.

Engine (via SELFIE_KTP_ENGINE):
  - "stub"              : deterministik, tanpa model (default).
  - "nanonets_tesseract" : Nanonets-OCR-s (in-process, lihat nanonets_engine.py)
                           sebagai baca teks UTAMA. Tesseract dipakai HANYA
                           sebagai fallback ringan kalau Nanonets gagal
                           inisialisasi/inference atau tidak berhasil
                           menemukan NIK 16 digit di foto.

Catatan: Google ML Kit TIDAK dipakai di sini — ML Kit adalah SDK on-device
Android/iOS, tidak bisa dipanggil dari backend Python. PaddleOCR sudah
dihapus total (diganti Nanonets).

Import berat (insightface/opencv/pytesseract) dilakukan LAZY di dalam fungsi
agar service tetap bisa di-import tanpa dependency model.
"""
from __future__ import annotations

import re

from app.config import settings
from app.services.errors import EngineUnavailableError, InferenceError

NIK_RE = re.compile(r"\b\d{16}\b")

_FACE_APP = None


def analyze_selfie_with_ktp(image_bytes: bytes, expected_nik: str | None) -> dict:
    """Cek apakah ada KTP ikut kefoto bareng selfie, lalu validasi NIK & wajah."""
    if settings.SELFIE_KTP_ENGINE != "nanonets_tesseract":
        return _stub_result()

    try:
        return _analyze_nanonets_tesseract(image_bytes, expected_nik)
    except (EngineUnavailableError, InferenceError):
        if not settings.ALLOW_STUB_FALLBACK:
            raise
        return _stub_result()


def _stub_result() -> dict:
    return {
        "ktp_detected": False,
        "nik_in_photo": None,
        "nik_match": None,
        "id_face_match": None,
        "id_face_match_score": None,
        "ktp_check_engine": "stub",
    }


# --------------------------------------------------------------------------- #
# Loader model wajah (lazy, singleton — dipisah dari face_match.py karena
# fitur ini punya siklus hidup & config sendiri)
# --------------------------------------------------------------------------- #
def _face_app():
    global _FACE_APP
    if _FACE_APP is None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover
            raise EngineUnavailableError(
                "insightface belum terpasang. `pip install insightface onnxruntime`."
            ) from exc
        try:
            app = FaceAnalysis(
                name=settings.INSIGHTFACE_MODEL_NAME,
                root=str(settings.INSIGHTFACE_MODEL_ROOT),
                allowed_modules=["detection", "recognition"],
            )
            det = settings.INSIGHTFACE_DET_SIZE
            app.prepare(ctx_id=0 if settings.USE_GPU else -1, det_size=(det, det))
        except Exception as exc:  # pragma: no cover
            raise EngineUnavailableError(
                f"Gagal inisialisasi InsightFace untuk cek selfie+KTP: {exc}"
            ) from exc
        _FACE_APP = app
    return _FACE_APP


def _nik_from_nanonets(image_bytes: bytes) -> tuple[str | None, str | None]:
    """Baca NIK (+ nama) dari foto lewat Nanonets. Raise kalau engine benar2
    tidak tersedia; return (None, None) kalau engine jalan tapi NIK tidak
    ketemu (supaya bisa lanjut ke fallback tesseract)."""
    from app.services.nanonets_engine import extract as nanonets_extract

    fields = nanonets_extract(image_bytes, "selfie_ktp")
    nik = fields.get("nik")
    if nik and not NIK_RE.fullmatch(re.sub(r"\s", "", str(nik))):
        nik = None  # buang hasil yang bukan 16 digit murni
    return nik, fields.get("name")


def _nik_from_tesseract(image) -> str | None:
    """Fallback ringan: Tesseract saja (tanpa Paddle) kalau Nanonets gagal
    atau tidak menemukan NIK. Hanya dipakai sebagai pelengkap, bukan jalur
    utama."""
    try:
        import pytesseract
    except ImportError:
        return None  # fallback opsional; tidak menggagalkan proses
    if settings.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
    try:
        text = pytesseract.image_to_string(image, lang=settings.TESSERACT_LANG)
    except Exception:
        return None
    match = NIK_RE.search(re.sub(r"\s", "", text))
    return match.group(0) if match else None


def _analyze_nanonets_tesseract(image_bytes: bytes, expected_nik: str | None) -> dict:
    from app.services.image_utils import decode_image

    # --- 1. Baca NIK: Nanonets dulu (utama), Tesseract kalau perlu (fallback) --
    engine_used = "nanonets"
    try:
        nik_in_photo, _name = _nik_from_nanonets(image_bytes)
    except (EngineUnavailableError, InferenceError):
        nik_in_photo = None
        engine_used = "tesseract"  # Nanonets sama sekali tidak jalan

    if nik_in_photo is None:
        image = decode_image(image_bytes)  # BGR np.ndarray
        fallback_nik = _nik_from_tesseract(image)
        if fallback_nik:
            nik_in_photo = fallback_nik
            engine_used = "nanonets+tesseract" if engine_used == "nanonets" else "tesseract"

    # --- 2. Deteksi wajah: >=2 wajah dianggap "orang + foto KTP" --------------
    image = decode_image(image_bytes)
    faces = _face_app().get(image)
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )

    ktp_detected = bool(nik_in_photo) or len(faces_sorted) >= 2

    id_face_match = None
    id_face_match_score = None
    if len(faces_sorted) >= 2:
        import numpy as np

        live_face, id_face = faces_sorted[0], faces_sorted[1]
        cosine = float(np.dot(live_face.normed_embedding, id_face.normed_embedding))
        similarity = max(0.0, min(1.0, (cosine + 1) / 2))
        id_face_match_score = int(round(similarity * 100))
        id_face_match = similarity >= settings.FACE_MATCH_THRESHOLD

    nik_match = None
    if expected_nik and nik_in_photo:
        nik_match = nik_in_photo == expected_nik

    return {
        "ktp_detected": ktp_detected,
        "nik_in_photo": nik_in_photo,
        "nik_match": nik_match,
        "id_face_match": id_face_match,
        "id_face_match_score": id_face_match_score,
        "ktp_check_engine": engine_used,
    }
