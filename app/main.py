"""
Victoria Sekuritas — eKYC AI Service (FastAPI)

Menyediakan endpoint AI yang dipanggil oleh Laravel (FastApiProvider):
    POST /ocr         (multipart file)        -> hasil OCR KTP
    POST /liveness    (multipart file)        -> passive liveness
    POST /face-match  (multipart selfie, ktp) -> face match

Tahap awal: implementasi memakai model open-source & self-hosted:
    - OCR KTP      : Nanonets-OCR-s (GGUF via llama-cpp-python), di-load
                     IN-PROCESS lewat app/services/nanonets_engine.py (bukan
                     lagi service Flask terpisah)
    - Face match   : InsightFace (app/services/face_match.py)
    - Liveness     : Silent-Face-Anti-Spoofing (app/services/liveness.py)

Provider komersial (ADVANCE.AI/Sumsub/Veriff/dst) cukup ditambah sebagai
adapter baru di sisi Laravel — service ini tetap kontraknya sama.
"""
from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.errors import ServiceError
from app.services.ocr import extract_ktp
from app.services.liveness import check_liveness
from app.services.face_match import match_faces

app = FastAPI(title="Victoria Sekuritas eKYC AI", version="0.1.0")


@app.on_event("startup")
async def _preload_models() -> None:
    # Preload model Nanonets saat startup kalau NANONETS_PRELOAD_ON_START=true,
    # supaya request pertama tidak kena delay loading model (~beberapa detik).
    # Aman dibiarkan false di dev (lazy-load saat request pertama).
    from app.services.nanonets_engine import preload

    preload()


@app.exception_handler(ServiceError)
async def _service_error_handler(_: Request, exc: ServiceError):
    # Terjemahkan error domain (gambar invalid, wajah tak terdeteksi, engine
    # belum siap, dll) ke status code yang sesuai.
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.code, "message": exc.message},
    )


def verify_api_key(x_api_key: str | None = Header(default=None)):
    """Autentikasi sederhana antar-service (Laravel <-> FastAPI)."""
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok", "service": "ekyc-ai", "version": app.version}


@app.post("/ocr", dependencies=[Depends(verify_api_key)])
async def ocr(file: UploadFile = File(...)):
    image = await file.read()
    return JSONResponse(extract_ktp(image))


@app.post("/liveness", dependencies=[Depends(verify_api_key)])
async def liveness(
    file: UploadFile = File(...),
    expected_nik: str | None = Form(default=None),
):
    image = await file.read()
    return JSONResponse(check_liveness(image, expected_nik))


@app.post("/face-match", dependencies=[Depends(verify_api_key)])
async def face_match(selfie: UploadFile = File(...), ktp: UploadFile = File(...)):
    return JSONResponse(match_faces(await selfie.read(), await ktp.read()))
