# Victoria Sekuritas — eKYC AI Service (FastAPI)

Microservice AI untuk modul eKYC. Dipanggil oleh `sekuritas-api` (Laravel) melalui
`FastApiProvider` bila `EKYC_PROVIDER=fastapi`.

## Endpoint
| Method | Path          | Input (multipart)     | Output |
|--------|---------------|-----------------------|--------|
| GET    | `/health`     | —                     | status |
| POST   | `/ocr`        | `file` (KTP)          | field KTP + `confidence` + flag kualitas |
| POST   | `/liveness`   | `file` (selfie)       | `passed`, `score`, `is_printed_photo`, `is_replay` |
| POST   | `/face-match` | `selfie`, `ktp`       | `matched`, `score`, `embedding` |

Semua endpoint (kecuali `/health`) butuh header `X-Api-Key` = `EKYC_AI_API_KEY`.

## Status saat ini
Semua service masih **STUB** (deterministik, tanpa model berat) agar bisa jalan &
diintegrasikan dulu. Titik pasang model nyata ada di:
- OCR KTP → `app/services/ocr.py` → **PaddleOCR**
- Face match → `app/services/face_match.py` → **InsightFace (ArcFace)**
- Liveness → `app/services/liveness.py` → **Silent-Face-Anti-Spoofing**

Uncomment dependency di `requirements.txt` lalu ganti body fungsi sesuai komentar `TODO`.

## Jalankan (dev)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

## Docker
```bash
docker build -t victoria-ekyc-ai .
docker run -p 8000:8000 --env-file .env victoria-ekyc-ai
```
Di production jalankan sebagai service `ekyc-ai` dalam docker-compose bersama
Laravel/Nuxt/PostgreSQL/MinIO/Nginx (lihat MASTER_PROMPT_EKYC).
