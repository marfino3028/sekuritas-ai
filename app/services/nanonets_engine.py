"""
Engine Nanonets-OCR-s (GGUF, via llama-cpp-python) — dijalankan IN-PROCESS di
dalam service FastAPI ini (bukan service Flask terpisah lagi).

Dipakai bersama oleh:
  - app/services/ocr.py       (OCR KTP murni, endpoint /ocr)
  - app/services/selfie_ktp.py (baca NIK dari foto selfie+KTP, pelengkap /liveness)

Model & mmproj (~2GB) di-load sekali sebagai singleton dan dipakai ulang lintas
request. Import berat (llama_cpp, huggingface_hub) LAZY di dalam fungsi supaya
service tetap bisa start dengan OCR_ENGINE=stub tanpa dependency model.

Auto-download dari Hugging Face hanya jalan kalau ALLOW_MODEL_DOWNLOADS=true
(untuk hindari download tidak sengaja di production).
"""
from __future__ import annotations

import json
import re
import sys
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from app.config import settings
from app.services.errors import EngineUnavailableError, InferenceError

_MODEL = None
_MODEL_LOCK = threading.Lock()

# Schema per template. Sengaja hardcode di sini (bukan file YAML terpisah)
# supaya modul ini mandiri — tidak bergantung pada folder ocr-engine/ lagi.
TEMPLATES: dict[str, dict] = {
    "ktp": {
        "province": None,
        "city": None,
        "nik": None,
        "name": None,
        "birth_place": None,
        "birth_date": None,
        "gender": None,
        "blood_type": None,
        "address": None,
        "rt_rw": None,
        "kelurahan_desa": None,
        "kecamatan": None,
        "religion": None,
        "marital_status": None,
        "occupation": None,
        "nationality": None,
        "valid_until": None,
    },
    "selfie_ktp": {
        "nik": None,
        "name": None,
    },
}

_PROMPT_INSTRUCTION = (
    "Return JSON matching schema exactly. Same keys only. Values only, "
    "no labels. Use null if unreadable."
)


def _load_llama_cpp():
    try:
        from llama_cpp import Llama
        from llama_cpp import llama_chat_format
    except ImportError as exc:
        raise EngineUnavailableError(
            "llama-cpp-python belum terinstall. "
            "`pip install -r requirements.txt -r requirements-models.txt`."
        ) from exc
    return Llama, llama_chat_format


def _resolve_chat_handler(chat_format, mmproj_path: Path):
    handler_name = (settings.NANONETS_CHAT_HANDLER or "qwen2.5-vl").lower()
    handlers = {
        "mtmd": "MTMDChatHandler",
        "llava-1-5": "Llava15ChatHandler",
        "llava": "Llava15ChatHandler",
        "qwen2.5-vl": "Qwen25VLChatHandler",
        "qwen25-vl": "Qwen25VLChatHandler",
        "gemma4": "Gemma4ChatHandler",
        "minicpm-v-2.6": "MiniCPMv26ChatHandler",
        "nanollava": "NanoLlavaChatHandler",
    }
    class_name = handlers.get(handler_name)
    if not class_name:
        raise EngineUnavailableError(
            f"NANONETS_CHAT_HANDLER '{handler_name}' tidak dikenal."
        )
    handler_cls = getattr(chat_format, class_name, None)
    if handler_cls is None:
        raise EngineUnavailableError(
            f"llama-cpp-python yang terinstall belum punya {class_name}. "
            "Upgrade llama-cpp-python."
        )
    try:
        return handler_cls(clip_model_path=str(mmproj_path), verbose=settings.NANONETS_VERBOSE)
    except TypeError:
        return handler_cls(clip_model_path=str(mmproj_path))


def _download_model_file(repo_id: str, filename: str, target_path: Path, revision: str):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename} from Hugging Face repo {repo_id}...", file=sys.stderr, flush=True)
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=str(target_path.parent),
            local_dir_use_symlinks=False,
        )
    except ImportError:
        repo_quoted = urllib.parse.quote(repo_id, safe="/")
        filename_quoted = urllib.parse.quote(filename)
        url = f"https://huggingface.co/{repo_quoted}/resolve/{revision}/{filename_quoted}"
        downloaded = str(target_path)
        urllib.request.urlretrieve(url, downloaded)
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != target_path.resolve():
        downloaded_path.replace(target_path)


def _ensure_model_files():
    if settings.NANONETS_MODEL_PATH.exists() and settings.NANONETS_MMPROJ_PATH.exists():
        return
    if not settings.ALLOW_MODEL_DOWNLOADS:
        missing = [
            str(p) for p in (settings.NANONETS_MODEL_PATH, settings.NANONETS_MMPROJ_PATH)
            if not p.exists()
        ]
        raise EngineUnavailableError(
            "File model Nanonets belum ada dan ALLOW_MODEL_DOWNLOADS=false: "
            f"{', '.join(missing)}. Set ALLOW_MODEL_DOWNLOADS=true atau taruh "
            "manual file .gguf di path tsb."
        )
    if not settings.NANONETS_MODEL_PATH.exists():
        _download_model_file(
            settings.NANONETS_REPO_ID, settings.NANONETS_MODEL_FILE,
            settings.NANONETS_MODEL_PATH, settings.NANONETS_REVISION,
        )
    if not settings.NANONETS_MMPROJ_PATH.exists():
        _download_model_file(
            settings.NANONETS_REPO_ID, settings.NANONETS_MMPROJ_FILE,
            settings.NANONETS_MMPROJ_PATH, settings.NANONETS_REVISION,
        )


def _get_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            _ensure_model_files()

            Llama, chat_format = _load_llama_cpp()
            chat_handler = _resolve_chat_handler(chat_format, settings.NANONETS_MMPROJ_PATH)

            _MODEL = Llama(
                model_path=str(settings.NANONETS_MODEL_PATH),
                chat_handler=chat_handler,
                n_ctx=settings.NANONETS_CTX_SIZE,
                n_gpu_layers=settings.NANONETS_N_GPU_LAYERS,
                n_threads=settings.NANONETS_N_THREADS,
                n_batch=settings.NANONETS_N_BATCH,
                flash_attn=settings.NANONETS_FLASH_ATTN,
                use_mmap=settings.NANONETS_USE_MMAP,
                use_mlock=settings.NANONETS_USE_MLOCK,
                verbose=settings.NANONETS_VERBOSE,
            )
        except EngineUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - tergantung environment runtime
            raise EngineUnavailableError(f"Gagal memuat model Nanonets: {exc}") from exc
    return _MODEL


def preload():
    """Panggil saat startup FastAPI kalau NANONETS_PRELOAD_ON_START=true."""
    if not settings.NANONETS_PRELOAD_ON_START:
        return
    if settings.OCR_ENGINE != "nanonets" and settings.SELFIE_KTP_ENGINE != "nanonets_tesseract":
        return
    print("Loading Nanonets-OCR-s model...", file=sys.stderr, flush=True)
    _get_model()
    print("Nanonets-OCR-s model ready.", file=sys.stderr, flush=True)


def _extract_json_object(text: str) -> dict:
    raw = (text or "").strip().replace("```json", "").replace("```", "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise InferenceError(f"Respons model bukan JSON valid: {exc}") from exc
        preview = raw[:300].replace("\n", "\\n")
        raise InferenceError(f"Respons model bukan JSON valid. Preview: {preview}")


def _apply_schema(data: dict, schema: dict) -> dict:
    if not isinstance(data, dict):
        data = {}
    normalized = {}
    for key in schema:
        value = data.get(key)
        normalized[key] = value if value not in ("", "Not found") else None
    return normalized


def _image_to_data_uri(image_bytes: bytes) -> str:
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")


def extract(image_bytes: bytes, template_name: str) -> dict:
    """Ekstrak field dari gambar sesuai template. Raise EngineUnavailableError /
    InferenceError bila gagal (biar caller bisa fallback ke stub)."""
    schema = TEMPLATES.get(template_name)
    if schema is None:
        raise InferenceError(f"Template Nanonets '{template_name}' tidak dikenal.")

    model = _get_model()
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    prompt_text = f"{_PROMPT_INSTRUCTION}\nTemplate: {template_name}\nSchema:\n{schema_text}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image_bytes)}},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    try:
        response = model.create_chat_completion(
            messages=messages,
            temperature=settings.NANONETS_TEMPERATURE,
            max_tokens=settings.NANONETS_MAX_TOKENS,
        )
    except Exception as exc:  # pragma: no cover - tergantung environment runtime
        raise InferenceError(f"Inference Nanonets gagal: {exc}") from exc

    choices = response.get("choices") or []
    if not choices:
        raise InferenceError("Respons Nanonets tidak punya choices.")
    message = choices[0].get("message") or {}
    content = message.get("content") or choices[0].get("text")
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if not content:
        raise InferenceError("Respons Nanonets tidak punya konten teks.")

    data = _extract_json_object(content)
    return _apply_schema(data, schema)
