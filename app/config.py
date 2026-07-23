"""Configuration for the eKYC service.

The model runtimes are optional on purpose.  A developer can run the API with
the deterministic stub engines, while a production deployment can opt in to
each real engine independently.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Muat file .env (bila ada) ke os.environ.

    Tanpa dependency eksternal. Memakai setdefault agar variabel yang sudah
    diekspor di shell tetap menang (perilaku 'immutable' seperti dotenv).
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


_load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_float(
    name: str, default: float, *, minimum: float = 0.0, maximum: float = 1.0
) -> float:
    value = float(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {expected}")
    return value


@dataclass(frozen=True)
class Settings:
    API_KEY: str
    MAX_UPLOAD_BYTES: int
    ALLOWED_CONTENT_TYPES: tuple[str, ...]
    ALLOW_STUB_FALLBACK: bool
    ALLOW_MODEL_DOWNLOADS: bool
    USE_GPU: bool

    OCR_ENGINE: str
    NANONETS_OCR_TEMPLATE: str

    SELFIE_KTP_CHECK_ENABLED: bool
    SELFIE_KTP_ENGINE: str
    TESSERACT_LANG: str
    TESSERACT_CMD: str | None

    NANONETS_MODEL_PATH: Path
    NANONETS_MMPROJ_PATH: Path
    NANONETS_REPO_ID: str
    NANONETS_REVISION: str
    NANONETS_MODEL_FILE: str
    NANONETS_MMPROJ_FILE: str
    NANONETS_CHAT_HANDLER: str
    NANONETS_CTX_SIZE: int
    NANONETS_N_GPU_LAYERS: int
    NANONETS_N_THREADS: int
    NANONETS_N_BATCH: int
    NANONETS_FLASH_ATTN: bool
    NANONETS_USE_MMAP: bool
    NANONETS_USE_MLOCK: bool
    NANONETS_MAX_TOKENS: int
    NANONETS_TEMPERATURE: float
    NANONETS_VERBOSE: bool
    NANONETS_PRELOAD_ON_START: bool

    FACE_MATCH_ENGINE: str
    FACE_MATCH_THRESHOLD: float
    INSIGHTFACE_MODEL_NAME: str
    INSIGHTFACE_MODEL_ROOT: Path
    INSIGHTFACE_DET_SIZE: int
    FACE_REQUIRE_SINGLE_FACE: bool
    RETURN_FACE_EMBEDDING: bool

    LIVENESS_ENGINE: str
    LIVENESS_THRESHOLD: float
    LIVENESS_MODEL_DIR: Path | None
    LIVENESS_INPUT_WIDTH: int
    LIVENESS_INPUT_HEIGHT: int
    LIVENESS_LIVE_CLASS_INDEX: int
    LIVENESS_PRINT_CLASS_INDEX: int
    LIVENESS_REPLAY_CLASS_INDEX: int
    LIVENESS_REQUIRE_FACE: bool

    @classmethod
    def from_env(cls) -> "Settings":
        content_types = tuple(
            item.strip().lower()
            for item in os.getenv(
                "ALLOWED_IMAGE_CONTENT_TYPES", "image/jpeg,image/png,image/webp"
            ).split(",")
            if item.strip()
        )
        model_dir = os.getenv("LIVENESS_MODEL_DIR", "").strip()
        return cls(
            API_KEY=os.getenv("EKYC_AI_API_KEY", "").strip(),
            MAX_UPLOAD_BYTES=_env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024, minimum=1024),
            ALLOWED_CONTENT_TYPES=content_types,
            ALLOW_STUB_FALLBACK=_env_bool("ALLOW_STUB_FALLBACK", True),
            ALLOW_MODEL_DOWNLOADS=_env_bool("ALLOW_MODEL_DOWNLOADS", False),
            USE_GPU=_env_bool("USE_GPU", False),
            OCR_ENGINE=_env_choice("OCR_ENGINE", "stub", {"stub", "nanonets"}),
            NANONETS_OCR_TEMPLATE=os.getenv("NANONETS_OCR_TEMPLATE", "ktp").strip() or "ktp",
            SELFIE_KTP_CHECK_ENABLED=_env_bool("SELFIE_KTP_CHECK_ENABLED", True),
            SELFIE_KTP_ENGINE=_env_choice(
                "SELFIE_KTP_ENGINE", "stub", {"stub", "nanonets_tesseract"}
            ),
            TESSERACT_LANG=os.getenv("TESSERACT_LANG", "ind+eng").strip() or "ind+eng",
            TESSERACT_CMD=os.getenv("TESSERACT_CMD", "").strip() or None,
            NANONETS_MODEL_PATH=Path(
                os.getenv("NANONETS_MODEL_PATH", "./model/Nanonets-OCR-s-Q4_0.gguf")
            ).expanduser().resolve(),
            NANONETS_MMPROJ_PATH=Path(
                os.getenv("NANONETS_MMPROJ_PATH", "./model/Nanonets-OCR-s-mmproj-F16.gguf")
            ).expanduser().resolve(),
            NANONETS_REPO_ID=os.getenv("NANONETS_REPO_ID", "unsloth/Nanonets-OCR-s-GGUF").strip(),
            NANONETS_REVISION=os.getenv("NANONETS_REVISION", "main").strip() or "main",
            NANONETS_MODEL_FILE=os.getenv("NANONETS_MODEL_FILE", "Nanonets-OCR-s-Q4_0.gguf").strip(),
            NANONETS_MMPROJ_FILE=os.getenv("NANONETS_MMPROJ_FILE", "mmproj-F16.gguf").strip(),
            NANONETS_CHAT_HANDLER=os.getenv("NANONETS_CHAT_HANDLER", "qwen2.5-vl").strip() or "qwen2.5-vl",
            NANONETS_CTX_SIZE=_env_int("NANONETS_CTX_SIZE", 8192, minimum=512),
            NANONETS_N_GPU_LAYERS=_env_int("NANONETS_N_GPU_LAYERS", -1, minimum=-1),
            NANONETS_N_THREADS=_env_int("NANONETS_N_THREADS", 4, minimum=1),
            NANONETS_N_BATCH=_env_int("NANONETS_N_BATCH", 2048, minimum=1),
            NANONETS_FLASH_ATTN=_env_bool("NANONETS_FLASH_ATTN", True),
            NANONETS_USE_MMAP=_env_bool("NANONETS_USE_MMAP", True),
            NANONETS_USE_MLOCK=_env_bool("NANONETS_USE_MLOCK", False),
            NANONETS_MAX_TOKENS=_env_int("NANONETS_MAX_TOKENS", 512, minimum=16),
            NANONETS_TEMPERATURE=_env_float("NANONETS_TEMPERATURE", 0.0, minimum=0.0, maximum=2.0),
            NANONETS_VERBOSE=_env_bool("NANONETS_VERBOSE", False),
            NANONETS_PRELOAD_ON_START=_env_bool("NANONETS_PRELOAD_ON_START", False),
            FACE_MATCH_ENGINE=_env_choice(
                "FACE_MATCH_ENGINE", "stub", {"stub", "insightface"}
            ),
            FACE_MATCH_THRESHOLD=_env_float("FACE_MATCH_THRESHOLD", 0.65),
            INSIGHTFACE_MODEL_NAME=os.getenv(
                "INSIGHTFACE_MODEL_NAME", "buffalo_l"
            ).strip()
            or "buffalo_l",
            INSIGHTFACE_MODEL_ROOT=Path(
                os.getenv("INSIGHTFACE_MODEL_ROOT", "~/.insightface")
            ).expanduser(),
            INSIGHTFACE_DET_SIZE=_env_int("INSIGHTFACE_DET_SIZE", 640, minimum=160),
            FACE_REQUIRE_SINGLE_FACE=_env_bool("FACE_REQUIRE_SINGLE_FACE", True),
            RETURN_FACE_EMBEDDING=_env_bool("RETURN_FACE_EMBEDDING", True),
            LIVENESS_ENGINE=_env_choice(
                "LIVENESS_ENGINE", "stub", {"stub", "silent-face-onnx"}
            ),
            LIVENESS_THRESHOLD=_env_float("LIVENESS_THRESHOLD", 0.80),
            LIVENESS_MODEL_DIR=Path(model_dir).expanduser() if model_dir else None,
            LIVENESS_INPUT_WIDTH=_env_int("LIVENESS_INPUT_WIDTH", 80, minimum=16),
            LIVENESS_INPUT_HEIGHT=_env_int("LIVENESS_INPUT_HEIGHT", 80, minimum=16),
            LIVENESS_LIVE_CLASS_INDEX=_env_int(
                "LIVENESS_LIVE_CLASS_INDEX", 1, minimum=0
            ),
            LIVENESS_PRINT_CLASS_INDEX=_env_int(
                "LIVENESS_PRINT_CLASS_INDEX", 0, minimum=0
            ),
            LIVENESS_REPLAY_CLASS_INDEX=_env_int(
                "LIVENESS_REPLAY_CLASS_INDEX", 2, minimum=0
            ),
            LIVENESS_REQUIRE_FACE=_env_bool("LIVENESS_REQUIRE_FACE", True),
        )


settings = Settings.from_env()
