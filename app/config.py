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
    PADDLE_LANG: str
    PADDLE_USE_ANGLE_CLS: bool
    PADDLE_DET_MODEL_DIR: Path | None
    PADDLE_REC_MODEL_DIR: Path | None
    PADDLE_CLS_MODEL_DIR: Path | None

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
        paddle_det_dir = os.getenv("PADDLE_DET_MODEL_DIR", "").strip()
        paddle_rec_dir = os.getenv("PADDLE_REC_MODEL_DIR", "").strip()
        paddle_cls_dir = os.getenv("PADDLE_CLS_MODEL_DIR", "").strip()
        return cls(
            API_KEY=os.getenv("EKYC_AI_API_KEY", "").strip(),
            MAX_UPLOAD_BYTES=_env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024, minimum=1024),
            ALLOWED_CONTENT_TYPES=content_types,
            ALLOW_STUB_FALLBACK=_env_bool("ALLOW_STUB_FALLBACK", True),
            ALLOW_MODEL_DOWNLOADS=_env_bool("ALLOW_MODEL_DOWNLOADS", False),
            USE_GPU=_env_bool("USE_GPU", False),
            OCR_ENGINE=_env_choice("OCR_ENGINE", "stub", {"stub", "paddle"}),
            PADDLE_LANG=os.getenv("PADDLE_LANG", "id").strip() or "id",
            PADDLE_USE_ANGLE_CLS=_env_bool("PADDLE_USE_ANGLE_CLS", True),
            PADDLE_DET_MODEL_DIR=(
                Path(paddle_det_dir).expanduser() if paddle_det_dir else None
            ),
            PADDLE_REC_MODEL_DIR=(
                Path(paddle_rec_dir).expanduser() if paddle_rec_dir else None
            ),
            PADDLE_CLS_MODEL_DIR=(
                Path(paddle_cls_dir).expanduser() if paddle_cls_dir else None
            ),
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
