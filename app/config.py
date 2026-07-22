import os


class Settings:
    API_KEY: str = os.getenv("EKYC_AI_API_KEY", "")
    # Ambang default (Laravel tetap punya keputusan final)
    LIVENESS_THRESHOLD: float = float(os.getenv("LIVENESS_THRESHOLD", "0.80"))
    FACE_MATCH_THRESHOLD: float = float(os.getenv("FACE_MATCH_THRESHOLD", "0.65"))
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"


settings = Settings()
