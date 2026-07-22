"""Errors that can safely be translated to API responses."""
from __future__ import annotations


class ServiceError(Exception):
    status_code = 500
    code = "service_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidImageError(ServiceError):
    status_code = 422
    code = "invalid_image"


class FaceNotFoundError(ServiceError):
    status_code = 422
    code = "face_not_found"


class MultipleFacesError(ServiceError):
    status_code = 422
    code = "multiple_faces"


class EngineUnavailableError(ServiceError):
    status_code = 503
    code = "engine_unavailable"


class InferenceError(ServiceError):
    status_code = 503
    code = "inference_failed"
