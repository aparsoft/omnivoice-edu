"""Errors raised by the synthesis engine.

The engine stays independent of the web framework: it raises these, and the
API layer turns them into HTTP responses with the matching status code.
"""


class TTSError(Exception):
    """Base error; ``status_code`` is the HTTP status the API should return."""

    status_code = 500


class InvalidRequest(TTSError):
    """The request cannot be synthesized as given (HTTP 400)."""

    status_code = 400


class ModelNotReady(TTSError):
    """The model is not loaded yet (HTTP 503)."""

    status_code = 503
