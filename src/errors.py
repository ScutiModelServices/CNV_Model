class EngineError(RuntimeError):
    """The model could not answer. `status` is the HTTP status the service maps it to."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class UnknownModel(LookupError):
    """The request names a model that is not in the catalog."""


class CapabilityError(ValueError):
    """The request uses a field the chosen model does not support (HTTP 422)."""


class ModelStopped(EngineError):
    """The model was stopped on the admin page; requests are refused until it is started (HTTP 503)."""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is stopped; start it on the admin page.", 503)
