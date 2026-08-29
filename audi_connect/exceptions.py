"""Custom exceptions for the Audi Connect client."""


class AudiConnectError(Exception):
    """Base exception for all Audi Connect errors."""


class AuthenticationError(AudiConnectError):
    """Raised when authentication fails (bad credentials, expired session, etc.)."""


class TokenRefreshError(AudiConnectError):
    """Raised when token refresh fails."""


class VehicleNotFoundError(AudiConnectError):
    """Raised when the requested VIN is not found."""


class ActionFailedError(AudiConnectError):
    """Raised when a vehicle action (lock, climate, etc.) fails."""


class AmbiguousActionError(ActionFailedError):
    """Raised when Audi may have accepted an action but no result was received."""


class ActionInProgressError(ActionFailedError):
    """Raised when another mutually exclusive vehicle action is unresolved."""


class CapabilityNotSupportedError(ActionFailedError):
    """Raised when a vehicle does not advertise a required capability."""


class ActionPersistenceError(ActionFailedError):
    """Raised when durable action state cannot be safely read or written."""


class ActionNotFoundError(ActionFailedError):
    """Raised when an action ID is not associated with the requested vehicle."""


class InvalidActionRequestError(ActionFailedError):
    """Raised when an action or request identifier is invalid."""


class SpinRequiredError(AudiConnectError):
    """Raised when an action requires S-PIN but none was provided."""


class CountryNotSupportedError(AudiConnectError):
    """Raised when the configured country is not in Audi's market list."""


class RequestTimeoutError(AudiConnectError):
    """Raised when an API request times out."""
