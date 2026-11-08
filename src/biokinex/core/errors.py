"""Exception hierarchy for BioKineX.

Every domain error derives from :class:`BioKineXError`, so callers can catch all library failures
with one ``except`` clause. Plain programming mistakes, such as passing an array with the wrong
shape, raise the standard built-in exceptions (``ValueError``, ``TypeError``).
"""

from __future__ import annotations


class BioKineXError(Exception):
    """Base class for all BioKineX domain errors."""


class ConfigurationError(BioKineXError, ValueError):
    """A pipeline, filter, or transport configuration is invalid."""


class TelemetryError(BioKineXError):
    """Incoming telemetry was rejected at the trust boundary."""


class TransportError(BioKineXError):
    """A source or sink failed to receive, decode, or send data."""


class DegenerateQuaternionError(BioKineXError, ValueError):
    """A quaternion is zero, nearly zero, or non-finite, so it has no valid orientation."""
