"""Tests for biokinex.core.errors."""

from __future__ import annotations

import pytest

from biokinex.core.errors import (
    BioKineXError,
    ConfigurationError,
    DegenerateQuaternionError,
    TelemetryError,
    TransportError,
)


@pytest.mark.parametrize(
    "error_type",
    [ConfigurationError, TelemetryError, TransportError, DegenerateQuaternionError],
)
def test_all_errors_share_base_class(error_type: type[Exception]) -> None:
    assert issubclass(error_type, BioKineXError)
    with pytest.raises(BioKineXError, match="boom"):
        raise error_type("boom")


@pytest.mark.parametrize("error_type", [ConfigurationError, DegenerateQuaternionError])
def test_value_related_errors_are_value_errors(error_type: type[Exception]) -> None:
    assert issubclass(error_type, ValueError)
