"""Second-order IIR (biquad) low-pass and notch filters.

Coefficients follow R. Bristow-Johnson, "Cookbook formulae for audio EQ biquad filter
coefficients", which uses the bilinear transform with frequency pre-warping. The filter runs in
transposed direct form II, which needs only two state values per channel and behaves well in
floating point.

A biquad is designed for one fixed sample rate. Timestamps are still checked so that the filter
fits the common :class:`~biokinex.dsp.base.Filter` interface, but the time between samples is not
used. The gap filler upstream keeps the stream close to the nominal rate.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from biokinex.core.types import FloatArray
from biokinex.dsp.base import check_timestamp, elapsed
from biokinex.schemas.config import LowPassConfig, NotchConfig

__all__ = ["BiquadCoefficients", "BiquadFilter"]

_BUTTERWORTH_Q = 1.0 / math.sqrt(2.0)


@dataclass(frozen=True, slots=True)
class BiquadCoefficients:
    """Normalized biquad coefficients, with ``a0`` scaled to 1.

    The transfer function is ``H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)``.
    """

    b0: float
    b1: float
    b2: float
    a1: float
    a2: float

    @classmethod
    def low_pass(
        cls, cutoff_hz: float, sample_rate_hz: float, q: float = _BUTTERWORTH_Q
    ) -> BiquadCoefficients:
        """Design a low-pass filter; its gain at ``cutoff_hz`` equals ``q``."""
        cos_w0, alpha = _prepare(cutoff_hz, sample_rate_hz, q)
        a0 = 1.0 + alpha
        b1 = (1.0 - cos_w0) / a0
        return cls(b0=0.5 * b1, b1=b1, b2=0.5 * b1, a1=-2.0 * cos_w0 / a0, a2=(1.0 - alpha) / a0)

    @classmethod
    def notch(cls, center_hz: float, sample_rate_hz: float, q: float) -> BiquadCoefficients:
        """Design a notch filter with zero gain at ``center_hz``; higher ``q`` is narrower."""
        cos_w0, alpha = _prepare(center_hz, sample_rate_hz, q)
        a0 = 1.0 + alpha
        b1 = -2.0 * cos_w0 / a0
        return cls(b0=1.0 / a0, b1=b1, b2=1.0 / a0, a1=b1, a2=(1.0 - alpha) / a0)

    @property
    def dc_gain(self) -> float:
        """Gain for a constant input, which is ``H(z)`` at ``z = 1``."""
        return (self.b0 + self.b1 + self.b2) / (1.0 + self.a1 + self.a2)

    def frequency_response(self, frequency_hz: float, sample_rate_hz: float) -> complex:
        """Return the complex gain ``H(e^{jw})`` at ``frequency_hz``."""
        z_inv = cmath.exp(-2j * math.pi * frequency_hz / sample_rate_hz)
        numerator = self.b0 + self.b1 * z_inv + self.b2 * z_inv * z_inv
        denominator = 1.0 + self.a1 * z_inv + self.a2 * z_inv * z_inv
        return numerator / denominator


def _prepare(frequency_hz: float, sample_rate_hz: float, q: float) -> tuple[float, float]:
    """Validate the design inputs and return ``cos(w0)`` and ``alpha`` from the cookbook."""
    if not (math.isfinite(sample_rate_hz) and sample_rate_hz > 0.0):
        msg = f"sample_rate_hz must be finite and positive, got {sample_rate_hz!r}"
        raise ValueError(msg)
    nyquist_hz = 0.5 * sample_rate_hz
    if not 0.0 < frequency_hz < nyquist_hz:
        msg = f"frequency must be in (0, {nyquist_hz}) Hz, got {frequency_hz!r}"
        raise ValueError(msg)
    if not (math.isfinite(q) and q > 0.0):
        msg = f"q must be finite and positive, got {q!r}"
        raise ValueError(msg)
    w0 = 2.0 * math.pi * frequency_hz / sample_rate_hz
    return math.cos(w0), math.sin(w0) / (2.0 * q)


class BiquadFilter:
    """Runs one biquad on every element of a fixed-shape vector."""

    __slots__ = ("_coefficients", "_state1", "_state2", "_timestamp")

    def __init__(self, coefficients: BiquadCoefficients) -> None:
        self._coefficients = coefficients
        self._state1: FloatArray | None = None
        self._state2: FloatArray | None = None
        self._timestamp = -math.inf

    @classmethod
    def from_config(
        cls, config: LowPassConfig | NotchConfig, sample_rate_hz: float
    ) -> BiquadFilter:
        """Build a low-pass or notch filter from validated configuration."""
        if isinstance(config, LowPassConfig):
            return cls(BiquadCoefficients.low_pass(config.cutoff_hz, sample_rate_hz, config.q))
        return cls(BiquadCoefficients.notch(config.center_hz, sample_rate_hz, config.q))

    @property
    def coefficients(self) -> BiquadCoefficients:
        """The coefficients in use."""
        return self._coefficients

    def step(self, value: npt.ArrayLike, timestamp: float) -> FloatArray:
        """Filter one sample and return a new array with the filtered value."""
        x = np.array(value, dtype=np.float64)
        c = self._coefficients
        if self._state1 is None or self._state2 is None:
            self._timestamp = check_timestamp(timestamp)
            # Starting in the steady state for this value avoids a start-up transient, which would
            # otherwise ring for many samples on signals with a large offset, such as gravity.
            y = c.dc_gain * x
            self._state1 = y - c.b0 * x
            self._state2 = c.b2 * x - c.a2 * y
            return y
        if x.shape != self._state1.shape:
            msg = f"sample shape {x.shape} does not match filter shape {self._state1.shape}"
            raise ValueError(msg)

        # Only the order of samples is checked, because the coefficients assume a fixed rate.
        elapsed(self._timestamp, timestamp)
        self._timestamp = timestamp
        y = c.b0 * x + self._state1
        self._state1 = c.b1 * x - c.a1 * y + self._state2
        self._state2 = c.b2 * x - c.a2 * y
        return np.asarray(y)

    def reset(self) -> None:
        """Forget all history; the next sample starts the filter in its steady state."""
        self._state1 = None
        self._state2 = None
        self._timestamp = -math.inf
