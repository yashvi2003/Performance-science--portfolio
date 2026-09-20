"""Zero-phase Butterworth low-pass digital filter for force plate telemetry.

Governed by specs/01_neuromuscular_pipeline.md and PLAN.md.
"""
from __future__ import annotations

import logging
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def butterworth_lowpass_filter(
    data: np.ndarray,
    fs_hz: float,
    cutoff_hz: float = 30.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase low-pass Butterworth filter to force-time data.

    Formula
    -------
    The continuous-time low-pass Butterworth filter transfer function is:
        |H(j*omega)|^2 = 1 / (1 + (omega / omega_c)^(2*N))

    Discrete-time implementation uses the bilinear transform to compute second-order sections (SOS).
    Zero phase distortion is guaranteed by forward-backward filtering (`scipy.signal.filtfilt`),
    which doubles the effective filter order (e.g., an order=4 specification executes two 2nd-order passes).

    Parameters
    ----------
    data : np.ndarray
        Raw 1D vertical force time series.
    fs_hz : float
        Sampling frequency in Hertz.
    cutoff_hz : float
        Low-pass cutoff frequency (-3dB point) in Hertz. Default is 30.0 Hz.
    order : int
        Filter order before bidirectional pass. Default is 4.

    Returns
    -------
    np.ndarray
        Zero-phase low-pass filtered vertical force trace.

    Raises
    ------
    ValueError
        If cutoff_hz >= fs_hz / 2 (violates Nyquist criterion) or fs_hz <= 0.
    """
    if fs_hz <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs_hz} Hz")

    nyquist_hz = 0.5 * fs_hz
    if cutoff_hz >= nyquist_hz:
        raise ValueError(
            f"Cutoff frequency ({cutoff_hz} Hz) must be strictly below Nyquist frequency ({nyquist_hz} Hz for fs={fs_hz} Hz)"
        )

    if len(data) == 0:
        return np.array([], dtype=float)

    # Normalized cutoff Wn in [0, 1] relative to Nyquist
    wn = cutoff_hz / nyquist_hz

    # Design filter using second-order sections (SOS) for numerical stability
    # filtfilt with SOS is numerically superior to (b, a) polynomials for higher orders
    sos = signal.butter(order, wn, btype="low", output="sos")

    # Determine pad length (cannot exceed data length - 1)
    padlen = min(3 * order, len(data) - 1)
    if padlen <= 0:
        return data.copy()

    filtered = signal.sosfiltfilt(sos, data, padlen=padlen)
    return filtered
