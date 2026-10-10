"""Wavenumber-band velocity errors for residual IFWI.

The error field is expanded with an orthonormal 2D DCT-II (even extension, no
periodic edge leakage). DCT index m on an axis of n cells with spacing d has
wavenumber m / (2 n d); radial k = sqrt(kx^2 + kz^2) in cycles/km. By Parseval
the squared band RMSEs sum exactly to the squared total RMSE.

The fixed background already contains the truth's long wavelengths, so each
band is also reported relative to the background error in that band
(< 1 means that band improved).
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn

# cycles/km. sigma=15 cells (225 m) Gaussian background keeps ~k < 0.8 c/km;
# 15 m grid Nyquist is 33.3 c/km.
DEFAULT_EDGES_CPKM = (0., 1., 3., 8., 16., np.inf)


def radial_wavenumber_cpkm(shape, spacing_m):
    nz, nx = shape
    kz = np.arange(nz) / (2 * nz * spacing_m / 1000)
    kx = np.arange(nx) / (2 * nx * spacing_m / 1000)
    return np.hypot(kz[:, None], kx[None, :])


def band_label(low, high):
    return f"{low:g}-{high:g}cpkm" if np.isfinite(high) else f">{low:g}cpkm"


def band_rmse(field_mps, spacing_m, edges=DEFAULT_EDGES_CPKM):
    """RMSE (m/s) of a [nz,nx] field restricted to each radial wavenumber band."""
    field = np.asarray(field_mps, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("Expected a [nz,nx] field")
    if any(b <= a for a, b in zip(edges[:-1], edges[1:])) or edges[0] != 0:
        raise ValueError("Band edges must start at 0 and increase")
    power = np.square(dctn(field, type=2, norm="ortho"))
    k = radial_wavenumber_cpkm(field.shape, spacing_m)
    result = {}
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (k >= low) & (k < high)
        result[band_label(low, high)] = float(np.sqrt(power[mask].sum() / field.size))
    return result


def band_errors(prediction_mps, truth_mps, background_mps, spacing_m, edges=DEFAULT_EDGES_CPKM):
    """Per-band RMSE of prediction and background against truth, and their ratio."""
    prediction, truth, background = (np.asarray(a, dtype=np.float64).squeeze() for a in
                                     (prediction_mps, truth_mps, background_mps))
    if not prediction.shape == truth.shape == background.shape:
        raise ValueError("prediction, truth and background must share one [nz,nx] shape")
    predicted = band_rmse(prediction - truth, spacing_m, edges)
    reference = band_rmse(background - truth, spacing_m, edges)
    return {"edges_cpkm": [float(e) if np.isfinite(e) else None for e in edges],
            "rmse_mps": predicted, "background_rmse_mps": reference,
            "ratio_to_background": {key: (predicted[key] / reference[key] if reference[key] > 0 else None)
                                    for key in predicted}}
