"""Load the stage-1 grid and reproducibly generate its smooth background.

Velocity outputs use m/s and float32. ``sigma`` is a nonnegative scalar in
inversion-grid points; zero explicitly requests an identity background. The
configured spacing describes the inversion grid, not the unknown raw CSV
spacing. Neither source CSV nor the historical smooth400 model is modified.
"""

import hashlib
import io
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter


def _scalar(value, name, *, zero_allowed=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite scalar")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not zero_allowed):
        condition = "nonnegative" if zero_allowed else "positive"
        raise ValueError(f"{name} must be finite and {condition}")
    return value


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _read_velocity(path, units, header):
    payload = path.read_bytes()
    try:
        table = pd.read_csv(io.BytesIO(payload), header=None)
        raw_shape = list(table.shape)
        first_row_numeric = True
        try:
            np.asarray(table.iloc[0], dtype=np.float64)
        except (ValueError, TypeError):
            first_row_numeric = False
        # For these CSVs legacy pandas header=0 consumes a numeric model row.
        # Retaining the unconsumed table also lets provenance report that loss.
        values = table.iloc[1:] if header == "legacy" else table
        values = values.to_numpy(dtype=np.float64)
    except (ValueError, TypeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError(f"velocity CSV must contain a rectangular numeric grid: {path}") from error
    if values.ndim != 2 or not all(values.shape):
        raise ValueError("velocity data must be a nonempty 2D grid")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("velocity values must be finite and positive")
    # Broad seismic bounds catch the common factor-of-1000 declaration error.
    # They do not infer units or silently repair a supplied declaration.
    lower, upper = (100., 15000.) if units == "m/s" else (.1, 15.)
    if values.min() < lower or values.max() > upper:
        raise ValueError(f"velocity values are implausible for declared model_units={units}")
    if units == "km/s":
        values = values * 1000.
    source = {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
              "raw_shape": raw_shape, "loaded_shape": list(values.shape),
              "model_units": units, "csv_header": header,
              "consumed_numeric_first_row": header == "legacy" and first_row_numeric}
    return values, source


def _sample(values, stride, crop):
    sampled = values[::stride, ::stride]
    sampled_shape = list(sampled.shape)
    if crop is not None:
        if not isinstance(crop, (list, tuple)) or len(crop) != 2:
            raise ValueError("crop_shape must be [nz, nx] or null")
        shape = [_positive_integer(item, "crop_shape") for item in crop]
        if any(requested > available for requested, available in zip(shape, sampled.shape)):
            raise ValueError("crop_shape must fit inside the sampled grid")
        sampled = sampled[:shape[0], :shape[1]]
    return np.ascontiguousarray(sampled, dtype=np.float32), sampled_shape


def _array_record(values):
    # Hashes describe row-major little-endian float32 m/s array bytes.
    serialized = np.asarray(values, dtype="<f4", order="C").tobytes(order="C")
    return {"units": "m/s", "dtype": "float32", "shape": list(values.shape),
            "sha256": hashlib.sha256(serialized).hexdigest(),
            "sha256_encoding": "C-order little-endian float32 array bytes",
            "min_mps": float(values.min()), "max_mps": float(values.max())}


def load_velocity_data(config: dict, repo_root: Path):
    """Return ``(truth_mps, init_mps, provenance)`` without generating shots.

    The background is generated after both sampling and any top-left crop.
    ``smooth400`` is audited only; its undocumented recipe never defines the
    new background. The provenance dictionary contains only JSON-safe values.
    """
    data = config["data"]
    initialization = config["initialization"]
    units = data.get("model_units", "m/s")
    if units not in ("m/s", "km/s"):
        raise ValueError("model_units must be 'm/s' or 'km/s'")
    header = data.get("csv_header", "legacy")
    if header not in ("legacy", "none"):
        raise ValueError("csv_header must be 'legacy' or 'none'")
    stride = _positive_integer(data.get("downsample", 4), "downsample")
    spacing = _scalar(data.get("grid_spacing_m", 15.), "grid_spacing_m")
    sigma = _scalar(initialization.get("sigma", 15.), "sigma", zero_allowed=True)
    truncate = _scalar(initialization.get("gaussian_truncate", 4.), "gaussian_truncate")
    mode = initialization.get("gaussian_mode", "reflect")
    if mode not in ("reflect", "constant", "nearest", "mirror", "wrap"):
        raise ValueError("gaussian_mode must be reflect, constant, nearest, mirror, or wrap")
    crop = data.get("crop_shape")
    source_path = Path(repo_root) / "data" / data.get("model_file", "vel_marmousi_376x1151.csv")
    raw_mps, source = _read_velocity(source_path, units, header)
    truth, sampled_shape = _sample(raw_mps, stride, crop)
    initial = gaussian_filter(truth, sigma=sigma, mode=mode, truncate=truncate).astype(np.float32)
    if not np.isfinite(initial).all() or np.any(initial <= 0):
        raise ValueError("generated background velocity must be finite and positive")
    nz, nx = truth.shape
    # Stable, readable provenance schema used by preparation artifacts:
    # source: CSV identity, raw/loaded shapes, declared units and consumed row.
    # sampling: stride, pre-crop shape, top-left crop, final inversion shape.
    # grid: actual inversion spacing, node extents and plotting cell extents.
    # initialization: filter recipe/order and sigma in grid points and meters.
    # truth/initial: output units, dtype, range, shape and canonical byte hashes.
    # smooth400_audit: identity, unknown recipe, same-grid comparison or error.
    provenance = {
        "schema_version": 1,
        "source": source,
        "sampling": {"downsample": stride, "sampled_shape": sampled_shape,
                     "crop_shape": None if crop is None else [int(v) for v in crop],
                     "crop_origin": [0, 0], "output_shape": [nz, nx]},
        "grid": {"axis_order": ["z", "x"], "grid_spacing_m": spacing,
                 "raw_grid_spacing_m": None,
                 "node_extent_m": {"x": [0., (nx - 1) * spacing],
                                   "z": [0., (nz - 1) * spacing]},
                 "display_extent_m": [0., nx * spacing, nz * spacing, 0.]},
        "initialization": {"method": "scipy.ndimage.gaussian_filter",
                           "filter_order": ["read_csv", "convert_units_to_m/s", "downsample",
                                            "crop", "float32", "gaussian_filter"],
                           "sigma_gridpoints": sigma, "sigma_physical_m": sigma * spacing,
                           "gaussian_mode": mode, "gaussian_truncate": truncate,
                           "scipy_version": scipy.__version__},
        "truth": _array_record(truth), "initial": _array_record(initial),
    }
    smooth_path = Path(repo_root) / "data" / "vel_marmousi_smooth400_376x1151.csv"
    audit = {"available": smooth_path.is_file(), "path": str(smooth_path.resolve()),
             "generation_recipe": "unknown", "comparable_grid": False,
             "rmse_difference_mps": None, "max_abs_difference_mps": None}
    if audit["available"]:
        try:
            audit["sha256"] = hashlib.sha256(smooth_path.read_bytes()).hexdigest()
            smooth_raw, smooth_source = _read_velocity(smooth_path, units, header)
            audit.update(smooth_source)
            smooth, _ = _sample(smooth_raw, stride, crop)
            audit["output_shape"] = list(smooth.shape)
            audit["comparable_grid"] = smooth_raw.shape == raw_mps.shape and smooth.shape == truth.shape
            if audit["comparable_grid"]:
                difference = smooth.astype(np.float64) - initial.astype(np.float64)
                audit["comparison_target"] = "generated_background_mps"
                audit["rmse_difference_mps"] = float(np.sqrt(np.mean(difference ** 2)))
                audit["max_abs_difference_mps"] = float(np.abs(difference).max())
        except (ValueError, OSError) as error:
            audit["error"] = str(error)
    provenance["smooth400_audit"] = audit
    return truth, initial, provenance
