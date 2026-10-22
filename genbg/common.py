"""Shared helpers: template spec, inversion grid and truth, candidate manifest."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from experiments.residual_ifwi_data import load_velocity_data

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = "manifest.json"


def load_template(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec["experiment"]["parameterization"] != "residual" or spec["control"]["kind"] != "zero_output":
        raise ValueError("P0 uses the residual E2 protocol (zero-output residual SIREN) as its template")
    return spec


def truth_and_default(spec):
    """Truth and the spec's own Gaussian background on the inversion grid (truth is for observations,
    the controlled P0 candidates and evaluation only)."""
    truth, background, provenance = load_velocity_data(copy.deepcopy(spec["experiment"]), ROOT)
    return truth, background, provenance


def read_manifest(directory):
    directory = Path(directory)
    entries = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    for entry in entries:
        entry["array"] = np.load(ROOT / entry["file"], allow_pickle=False)
    return entries


def twt_bottom_ms(velocity_mps, dz_m):
    """Vertical two-way traveltime from the surface to the bottom row, per column (ms)."""
    slowness = 1. / np.asarray(velocity_mps, dtype=np.float64)
    pairs = .5 * (slowness[1:] + slowness[:-1]) * dz_m
    return 2000. * pairs.sum(axis=0)
