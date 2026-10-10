"""CPU data-contract checks using independent numeric fixtures."""

import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def configuration(**data_overrides):
    data = dict(model_file="velocity.csv", model_units="m/s", csv_header="none",
                downsample=4, crop_shape=None, grid_spacing_m=15.)
    data.update(data_overrides)
    return {"data": data, "initialization": dict(
        sigma=15., gaussian_mode="reflect", gaussian_truncate=4.)}


def write_velocity(tmp_path, values, filename="velocity.csv"):
    folder = tmp_path / "data"
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / filename
    np.savetxt(source, values, delimiter=",")
    return source


def load(config, root):
    # Import during the test so an absent feature gives an ordinary test failure.
    module = importlib.import_module("experiments.residual_ifwi_data")
    return module.load_velocity_data(config, root)


def test_sigma15_smooths_sampled_grid_and_preserves_source(tmp_path):
    z, x = np.indices((161, 181))
    raw = 1800. + 7 * z + 3 * x + 1200. * (x > 73) + 500. * (z > 91)
    source = write_velocity(tmp_path, raw)
    before = source.read_bytes()
    truth, initial, provenance = load(configuration(), tmp_path)
    sampled = raw[::4, ::4].astype(np.float32)
    expected = gaussian_filter(sampled, sigma=15., mode="reflect", truncate=4.)
    wrong_order = gaussian_filter(raw, sigma=15.)[::4, ::4].astype(np.float32)
    np.testing.assert_array_equal(truth, sampled)
    np.testing.assert_allclose(initial, expected, rtol=0., atol=1e-4)
    assert np.sqrt(np.mean((initial - wrong_order) ** 2)) > 100.
    assert source.read_bytes() == before
    assert truth.dtype == initial.dtype == np.float32
    assert provenance["source"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert provenance["initialization"]["sigma_physical_m"] == 225.


def test_legacy_header_consumes_numeric_first_row(tmp_path):
    raw = np.arange(35, dtype=float).reshape(5, 7) * 10 + 2000
    write_velocity(tmp_path, raw)
    cfg = configuration(csv_header="legacy", downsample=2)
    cfg["initialization"]["sigma"] = 0.
    truth, initial, provenance = load(cfg, tmp_path)
    np.testing.assert_array_equal(truth, [[2070., 2090., 2110., 2130.],
                                         [2210., 2230., 2250., 2270.]])
    np.testing.assert_array_equal(initial, truth)
    assert provenance["source"]["raw_shape"] == [5, 7]
    assert provenance["source"]["consumed_numeric_first_row"] is True
    cfg["data"]["csv_header"] = "none"
    retained, _, report = load(cfg, tmp_path)
    assert retained.shape == (3, 4)
    assert retained[0, 0] == 2000.
    assert report["source"]["consumed_numeric_first_row"] is False


def test_declared_km_s_converts_to_same_m_s_background(tmp_path):
    raw = 2200. + np.arange(255).reshape(15, 17) * 5.
    m_root, km_root = tmp_path / "m", tmp_path / "km"
    write_velocity(m_root, raw)
    write_velocity(km_root, raw / 1000.)
    m_truth, m_initial, _ = load(configuration(), m_root)
    km_truth, km_initial, provenance = load(configuration(model_units="km/s"), km_root)
    np.testing.assert_array_equal(km_truth, m_truth)
    np.testing.assert_array_equal(km_initial, m_initial)
    assert provenance["truth"]["units"] == "m/s"


def test_crop_precedes_smoothing_and_records_physical_extents(tmp_path):
    raw = np.arange(99).reshape(9, 11) * 15. + 1800.
    write_velocity(tmp_path, raw)
    cfg = configuration(downsample=2, crop_shape=[3, 4])
    cfg["initialization"]["sigma"] = 1.
    truth, initial, provenance = load(cfg, tmp_path)
    expected_truth = raw[::2, ::2][:3, :4].astype(np.float32)
    np.testing.assert_array_equal(truth, expected_truth)
    np.testing.assert_allclose(initial, gaussian_filter(expected_truth, 1.))
    assert provenance["sampling"]["sampled_shape"] == [5, 6]
    assert provenance["sampling"]["output_shape"] == [3, 4]
    assert provenance["grid"]["node_extent_m"] == {"x": [0., 45.], "z": [0., 30.]}
    assert provenance["grid"]["display_extent_m"] == [0., 60., 45., 0.]


@pytest.mark.parametrize("sigma", [-1., float("nan"), float("inf"), True, "invalid"])
def test_rejects_invalid_sigma(tmp_path, sigma):
    write_velocity(tmp_path, np.full((9, 9), 2000.))
    cfg = configuration()
    cfg["initialization"]["sigma"] = sigma
    with pytest.raises(ValueError, match="sigma"):
        load(cfg, tmp_path)


@pytest.mark.parametrize("stride", [0, -1, 1.5, True])
def test_rejects_invalid_downsample(tmp_path, stride):
    write_velocity(tmp_path, np.full((9, 9), 2000.))
    with pytest.raises(ValueError, match="downsample"):
        load(configuration(downsample=stride), tmp_path)


@pytest.mark.parametrize("spacing", [0., -1., float("nan"), float("inf")])
def test_rejects_invalid_grid_spacing(tmp_path, spacing):
    write_velocity(tmp_path, np.full((9, 9), 2000.))
    with pytest.raises(ValueError, match="grid_spacing_m"):
        load(configuration(grid_spacing_m=spacing), tmp_path)


@pytest.mark.parametrize("crop", [[0, 2], [4, 2], [2], [2.5, 2], [True, 2]])
def test_rejects_crop_that_cannot_fit_sampled_grid(tmp_path, crop):
    write_velocity(tmp_path, np.full((9, 9), 2000.))
    with pytest.raises(ValueError, match="crop_shape"):
        load(configuration(crop_shape=crop), tmp_path)


@pytest.mark.parametrize("units,value", [("m/s", 2.), ("km/s", 2000.), ("feet/s", 2000.)])
def test_rejects_declared_unit_mismatch(tmp_path, units, value):
    write_velocity(tmp_path, np.full((9, 9), value))
    with pytest.raises(ValueError, match="units"):
        load(configuration(model_units=units), tmp_path)


@pytest.mark.parametrize("bad", [0., -2000., float("nan"), float("inf")])
def test_rejects_invalid_velocity_even_outside_sampled_points(tmp_path, bad):
    raw = np.full((9, 9), 2000.)
    raw[1, 1] = bad
    write_velocity(tmp_path, raw)
    with pytest.raises(ValueError, match="velocity"):
        load(configuration(), tmp_path)


def test_smooth400_audit_records_unknown_recipe_and_observed_difference(tmp_path):
    raw = np.full((9, 9), 2000.)
    write_velocity(tmp_path, raw)
    smooth = write_velocity(tmp_path, raw + 300., "vel_marmousi_smooth400_376x1151.csv")
    before = smooth.read_bytes()
    _, _, provenance = load(configuration(), tmp_path)
    audit = provenance["smooth400_audit"]
    assert audit["available"] is True
    assert audit["generation_recipe"] == "unknown"
    assert audit["comparable_grid"] is True
    assert audit["rmse_difference_mps"] == pytest.approx(300.)
    assert audit["sha256"] == hashlib.sha256(before).hexdigest()
    assert smooth.read_bytes() == before


def test_smooth400_shape_mismatch_is_disclosed_without_replacing_background(tmp_path):
    write_velocity(tmp_path, np.full((9, 9), 2000.))
    write_velocity(tmp_path, np.full((17, 17), 2500.), "vel_marmousi_smooth400_376x1151.csv")
    _, initial, provenance = load(configuration(), tmp_path)
    assert np.all(initial == 2000.)
    assert provenance["smooth400_audit"]["comparable_grid"] is False
    assert provenance["smooth400_audit"]["rmse_difference_mps"] is None


def test_smooth400_unit_failure_still_records_identity(tmp_path):
    write_velocity(tmp_path, np.full((9, 9), 2000.))
    smooth = write_velocity(tmp_path, np.full((9, 9), 2.),
                            "vel_marmousi_smooth400_376x1151.csv")
    _, initial, report = load(configuration(), tmp_path)
    audit = report["smooth400_audit"]
    assert np.all(initial == 2000.)
    assert audit["sha256"] == hashlib.sha256(smooth.read_bytes()).hexdigest()
    assert audit["comparable_grid"] is False
    assert "units" in audit["error"]


def test_standard_constant_mode_matches_explicit_scipy_recipe(tmp_path):
    write_velocity(tmp_path, np.full((21, 25), 2000.))
    cfg = configuration()
    cfg["initialization"].update(sigma=1., gaussian_mode="constant")
    truth, initial, report = load(cfg, tmp_path)
    np.testing.assert_array_equal(initial, gaussian_filter(truth, 1., mode="constant"))
    assert np.all(initial > 0)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("truncate", [0., -1., float("nan"), True])
def test_rejects_invalid_gaussian_truncation(tmp_path, truncate):
    write_velocity(tmp_path, np.full((9, 9), 2000.))
    cfg = configuration()
    cfg["initialization"]["gaussian_truncate"] = truncate
    with pytest.raises(ValueError, match="gaussian_truncate"):
        load(cfg, tmp_path)


@pytest.mark.parametrize("text", ["", "not,velocity\n2,3\n", "2000,2100\n2000,2100,2200\n"])
def test_rejects_empty_nonnumeric_or_ragged_csv(tmp_path, text):
    folder = tmp_path / "data"
    folder.mkdir(parents=True)
    (folder / "velocity.csv").write_text(text)
    with pytest.raises(ValueError, match="velocity"):
        load(configuration(), tmp_path)
