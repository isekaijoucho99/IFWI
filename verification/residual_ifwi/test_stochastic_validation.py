"""Reject invalid sampling budgets without changing the legacy config contract."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.residual_ifwi_experiment import default_config, validate_config


def test_random_shots_are_an_optional_valid_configuration():
    config = default_config()
    config["training"].update(shots_per_update=8, full_eval_interval=50)
    validate_config(config)
    legacy = default_config()
    validate_config(legacy)
    assert "shots_per_update" not in legacy["training"]


@pytest.mark.parametrize("shots", [0, -1, 14, True, 1.5])
def test_invalid_sampling_budget_is_rejected(shots):
    config = default_config()
    config["training"]["shots_per_update"] = shots
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_invalid_full_evaluation_interval_is_rejected(interval):
    config = default_config()
    config["training"].update(shots_per_update=8, full_eval_interval=interval)
    with pytest.raises(ValueError):
        validate_config(config)
