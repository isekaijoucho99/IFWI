import json

import numpy as np

from experiments.direction_a import runner


def test_sub_float32_trial_cannot_claim_strict_misfit_descent(tmp_path):
    out = runner.main(["--datasets", "marmousi", "--methods", "adam_guarded", "--lr", "1e-14",
                       "--seed", "12", "--steps", "1", "--lr-modes", "native", "--scales", "1",
                       "--representations", "fr", "--output-dir", str(tmp_path)])
    run = json.loads((out / "report.json").read_text())["datasets"][0]["runs"][0]
    assert run["completed_steps"] == 0
    assert all(t.get("trial_loss", run["initial_misfit"]) == run["initial_misfit"]
               for t in run["steps"][0]["trials"])


def test_real_survey_starts_match_and_costs_include_extra_forwards(tmp_path):
    out = runner.main(["--output-dir", str(tmp_path), "--datasets", "marmousi",
                       "--shape", "8", "12", "--shots", "1", "--nt", "85", "--frequency", "20",
                       "--steps", "3", "--warmup-steps", "1", "--width", "6",
                       "--methods", "sgd", "adam_no_momentum", "adam", "adam_guarded",
                       "--representations", "siren", "fr", "--scales", "1", "4",
                       "--lr-modes", "matched", "--first-rms", "1", "--save-steps", "2"])
    result = json.loads((out / "report.json").read_text())["datasets"][0]
    assert len(result["runs"]) == 16
    assert result["setup"]["forward_shots"] == 3
    assert result["setup"]["adjoint_shots"] == 1
    for run in result["runs"]:
        assert run["start_max_diff_mps"] < 1e-9
        assert run["completed_steps"] >= 2
        first, second = run["steps"][:2]
        assert first["misfit_before"] == result["setup"]["initial_misfit"]
        assert second["misfit_before"] == first["misfit_after"]
        assert abs(first["diagnostics"]["linear_rms_mps"] - 1.) < 1e-8
        assert second["diagnostics"]["decomposition_error_rms_mps"] < 1e-10
        assert second["adjoint_shots"] == 2
        assert second["total_forward_shots"] >= result["setup"]["forward_shots"] + 4
        if run["method"] == "adam_guarded":
            assert second["extra_forward_shots"] >= 1
            assert second["report_forward_shots"] == 0
            assert second["misfit_after"] < second["misfit_before"]
        else:
            assert second["extra_forward_shots"] == 0
            assert second["report_forward_shots"] == 2
        with np.load(out / second["fields_path"]) as maps:
            np.testing.assert_allclose(maps["actual"], maps["after"] - maps["before"])
            np.testing.assert_allclose(maps["linear"], maps["history"] + maps["current"], atol=1e-9)
        if run["method"] == "adam_no_momentum":
            assert second["diagnostics"]["history_rms_mps"] == 0
        if run["method"] == "adam":
            assert second["diagnostics"]["history_rms_mps"] > 0
