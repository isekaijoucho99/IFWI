import csv
import gzip
import json

import numpy as np

from experiments.direction_c.__main__ import main


def test_public_entry_point_runs_all_tiers_and_calibration(tmp_path):
    run = main(["--output-dir", str(tmp_path), "--datasets", "overthrust",
                "--particles", "4", "--seismic-particles", "3", "--sigmas", ".5", ".2",
                "--budget", "8", "--sbc-replicates", "2", "--sbc-particles", "4",
                "--importance-per-component", "2"])
    with (run / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3*7
    for case in ("gaussian", "layered", "overthrust"):
        report = json.loads((run / case / "report.json").read_text())
        for label, method in report["methods"].items():
            ledger = method["ledger"]
            assert ledger["used"] <= ledger["budget"]
            if case != "overthrust":
                assert ledger["forward_solves"] == ledger["adjoint_solves"] == 0
                assert method["metrics"]["posterior_predictive"]["evaluation_forward_solves"] == 0
            with np.load(run / case / (label + ".npz")) as samples:
                assert samples["latent"].shape[0] == report["particles"]
                assert np.isfinite(samples["quantiles_05_50_95"]).all()
            with gzip.open(run / case / (label + "_stages.jsonl.gz"), "rt") as handle:
                stages = [json.loads(line) for line in handle]
            assert len(stages) == report["particles"]*2
            assert sum(stage["used"] for stage in stages) == ledger["used"]
        if case == "gaussian":
            moments = report["methods"]["refresh_wide"]["metrics"]["moment_laws"]
            assert "encoding_at_matched_schedule" in moments["discrepancies"]
        if case == "layered":
            assert np.isclose(sum(report["methods"]["fixed"]["metrics"]["modes"]["reference_mass"]), 1.)
        if case == "overthrust":
            work = report["pde_work_all_phases"]
            assert work["prediction"]["forward_rhs"] == report["reference"]["forward_solves"]
            assert work["likelihood"]["adjoint_rhs"] == sum(m["pde_work"]["adjoint_rhs"]
                                                            for m in report["methods"].values())
            assert len(report["reference"]["conditional_approximation"]) == 2
    calibration = json.loads((run / "calibration.json").read_text())
    assert set(calibration["conditions"]) == {"matched", "mismatched"}
    assert len(calibration["conditions"]["matched"]["methods"]["exact"]["ranks_by_dimension"][0]) == 2
