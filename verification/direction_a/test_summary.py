import csv

import pytest

from experiments.direction_a.report import write_summary


def _costs(n):
    return dict(forward_shots=2 * n, adjoint_shots=n, online_seconds=0.5 * n,
                total_forward_shots=3 * n + 1, total_adjoint_shots=n + 1,
                total_seconds=0.75 * n + 1)


def _accepted(i, linear, instant):
    return dict(step=i, status="accepted", restarted=i == 2, misfit_increased=i == 3,
                misfit_after=10 - 0.5 * i, rmse_mps=100 - i,
                diagnostics=dict(linear_relative_error=linear, instant_relative_error=instant))


def _arm(steps):
    return dict(representation="fr", method="adam_guarded", scale=4, lr_mode="matched",
                effective_lr=0.02, initial_misfit=10, initial_rmse_mps=100, steps=steps,
                completed_steps=sum(s["status"] == "accepted" for s in steps),
                first_misfit_target=None, first_rmse_target=None, costs=_costs(len(steps)),
                stop_reason="no_linear_descent")


def test_summary_uses_paired_postfirst_errors_and_cost_at_each_target(tmp_path):
    rejected = dict(step=6, status="no_linear_descent", restarted=True)
    run = _arm([_accepted(1, 99, 0), _accepted(2, 0.1, 0.3), _accepted(3, 0.2, 0.2),
                _accepted(4, 0.9, 0.1), _accepted(5, None, None), rejected])
    run["first_misfit_target"] = dict(step=2, **_costs(2))
    run["first_rmse_target"] = dict(step=4, **_costs(4))
    empty = _arm([dict(step=1, status="no_linear_descent", restarted=False)])
    report = {"datasets": [{"dataset": "marmousi", "runs": [run]},
                           {"dataset": "overthrust", "runs": [empty]}]}
    path = tmp_path / "summary.csv"
    write_summary(report, path)
    with path.open(newline="", encoding="utf-8") as stream:
        row, rejected_row = csv.DictReader(stream)

    assert [row[k] for k in ("dataset", "representation", "method", "scale", "lr_mode",
                             "effective_lr", "completed_steps")] == [
        "marmousi", "fr", "adam_guarded", "4", "matched", "0.02", "5"]
    assert [row[k] for k in ("misfit_increases", "restart_count", "accepted_after_first",
                             "comparable_after_first", "linear_better_than_instant_after_first",
                             "median_linear_relative_error_after_first",
                             "median_instant_relative_error_after_first")] == [
        "1", "2", "4", "3", "1", "0.2", "0.2"]
    for metric, n in (("misfit", 2), ("rmse", 4)):
        prefix = "first_" + metric + "_target_"
        assert row[prefix + "step"] == str(n)
        for name in ("forward_shots", "adjoint_shots", "seconds"):
            online_key = "online_seconds" if name == "seconds" else name
            assert row[prefix + "online_" + name] == str(_costs(n)[online_key])
            assert row[prefix + "total_" + name] == str(_costs(n)["total_" + name])
    assert (row["final_misfit"], row["final_rmse_mps"], row["stop_reason"]) == (
        "7.5", "95", "no_linear_descent")
    assert row["final_online_forward_shots"] == str(_costs(6)["forward_shots"])
    assert row["final_total_seconds"] == str(_costs(6)["total_seconds"])
    assert (rejected_row["dataset"], rejected_row["completed_steps"], rejected_row["final_misfit"],
            rejected_row["final_rmse_mps"]) == ("overthrust", "0", "10", "100")
    assert rejected_row["median_linear_relative_error_after_first"] == ""
    assert rejected_row["comparable_after_first"] == "0"
    assert all(value == "" for name, value in rejected_row.items() if name.startswith("first_"))


def test_incomplete_report_leaves_existing_summary_intact(tmp_path):
    path = tmp_path / "summary.csv"
    path.write_text("previous summary\n", encoding="utf-8")
    with pytest.raises(KeyError):
        write_summary({"datasets": [{"dataset": "marmousi", "runs": [_arm([]), {}]}]}, path)
    assert path.read_text(encoding="utf-8") == "previous summary\n"
    assert list(tmp_path.iterdir()) == [path]
