"""Write one CSV row per optimizer trajectory in a Direction A report."""

import csv
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile


_ARM_FIELDS = ("representation", "method", "scale", "lr_mode", "effective_lr", "completed_steps")
_COST_KEYS = {
    "online_forward_shots": "forward_shots",
    "online_adjoint_shots": "adjoint_shots",
    "online_seconds": "online_seconds",
    "total_forward_shots": "total_forward_shots",
    "total_adjoint_shots": "total_adjoint_shots",
    "total_seconds": "total_seconds",
}
_TARGETS = ("first_misfit_target", "first_rmse_target")
_FIELDNAMES = (
    "dataset", *_ARM_FIELDS, "misfit_increases", "restart_count", "accepted_after_first",
    "comparable_after_first", "linear_better_than_instant_after_first",
    "median_linear_relative_error_after_first", "median_instant_relative_error_after_first",
    "final_misfit", "final_rmse_mps", "stop_reason",
    *(f"{target}_{field}" for target in _TARGETS for field in ("step", *_COST_KEYS)),
    *(f"final_{field}" for field in _COST_KEYS),
)


def _summary_row(dataset, run):
    accepted = [s for s in run["steps"] if s["status"] == "accepted"]
    pairs = []
    for step in accepted[1:]:
        diagnostics = step["diagnostics"]
        linear = diagnostics["linear_relative_error"]
        instant = diagnostics["instant_relative_error"]
        if linear is not None and instant is not None:
            pairs.append((linear, instant))

    row = {key: run[key] for key in _ARM_FIELDS}
    row.update(
        dataset=dataset,
        misfit_increases=sum(s["misfit_increased"] for s in accepted),
        restart_count=sum(s["restarted"] for s in run["steps"]),
        accepted_after_first=len(accepted[1:]),
        comparable_after_first=len(pairs),
        linear_better_than_instant_after_first=sum(linear < instant for linear, instant in pairs),
        median_linear_relative_error_after_first=median(p[0] for p in pairs) if pairs else None,
        median_instant_relative_error_after_first=median(p[1] for p in pairs) if pairs else None,
        final_misfit=accepted[-1]["misfit_after"] if accepted else run["initial_misfit"],
        final_rmse_mps=accepted[-1]["rmse_mps"] if accepted else run["initial_rmse_mps"],
        stop_reason=run["stop_reason"],
    )
    for target in _TARGETS:
        hit = run[target]
        if hit is not None:
            row[f"{target}_step"] = hit["step"]
            row.update({f"{target}_{name}": hit[key] for name, key in _COST_KEYS.items()})
    row.update({f"final_{name}": run["costs"][key] for name, key in _COST_KEYS.items()})
    return row


def write_summary(report, path):
    """Atomically replace the CSV at path. Missing targets and undefined medians are blank.

    Both medians and the strict win count use the same accepted updates after
    the first with defined full-JVP and instantaneous-JVP relative errors.
    Restarts include rejected attempts. Final costs include the last attempt.
    """
    path = Path(path)
    stream = NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                prefix=f".{path.name}.", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            writer = csv.DictWriter(stream, fieldnames=_FIELDNAMES)
            writer.writeheader()
            for dataset in report["datasets"]:
                for run in dataset["runs"]:
                    writer.writerow(_summary_row(dataset["dataset"], run))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
