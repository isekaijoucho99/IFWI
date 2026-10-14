"""Muon on hidden INR matrices with Adam elsewhere; rank diagnostics; tiny CPU FD runs."""
import copy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

SMALL_FR = {"high_freq_num": 8, "low_freq_num": 8, "phi_num": 8, "alpha": .01, "lambda_init": "fr_inr"}
MUON = {"name": "muon", "learning_rate": 1e-3, "momentum": .95, "nesterov": True, "ns_steps": 5}


def mod():
    return importlib.import_module("experiments.muon_optimizer")


def api():
    return importlib.import_module("experiments.residual_ifwi_experiment")


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (48, 48)])
def test_newton_schulz_flattens_singular_values_and_is_scale_invariant(shape):
    torch.manual_seed(0)
    k = min(shape)
    left, _ = torch.linalg.qr(torch.randn(shape[0], k))
    right, _ = torch.linalg.qr(torch.randn(shape[1], k))
    G = left @ torch.diag(torch.linspace(1, 30, k)) @ right.T           # condition number 30
    X = mod().newton_schulz_orthogonalize(G).float()
    s = torch.linalg.svdvals(X)
    assert s.min() > .4 and s.max() < 1.6               # reference: roughly U(0.5, 1.5), not exactly 1
    scaled = mod().newton_schulz_orthogonalize(7.5 * G).float()          # scale invariant up to bf16 rounding
    assert (scaled - X).norm() / X.norm() < .05
    zero = mod().newton_schulz_orthogonalize(torch.zeros(*shape))
    assert torch.isfinite(zero).all() and zero.abs().max() == 0


def test_muon_step_matches_the_reference_arithmetic_and_keeps_the_gradient():
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(16, 8))
    g = torch.randn(16, 8)
    p.grad = g.clone()
    start = p.detach().clone()
    opt = mod().Muon([p], lr=.1, momentum=.9)
    opt.step()
    buffer = .1 * g                                     # lerp(0, g, 1 - 0.9)
    blended = g + .9 * (buffer - g)                     # Nesterov: grad.lerp(buffer, 0.9)
    expected = start - .1 * mod().newton_schulz_orthogonalize(blended).float() * (16 / 8) ** .5
    assert torch.allclose(p.detach(), expected, atol=1e-6)
    assert torch.equal(p.grad, g)                       # not modified in place
    assert torch.allclose(opt.state[p]["momentum_buffer"], buffer)


def models():
    initial = np.full((5, 6), 3000., np.float32)
    plain = api().VelocityParameterization(initial, 15, [2, 8, 8, 8, 1])
    fr = api().VelocityParameterization(initial, 15, [2, 8, 8, 8, 1], fourier=SMALL_FR, trainable_omega=True)
    lr = api().VelocityParameterization(initial, 15, None, low_rank={
        "rank_z": 3, "rank_x": 4, "hidden": [8, 8], "coord_scale": .5, "output_gain": 1.}, fourier=SMALL_FR)
    return plain, fr, lr


def test_hidden_matrices_are_chosen_by_layer_position():
    plain, fr, lr = models()
    hidden = mod().hidden_matrices(plain)
    # [2,8,8,8,1]: layers 1 and 2 are hidden-to-hidden; layer 0 (2 inputs) and the 1 x 8 head stay in Adam.
    assert [id(p) for p in hidden] == [id(plain.net.linear[1].weight), id(plain.net.linear[2].weight)]
    fr_hidden = mod().hidden_matrices(fr)
    assert [id(p) for p in fr_hidden] == [id(fr.net.linear[i].layer.lamb) for i in (1, 2)]
    lr_hidden = mod().hidden_matrices(lr)
    assert [id(p) for p in lr_hidden] == [id(lr.low_rank.net_z.linear[1].lamb), id(lr.low_rank.net_x.linear[1].lamb)]
    assert all(id(lr.low_rank.core) != id(p) for p in lr_hidden)


def adam_config(model_settings=None):
    cfg = api().default_config()
    cfg["model"].update(neurons=[2, 8, 8, 8, 1], **(model_settings or {}))
    return cfg


def test_adam_path_is_the_original_construction_and_muon_covers_every_parameter_once():
    plain, fr, _ = models()
    cfg = adam_config({"fourier": SMALL_FR, "trainable_omega": {"learning_rate": 1e-2}})
    opt = mod().build_optimizer(fr, cfg)
    assert isinstance(opt, torch.optim.Adam)
    reference = torch.optim.Adam(importlib.import_module("experiments.trainable_omega").parameter_groups(fr, 1e-4, {"learning_rate": 1e-2}))
    assert [[id(p) for p in g["params"]] for g in opt.param_groups] == [[id(p) for p in g["params"]] for g in reference.param_groups]
    cfg["training"]["optimizer"] = dict(MUON)
    opt = mod().build_optimizer(fr, cfg)
    muon_ids = [id(p) for g in opt.muon.param_groups for p in g["params"]]
    adam_ids = [id(p) for g in opt.adam.param_groups for p in g["params"]]
    assert sorted(muon_ids + adam_ids) == sorted(id(p) for p in fr.parameters())
    assert not set(muon_ids) & set(adam_ids)
    assert [g["lr"] for g in opt.adam.param_groups] == [1e-4, 1e-2]       # network aux, omega
    assert all(g["betas"] == (.9, .999) and g["eps"] == 1e-8 for g in opt.adam.param_groups)


def test_optimizer_validation_and_resume_normalization():
    cfg = adam_config()
    cfg["training"]["optimizer"] = dict(MUON)
    api().validate_config(cfg)
    for bad in ({"name": "sgd"}, {"name": "adam", "learning_rate": 1e-3}, {**MUON, "learning_rate": 0},
                {**MUON, "momentum": 1.}, {**MUON, "ns_steps": 0}, {**MUON, "nesterov": 1}, {k: v for k, v in MUON.items() if k != "ns_steps"}):
        broken = copy.deepcopy(cfg)
        broken["training"]["optimizer"] = bad
        with pytest.raises(ValueError):
            api().validate_config(broken)
    shallow = copy.deepcopy(cfg)
    shallow["model"]["neurons"] = [2, 8, 1]
    with pytest.raises(ValueError, match="hidden-to-hidden"):
        api().validate_config(shallow)
    plain = api().default_config()
    explicit = copy.deepcopy(plain)
    explicit["training"]["optimizer"] = {"name": "adam"}
    assert api()._resume_configuration(plain) == api()._resume_configuration(explicit)
    assert api()._resume_configuration(plain) != api()._resume_configuration(cfg)


def tiny_config(tmp_path, model_settings=None, optimizer=MUON):
    truth = np.linspace(2300, 3200, 8, dtype=np.float32)[:, None] + np.zeros((8, 9), np.float32)
    truth[5:7, 3:6] = 3400.
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = adam_config(model_settings)
    cfg["acquisition"].update(num_shots=2, source_x_indices=[2, 6])
    cfg["forward"].update(dt_s=.001, nt=32, frequency_hz=60, npad=3)
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    cfg["initialization"]["sigma"] = 1
    cfg["training"].update(epochs=3, checkpoint_interval=1)
    if optimizer is not None:
        cfg["training"]["optimizer"] = dict(optimizer)
    return cfg


@pytest.mark.parametrize("model_settings", [None, {"fourier": SMALL_FR, "trainable_omega": {"learning_rate": 1e-2}}])
def test_muon_run_logs_ranks_and_resumes_exactly(tmp_path, model_settings):
    cfg = tiny_config(tmp_path, model_settings)
    full = api().run_experiment(cfg, tmp_path / "full", "cpu", save_figures=False)
    history = json.loads((full / "history.json").read_text())
    # Zero residual head: no hidden gradient on update 1, so its update has no stable rank yet.
    assert history[0]["hidden_update_srank"] == [None, None]
    assert all(r is not None and r > 1 for r in history[1]["hidden_update_srank"])
    assert len(history[0]["hidden_weight_srank"]) == 2
    saved = torch.load(full / "checkpoint.pt", weights_only=False)
    assert set(saved["optimizer"]) == {"muon", "adam"}
    partial = copy.deepcopy(cfg)
    partial["training"]["epochs"] = 1
    first = api().run_experiment(partial, tmp_path / "partial", "cpu", save_figures=False)
    resumed = api().run_experiment(cfg, tmp_path / "resumed", "cpu", resume=first / "checkpoint.pt", save_figures=False)
    assert np.array_equal(np.load(full / "final_velocity.npy"), np.load(resumed / "final_velocity.npy"))
    adam = copy.deepcopy(cfg)
    adam["training"].pop("optimizer")
    with pytest.raises(ValueError, match="configuration differs"):
        api().run_experiment(adam, tmp_path / "wrong", "cpu", resume=first / "checkpoint.pt", save_figures=False)


def test_rank_report_reads_checkpoints(tmp_path):
    cfg = tiny_config(tmp_path, {"fourier": SMALL_FR})
    run = api().run_experiment(cfg, tmp_path / "run", "cpu", save_figures=False)
    rows = importlib.import_module("experiments.rank_report").run_series([run])
    assert sorted({r["update"] for r in rows}) == [1, 2, 3]
    weights = [r for r in rows if r["kind"] == "weight" and r["update"] == 3]
    features = [r for r in rows if r["kind"] == "feature" and r["update"] == 3]
    assert [r["layer"] for r in weights] == ["net0.1", "net0.2"]
    assert [r["layer"] for r in features] == ["net0.1", "net0.2", "net0.3"]
    assert all(1 <= r["stable_rank"] <= 8 for r in weights + features)


def test_muon_configs_change_only_the_optimizer():
    yaml = pytest.importorskip("yaml")
    folder = ROOT / "experiments/configs"
    load = lambda n: yaml.safe_load((folder / f"{n}.yaml").read_text(encoding="utf-8"))
    for name, base, lr in (("paper13_ifwi_muon", "paper13_ifwi", 1e-3), ("paper13_fr_ifwi_muon", "paper13_fr_ifwi", 1e-2),
                           ("residual_ifwi_baseline1000_muon", "residual_ifwi_baseline1000", 1e-3)):
        cfg, ref = load(name), load(base)
        api().validate_config(cfg)
        assert cfg["training"].pop("optimizer") == {**MUON, "learning_rate": lr}
        cfg["experiment_name"] = ref["experiment_name"]
        assert cfg == ref
