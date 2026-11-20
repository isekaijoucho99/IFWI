from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.direction_a.optimizer import UpdateRule


GRADS = torch.tensor([[3., -2., 0.], [-1., 4., 2.], [0., 0., 0.], [2., -1., 3.]],
                     dtype=torch.float64)


@pytest.mark.parametrize("method", ["sgd", "adam_no_momentum", "adam"])
@pytest.mark.parametrize("clip_norm", [None, 1.5])
def test_multistep_deltas_match_torch(method, clip_norm):
    lr, beta1, beta2, eps = 0.03, 0.76, 0.93, 1e-3
    rule = UpdateRule(method, lr, beta1=beta1, beta2=beta2, eps=eps, clip_norm=clip_norm)
    parameter = torch.nn.Parameter(torch.zeros_like(GRADS[0]))
    if method == "sgd":
        reference = torch.optim.SGD([parameter], lr)
    else:
        reference = torch.optim.Adam([parameter], lr, betas=(beta1 if method == "adam" else 0., beta2),
                                     eps=eps)

    for grad in GRADS:
        parameter.grad = grad.clone()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([parameter], clip_norm)
        before = parameter.detach().clone()
        reference.step()
        proposal = rule.propose(grad)
        torch.testing.assert_close(proposal.delta, parameter.detach() - before, atol=1e-15, rtol=1e-13)
        torch.testing.assert_close(proposal.delta, proposal.current + proposal.history)
        assert proposal.raw_norm == pytest.approx(torch.linalg.vector_norm(grad).item())
        torch.testing.assert_close(grad * proposal.clip_factor, parameter.grad)
        if method != "adam":
            torch.testing.assert_close(proposal.current, proposal.instant)
            assert torch.count_nonzero(proposal.history) == 0
        rule.accept(proposal)


def test_components_use_current_denominator_and_separate_history():
    rule = UpdateRule("adam", 0.1, beta1=0.5, beta2=0.8, eps=0.03)
    g1, g2 = GRADS[:2]
    rule.accept(rule.propose(g1))
    proposal = rule.propose(g2)
    v2 = (0.8 * g1.square() + g2.square()) / 1.8
    operator = -0.1 / (v2.sqrt() + 0.03)
    torch.testing.assert_close(proposal.instant, operator * g2)
    torch.testing.assert_close(proposal.current, operator * g2 / 1.5)
    torch.testing.assert_close(proposal.history, operator * g1 * 0.5 / 1.5)


def test_restart_clocks_and_discarded_proposals():
    rule = UpdateRule("adam", 0.1, beta1=0.5, beta2=0.8, eps=0.03)
    g1, g2, _, g3 = GRADS
    rule.accept(rule.propose(g1))
    ordinary = rule.propose(g2)
    rule.propose(10 * g1, restart=True)
    torch.testing.assert_close(rule.propose(g2).delta, ordinary.delta)
    restarted = rule.propose(g2, restart=True)
    v2 = (0.8 * g1.square() + g2.square()) / 1.8
    torch.testing.assert_close(restarted.delta, -0.1 * g2 / (v2.sqrt() + 0.03))
    torch.testing.assert_close(restarted.current, restarted.instant)
    assert torch.count_nonzero(restarted.history) == 0
    rule.accept(restarted)
    next_step = rule.propose(g3)
    v3 = (0.8**2 * g1.square() + 0.8 * g2.square() + g3.square()) / (1 + 0.8 + 0.8**2)
    operator = -0.1 / (v3.sqrt() + 0.03)
    torch.testing.assert_close(next_step.current, operator * g3 / 1.5)
    torch.testing.assert_close(next_step.history, operator * g2 * 0.5 / 1.5)


@pytest.mark.parametrize("method", ["sgd", "adam_no_momentum"])
def test_restart_is_identity_without_momentum(method):
    rule = UpdateRule(method, 0.1)
    rule.accept(rule.propose(GRADS[0]))
    normal = rule.propose(GRADS[1])
    restarted = rule.propose(GRADS[1], restart=True)
    torch.testing.assert_close(normal.delta, restarted.delta)


def test_stale_and_foreign_proposals_cannot_overwrite_accepted_moments():
    rule = UpdateRule("adam", 0.1)
    other = UpdateRule("adam", 0.1)
    proposal = rule.propose(GRADS[0])
    with pytest.raises(ValueError):
        other.accept(proposal)
    alternative = rule.propose(GRADS[1])
    rule.accept(proposal)
    with pytest.raises(ValueError):
        rule.accept(alternative)


@pytest.mark.parametrize("kwargs", [dict(lr=0), dict(lr=float("nan")), dict(beta1=1),
                                    dict(beta2=-0.1), dict(eps=0), dict(clip_norm=0)])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        UpdateRule("adam", **({"lr": 0.1} | kwargs))
