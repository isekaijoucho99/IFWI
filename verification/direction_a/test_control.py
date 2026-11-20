import torch
from torch import nn

from experiments.direction_a.control import choose_step
from experiments.direction_a.model import pullback, velocity
from experiments.direction_a.optimizer import UpdateRule


class Pixel(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor([[[3.]]], dtype=torch.double))

    def forward(self):
        return self.parameter * 1.


class Survey:
    shots = 2
    def is_valid(self, v):
        return bool((v > 0).all())

    def evaluate(self, v, *, gradient=False):
        return float(.5 * v.square().sum()), None


def test_momentum_restart_checks_actual_loss_without_committing():
    model, survey = Pixel(), Survey()
    rule = UpdateRule("adam", .2)
    rule.accept(rule.propose(torch.tensor([-20.], dtype=torch.double)))
    v = velocity(model)
    grad = pullback(model, v)
    before = rule.propose(grad).delta.clone()
    result = choose_step(model, v, v, grad, rule, 4.5, survey, policy="guarded",
                         max_rms=2., linear_tol=.3, max_halvings=5)
    assert result.restarted and result.reason == "accepted" and result.forward_shots == 2
    assert result.verified_loss < 4.5
    assert torch.equal(rule.propose(grad).delta, before)
    assert torch.equal(velocity(model), v)


def test_failed_backtracking_charges_attempts_and_leaves_state_untouched():
    model, survey = Pixel(), Survey()
    survey.evaluate = lambda *args, **kwargs: (6., None)
    rule = UpdateRule("sgd", .1)
    v = velocity(model)
    result = choose_step(model, v, v, pullback(model, v), rule, 4.5, survey,
                         policy="guarded", max_rms=2., linear_tol=.3, max_halvings=2)
    assert result.reason == "backtrack_exhausted" and result.trial is None
    assert result.forward_shots == 6 and len(result.trials) == 3
    assert torch.equal(velocity(model), v)


def test_armijo_ignores_velocity_cap_and_accepts_a_measured_decrease():
    model, survey = Pixel(), Survey()
    rule = UpdateRule("adam", .3)
    v = velocity(model)
    result = choose_step(model, v, v, pullback(model, v), rule, 4.5, survey,
                         policy="armijo", max_rms=.001, linear_tol=.01, max_halvings=3)
    assert result.reason == "accepted" and result.factor == 1.
    assert result.forward_shots == survey.shots and not result.restarted


def test_turning_off_remainder_check_isolates_its_backtrack():
    class Square(Pixel):
        def forward(self):
            return self.parameter.square()

    model, survey = Square(), Survey()
    model.parameter.data.fill_(2.)
    v = velocity(model)
    grad = pullback(model, v)
    rule = UpdateRule("sgd", 1. / 16.)
    kwargs = dict(max_rms=10., linear_tol=.2, max_halvings=2)
    guarded = choose_step(model, v, v, grad, rule, 8., survey, policy="guarded", **kwargs)
    ablated = choose_step(model, v, v, grad, rule, 8., survey, policy="guarded_no_remainder", **kwargs)
    assert guarded.factor == .5 and guarded.trials[0]["reason"] == "linearization_error"
    assert ablated.factor == 1. and ablated.verified_loss == .5
    assert guarded.forward_shots == ablated.forward_shots == survey.shots
