import pytest
import torch
from experiments.rom.benchmark import evaluate,error_metrics


class NonfiniteDerivative(torch.nn.Module):
    def forward(self,vmodel,**kwargs):
        prediction=torch.sqrt((vmodel-1).sum()).expand(1,1,2,1)
        return torch.zeros(1),torch.zeros(1),prediction,torch.zeros(1)


def test_finite_prediction_with_infinite_gradient_is_rejected_before_logging():
    problem=dict(propagators=[dict(solver=NonfiniteDerivative(),first=0,last=1)],
        wavelet=torch.zeros(2),observed=torch.ones(1,1,2,1))
    with pytest.raises(FloatingPointError):evaluate(torch.ones(1,2,2),problem,backward=True)


def test_invalid_metrics_are_rejected_before_poisoning_json_rows():
    with pytest.raises(FloatingPointError):error_metrics([float('inf')],[1.])
