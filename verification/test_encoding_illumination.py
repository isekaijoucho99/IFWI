import importlib.util
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'ifwi_research/encoding_illumination.py'
if path.exists():
    spec=importlib.util.spec_from_file_location('encoding_illumination_test_target',path)
    lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
else:lib=None


class Echo(torch.nn.Module):
    def forward(self,value):return None,value,None,None


class IlluminationTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(lib,'illumination observer not ported yet')

    def test_observer_preserves_gradient_and_sums_model_and_field_axes(self):
        x=torch.arange(1,49,dtype=torch.float32).reshape(2,2,3,4).requires_grad_()
        baseline=x.detach().clone().requires_grad_();(baseline.square().sum()/13).backward()
        ob=lib.PressureEnergy(3,4,0,True)
        echo=Echo()
        with echo.register_forward_hook(ob.observe):pressure=echo(x)[1];loss=pressure.square().sum()/13
        loss.backward()
        self.assertTrue(torch.equal(x.grad,baseline.grad))
        self.assertTrue(torch.equal(ob.source,x.detach().double().square().sum((0,1))))
        self.assertTrue(torch.equal(ob.receiver,x.grad.detach().double().square().sum((0,1))))
        self.assertEqual((ob.forward_samples,ob.adjoint_samples),(1,1))

    def test_batch_and_sum_of_serial_energies_match_same_loss_coefficients(self):
        values=torch.linspace(-1,2,24).reshape(1,2,3,4)
        observers=[]
        for j in range(2):
            x=values[:,j:j+1].clone().requires_grad_();ob=lib.PressureEnergy(3,4,0,True);m=Echo()
            with m.register_forward_hook(ob.observe):loss=m(x)[1].square().sum()/13
            loss.backward();observers.append(ob)
        x=values.clone().requires_grad_();ob=lib.PressureEnergy(3,4,0,True);m=Echo()
        with m.register_forward_hook(ob.observe):loss=m(x)[1].square().sum()/13
        loss.backward()
        self.assertTrue(torch.allclose(ob.source,sum(v.source for v in observers),rtol=0,atol=1e-14))
        self.assertTrue(torch.allclose(ob.receiver,sum(v.receiver for v in observers),rtol=0,atol=1e-14))

    def test_direction_matches_existing_water_level_and_rms_formula(self):
        g=torch.tensor([[[1.,-2.],[3.,-4.]]]);s=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64);a=s.square()
        actual,report=lib.illumination_direction(g,s,a)
        h=(s*a).sqrt();q=g.double()/(h/(h.max()+1e-30)+.1)
        expected=(q*g.double().norm()/q.norm().clamp_min(1e-30)).float()
        self.assertTrue(torch.equal(actual,expected))
        self.assertAlmostEqual(float(actual.double().norm()),float(g.double().norm()),places=6)
        self.assertEqual(report['water_level'],.1)

    def test_zero_gradient_and_zero_adjoint_are_valid_noops(self):
        g=torch.zeros(1,3,4);s=torch.ones(3,4,dtype=torch.float64);a=torch.zeros_like(s)
        actual,_=lib.illumination_direction(g,s,a)
        self.assertTrue(torch.equal(actual,g))

    def test_invalid_energy_is_not_silently_repaired(self):
        g=torch.ones(1,2,2);s=torch.ones(2,2,dtype=torch.float64)
        with self.assertRaises(ValueError):lib.illumination_direction(g,s,-s)


if __name__=='__main__':unittest.main(verbosity=2)
