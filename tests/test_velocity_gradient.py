import unittest
import torch
from helpers import tiny
from improved_modules.losses import make_depth_weights, attach_velocity_preconditioner

class GradientTests(unittest.TestCase):
    def test_hook_removed_when_loss_raises(self):
        from unittest.mock import patch
        import run_experiment
        model,data,_=tiny({'gradient_preconditioner':{'enabled':True}})
        captured=[]
        def attach(v,w):
            captured.append(v)
            return attach_velocity_preconditioner(v,w)
        def broken(*a,**kw): raise RuntimeError('loss failed')
        model.loss_fn=broken
        with patch.object(run_experiment,'attach_velocity_preconditioner',attach):
            with self.assertRaisesRegex(RuntimeError,'loss failed'):
                model.train_one_epoch(torch.optim.Adam(model.vel_net.parameters()),wavelet=data['wavelet'],shots=data['shots'])
        self.assertFalse(captured[0]._backward_hooks)

    def test_piecewise_and_identity(self):
        w=make_depth_weights(8,{'weight_type':'piecewise'},'cpu',torch.float64)
        raw=torch.tensor([1,1,1,1,2,2,5,5],dtype=torch.float64)
        torch.testing.assert_close(w,raw/raw.mean())
        torch.testing.assert_close(make_depth_weights(8,{'weight_type':'identity'},'cpu',torch.float32),torch.ones(8))

    def test_wave_branch_only_and_chain_rule(self):
        p=torch.tensor(2.,requires_grad=True)
        v=p*torch.ones(1,8,3); wave=v.clone()
        w=make_depth_weights(8,{'weight_type':'piecewise'},'cpu',v.dtype)
        h=attach_velocity_preconditioner(wave,w)
        coeff=torch.arange(1,9.)[None,:,None]
        loss=(wave*coeff).sum()+v.square().sum()
        loss.backward(); h.remove()
        expected=(coeff*w[None,:,None]*torch.ones_like(v)).sum()+2*v.detach().sum()
        torch.testing.assert_close(p.grad,expected)

    def test_invalid_and_wrong_shape(self):
        for config in [{'deep_weight':-1},{'scale':0,'weight_type':'exponential'}]:
            with self.assertRaises(ValueError): make_depth_weights(8,config,'cpu',torch.float32)
        with self.assertRaises(ValueError):
            attach_velocity_preconditioner(torch.ones(256,2,requires_grad=True),torch.ones(94))

    def test_real_identity_matches_baseline(self):
        a,data,_=tiny(); b,_,_=tiny({'gradient_preconditioner':{'enabled':True,'weight_type':'identity'}})
        for m in (a,b):
            m.train_one_epoch(torch.optim.Adam(m.vel_net.parameters(),lr=1e-4),wavelet=data['wavelet'],shots=data['shots'])
        for p,q in zip(a.vel_net.parameters(),b.vel_net.parameters()): torch.testing.assert_close(p,q)
