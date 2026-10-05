import copy
import unittest
import torch
from helpers import tiny

class TrainingTests(unittest.TestCase):
    def test_real_update_matches_reference(self):
        model,data,_=tiny()
        reference=copy.deepcopy(model)
        opt=torch.optim.Adam(model.vel_net.parameters(),lr=1e-4)
        refopt=torch.optim.Adam(reference.vel_net.parameters(),lr=1e-4)
        before=[p.detach().clone() for p in model.vel_net.parameters()]
        normalized,_=reference.vel_net(reference.coords)
        v=(normalized.squeeze(-1)+3)*1000
        predicted=reference.rnn(v,data['wavelet'])[2]
        loss=(predicted-data['shots']).square().mean()
        loss.backward(); refopt.step()
        _,losses=model.train_one_epoch(opt,wavelet=data['wavelet'],shots=data['shots'])
        self.assertAlmostEqual(losses[0],loss.item(),places=6)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.vel_net.parameters())))
        for p,q in zip(model.vel_net.parameters(),reference.vel_net.parameters()):
            self.assertTrue(torch.isfinite(p.grad).all())
            torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)

    def test_nonfinite_raw_solver_output_is_rejected(self):
        model,data,_=tiny()
        def corrupt(module,args,output):
            values=list(output); values[2]=values[2]*float('nan'); return tuple(values)
        hook=model.rnn.register_forward_hook(corrupt)
        with self.assertRaises(FloatingPointError):
            model.train_one_epoch(torch.optim.Adam(model.vel_net.parameters()),
                                  wavelet=data['wavelet'],shots=data['shots'])
        hook.remove()

    def test_invalid_clip_rejected(self):
        with self.assertRaises(ValueError):
            tiny({'training':{'clip_grad':-1}})

if __name__=='__main__': unittest.main()
