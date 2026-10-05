import unittest
import torch
from helpers import tiny
from ifwi_modules import IRN
from improved_modules.networks import create_improved_network
from improved_modules.optimizers import CosineAnnealingWithWarmup, create_improved_optimizer
from improved_modules.losses import DeepLayerPriorLoss

class SemanticsTests(unittest.TestCase):
    def test_prior_components_sum_to_reported_total(self):
        from improved_modules.losses import CombinedLoss
        loss=CombinedLoss(8,6,{'use_prior':True,'lambda_prior':.1})
        v=torch.arange(6.).expand(1,8,6)*50+6000
        shots=torch.zeros(1,2,3,6)
        _,parts=loss(shots,shots,v)
        self.assertIn('prior_horizontal',parts)
        torch.testing.assert_close(parts['prior_loss'],sum(parts[k] for k in ('prior_horizontal','prior_range','prior_monotonic')))

    def test_runner_preserves_backbone_when_attention_replaced(self):
        a,_,_=tiny()
        b,_,_=tiny({'model':{'network_type':'attention','neuron':[2,12,12,1],
                            'use_attention':False}})
        for p,q in zip(a.vel_net.linear.parameters(),b.vel_net.linear.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_layerwise_config_is_single_factor(self):
        import yaml
        from helpers import ROOT
        a=yaml.safe_load((ROOT/'experiments/configs/baseline.yaml').read_text())
        b=yaml.safe_load((ROOT/'experiments/configs/adaptive_lr.yaml').read_text())
        self.assertEqual(a['optimizer']['use_scheduler'],b['optimizer']['use_scheduler'])

    def test_attention_off_matches_author(self):
        ref=IRN(neuron=[2,12,12,1],outermost_linear=True)
        net=create_improved_network({'network_type':'attention','neuron':[2,12,12,1],
            'use_attention':False,'outermost_linear':True,'depth_max':1.4})
        net.linear.load_state_dict(ref.linear.state_dict())
        coords=torch.rand(1,8,9,2)
        torch.testing.assert_close(net(coords)[0],ref(coords)[0])

    def test_attention_coordinate_subset_consistent(self):
        net=create_improved_network({'network_type':'attention','neuron':[2,12,12,1],
            'depth_max':1.4,'outermost_linear':True})
        coords=torch.rand(1,8,9,2); coords[...,1]*=1.4
        full=net(coords)[0]; part=net(coords[:,2:4,3:5])[0]
        torch.testing.assert_close(full[:,2:4,3:5],part)

    def test_schedule_endpoints_and_resume(self):
        p=torch.nn.Parameter(torch.ones(1)); opt=torch.optim.Adam([p],lr=1e-4)
        s=CosineAnnealingWithWarmup(opt,warmup_epochs=2,max_epochs=6,eta_min=1e-6)
        for i,expected in [(0,5e-5),(1,1e-4),(5,1e-6)]:
            s.step(i); self.assertAlmostEqual(opt.param_groups[0]['lr'],expected,places=12)
        other=CosineAnnealingWithWarmup(torch.optim.Adam([p],lr=1e-4),2,6,1e-6)
        other.load_state_dict(s.state_dict()); self.assertEqual(other.last_epoch,5)
        with self.assertRaises(ValueError): CosineAnnealingWithWarmup(opt,6,6)

    def test_layerwise_groups_cover_network(self):
        net=create_improved_network({'network_type':'attention','neuron':[2,12,12,1]})
        opt,_,_=create_improved_optimizer(net,{'optimizer_type':'layerwise_lr','learning_rate':1e-4,'lr_deep':5e-4})
        ids=[id(p) for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),set(map(id,net.parameters())))
        self.assertEqual(len(opt.param_groups),2)

    def test_prior_shallow_changes_do_not_affect_deep_prior(self):
        prior=DeepLayerPriorLoss(8,6,deep_start=.5)
        v=torch.ones(1,8,6)*3000; other=v.clone(); other[:,:2]=5000; other[:,2:4]=1000
        torch.testing.assert_close(prior(v),prior(other))
