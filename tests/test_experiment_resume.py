import tempfile
import unittest
from pathlib import Path
import torch
from helpers import tiny
from experiment_runtime import train_loop

class ResumeTests(unittest.TestCase):
    def test_resume_preserves_best_file_without_improvement(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); m,data,c=tiny()
            c['training'].update(max_iterations=2,log_interval=1)
            c['optimizer']['learning_rate']=1e-30
            train_loop(m,data,c,root/'first',stop_after=1)
            n,_,_=tiny()
            result=train_loop(n,data,c,root/'resume',resume=root/'first/last.pth')
            self.assertEqual(result['best_update'],1)
            self.assertTrue((root/'resume/best.pth').is_file())

    def config(self,c):
        c['training'].update(max_iterations=4,log_interval=2,alpha=0)
        c['optimizer'].update(use_scheduler=True,scheduler_params={'warmup_epochs':1,'max_epochs':4,'eta_min':1e-6})
        c['seed']=42
        return c

    def test_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            a,data,c=tiny(); c=self.config(c)
            full=train_loop(a,data,c,root/'full')
            b,_,_=tiny(); train_loop(b,data,c,root/'first',stop_after=2)
            cmodel,_,_=tiny()
            resumed=train_loop(cmodel,data,c,root/'resumed',resume=root/'first/last.pth')
            self.assertEqual(resumed['completed_updates'],4)
            self.assertEqual(full['history'],resumed['history'])
            for p,q in zip(a.vel_net.parameters(),cmodel.vel_net.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)
            ck=torch.load(root/'first/last.pth',weights_only=False)
            self.assertEqual(ck['completed_updates'],2)

    def test_best_snapshot_independent_and_loss_matches(self):
        with tempfile.TemporaryDirectory() as d:
            model,data,c=tiny(); result=train_loop(model,data,self.config(c),Path(d)/'run')
            best={k:v.clone() for k,v in result['best_state'].items()}
            with torch.no_grad():
                for p in model.vel_net.parameters(): p.add_(100)
            for k in best: torch.testing.assert_close(best[k],result['best_state'][k],rtol=0,atol=0)
            model.vel_net.load_state_dict(best)
            with torch.no_grad(): _,loss,_,_=model.objective(data['wavelet'],data['shots'])
            self.assertAlmostEqual(float(loss),result['best_loss'],places=6)

    def test_reject_changed_observations_and_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); m,data,c=tiny(); c=self.config(c)
            train_loop(m,data,c,root/'a',stop_after=2)
            n,changed,_=tiny(); changed['shots']=changed['shots']+1
            with self.assertRaisesRegex(ValueError,'contract'):
                train_loop(n,changed,c,root/'b',resume=root/'a/last.pth')
            torch.save({'epoch':1},root/'old.pth')
            with self.assertRaisesRegex(ValueError,'checkpoint'):
                train_loop(n,data,c,root/'old',resume=root/'old.pth')
