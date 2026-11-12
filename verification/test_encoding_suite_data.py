from pathlib import Path
import hashlib,json,tempfile,unittest
import numpy as np
import torch
try:
    from experiments.encoding_suite_data import load_assets,make_model,build_problem,ARMS
except ImportError:load_assets=make_model=build_problem=ARMS=None


class SuiteDataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(load_assets,'suite data contract not implemented')
        scratch=Path(__file__).resolve().parents[1]/'.pytest_cache'/'suite_data';scratch.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch);self.root=Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(scratch.resolve()));self.addCleanup(self.temp.cleanup)
        clean=np.full((1,13,8,16),.1,dtype=np.float32)
        self.arrays=dict(truth=np.full((3,16),1700,dtype=np.float32),background=np.full((3,16),1500,dtype=np.float32),wavelet=np.linspace(-.1,.2,8,dtype=np.float32),clean=clean,noisy=clean.astype(np.float64)+.4)
        meta={}
        for name,a in self.arrays.items():
            p=self.root/(name+'.npy');np.save(p,a);meta[name]=dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),shape=list(a.shape),dtype=str(a.dtype))
        index={}
        for key,obs in [('marmousi_clean','clean'),('marmousi_noise_sigma4','noisy')]:index[key]={n:meta[obs if n=='observed'else n]for n in ('truth','background','wavelet','observed')}
        self.index=self.root/'dataset_index.json';self.index.write_text(json.dumps(index))
        cfg=dict(parameterization='residual',seed=3,model=dict(neurons=[2,8,8,1],omega_0=30.,mean_kmps=3.,std_kmps=1.),data=dict(grid_spacing_m=15.),acquisition=dict(num_shots=13,source_x_indices=list(range(13)),source_depth_index=1,receiver_depth_index=2,receiver_stride=1),forward=dict(dt_s=.0019,nt=8,frequency_hz=8.,npad=2,order=2,free_surface=True,pml_reflection=1e-6),training=dict(epochs=1000,learning_rate=1e-4,shot_batch_size=13))
        self.config=self.root/'config.json';self.config.write_text(json.dumps(cfg))

    def test_cached_assets_and_actual_snr_are_verified(self):
        a=load_assets(self.index,self.config)
        self.assertEqual(a['clean'].dtype,np.float32);self.assertEqual(a['noisy'].dtype,np.float32)
        self.assertTrue(np.array_equal(a['background'],self.arrays['background']))
        self.assertAlmostEqual(a['measured_snr_db'],-12.0411998,places=5)
        self.assertEqual(len(ARMS),6)

    def test_changed_input_hash_is_rejected_before_model(self):
        (self.root/'clean.npy').write_bytes(b'changed')
        with self.assertRaises(ValueError):load_assets(self.index,self.config)

    def test_all_models_have_exact_common_background_and_empty_adam(self):
        a=load_assets(self.index,self.config)
        for arm in ARMS:
            model,opt=make_model(a,arm,'cpu')
            self.assertTrue(torch.equal(model().detach(),torch.from_numpy(a['background'])[None]))
            self.assertEqual(len(opt.state),0)

    def test_problem_uses_cached_data_without_synthetic_generation(self):
        a=load_assets(self.index,self.config);p=build_problem(a,'noisy','cpu')
        self.assertEqual(tuple(p['observed'].shape),(1,13,8,16))
        self.assertTrue(torch.equal(p['observed'],torch.from_numpy(a['noisy'])))
        self.assertEqual(len(p['propagators']),1)


if __name__=='__main__':unittest.main(verbosity=2)
