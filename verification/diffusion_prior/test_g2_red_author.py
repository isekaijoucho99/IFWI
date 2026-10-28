"""G2IFWI author U-Net, RED prior direction, LoRA, and safe CPU test suite."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.diffusion_prior.model import DDPM

torch.set_num_threads(1)


def tiny_cfg():
    return dict(backbone='g2_author', in_channels=1, out_channels=1,
                model_channels=32, channel_mult=[1, 1], num_res_blocks=1,
                attention_resolutions=[1, 2], num_heads=4)


def tiny_meta():
    return dict(units='m/s', mean_mps=3000., scale_mps=1500.,
                grid_spacing_m=15., patch_size=8)


def test_author_unet_attention_and_checkpoint_backward():
    from experiments.diffusion_prior.author_unet import G2AuthorUNet, AttentionBlock
    net = G2AuthorUNet(**{k:v for k,v in tiny_cfg().items() if k!='backbone'})
    assert isinstance(net.middle_block[1], AttentionBlock)
    assert net.middle_block[1].qkv.weight.shape == (96, 32, 1, 1)
    x = torch.randn(1, 1, 8, 8)
    t = torch.tensor([1])
    assert net(x,t).shape == x.shape
    net.gradient_checkpointing = True
    net(x,t).square().mean().backward()
    assert any(p.grad is not None for p in net.parameters())


def test_author_checkpoint_lora_roundtrip(tmp_path):
    from experiments.diffusion_prior.model import build_model, inject_lora
    from experiments.diffusion_prior.io import save_base, load_prior, save_adapter
    cfg = tiny_cfg()
    model = build_model(cfg)
    path = tmp_path/'base.pt'
    save_base(path, model, cfg, dict(steps=20,beta_start=1e-4,beta_end=.02),
              tiny_meta(),updates=1)
    params = inject_lora(model,rank=2,alpha=2.)
    assert params and all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    adapter = tmp_path/'adapter.pt'
    save_adapter(adapter,model,path,rank=2,alpha=2.,updates=1)
    loaded,_,_,_=load_prior(path,'cpu',15.,adapter=adapter)
    model.eval()
    x=torch.randn(1,1,8,8)
    with torch.no_grad():
        assert torch.equal(model(x,torch.tensor([1])),loaded(x,torch.tensor([1])))


def test_memory_preflight_guards_128_channel_author():
    from experiments.diffusion_prior.author_unet import memory_requirements, check_memory_budget
    from experiments.diffusion_prior.author_unet import paper_configuration
    full = paper_configuration(128)
    assert memory_requirements(full)['parameters']==1_307_810_433
    for mode in ('base_train','adapt'):
        with pytest.raises(MemoryError, match='16'):
            check_memory_budget(full,16*2**30,mode=mode,microbatch=1)
    assert memory_requirements(paper_configuration(32))['parameters'] < 100_000_000


class ZeroNoise(torch.nn.Module):
    def forward(self,x,t):
        return torch.zeros_like(x)


def test_g2_red_single_step_deterministic_mean():
    from experiments.diffusion_prior.guidance import g2_red_direction
    ddpm=DDPM(steps=100)
    v=torch.tensor([[[2700.,3000.],[3300.,3600.]]])
    direction,report=g2_red_direction(v,ZeroNoise(),ddpm,tiny_meta(),
                    image_size=2,steps=1,normalization='fixed',geometry='resize')
    clean=(((v-3000.)/1500.)/ddpm.abar[0].sqrt()).clamp(-1,1)*1500.+3000.
    assert torch.allclose(direction,v-clean,atol=5e-4)
    assert report['red_steps']==1 and report['unet_calls']==1


def test_native_patch_g2_red_zero_residual_for_constant_field():
    from experiments.diffusion_prior.guidance import g2_red_direction
    v=torch.full((1,9,17),3000.)
    direction,report=g2_red_direction(v,ZeroNoise(),DDPM(steps=20),
                   tiny_meta(),image_size=8,steps=1,geometry='patch',
                   normalization='fixed',patch_batch_size=1)
    assert direction.shape==v.shape and direction.abs().max()<1e-5
    assert report['unet_calls']>1


def test_author_train_one_step_accumulates_and_adapts(tmp_path):
    from experiments.diffusion_prior.training import train
    from experiments.diffusion_prior.io import load_prior
    root=tmp_path/'models'
    for split,offset in [('train',0),('val',500)]:
        folder=root/split
        folder.mkdir(parents=True)
        a=np.linspace(2000+offset,4500+offset,16*16,dtype=np.float32).reshape(16,16)
        np.save(folder/'0.npy',a)
    common=dict(train_dir=root/'train',val_dir=root/'val',out=tmp_path/'base_train',
                steps=1,batch_size=1,accumulate_steps=2,eval_every=1,val_batches=1,
                device='cpu',meta=tiny_meta(),model_config=tiny_cfg(),
                schedule_config=dict(steps=20,beta_start=1e-4,beta_end=.02))
    base=train(**common)
    assert base.is_file()
    net,_,_,_=load_prior(base,'cpu',15.)
    assert net.__class__.__name__=='G2AuthorUNet'
    adapter=train(**{**common,'out':tmp_path/'adapter_train','base':base,'rank':2,'alpha':2.})
    assert adapter.is_file()
    load_prior(base,'cpu',15.,adapter=adapter)


def test_mixed_precision_on_cpu_rejected_before_training(tmp_path):
    from experiments.diffusion_prior.training import train
    with pytest.raises(ValueError,match='CUDA'):
        train(tmp_path/'missing',tmp_path/'missing',tmp_path/'out',mixed_precision='bf16')


def test_pms_three_specs_hold_fixed_physics():
    import core_runner
    root=Path(__file__).resolve().parents[2]
    paths=sorted((root/'experiments/configs/diffusion_prior/g2_pms').glob('*.json'))
    assert len(paths)==3
    specs=[json.loads(p.read_text()) for p in paths]
    for spec in specs:core_runner.check_spec(spec)
    initial=specs[0]
    for item in specs[1:]:
        for key in ('experiment','control','robustness','background'):
            assert item[key]==initial[key]
        assert item['diffusion_prior']['mode']=='g2_red'
    assert 'diffusion_prior' not in initial


def test_g2_red_core_runner_tiny_physical_forward(tmp_path):
    import core_runner
    from verification.residual_ifwi.test_paper_supplement import tiny_spec
    from experiments.diffusion_prior.model import build_model
    from experiments.diffusion_prior.io import save_base
    cfg=tiny_cfg()
    path=tmp_path/'prior.pt'
    save_base(path,build_model(cfg),cfg,
              dict(steps=20,beta_start=1e-4,beta_end=.02),
              {**tiny_meta(),'patch_size':128},updates=0)
    spec=tiny_spec(tmp_path,'g2red_tiny',method='E2')
    spec['diffusion_prior']=dict(mode='g2_red',checkpoint=str(path),
                   allow_untrained=True,warmup=0,every=1,
                   red_steps=1,red_image_size=8,red_geometry='resize',
                   red_normalization='fixed')
    out=core_runner.run(spec,tmp_path/'run','cpu',
                        save_figures=False,shared_dir=tmp_path/'shared')
    history=json.loads((out/'history.json').read_text())
    assert len(history)==3
    assert history[0]['diffusion_prior']['active']
    assert history[0]['diffusion_prior']['red_steps']==1
    assert history[0]['diffusion_prior']['unet_calls']==1
