import copy
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from run_experiment import ImprovedIFWI

torch.set_num_threads(1)

def tiny(config=None):
    torch.manual_seed(42)
    cfg = {'model': {'network_type': 'vanilla', 'neuron': [2, 12, 12, 1]},
           'loss': {'type': 'standard_mse'}, 'training': {'clip_grad': None},
           'optimizer': {'optimizer_type': 'adam', 'learning_rate': 1e-4}}
    if config:
        cfg.update(copy.deepcopy(config))
    nz, nx, nt = 10, 12, 24
    xs = torch.tensor([[3, 8]])
    model = ImprovedIFWI(improved_config=cfg, mean=3., std=1.,
        neuron=[2,12,12,1], omega_0=30, outermost_linear=True,
        nz=nz,nx=nx,zs=torch.ones_like(xs),xs=xs,
        zr=torch.full((1,2,nx),2),xr=torch.arange(nx)[None,None].repeat(1,2,1),
        dz=15.,dt=.001,npad=3,order=2,vmax=4000.,freeSurface=True,
        regularization='TV',segment_size=nt,device='cpu',netOpt='IFWI')
    # Nonzero source with arrivals inside a short genuine wave simulation.
    wave=torch.zeros(nt); wave[2:5]=torch.tensor([.5,1.,.5])
    truth=torch.linspace(2800,3400,nz)[:,None].expand(nz,nx)[None].contiguous()
    with torch.no_grad():
        shots=model.rnn(truth,wave)[2].clone()
    data={'vp_true':truth,'shots':shots,'wavelet':wave,
          'geometry':{'nz':nz,'nx':nx},'params':{'dz':15.,'dt':.001,'nt':nt}}
    return model,data,cfg
