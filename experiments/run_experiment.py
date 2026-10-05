"""Checked IFWI experiments. Explicit configs; full-record propagation; no hidden clipping."""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ifwi_modules import IFWI2D
from rnn_fd import rnn2D
from generator import wGenerator
from improved_modules.losses import CombinedLoss, make_depth_weights, attach_velocity_preconditioner
from improved_modules.networks import create_improved_network
from improved_modules.spatiotemporal import SpatiotemporalController
from improved_modules.evaluate_deep import evaluate_deep_layers
from experiment_runtime import train_loop, write_json, source_hashes, velocity, evaluate_loss
from parameter_sweep import source_positions, backward_shot_batches

class ImprovedIFWI(IFWI2D):
    """Checked full-record IFWI; author solver and network modules stay unchanged."""
    def __init__(self, *args, improved_config=None, **kwargs):
        import copy
        self.improved_config = copy.deepcopy(improved_config or {})
        super().__init__(*args, **kwargs)
        config = self.improved_config
        mc = config.get('model', {})
        if mc.get('network_type', 'vanilla') != 'vanilla':
            mc.update(neuron=kwargs['neuron'], omega_0=kwargs.get('omega_0',30),
                      outermost_linear=kwargs.get('outermost_linear',True),
                      depth_min=0., depth_max=(kwargs['nz']-1)*kwargs['dz']/1000)
            original_linear = self.vel_net.linear.state_dict()
            self.vel_net = create_improved_network(mc, device=self.device)
            self.vel_net.linear.load_state_dict(original_linear)
        self.clip = config.get('training',{}).get('clip_grad')
        if self.clip is not None and (not np.isfinite(self.clip) or self.clip<=0):
            raise ValueError('clip_grad must be null or finite and positive')
        self.loss_fn = CombinedLoss(kwargs['nz'],kwargs['nx'],config.get('loss',{}))
        self.last_gradient_norm = None
        self.sample_dt = kwargs["dt"]
        pc=config.get('gradient_preconditioner',{})
        self.depth_weights=make_depth_weights(kwargs['nz'],pc,self.device,self.dtype) if pc.get('enabled') else None
        st = config.get('spatiotemporal', {})
        self.attention = SpatiotemporalController(st, self.sample_dt) if st.get('enabled', False) else None
        self.selection_metric = 'data_mse' if self.attention is not None else 'total_loss'
        self.objective_step = 0
        if self.attention is not None and self.depth_weights is not None:
            raise ValueError('Do not combine legacy depth weights with the spatiotemporal probe')
        if self.attention is not None and self.loss_fn.data_objective != 'mse':
            raise ValueError('Spatiotemporal probe requires the MSE data objective')
        self.shot_batch_size = config.get('training', {}).get('shot_batch_size')
        if self.shot_batch_size is not None:
            if isinstance(self.shot_batch_size, bool) or not isinstance(self.shot_batch_size, int) or self.shot_batch_size < 1:
                raise ValueError('shot_batch_size must be a positive integer')
            if self.attention is not None or self.depth_weights is not None or config.get('loss', {}).get('use_prior') or self.loss_fn.data_objective != 'mse' or config.get('training', {}).get('alpha', 0):
                raise ValueError('Shot accumulation currently supports plain MSE without priors or hooks')

    def set_training_step(self, completed_updates):
        self.objective_step = completed_updates
        if self.attention is not None:
            self.attention.set_step(completed_updates)

    @staticmethod
    def finite(tensor, label):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError('Non-finite '+label)
        return tensor

    def objective(self, wavelet, shots, trade_off=0, precondition=False):
        # Full record only: preserve raw solver outputs and reject nonfinite values.
        normalized,coords = self.vel_net(self.coords)
        v = (normalized.squeeze(-1)*self.std+self.mean)*1000
        self.finite(v,'velocity')
        v_wave = v.clone()
        handle=attach_velocity_preconditioner(v_wave,self.depth_weights) if precondition and self.depth_weights is not None else None
        if precondition and self.attention is not None:
            handle = v_wave.register_hook(self.attention.spatial_gradient)
        try:
            outputs = self.rnn(v_wave,wavelet)
            for value in outputs:
                if torch.is_tensor(value): self.finite(value,'raw solver output')
            self.finite(shots,'observations')
            loss,parts = self.loss_fn(outputs[2],shots,v,dt=self.sample_dt)
            if self.attention is not None:
                selected = self.attention.data_loss(outputs[2], shots)
                loss = selected + self.loss_fn.lambda_prior * parts['prior_loss']
                parts.update(data_loss=selected, total_loss=loss)
            if trade_off:
                tv=(self.gradient(normalized,coords).square()+1e-6).sqrt().mean()
                loss=loss+trade_off*tv
            self.finite(loss,'loss')
            return v,loss,parts,handle
        except BaseException:
            if handle is not None: handle.remove()
            raise

    def train_one_epoch(self, optimizer, vmodel=None, wavelet=None, shots=None,
                        trade_off=0, option=0):
        if self.netOpt!='IFWI' or option!=0 or len(wavelet)!=self.segment_size:
            raise ValueError('Experiment runner supports full-record IFWI only')
        self.vel_net.train()
        self.params = [p for group in optimizer.param_groups for p in group['params']]
        current = [p for p in self.vel_net.parameters() if p.requires_grad]
        if len(self.params)!=len(set(map(id,self.params))) or set(map(id,current))!=set(map(id,self.params)):
            raise ValueError('Optimizer must cover current network exactly once')
        optimizer.zero_grad(set_to_none=True)
        handle = None
        try:
            if self.shot_batch_size is not None:
                if trade_off:
                    raise ValueError('Shot accumulation requires trade_off=0')
                v,loss,parts,handle = backward_shot_batches(self,wavelet,shots.to(self.device))
            else:
                v,loss,parts,handle = self.objective(wavelet,shots.to(self.device),trade_off,True)
                loss.backward()
            grads=[p.grad for p in self.params if p.grad is not None]
            if not grads: raise FloatingPointError('No parameter gradients')
            for grad in grads: self.finite(grad,'parameter gradient')
            norm=torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in grads]))
            self.finite(norm,'gradient norm')
            self.last_gradient_norm=float(norm)
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.params,self.clip,error_if_nonfinite=True)
            optimizer.step()
            for p in self.params: self.finite(p,'updated parameter')
            return v.detach(),[float(loss.detach()),float(parts['data_loss'].detach()),
                               float(parts['prior_loss'].detach()),float(parts['data_mse'].detach())]
        finally:
            if handle is not None: handle.remove()


def prepare_data(config, device):
    """Prepare training data."""
    data_config = config['data']

    # Load true velocity model
    data_path = Path(__file__).parent.parent / 'data' / data_config['model_file']
    truth = np.array(pd.read_csv(data_path, header=0))

    # Downsample
    downsample = data_config.get('downsample', 4)
    truth = truth[::downsample, ::downsample].astype(np.float32)

    vp = torch.from_numpy(truth[None]).to(device)
    nv, nz, nx = vp.shape

    # Setup acquisition geometry
    xs = torch.from_numpy(source_positions(nx, data_config)).repeat(nv, 1)
    ns = xs.shape[1]
    xr = torch.arange(nx, dtype=torch.long).repeat(nv, ns, 1)
    zs = torch.full((nv, ns), 1, dtype=torch.long)  # source_depth_index
    zr = torch.full((nv, ns, nx), 2, dtype=torch.long)  # receiver_depth_index

    # Generate wavelet
    dt = data_config['dt']
    nt = data_config['nt']
    t = dt * torch.arange(nt, dtype=torch.float32)
    wavelet = wGenerator(t, 8).ricker().to(device)

    # Forward modeling to generate observed data
    print("Generating observed shots...")
    forward = rnn2D(
        nz=nz, nx=nx, zs=zs, xs=xs, zr=zr, xr=xr,
        dz=data_config['dz'], dt=dt,
        npad=15, order=2, vmax=vp.max(), log_para=1e-6,
        freeSurface=True, dtype=torch.float32, device=device
    ).to(device)

    with torch.no_grad():
        _, _, shots, _ = forward(vmodel=vp, segment_wavelet=wavelet)

    for value in (vp,shots,wavelet):
        if not torch.isfinite(value).all(): raise FloatingPointError("Nonfinite generated data")

    # Add noise if specified
    noise_level = data_config.get('noise_level', 0.0)
    if noise_level > 0:
        noise = torch.randn_like(shots) * (noise_level * shots.std())
        shots = shots + noise
        print(f"Added noise: {noise_level * 100:.1f}% of signal std")

    return {
        'vp_true': vp,
        'shots': shots,
        'wavelet': wavelet,
        'geometry': {'nz': nz, 'nx': nx, 'zs': zs, 'xs': xs, 'zr': zr, 'xr': xr},
        'params': {'dz': data_config['dz'], 'dt': dt, 'nt': nt}
    }


def load_config(path):
    with open(path,encoding='utf-8') as f: config=yaml.safe_load(f)
    for section in ('model','data','training','optimizer','evaluation'):
        if section not in config: raise ValueError('Missing config section: '+section)
    t=config['training']; d=config['data']
    if any(not isinstance(t[k],int) or t[k]<1 for k in ('max_iterations','log_interval')):
        raise ValueError('Iterations/log_interval must be positive integers')
    if d['dt']<=0 or d['dz']<=0 or d['nt']<2 or d.get('downsample',4)<1:
        raise ValueError('Invalid data sampling')
    if config['model'].get('network_type','vanilla') not in ('vanilla','attention'):
        raise ValueError('Only vanilla/attention supported by this protocol')
    if config['model'].get('dropout',False): raise ValueError('Dropout outside this deterministic comparison protocol')
    if config['loss'].get('use_depth_weight') or config['optimizer'].get('use_grad_modifier'):
        raise ValueError('Legacy gradient settings: use gradient_preconditioner instead')
    return config


def build_model(config,data,device):
    geom=data['geometry']; params=data['params']; mc=config['model']
    return ImprovedIFWI(improved_config=config,mean=3.,std=1.,neuron=mc['neuron'],
        omega_0=mc.get('omega_0',30),prob=mc.get('prob',.2),activation=mc.get('activation','sine'),
        bias=mc.get('bias',True),dropout=mc.get('dropout',False),outermost_linear=mc.get('outermost_linear',True),
        nz=geom['nz'],nx=geom['nx'],zs=geom['zs'],xs=geom['xs'],zr=geom['zr'],xr=geom['xr'],
        dz=params['dz'],dt=params['dt'],npad=15,order=2,vmax=float(data['vp_true'].max()),
        log_para=1e-6,segment_size=params['nt'],freeSurface=True,regularization='TV',
        dtype=torch.float32,device=device,netOpt='IFWI')


def save_plot(truth,pred,dz,path,model_title='Best saved model'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    extent=[0,truth.shape[1]*dz/1000,truth.shape[0]*dz/1000,0]
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,v,title in zip(axes[:2],[truth,pred],['Truth',model_title]):
        im=ax.imshow(v/1000,extent=extent,aspect='auto',cmap='RdBu_r',vmin=truth.min()/1000,vmax=truth.max()/1000)
        ax.set(title=title,xlabel='Distance (km)',ylabel='Depth (km)')
        fig.colorbar(im,ax=ax,label='km/s')
    error=pred-truth; bound=max(float(np.abs(error).max()),1.)
    im=axes[2].imshow(error,extent=extent,aspect='auto',cmap='RdBu_r',vmin=-bound,vmax=bound)
    axes[2].set(title='Prediction minus truth',xlabel='Distance (km)',ylabel='Depth (km)')
    fig.colorbar(im,ax=axes[2],label='m/s'); fig.savefig(path,dpi=160); plt.close(fig)


def evaluate_model_metrics(model, data, config):
    """Report raw MSE alongside the selected objective and velocity diagnostics."""
    pred=velocity(model); truth=data['vp_true'].squeeze().cpu().numpy()
    settings=config['evaluation']; depth=settings.get('depth_threshold',.5)
    metrics=evaluate_deep_layers(truth,pred,depth_threshold=depth,
        corner_size=settings.get('corner_size',.25),dz=data['params']['dz'])
    _,parts=evaluate_loss(model,data,config['training'].get('alpha',0))
    metrics.update({k:v for k,v in parts.items() if k.startswith('prior_')})
    deep=pred[int(pred.shape[0]*depth):]
    metrics.update(data_mse=parts['data_mse'],data_objective=parts['data_loss'],
        velocity_min_mps=float(pred.min()),velocity_max_mps=float(pred.max()),
        deep_velocity_std_mps=float(deep.std()),
        deep_horizontal_tv_mps=float(np.abs(np.diff(deep,axis=1)).mean()))
    metrics.update(selection_metric=model.selection_metric, objective_step=model.objective_step,
                   cutoff_hz=model.attention.current_cutoff_hz if model.attention is not None else None)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output-dir',default='results')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--allow-execution-change',action='store_true',help='Verified resume across shot-batching/runner changes; core physics and objective must match')
    parser.add_argument('--iterations',type=int,help='Override total update budget, not scheduler horizon')
    parser.add_argument('--log-interval',type=int)
    parser.add_argument('--preliminary',action='store_true',help='Explicit short original-protocol validation budget')
    args=parser.parse_args()
    if args.allow_execution_change and args.resume is None:
        parser.error('--allow-execution-change requires --resume')
    config=load_config(args.config)
    if args.iterations is not None: config['training']['max_iterations']=args.iterations
    if args.log_interval is not None: config['training']['log_interval']=args.log_interval
    if config['training']['max_iterations']<1 or config['training']['log_interval']<1:
        parser.error('iterations and log-interval must be positive')
    args.seed = args.seed if args.seed is not None else config.get('seed', 42)
    config['seed']=args.seed
    if config.get('execution', {}).get('protocol') == 'original_baseline':
        if args.allow_execution_change:
            parser.error('Original baseline resumes require the same original code and full-shot input')
        from baseline_experiment import run_config
        run_config(config, Path(args.output_dir), device=args.device, seed=args.seed,
                   resume=args.resume, preliminary=args.preliminary)
        return 0
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    out=Path(args.output_dir)/(config['experiment_name']+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',config)
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    write_json(out/'status.json',{'state':'preparing','completed_updates':0})
    import importlib.metadata as metadata
    write_json(out/'environment.json',{'python':sys.version,'device':args.device,
        'packages':{n:metadata.version(n) for n in ('torch','torchvision','numpy','pandas','scipy','matplotlib','scikit-image','PyYAML')},
        'source_hashes':source_hashes()})
    print('Output directory: '+str(out),flush=True)
    try:
        data=prepare_data(config,args.device)
        np.save(out/'observed.npy',data['shots'].cpu().numpy())
        write_json(out/'acquisition.json', {
            'num_shots': int(data['geometry']['xs'].shape[1]),
            'source_x_indices': data['geometry']['xs'].tolist(),
            'source_z_indices': data['geometry']['zs'].tolist(),
            'receiver_x_indices': data['geometry']['xr'].tolist(),
            'receiver_z_indices': data['geometry']['zr'].tolist()})
        # Decouple network initialization from acquisition/noise RNG consumption.
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        model=build_model(config,data,args.device)
        if model.depth_weights is not None: np.save(out/'depth_weights.npy',model.depth_weights.cpu().numpy())
        result=train_loop(model,data,config,out,resume=args.resume,allow_execution_change=args.allow_execution_change)
        final_metrics=evaluate_model_metrics(model,data,config)
        final_metrics['completed_updates']=result['completed_updates']
        write_json(out/'final_metrics.json',final_metrics)
        model.vel_net.load_state_dict(result['best_state'])
        model.set_training_step(result['best_update'] - 1)
        pred=velocity(model); truth=data['vp_true'].squeeze().cpu().numpy()
        settings=config['evaluation']
        metrics=evaluate_model_metrics(model,data,config)
        metrics.update(parameter_count=result['parameter_count'],best_update=result['best_update'])
        metrics['selection_metric'] = result['selection_metric']
        write_json(out/'metrics.json',metrics)
        np.save(out/'v_pred.npy',pred)
        if settings.get('save_plots',True): save_plot(truth,pred,data['params']['dz'],out/'result.png')
        write_json(out/'status.json',{'state':'completed','completed_updates':result['completed_updates']})
        print(f"Completed: best_update={result['best_update']}, deep_rmse={metrics['deep_rmse']:.2f} m/s; {out}",flush=True)
    except BaseException as exc:
        previous=json.loads((out/'status.json').read_text(encoding='utf-8'))
        write_json(out/'status.json',{'state':'failed','completed_updates':previous.get('completed_updates',0),'error':str(exc)})
        raise
    return 0

if __name__=='__main__': raise SystemExit(main())
