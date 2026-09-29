"""Traditional FWI entry point, independent of main.py; shared author solver and training kernel.

python fwi.py --experiment fwi_smooth --epochs 4000 --plots
python fwi.py --experiment fwi_noisy --noise 2 --epochs 4000 --plots
python fwi.py --experiment fwi_random --epochs 4000 --plots
"""
import argparse
import contextlib
import ctypes
import json
import os
from pathlib import Path
import sys
import time
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from generator import wGenerator
from ifwi_modules import IFWI2D
from rnn_fd import rnn2D
ROOT=Path(__file__).resolve().parent
EXPERIMENTS=('fwi_random','fwi_smooth','fwi_noisy')

class TraditionalFWI(IFWI2D):
    """Traditional FWI with the existing velocity projection and finite-value guards."""
    gradient_clip = None
    fwi_velocity_bounds = None

    @staticmethod
    def require_finite(value, label):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'Non-finite {label}; training stopped instead of masking invalid data')
        return value

    def check_fwi_forward(self, module, inputs, output):
        # Runs before the author's forward_process replaces NaN predictions by zero.
        for value in output:
            if torch.is_tensor(value):
                self.require_finite(value, 'FWI wavefield or predicted shots')

    def train_one_epoch(self, optimizer, *args, **kwargs):
        if self.gradient_clip is not None:
            self.params = [p for group in optimizer.param_groups for p in group['params']]
            self.clip = self.gradient_clip
        if self.fwi_velocity_bounds is None:
            return super().train_one_epoch(optimizer, *args, **kwargs)
        self.require_finite(self.vmodel, 'FWI velocity')
        lo, hi = self.fwi_velocity_bounds
        with torch.no_grad():
            self.vmodel.clamp_(lo / 1000, hi / 1000)
        handle = self.vmodel.register_hook(lambda grad: self.require_finite(grad, 'FWI velocity gradient'))
        try:
            result = super().train_one_epoch(optimizer, *args, **kwargs)
        finally:
            handle.remove()
        self.require_finite(self.vmodel, 'updated FWI velocity')
        if not np.isfinite(result[1]).all():
            raise FloatingPointError('Non-finite FWI loss')
        with torch.no_grad():
            self.vmodel.clamp_(lo / 1000, hi / 1000)
        return result

def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")
    os.replace(temporary, path)

@contextlib.contextmanager
def prevent_idle_sleep():
    enabled = os.name == "nt" and bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        yield
    finally:
        if enabled:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def new_output(path, label):
    out = (path or ROOT / "outputs" / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"输出目录非空，避免覆盖：{out}")
    out.mkdir(parents=True, exist_ok=True)
    return out

class EpochLog:
    """借用原训练函数的日志回调，每轮打印；不连接 W&B。"""
    def __init__(self, completed=0):
        self.completed = completed

    def init(self, **kwargs):
        return self

    def log(self, values):
        self.completed += 1
        print(f"Completed: {self.completed}, Loss: {values['Total Loss']:.6e}, "
              f"DataLoss: {values['Data Loss']:.6e}, RegLoss: {values['Regularization Loss']:.6e}",
              flush=True)

    def finish(self):
        pass

class LiveLog:
    """同一份原版 print 同时写到终端和日志，立即刷新。"""
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

def plot_result(out, truth, initial, best, history, dz=15):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout="constrained")
    extent = [0, truth.shape[1] * dz/1000, truth.shape[0] * dz/1000, 0]
    for ax, data, title in zip([axes[0, 0], axes[0, 1], axes[1, 0]],
                                [truth, initial, best],
                                ["Ground truth", "Initial model", "Best saved model"]):
        im = ax.imshow(data/1000, extent=extent, aspect="auto", cmap="RdBu_r",
                       vmin=1 if dz == 15 else float(truth.min()/1000),
                       vmax=4.7 if dz == 15 else float(truth.max()/1000))
        ax.set(title=title, xlabel="Distance (km)", ylabel="Depth (km)")
        fig.colorbar(im, ax=ax, label="km/s", extend="both")
    axes[1, 1].semilogy(np.arange(1, len(history)+1), history[:, 1])
    axes[1, 1].set(title="Waveform loss", xlabel="Update (loss before update)", ylabel="MSE")
    fig.savefig(out / "result.png", dpi=160)
    plt.close(fig)

def run_experiment(args):
    mode = args.experiment
    args.epochs = args.epochs if args.epochs is not None else 4000
    pretrained = None
    config = dict(mode=mode, seed=args.seed, mean=3., std=1., dz=15, dt=.0019, nt=1000,
                  frequency=8, source_depth_index=1, receiver_depth_index=2,
                  learning_rate=1e-4, alpha=0,
                  noise=args.noise if mode in ('noisy', 'fwi_noisy') else 0,
                  dropout=0,
                  epochs=args.epochs, log_interval=args.log_interval, pretrained=pretrained,
                  backend='reference', gradient_clip=args.clip_grad)
    if mode == 'fwi_random':
        config.update(learning_rate=.001, netOpt='FWI',
                      initial_source=str(ROOT.parent/'IFWI 1/11479806/runs/overnight_20260920_215223/random/initial_velocity.npy'))
    if mode in ('fwi_smooth', 'fwi_noisy'):
        config.update(learning_rate=.01, netOpt='FWI',
                      initial_source=str(ROOT/'data/vel_marmousi_smooth400_376x1151.csv'),
                      initial_gaussian_sigma=50, initial_stride=4, initial_csv_header=0)
    if config.get('netOpt') == 'FWI':
        config.update(fwi_velocity_bounds_mps=[1500., 5500.], fwi_finite_checks=True)
    resume = str(args.resume.resolve()) if args.resume else None
    if resume:
        previous = json.loads((Path(resume).parent.parent / 'config.json').read_text(encoding='utf8'))
        # 旧版没有这两个键，等价于默认未加速、未修复裁剪。
        previous.setdefault('backend', 'reference')
        previous.setdefault('gradient_clip', None)
        # A legacy FWI checkpoint may enable the new numerical safeguards on resume.
        for key in ('fwi_velocity_bounds_mps', 'fwi_finite_checks'):
            if key in config and key not in previous:
                previous[key] = config[key]
        for key, value in config.items():
            if key not in ('epochs', 'log_interval', 'pretrained') and previous.get(key) != value:
                raise ValueError(f"续训配置不一致：{key}，旧值={previous.get(key)}，新值={value}")
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        if args.epochs <= checkpoint['epoch']:
            raise ValueError("epochs 是总轮数，必须大于 checkpoint 已完成轮数")
        del checkpoint
    out = new_output(args.output_dir, mode)
    (out / 'checkpoints').mkdir()
    write_json(out / 'config.json', config)
    write_json(out / 'status.json', dict(state='running', mode=mode, pid=os.getpid()))
    with (out / 'progress.log').open('w', encoding='utf8', buffering=1) as log:
        with contextlib.redirect_stdout(LiveLog(sys.stdout, log)), contextlib.redirect_stderr(LiveLog(sys.stderr, log)):
            try:
                _execute(args, config, out, pretrained, resume)
                write_json(out / 'status.json', dict(state='completed', mode=mode))
            except BaseException as error:
                write_json(out / 'status.json', dict(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error)))
                traceback.print_exc()
                raise

def _execute(args, config, out, pretrained, resume):
    root = Path(__file__).resolve().parent
    mode = config["mode"]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 有意保留作者默认 header=0 的读取行为，不改变网格数据。
    truth = np.array(pd.read_csv(root / "data" / "vel_marmousi_376x1151.csv"))[::4, ::4].astype(np.float32)
    vp = torch.from_numpy(truth[None]).to(args.device)
    nv, nz, nx = vp.shape
    xs = torch.arange(20, nx-10, 20, dtype=torch.long).repeat(nv, 1)
    ns = xs.shape[1]
    xr = torch.arange(nx, dtype=torch.long).repeat(nv, ns, 1)
    zs = torch.full((nv, ns), config["source_depth_index"], dtype=torch.long)
    zr = torch.full((nv, ns, nx), config["receiver_depth_index"], dtype=torch.long)
    t = config["dt"] * torch.arange(config["nt"], dtype=torch.float32)
    wavelet = wGenerator(t, 8).ricker().to(args.device)
    common = dict(nz=nz, nx=nx, zs=zs, xs=xs, zr=zr, xr=xr, dz=config["dz"], dt=config["dt"],
                  npad=15, order=2, vmax=vp.max(), log_para=1e-6,
                  freeSurface=True, dtype=torch.float32, device=args.device)
    print(f"START {mode}: {args.epochs} updates, device={args.device}, output={out}", flush=True)
    print("Forward modeling: generating observed shots...", flush=True)
    forward = rnn2D(**common).to(args.device)
    with torch.no_grad():
        _, _, shots, _ = forward(vmodel=vp, segment_wavelet=wavelet)
    if mode in ("noisy", "fwi_noisy"):
        # 保留 notebook 的 NumPy 噪声生成顺序与 float64 含噪观测。
        observed = shots.cpu().numpy()
        noise = np.random.randn(nv, ns, 1000, nx) * (args.noise * observed.std())
        shots = torch.from_numpy(observed + noise).to(args.device)
    model = TraditionalFWI(**common, mean=config["mean"], std=config["std"], neuron=[2,128,128,128,128,1],
                   omega_0=30, prob=getattr(args, "dropout", .2), activation="sine",
                   bias=True, dropout=False, outermost_linear=True,
                   segment_size=config["nt"], vpadding=None, pretrained=pretrained, netOpt='FWI')
    model.gradient_clip = args.clip_grad
    if config.get('fwi_finite_checks'):
        model.fwi_velocity_bounds = config['fwi_velocity_bounds_mps']
        if model.fwi_velocity_bounds[1] * config['dt'] / config['dz'] >= 1 / np.sqrt(2):
            raise ValueError('FWI upper velocity bound violates the order-2 CFL limit')
        model.rnn.register_forward_hook(model.check_fwi_forward)
    initial_tensor = None
    if mode in ('fwi_smooth', 'fwi_noisy'):
        from scipy.ndimage import gaussian_filter
        smooth = pd.read_csv(config['initial_source'], header=config['initial_csv_header']).to_numpy()
        stride = config['initial_stride']
        initial = gaussian_filter(smooth, sigma=config['initial_gaussian_sigma'])[::stride, ::stride].astype(np.float32)
        baseline_truth = truth
    else:
        initial = np.load(config['initial_source']).squeeze().astype(np.float32)
        baseline_truth = np.load(Path(config['initial_source']).with_name('true_velocity.npy')).squeeze()
    if initial.shape != truth.shape or not np.isfinite(initial).all() or initial.min() <= 0 or not np.array_equal(baseline_truth, truth):
        raise ValueError('FWI initial grid or baseline truth mismatch')
    initial_tensor = torch.from_numpy(initial[None]/1000).to(args.device)
    np.save(out / 'observed.npy', shots.detach().cpu().numpy())
    np.save(out / "initial_velocity.npy", initial)
    np.save(out / "true_velocity.npy", truth)
    prefix = str(out / "checkpoints" / f"MarmousiI_{mode}-")
    completed = torch.load(resume, map_location="cpu", weights_only=False)['epoch'] if resume else 0
    history, _ = model.train(MaxIter=args.epochs, vmodel=initial_tensor, wavelet=wavelet, shots=shots,
                             alpha=config["alpha"], option=0, learning_rate=config['learning_rate'],
                             log_interval=args.log_interval, wandb=EpochLog(completed),
                             resume_file_name=resume, save_file_name=prefix)
    history = np.asarray(history)
    np.savetxt(out / "loss.csv", np.column_stack([np.arange(1,len(history)+1), history]),
               delimiter=",", header="completed_updates,total_loss,data_loss,regularization_statistic", comments="")
    checkpoint = prefix + f"checkpoint-{args.epochs}.pth"
    if config.get('netOpt') == 'FWI':
        np.save(out / 'final_velocity.npy', (model.vmodel.detach().squeeze()*1000).cpu().numpy())
    best, _ = model.predict(resume_file_name=checkpoint, best=True)
    best = best.detach().squeeze().cpu().numpy()
    np.save(out / "best_velocity.npy", best)
    if args.plots:
        plot_result(out, truth, initial, best, history, dz=config["dz"])
    metrics = dict(relative_model_error=float(np.linalg.norm(best-truth)/np.linalg.norm(truth)),
                   velocity_rmse_mps=float(np.sqrt(np.mean((best-truth)**2))),
                   final_data_loss=float(history[-1,1]))
    (out / "metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf8")
    print(f"COMPLETED: {metrics}\nResults: {out}",flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment',choices=EXPERIMENTS,default='fwi_smooth')
    parser.add_argument('--epochs',type=int,default=4000,help='Total optimizer updates, including resumed updates')
    parser.add_argument('--noise',type=float,default=2.,help='Noise SD multiplier; only used by fwi_noisy')
    parser.add_argument('--seed',type=int,default=3)
    parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log-interval',type=int,default=100)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--plots',action='store_true')
    parser.add_argument('--clip-grad',type=float,default=None,help='Optional effective gradient clipping; disabled by default')
    args=parser.parse_args()
    if args.epochs<1 or args.log_interval<1:parser.error('epochs and log-interval must be positive')
    if not np.isfinite(args.noise) or args.noise<0:parser.error('noise must be finite and nonnegative')
    if args.clip_grad is not None and (not np.isfinite(args.clip_grad) or args.clip_grad<=0):parser.error('clip-grad must be finite and positive')
    with prevent_idle_sleep():run_experiment(args)

if __name__=='__main__':main()
