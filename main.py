"""IFWI 主实验入口；原作者四个模块保持原样。

python main.py --experiment pretrain
python main.py --experiment random --accelerate --clip-grad 0.25
python main.py --sequence random noisy uncertainty
python main.py --validate-speed

mean=3/std=1 km/s；其余数据和正演设置沿用公开 notebook。
默认逐轮打印、不绘图、不启用加速或修复裁剪；详细差异见 README.md。
"""
import argparse
import contextlib
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from generator import wGenerator
from ifwi_modules import IFWI2D
from rnn_fd import rnn2D

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ("pretrain", "random", "noisy", "regularization", "uncertainty", "overthrust", "fwi_random", "fwi_smooth", "fwi_noisy")


class ExperimentIFWI(IFWI2D):
    """仅在明确开启裁剪时，向作者原训练步骤提供可重复遍历的参数列表。"""
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


def run_experiment(args):
    mode = args.experiment
    args.epochs = args.epochs if args.epochs is not None else (1001 if mode == "pretrain" else 4001)
    if args.accelerate and mode not in ('random', 'pretrain'):
        raise ValueError("当前加速入口限无噪声、无 dropout 的 random/pretrain 实验")
    pretrained = str(args.pretrained.resolve()) if mode == 'pretrain' and not args.resume else None
    if pretrained:
        meta = torch.load(pretrained, map_location="cpu", weights_only=False)
        if (meta.get('mean'), meta.get('std')) != (3., 1.):
            raise ValueError("预训练权重的 mean/std 必须为 3/1")
    config = dict(mode=mode, seed=args.seed, mean=3., std=1., dz=15, dt=.0019, nt=1000,
                  frequency=8, source_depth_index=1, receiver_depth_index=2,
                  learning_rate=1e-4, alpha='auto' if mode == 'regularization' else 0,
                  noise=args.noise if mode in ('noisy', 'fwi_noisy') else 0,
                  dropout=args.dropout if mode == 'uncertainty' else 0,
                  epochs=args.epochs, log_interval=args.log_interval, pretrained=pretrained,
                  backend='prepared' if args.accelerate else 'reference', gradient_clip=args.clip_grad)
    if mode == 'fwi_random':
        config.update(learning_rate=.001, netOpt='FWI',
                      initial_source=str(ROOT.parent/'IFWI 1/11479806/runs/overnight_20260920_215223/random/initial_velocity.npy'))
    if mode in ('fwi_smooth', 'fwi_noisy'):
        config.update(learning_rate=.01, netOpt='FWI',
                      initial_source=str(ROOT/'data/vel_marmousi_smooth400_376x1151.csv'),
                      initial_gaussian_sigma=50, initial_stride=4, initial_csv_header=0)
    if config.get('netOpt') == 'FWI':
        config.update(fwi_velocity_bounds_mps=[1500., 5500.], fwi_finite_checks=True)
    if mode == 'overthrust':
        config.update(mean=4.412, std=1.116, dz=20, dt=.002, nt=1500,
                      source_depth_index=1, receiver_depth_index=0,
                      source_start_index=20, source_spacing_index=40, num_shots=10,
                      data_file='overthrust2d.npz')
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


def launch(command, logfile, status, statusfile):
    """队列与验证共用：每次只启动一个子进程，日志立即刷新。"""
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    with logfile.open('w', encoding='utf8', buffering=1) as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf8', errors='replace')
        status['pid'] = child.pid
        write_json(statusfile, status)
        try:
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
            return child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
        finally:
            child.stdout.close()


def run_sequence(args):
    if args.resume:
        raise ValueError("顺序运行不共用一个续训文件，请单独续训对应实验")
    if args.accelerate and any(mode not in ('random','pretrain') for mode in args.sequence):
        raise ValueError("启用加速的队列只能包含 random/pretrain")
    out = new_output(args.output_dir, 'sequence')
    state = dict(state='running', experiments=[dict(mode=m, state='pending') for m in args.sequence])
    target = out / 'queue_status.json'
    write_json(target, state)
    try:
        for i, entry in enumerate(state['experiments']):
            folder = out / f"{i+1:02d}_{entry['mode']}"
            command = [sys.executable, '-u', str(ROOT/'main.py'), '--experiment', entry['mode'],
                       '--output-dir', str(folder), '--device', args.device, '--seed', str(args.seed),
                       '--log-interval', str(args.log_interval), '--noise', str(args.noise),
                       '--dropout', str(args.dropout), '--samples', str(args.samples),
                       '--pretrained', str(args.pretrained.resolve())]
            if args.epochs is not None:command += ['--epochs', str(args.epochs)]
            if args.clip_grad is not None:command += ['--clip-grad', str(args.clip_grad)]
            if args.accelerate:command.append('--accelerate')
            if args.plots:command.append('--plots')
            entry.update(state='running', output=str(folder), started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            print(f"START {i+1}/{len(args.sequence)}: {entry['mode']}", flush=True)
            rc = launch(command, out/f"{i+1:02d}_{entry['mode']}.log", state, target)
            entry.update(state='completed' if rc == 0 else 'failed', returncode=rc,
                         finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            write_json(target, state)
        state['state'] = 'completed' if all(e['state']=='completed' for e in state['experiments']) else 'finished_with_errors'
        write_json(target, state)
    except BaseException:
        state['state'] = 'interrupted'
        for entry in state['experiments']:
            if entry['state'] == 'running':entry['state'] = 'interrupted'
        write_json(target, state)
        raise
    print(f"QUEUE {state['state']}: {out}", flush=True)
    return 0 if state['state'] == 'completed' else 1


def run_validation(args):
    # 固定为已经验证过的无噪声随机初始化设置，不自动开始长训练。
    if args.device != 'cuda:0' or args.seed != 3 or args.clip_grad not in (None, .25):
        raise ValueError("加速验证固定使用 cuda:0、seed=3、clip=0.25；不支持另改这些参数")
    out = new_output(args.output_dir, 'speed_validation')
    states = {}
    for i, checkpoint in enumerate(args.validation_checkpoints):
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        states[f'checkpoint_{i+1}'] = str(checkpoint)
    write_json(out/'validation_states.json', states)
    state = dict(state='validating', training_will_start=False, gradient_clip=.25)
    target = out/'status.json'
    try:
        for backend in ['reference','prepared']:
            state['backend'] = backend
            rc = launch([sys.executable,'-u',str(ROOT/'main.py'),'--validation-worker',backend,
                         '--output-dir',str(out),'--validation-steps',str(args.validation_steps)],
                        out/f'{backend}.log', state, target)
            if rc:raise RuntimeError(f'{backend} 验证失败，退出码 {rc}')
        result = evaluate(out)
        write_json(out/'validation_decision.json', result)
        state.update(state='completed', selected_backend=result['selected_backend'], speedup=result['speedup'])
        write_json(target, state)
        print(f"VALIDATION: numerical={result['numerical_passed']}, speedup={result['speedup']:.3f}x, "
              f"selected={result['selected_backend']}\nNo long training started. Results: {out}", flush=True)
    except BaseException as error:
        state.update(state='failed', selected_backend='reference', error=str(error))
        write_json(target, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--experiment', choices=EXPERIMENTS)
    action.add_argument('--sequence', nargs='+', choices=EXPERIMENTS)
    action.add_argument('--validate-speed', action='store_true')
    action.add_argument('--validation-worker', choices=['reference','prepared'], help=argparse.SUPPRESS)
    parser.add_argument('--epochs', type=int, help='总轮数；默认 pretrain=1001，其他=4001')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log-interval', type=int, default=100, help='checkpoint 保存/最佳选择间隔；文本仍逐轮打印')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--pretrained', type=Path, default=ROOT/'weights/ifwi_pretrain_marmousi.pth')
    parser.add_argument('--noise', type=float, default=4.)
    parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--accelerate', action='store_true')
    parser.add_argument('--clip-grad', type=float, help='显式启用有效梯度裁剪，例如 0.25')
    parser.add_argument('--plots', action='store_true', help='可选输出结果图')
    parser.add_argument('--validation-checkpoints', nargs='*', type=Path, default=[])
    parser.add_argument('--validation-steps', type=int, default=20, help='短程训练对照轮数，至少 3；建议保持 20')
    args = parser.parse_args()
    if (args.epochs is not None and args.epochs < 1) or args.log_interval < 1:
        parser.error('epochs/log-interval 必须为正整数')
    if not np.isfinite(args.noise) or args.noise < 0 or not 0 <= args.dropout < 1 or args.samples < 2:
        parser.error('noise 非负且有限，dropout 在 [0,1)，samples 至少 2')
    if args.clip_grad is not None and (not np.isfinite(args.clip_grad) or args.clip_grad <= 0):
        parser.error('clip-grad 必须是有限正数')
    if args.validation_steps < 3:parser.error('validation-steps 至少为 3')
    args.experiment = args.experiment or 'random'
    with prevent_idle_sleep():
        if args.validation_worker:
            validate_worker(args.output_dir.resolve(), args.validation_worker, args.validation_steps)
        elif args.validate_speed:
            run_validation(args)
        elif args.sequence:
            return run_sequence(args)
        else:
            run_experiment(args)
    return 0


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


def _execute(args, config, out, pretrained, resume):
    root = Path(__file__).resolve().parent
    mode = config["mode"]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 有意保留作者默认 header=0 的读取行为，不改变网格数据。
    if mode == "overthrust":
        with np.load(root / "data" / config["data_file"]) as data:
            truth = data["vp"].astype(np.float32) * 1000  # archive is km/s
            if truth.shape != (94, 401) or float(data["dz"]) != config["dz"]:
                raise ValueError("Unexpected Overthrust grid or spacing")
        if not np.isfinite(truth).all() or truth.min() <= 0:
            raise ValueError("Invalid Overthrust velocities")
    else:
        truth = np.array(pd.read_csv(root / "data" / "vel_marmousi_376x1151.csv"))[::4, ::4].astype(np.float32)
    vp = torch.from_numpy(truth[None]).to(args.device)
    nv, nz, nx = vp.shape
    xs = torch.arange(20, nx-10, 20, dtype=torch.long).repeat(nv, 1)
    if mode == "overthrust":
        xs = (config["source_start_index"] + torch.arange(config["num_shots"], dtype=torch.long)
              * config["source_spacing_index"]).repeat(nv, 1)
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
    model = ExperimentIFWI(**common, mean=config["mean"], std=config["std"], neuron=[2,128,128,128,128,1],
                   omega_0=30, prob=getattr(args, "dropout", .2), activation="sine",
                   bias=True, dropout=mode == "uncertainty", outermost_linear=True,
                   segment_size=config["nt"], vpadding=None, pretrained=pretrained, netOpt=config.get('netOpt', 'IFWI'))
    model.gradient_clip = args.clip_grad
    if config.get('fwi_finite_checks'):
        model.fwi_velocity_bounds = config['fwi_velocity_bounds_mps']
        if model.fwi_velocity_bounds[1] * config['dt'] / config['dz'] >= 1 / np.sqrt(2):
            raise ValueError('FWI upper velocity bound violates the order-2 CFL limit')
        model.rnn.register_forward_hook(model.check_fwi_forward)
    if args.accelerate:
        install_prepared_forward(model)
    # 诊断输出关闭 dropout，不消耗训练随机数；此后恢复 train 模式。
    if resume and config.get('netOpt') != 'FWI':
        model.load_state(resume, best=False)
    initial_tensor = None
    if config.get('netOpt') == 'FWI':
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
    else:
        model.vel_net.eval()
        with torch.no_grad():
            initial, _ = model.vel_net(model.coords)
            initial = ((initial.squeeze()*model.std+model.mean)*1000).cpu().numpy()
        model.vel_net.train()
    np.save(out / "initial_velocity.npy", initial)
    np.save(out / "true_velocity.npy", truth)
    prefix = str(out / "checkpoints" / f"{'Overthrust' if mode == 'overthrust' else 'MarmousiI'}_{mode}-")
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
    if mode == "uncertainty":
        samples, _ = model.predict(resume_file_name=checkpoint, best=True,
                                    uncertainty=True, NoSim=args.samples)
        samples = samples.squeeze(-1).cpu().numpy()
        mean, std = samples.mean(axis=0), samples.std(axis=0, ddof=1)
        np.savez(out / "uncertainty.npz", mean_mps=mean, std_mps=std)
        if args.plots:
            fig, axes = plt.subplots(1,2,figsize=(11,3),layout="constrained")
            for ax, data, title in zip(axes, [mean, std], ["MC dropout mean (m/s)", "MC dropout std (m/s)"]):
                im=ax.imshow(data,extent=[0,nx*.015,nz*.015,0],aspect="auto",cmap="RdBu_r" if data is mean else "viridis")
                ax.set(title=title,xlabel="Distance (km)",ylabel="Depth (km)")
                fig.colorbar(im,ax=ax)
            fig.savefig(out / "uncertainty.png",dpi=160)
            plt.close(fig)
    (out / "metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf8")
    print(f"COMPLETED: {metrics}\nResults: {out}",flush=True)


def install_prepared_forward(model):
    """仅支持本实验：一个模型、整段记录、option=0、无分段状态传递。"""
    import types
    import torch
    rnn, fd = model.rnn, model.rnn.fd
    zs, xs = rnn.zs.cpu().numpy(), rnn.xs.cpu().numpy()
    shape = (1, rnn.zs.shape[1], rnn.xr.shape[-1])
    rows = torch.zeros(shape, dtype=torch.long, device=model.device)
    cols = torch.arange(shape[1], device=model.device)[None, :, None].expand(shape)
    zr = rnn.zr.to(model.device) + (0 if fd.freeSurface else fd.npad)
    xr = rnn.xr.to(model.device) + fd.npad

    def forward(self, vmodel, segment_wavelet, prev_state=None, curr_state=None, option=0):
        if option != 0 or vmodel.shape[0] != 1 or segment_wavelet.ndim != 1:
            raise ValueError("Prepared backend supports a single model, full 1D wavelet, option=0 only")
        fd.velocity = fd._Propagator2D___tensor_pad(vmodel[:, None])
        if prev_state is None:
            prev_state = vmodel.new_zeros((1, shape[1], self.nz_pad, self.nx_pad))
            curr_state = torch.zeros_like(prev_state)
        vx, vz = torch.zeros_like(prev_state), torch.zeros_like(prev_state)
        records = []
        for it in range(segment_wavelet.numel()):
            sources = [segment_wavelet[it:it+1], zs, xs]
            prev_state, curr_state, vx, vz = fd._Propagator2D___step_rnncell(
                sources, prev_state, curr_state, vx, vz)
            records.append(curr_state[rows, cols, zr, xr])
        return prev_state, curr_state, torch.stack(records, dim=-2), vmodel.new_zeros((1, 1))

    rnn.forward = types.MethodType(forward, rnn)


def build_model(device, small=False):
    import numpy as np
    import torch
    from ifwi_modules import IFWI2D
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    np.random.seed(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    nz, nx = (9, 12) if small else (94, 288)
    xs = torch.tensor([[3, 8]]) if small else torch.arange(20, nx-10, 20)[None]
    ns = xs.shape[1]
    return IFWI2D(mean=3., std=1., neuron=[2,16,16,1] if small else [2,128,128,128,128,1],
                  omega_0=30, dropout=False, outermost_linear=True, nz=nz, nx=nx,
                  zs=torch.ones((1, ns), dtype=torch.long), xs=xs,
                  zr=torch.full((1,ns,nx),2,dtype=torch.long),
                  xr=torch.arange(nx)[None,None].repeat(1,ns,1),
                  dz=15, dt=.001 if small else .0019, npad=3 if small else 15,
                  order=2, vmax=3500 if small else 5500, segment_size=24 if small else 1000,
                  freeSurface=True, device=device, dtype=torch.float32, netOpt="IFWI")


def objective(model, wavelet, observed):
    """与原版全时段 alpha=0 的运算及 loss 除法顺序一致。"""
    model.vel_net.zero_grad(set_to_none=True)
    velocity, _, pred, _, _ = model.forward_process(None, wavelet, None, None, 0)
    loss = (pred-observed).square().sum() / observed.shape[0] / observed.shape[1] / observed.shape[-2] / observed.shape[-1]
    loss.backward()
    return velocity, pred, loss


def validate_worker(work, backend, steps=20):
    import numpy as np
    import pandas as pd
    import torch
    from generator import wGenerator
    device = "cuda:0"
    model = build_model(device)
    wavelet = wGenerator(.0019*torch.arange(1000,dtype=torch.float32),8).ricker().to(device)
    initial = {k:v.detach().cpu().clone() for k,v in model.vel_net.state_dict().items()}
    observed_path = work / "observed.npy"
    if backend == "reference":
        truth = np.array(pd.read_csv(Path(__file__).parent/'data/vel_marmousi_376x1151.csv'))[::4,::4].astype(np.float32)
        with torch.no_grad():
            observed = model.rnn(torch.from_numpy(truth[None]).to(device),wavelet)[2]
        np.save(observed_path, observed.cpu().numpy())
    else:
        observed = torch.from_numpy(np.load(observed_path)).to(device)
        install_prepared_forward(model)
    artifact = {}
    reports = []
    snapshot = json.loads((work/'validation_states.json').read_text(encoding='utf8'))
    for label, filename in [("initial",None),*snapshot.items()]:
        state = initial if filename is None else torch.load(filename,map_location="cpu",weights_only=False)['state_dict']
        model.vel_net.load_state_dict(state)
        velocity,pred,loss = objective(model,wavelet,observed)
        gradients = torch.cat([p.grad.detach().flatten() for p in model.vel_net.parameters()])
        artifact[label+"_waveform"] = pred.detach().cpu().numpy()
        artifact[label+"_gradients"] = gradients.cpu().numpy()
        artifact[label+"_velocity"] = velocity.detach().cpu().numpy()
        if not all(np.isfinite(artifact[label+suffix]).all() for suffix in ['_waveform','_gradients','_velocity']):
            raise FloatingPointError(f"Nonfinite {backend}/{label}")
        reports.append(dict(state=label,loss=loss.item()))
        print(f"{backend} numerical check {label}: loss={loss.item():.8e}",flush=True)
        del velocity,pred,loss,gradients
    # 丢弃两次预热更新；两后端重置成完全相同的网络和 Adam 状态后测量。
    params = list(model.vel_net.parameters())
    model.params, model.clip = params, .25
    model.vel_net.load_state_dict(initial)
    optimizer = torch.optim.Adam(params,lr=1e-4)
    for _ in range(2):
        model.train_one_epoch(optimizer,None,wavelet,observed,0,0)
    model.vel_net.load_state_dict(initial)
    optimizer = torch.optim.Adam(params,lr=1e-4)
    times, losses = [], []
    for step in range(steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _, loss = model.train_one_epoch(optimizer,None,wavelet,observed,0,0)
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        losses.append(loss[1])
        with torch.no_grad():
            normalized,_=model.vel_net(model.coords)
            v=((normalized.squeeze(-1)+3)*1000).cpu().numpy()
        artifact[f"trajectory_{step}"]=v
        print(f"{backend} step {step+1}/{steps} loss={loss[1]:.8e} seconds={times[-1]:.3f}",flush=True)
    np.savez(work/f"{backend}_arrays.npz",**artifact)
    write_json(work/f"{backend}.json",dict(backend=backend,states=reports,losses=losses,
                                         seconds=times,median_seconds=float(np.median(times[2:])),
                                         torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(0)))


def evaluate(work):
    import numpy as np
    ref=np.load(work/'reference_arrays.npz')
    fast=np.load(work/'prepared_arrays.npz')
    a=json.loads((work/'reference.json').read_text())
    b=json.loads((work/'prepared.json').read_text())
    steps = len(a['losses'])
    checks=[]
    for name in ['initial',*json.loads((work/'validation_states.json').read_text())]:
        for suffix,tolerance in [('_waveform',1e-6),('_gradients',5e-5),('_velocity',1e-7)]:
            a,b=ref[name+suffix].astype(float),fast[name+suffix].astype(float)
            relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30))
            checks.append(dict(quantity=name+suffix,relative_l2=relative,tolerance=tolerance,
                               passed=bool(np.isfinite(relative) and relative<=tolerance)))
    for step in range(steps):
        diff=ref[f'trajectory_{step}'].astype(float)-fast[f'trajectory_{step}'].astype(float)
        rmse=float(np.sqrt(np.mean(diff**2)))
        maximum=float(np.abs(diff).max())
        checks.append(dict(quantity=f'training_velocity_{step+1}',rmse_mps=rmse,max_mps=maximum,
                           passed=bool(np.isfinite(rmse) and rmse<=.05 and maximum<=.5)))
    a=json.loads((work/'reference.json').read_text())
    b=json.loads((work/'prepared.json').read_text())
    loss_error=float(np.max(np.abs(np.array(a['losses'])-b['losses'])/np.maximum(np.abs(a['losses']),1e-30)))
    checks.append(dict(quantity='short_run_losses',max_relative=loss_error,passed=bool(loss_error<=1e-4)))
    speedup=a['median_seconds']/b['median_seconds']
    numerical=all(c['passed'] for c in checks)
    return dict(checks=checks,numerical_passed=numerical,speedup=speedup,
                reference_seconds=a['median_seconds'],prepared_seconds=b['median_seconds'],
                selected_backend='prepared' if numerical and speedup>=1.15 else 'reference',
                validation_steps=steps,
                limitation='Finite numerical and short-run trajectory checks do not prove identical 4000-step accuracy.')


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())

