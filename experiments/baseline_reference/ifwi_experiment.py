"""Marmousi notebook 的公共准备、原版训练调用和结果输出。

入口见 IFWI_Marmousi_*.py。默认论文归一化 mean=3/std=1 km/s；
采集几何、数据读取、正演和训练超参数沿用 notebook。
不修改 ifwi_modules.py、rnn_fd.py、generator.py、plot_functions.py。
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from generator import wGenerator
from ifwi_modules import IFWI2D
from rnn_fd import rnn2D


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


def watch_progress(pid, output):
    """旁路监视已运行的训练；不向训练进程注入代码，不改变参数。"""
    import ast
    import subprocess
    import psutil
    process = psutil.Process(pid)
    spy = Path(sys.executable).parent / "Scripts/py-spy.exe"
    if not spy.is_file():
        raise FileNotFoundError(f"Live monitor requires {spy}")
    previous = None
    print("Read-only live monitor: completed updates and their loss. Keep the training window open.", flush=True)
    with Path(output).open("a", encoding="utf8", buffering=1) as log:
        while process.is_running():
            try:
                result = subprocess.run([str(spy), "dump", "--pid", str(pid), "--locals", "--json", "--nonblocking"],
                                        capture_output=True, text=True, encoding="utf8", timeout=8)
                result.check_returncode()
                for thread in json.loads(result.stdout):
                    frames = thread.get("frames", [])
                    if not any(f.get("name") == "train_one_epoch" for f in frames):
                        continue
                    for frame in frames:
                        if frame.get("name") != "train" or not frame.get("filename", "").endswith("ifwi_modules.py"):
                            continue
                        values = {v['name']: v['repr'] for v in frame.get('locals', [])
                                  if v['name'] in ('epoch', 'MaxIter', 'loss')}
                        if not all(k in values for k in ('epoch', 'MaxIter', 'loss')):
                            continue
                        completed = int(values['epoch'])
                        loss = ast.literal_eval(values['loss'])
                        if completed == previous or completed == 0:
                            continue
                        if previous is not None and completed > previous + 1:
                            print(f"Monitor skipped {completed-previous-1} updates; full history remains in checkpoints.", flush=True)
                        previous = completed
                        line = (f"[{time.strftime('%H:%M:%S')}] Completed: {completed}/{values['MaxIter']} "
                                f"Loss: {loss[0]:.6e} DataLoss: {loss[1]:.6e} RegLoss: {loss[2]:.6e}")
                        print(line, flush=True)
                        log.write(line + '\n')
            except (subprocess.SubprocessError, ValueError, KeyError, OSError):
                pass  # transient sampling failures; the training process is untouched
            time.sleep(1)
    print("Training process exited. See the training window and progress.log for the final status.", flush=True)


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


def plot_result(out, truth, initial, best, history):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout="constrained")
    extent = [0, truth.shape[1] * .015, truth.shape[0] * .015, 0]
    for ax, data, title in zip([axes[0, 0], axes[0, 1], axes[1, 0]],
                                [truth, initial, best],
                                ["Ground truth", "Initial model", "Best saved model"]):
        im = ax.imshow(data/1000, extent=extent, aspect="auto", cmap="RdBu_r",
                       vmin=1, vmax=4.7)
        ax.set(title=title, xlabel="Distance (km)", ylabel="Depth (km)")
        fig.colorbar(im, ax=ax, label="km/s", extend="both")
    axes[1, 1].semilogy(np.arange(1, len(history)+1), history[:, 1])
    axes[1, 1].set(title="Waveform loss", xlabel="Update (loss before update)", ylabel="MSE")
    fig.savefig(out / "result.png", dpi=160)
    plt.close(fig)


def run(mode):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=f"Original notebook: IFWI-{mode}")
    parser.add_argument("--epochs", type=int, default=1001 if mode == "pretrain" else 4001,
                        help="总更新数；续训时也表示总数，不是新增轮数")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--plots", action="store_true", help="可选：训练结束后输出图片，默认只保存数值")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="本套脚本生成的反演 checkpoint")
    if mode == "pretrain":
        parser.add_argument("--pretrained", type=Path,
                            default=root / "pretrain_output/ifwi_pretrain_marmousi.pth")
    if mode == "noisy":
        parser.add_argument("--noise", type=float, default=4., help="噪声标准差/观测标准差")
    if mode == "uncertainty":
        parser.add_argument("--dropout", type=float, default=.4)
        parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1 or args.log_interval < 1:
        parser.error("epochs 和 log-interval 必须为正整数")
    if mode == "noisy" and (not np.isfinite(args.noise) or args.noise < 0):
        parser.error("noise 必须是有限非负数")
    if mode == "uncertainty" and (not 0 <= args.dropout < 1 or args.samples < 2):
        parser.error("dropout 必须在 [0,1)，samples 至少为 2")
    pretrained = None
    if mode == "pretrain":
        pretrained = str(args.pretrained.resolve())
        if not Path(pretrained).is_file():
            parser.error("未找到预训练权重，请先运行 pretrain_marmousi.py")
        meta = torch.load(pretrained, map_location="cpu", weights_only=False)
        if (meta.get("mean"), meta.get("std")) != (3., 1.):
            parser.error("预训练权重必须注明 mean=3、std=1，与本实验一致")
    config = dict(mode=mode, seed=args.seed, mean=3., std=1., dz=15, dt=.0019, nt=1000,
                  frequency=8, source_depth_index=1, receiver_depth_index=2,
                  learning_rate=1e-4, alpha="auto" if mode == "regularization" else 0,
                  noise=getattr(args, "noise", 0), dropout=getattr(args, "dropout", 0),
                  epochs=args.epochs, log_interval=args.log_interval, pretrained=pretrained)
    resume = str(args.resume.resolve()) if args.resume else None
    if resume:
        config_path = Path(resume).parent.parent / "config.json"
        if not config_path.is_file():
            parser.error("续训需要 checkpoint 所属运行目录的 config.json，以核对数据与归一化")
        previous = json.loads(config_path.read_text(encoding="utf8"))
        for key in config:
            if key not in ("epochs", "log_interval", "pretrained") and previous.get(key) != config[key]:
                parser.error(f"续训配置不一致: {key}")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if args.epochs <= checkpoint["epoch"]:
            parser.error("epochs 必须大于 checkpoint 中已完成的轮数")
        del checkpoint
    out = (args.output_dir or root / "runs" / f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("输出目录非空，请指定新目录，避免覆盖结果")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf8")
    with (out / "progress.log").open("w", encoding="utf8", buffering=1) as log:
        with contextlib.redirect_stdout(LiveLog(sys.stdout, log)):
            try:
                _execute(args, config, out, pretrained, resume)
            except Exception as error:
                print(f"FAILED: {type(error).__name__}: {error}", flush=True)
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
    truth = np.array(pd.read_csv(root / "vel_marmousi_376x1151.csv"))[::4, ::4].astype(np.float32)
    vp = torch.from_numpy(truth[None]).to(args.device)
    nv, nz, nx = vp.shape
    xs = torch.arange(20, nx-10, 20, dtype=torch.long).repeat(nv, 1)
    ns = xs.shape[1]
    xr = torch.arange(nx, dtype=torch.long).repeat(nv, ns, 1)
    zs = torch.full((nv, ns), 1, dtype=torch.long)
    zr = torch.full((nv, ns, nx), 2, dtype=torch.long)
    t = .0019 * torch.arange(1000, dtype=torch.float32)
    wavelet = wGenerator(t, 8).ricker().to(args.device)
    common = dict(nz=nz, nx=nx, zs=zs, xs=xs, zr=zr, xr=xr, dz=15, dt=.0019,
                  npad=15, order=2, vmax=vp.max(), log_para=1e-6,
                  freeSurface=True, dtype=torch.float32, device=args.device)
    print(f"START {mode}: {args.epochs} updates, device={args.device}, output={out}", flush=True)
    print("Forward modeling: generating observed shots...", flush=True)
    forward = rnn2D(**common).to(args.device)
    with torch.no_grad():
        _, _, shots, _ = forward(vmodel=vp, segment_wavelet=wavelet)
    if mode == "noisy":
        # 保留 notebook 的 NumPy 噪声生成顺序与 float64 含噪观测。
        observed = shots.cpu().numpy()
        noise = np.random.randn(nv, ns, 1000, nx) * (args.noise * observed.std())
        shots = torch.from_numpy(observed + noise).to(args.device)
    model = IFWI2D(**common, mean=3., std=1., neuron=[2,128,128,128,128,1],
                   omega_0=30, prob=getattr(args, "dropout", .2), activation="sine",
                   bias=True, dropout=mode == "uncertainty", outermost_linear=True,
                   segment_size=1000, vpadding=None, pretrained=pretrained, netOpt="IFWI")
    # 诊断输出关闭 dropout，不消耗训练随机数；此后恢复 train 模式。
    if resume:
        model.load_state(resume, best=False)
    model.vel_net.eval()
    with torch.no_grad():
        initial, _ = model.vel_net(model.coords)
        initial = ((initial.squeeze()*model.std+model.mean)*1000).cpu().numpy()
    model.vel_net.train()
    np.save(out / "initial_velocity.npy", initial)
    np.save(out / "true_velocity.npy", truth)
    prefix = str(out / "checkpoints" / f"MarmousiI_{mode}-")
    completed = torch.load(resume, map_location="cpu", weights_only=False)['epoch'] if resume else 0
    history, _ = model.train(MaxIter=args.epochs, vmodel=None, wavelet=wavelet, shots=shots,
                             alpha=config["alpha"], option=0, learning_rate=1e-4,
                             log_interval=args.log_interval, wandb=EpochLog(completed),
                             resume_file_name=resume, save_file_name=prefix)
    history = np.asarray(history)
    np.savetxt(out / "loss.csv", np.column_stack([np.arange(1,len(history)+1), history]),
               delimiter=",", header="completed_updates,total_loss,data_loss,regularization_statistic", comments="")
    checkpoint = prefix + f"checkpoint-{args.epochs}.pth"
    best, _ = model.predict(resume_file_name=checkpoint, best=True)
    best = best.squeeze().cpu().numpy()
    np.save(out / "best_velocity.npy", best)
    if args.plots:
        plot_result(out, truth, initial, best, history)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read-only live monitor for an existing IFWI process")
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--watch-log", type=Path, required=True)
    options = parser.parse_args()
    watch_progress(options.watch_pid, options.watch_log)
