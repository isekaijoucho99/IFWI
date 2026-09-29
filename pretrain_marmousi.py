"""平滑 Marmousi 模型的 SIREN 预训练；原作者模块不作修改。

运行：python pretrain_marmousi.py --epochs 5000
论文设置：mean=3、std=1 km/s，Adam lr=1e-4。
目标处理和网络结构沿用公开 notebook；5000 轮是工程预算，不是论文规定。
输出到本脚本旁的 weights；后续 IFWI 必须同样使用 mean=3、std=1。
这是补充实现，不是作者缺失的原始预训练脚本。
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

from ifwi_modules import IFWI2D


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=root / "weights")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 必须大于 0")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    weight_path = out / "ifwi_pretrain_marmousi.pth"
    if any((out / name).exists() for name in
           (weight_path.name, "loss.csv", "pretrain.png")):
        parser.error("输出文件已存在，请用 --output-dir 指定新目录，避免覆盖结果。")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 保留 notebook 的默认 CSV 表头处理、先平滑后每 4 点采样。
    smooth = np.array(pd.read_csv(root / "data" / "vel_marmousi_smooth400_376x1151.csv"))
    target_mps = gaussian_filter(smooth, sigma=50)[::4, ::4].astype(np.float32)
    nz, nx = target_mps.shape
    model = IFWI2D(mean=3.0, std=1.0, neuron=[2, 128, 128, 128, 128, 1],
                   omega_0=30, activation="sine", bias=True, dropout=False,
                   outermost_linear=True, nz=nz, nx=nx, dz=15,
                   device=args.device, netOpt="IRN")
    target = torch.from_numpy((target_mps / 1000 - model.mean) / model.std)
    target = target.to(args.device)[None, :, :, None]
    optimizer = torch.optim.Adam(model.vel_net.parameters(), lr=1e-4)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    print(f"device={args.device}, grid={nz}x{nx}, mean=3, std=1 km/s", flush=True)
    # IRN 的公共 train() 中存在未定义的 iseg，无法直接用于预训练。
    # 因此在本文件使用同一个 vel_net/coords 和标准 MSE、Adam 循环。
    # 直接取归一化网络输出；不调用会反归一化输出的 forward_process()。
    for epoch in range(args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        pred, _ = model.vel_net(model.coords)
        loss = (pred - target).square().mean()
        value = loss.item()
        if not np.isfinite(value):
            raise RuntimeError(f"第 {epoch} 轮 loss 非有限值，停止预训练。")
        history.append((epoch, value, np.sqrt(value) * 1000 * model.std))
        # 此处 loss 与保存的权重属于同一时刻；epoch 是已完成的更新数。
        if value < best_loss:
            best_loss, best_epoch = value, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.vel_net.state_dict().items()}
        if epoch % 100 == 0 or epoch == args.epochs:
            print(f"epoch={epoch}/{args.epochs} loss={value:.6e} "
                  f"RMSE={history[-1][2]:.2f} m/s", flush=True)
        if epoch < args.epochs:
            loss.backward()
            optimizer.step()

    torch.save(dict(state_dict=best_state, epoch=best_epoch, loss=best_loss,
                    mean=3.0, std=1.0, dz=15, seed=args.seed,
                    neuron=[2, 128, 128, 128, 128, 1], omega_0=30,
                    learning_rate=1e-4, requested_updates=args.epochs), weight_path)
    model.vel_net.load_state_dict(best_state)
    with torch.no_grad():
        pred, _ = model.vel_net(model.coords)
        fitted = (pred.squeeze() * model.std + model.mean).cpu().numpy() * 1000
    with (out / "loss.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["completed_updates", "normalized_mse", "velocity_rmse_mps"])
        writer.writerows(history)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout="constrained")
    extent = [0, nx * .015, nz * .015, 0]
    for ax, data, title in zip(axes[0], [target_mps, fitted],
                                ["Smooth target", f"Best SIREN: update {best_epoch}"]):
        im = ax.imshow(data / 1000, extent=extent, aspect="auto", cmap="RdBu_r",
                       vmin=target_mps.min()/1000, vmax=target_mps.max()/1000)
        ax.set(title=title, xlabel="Distance (km)", ylabel="Depth (km)")
        fig.colorbar(im, ax=ax, label="km/s")
    error = fitted - target_mps
    limit = max(float(np.abs(error).max()), 1e-6)
    im = axes[1, 0].imshow(error, extent=extent, aspect="auto", cmap="RdBu_r",
                            vmin=-limit, vmax=limit)
    axes[1, 0].set(title="Fit minus target", xlabel="Distance (km)", ylabel="Depth (km)")
    fig.colorbar(im, ax=axes[1, 0], label="m/s")
    axes[1, 1].semilogy(np.array(history)[:, 0], np.array(history)[:, 1])
    axes[1, 1].set(xlabel="Completed updates", ylabel="Normalized MSE", title="Pretraining loss")
    fig.savefig(out / "pretrain.png", dpi=160)
    plt.close(fig)
    print(f"Saved: {weight_path}\nBest RMSE: {np.sqrt(best_loss)*1000:.2f} m/s\n"
          "Load with IFWI2D(mean=3.0, std=1.0, pretrained=<this path>, ...).", flush=True)


if __name__ == "__main__":
    main()
