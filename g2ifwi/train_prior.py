"""Train the DDPM velocity prior on procedurally generated models (Table 2).

python -m g2ifwi.train_prior --preset small --steps 20000 --out weights/prior_small.pt
python -m g2ifwi.train_prior --preset paper --steps 250000 --batch 16   # 40000 crops x 100 epochs / 16 (A100-scale)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .common import pick_device
from .diffusion import DDPM, PRESETS, UNet
from .velocity_gen import generate_model, normalize, random_crop


class CropSampler:
    """Infinite stream of normalised 128x128 crops; each base model (152x708) yields several crops."""

    def __init__(self, seed, size=128, crops_per_model=20, nz=152, nx=708):
        self.rng, self.size, self.k, self.shape = np.random.default_rng(seed), size, crops_per_model, (nz, nx)
        self.model, self.left = None, 0

    def batch(self, n):
        out = []
        for _ in range(n):
            if self.left == 0:
                self.model, self.left = generate_model(*self.shape, rng=self.rng), self.k
            self.left -= 1
            out.append(normalize(random_crop(self.model, self.size, self.rng))[0])
        return torch.from_numpy(np.stack(out)[:, None]).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="small")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device")
    ap.add_argument("--out", type=Path, default=Path("weights/prior.pt"))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = pick_device(a.device)
    cfg = dict(PRESETS[a.preset])
    lr = a.lr or cfg.pop("lr")
    cfg.pop("lr", None)
    torch.manual_seed(a.seed)
    net, ddpm = UNet(**cfg).to(dev), DDPM(device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    step = 0
    if a.resume and a.out.exists():
        ck = torch.load(a.out, map_location=dev)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    print(f"device={dev} params={sum(p.numel() for p in net.parameters())/1e6:.1f}M lr={lr}", flush=True)
    data, t0, ema = CropSampler(a.seed + step), time.time(), None
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), step=step, cfg=cfg), a.out)

    while step < a.steps:
        loss = ddpm.loss(net, data.batch(a.batch).to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 100 == 0:
            print(f"step {step} loss {ema:.4f} {time.time()-t0:.0f}s", flush=True)
        if step % 1000 == 0:
            save()
    save()


if __name__ == "__main__":
    main()
