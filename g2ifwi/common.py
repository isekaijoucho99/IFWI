import numpy as np
import torch


def pick_device(name=None):
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def metrics(v, v_true):
    """RMSE (m/s), PSNR (dB) and SSIM, Eqs. (40)-(41)."""
    from skimage.metrics import structural_similarity
    v, v_true = np.asarray(v, np.float64), np.asarray(v_true, np.float64)
    mse = np.mean((v - v_true) ** 2)
    return dict(rmse=float(np.sqrt(mse)), psnr=float(10 * np.log10(v_true.max() ** 2 / mse)),
                ssim=float(structural_similarity(v, v_true, data_range=v_true.max() - v_true.min())))
