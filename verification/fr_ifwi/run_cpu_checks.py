"""Run repository CPU checks using the reference convolution backend.

The legacy suite asserts bitwise equality across grouped-convolution batch sizes.
PyTorch 2.10 oneDNN need not preserve those last bits. Disable it in THIS test
process, not in the training runner. CUDA is hidden to avoid touching an active
local GPU experiment. Extra arguments are passed to pytest.
"""
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    import torch
    import pytest
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    raise SystemExit(pytest.main(sys.argv[1:] or ["-q"]))
