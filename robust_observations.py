"""Reproducible legacy-amplitude noise; immutable observation cache per condition."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def array_hash(a):
    a = np.ascontiguousarray(a)
    header = json.dumps({'shape':list(a.shape),'dtype':str(a.dtype)},sort_keys=True).encode()
    return hashlib.sha256(header+a.tobytes()).hexdigest()


def make_observations(clean, sigma, seed):
    clean = np.asarray(clean)
    if clean.ndim != 4 or not np.isfinite(clean).all():
        raise ValueError('Clean observations must be finite [B,S,T,R]')
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError('Noise sigma must be finite and nonnegative')
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('Noise seed must be a nonnegative integer')
    # Match historical np.random.randn * (sigma * observed.std()), preserving
    # float32 std and float64 noisy observations without touching global RNG.
    std = clean.std()
    if sigma and std <= 0:
        raise ValueError('Nonzero noise needs positive clean observation std')
    noise = np.random.RandomState(seed).randn(*clean.shape) * (sigma * std) if sigma else np.zeros_like(clean)
    observed = clean + noise
    c64 = clean.astype(np.float64)
    p_signal, p_noise = float(np.mean(c64**2)), float(np.mean(noise**2))
    info = {'sigma':float(sigma),'noise_seed':seed,
        'sigma_definition':'noise_std / clean_observation_std',
        'noise_generator':'numpy.RandomState(seed).randn; independent from network RNG',
        'clean_std_for_generation':float(std),'requested_noise_std':float(sigma*std),
        'realized_noise_std':float(noise.std()),'realized_noise_mean':float(noise.mean()),
        'realized_std_ratio':float(noise.std()/std) if std else 0.,
        'snr_db_signal_power':float(10*np.log10(p_signal/p_noise)) if p_signal and p_noise else None,
        'clean_hash':array_hash(clean),'noise_hash':array_hash(noise),'observed_hash':array_hash(observed),
        'clean_dtype':str(clean.dtype),'observed_dtype':str(observed.dtype),
        'shape':list(clean.shape),'numpy_version':np.__version__}
    return observed, noise, info


def shared_observations(clean, sigma, seed, directory, identity):
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    manifest = directory/'observation_manifest.json'
    if manifest.exists():
        info = json.loads(manifest.read_text(encoding='utf-8'))
        if (info['identity'] != identity or info['sigma'] != float(sigma)
                or info['noise_seed'] != seed or info['clean_hash'] != array_hash(clean)):
            raise ValueError('Shared observation identity or clean data mismatch')
        arrays = {}
        for name, key in [('clean_observed','clean_hash'),('noise','noise_hash'),('observed','observed_hash')]:
            arrays[name] = np.load(directory/(name+'.npy'),allow_pickle=False)
            if array_hash(arrays[name]) != info[key]:
                raise ValueError('Shared observation hash mismatch: '+name)
        return arrays['observed'], info
    if any((directory/(name+'.npy')).exists() for name in ['clean_observed','noise','observed']):
        raise ValueError('Incomplete observation cache; inspect rather than overwrite')
    observed, noise, info = make_observations(clean,sigma,seed)
    info['identity'] = identity
    for name, values in [('clean_observed',clean),('noise',noise),('observed',observed)]:
        np.save(directory/(name+'.npy'),values)
    tmp = directory/'observation_manifest.json.tmp'
    tmp.write_text(json.dumps(info,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,manifest)
    return observed,info
