"""Measure actual recorded energy retained by the proposed low-pass stages."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dt', type=float, default=.0019)
    args = parser.parse_args()
    observed = np.load(args.observed, allow_pickle=False)
    if observed.ndim != 4 or not np.isfinite(observed).all() or args.dt <= 0:
        raise ValueError('Expected finite [batch,shot,time,receiver] observations and positive dt')
    n = observed.shape[2]
    transformed = np.fft.rfft(observed, n=2*n, axis=2)
    frequency = np.fft.rfftfreq(2*n, d=args.dt)
    energy = float(np.mean(observed.astype(np.float64)**2))
    if not energy > 0:
        raise ValueError('Zero observed energy')
    rows = []
    for cutoff in (5., 8.):
        response = np.exp(-.5*(frequency/cutoff)**8)
        filtered = np.fft.irfft(transformed*response[None,None,:,None], n=2*n, axis=2)[:,:,:n,:]
        rows.append({'cutoff_hz':cutoff, 'retained_observed_energy_fraction':float(np.mean(filtered**2)/energy)})
    result = {'observed_file':str(args.observed), 'dt':args.dt, 'shape':list(observed.shape),
              'observed_mean_square':energy, 'stages':rows,
              'interpretation':'Energy support only; not an SNR, depth sensitivity or recoverability estimate.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
