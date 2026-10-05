"""Read-only observer: preserve selected atomic training checkpoints for analysis."""
import argparse
import io
import json
import time
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--updates', type=int, nargs='+', default=[30, 70, 100])
    args = parser.parse_args()
    torch.set_num_threads(1)
    captured = set()
    while True:
        for path in sorted((args.suite/'runs').glob('*/last.pth')):
            # Atomic replacement means this byte string belongs to one whole checkpoint.
            try:
                raw = path.read_bytes()
                checkpoint = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
            except (FileNotFoundError, PermissionError):
                continue
            step = checkpoint['completed_updates']
            key = (path.parent.name, step)
            if step in args.updates and key not in captured:
                out = args.suite/'stage_snapshots'/path.parent.name
                out.mkdir(parents=True, exist_ok=True)
                (out/f'update{step:06d}.pth').write_bytes(raw)
                captured.add(key)
                print(f'Captured {path.parent.name}: {step}', flush=True)
        try:
            status = json.loads((args.suite/'active.json').read_text(encoding='utf-8'))
        except (FileNotFoundError, PermissionError):
            status = {}
        if status.get('state') in ('completed', 'failed'):
            break
        time.sleep(2)
    (args.suite/'stage_capture.json').write_text(json.dumps({
        'requested_updates': args.updates, 'captured': sorted(captured),
        'state': status.get('state')}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
