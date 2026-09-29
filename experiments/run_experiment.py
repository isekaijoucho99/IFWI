"""
Main Experiment Runner for Improved IFWI
----------------------------------------
Unified entry point for running experiments with different configurations.

Usage:
    python run_experiment.py --config configs/baseline.yaml
    python run_experiment.py --config configs/attention.yaml --device cuda:0
    python run_experiment.py --config configs/combined_best.yaml --resume results/exp_001/checkpoint.pth
"""

import os
import sys
import argparse
import yaml
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add parent directory to path to import IFWI modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ifwi_modules import IFWI2D, IRN
from rnn_fd import rnn2D
from generator import wGenerator

# Import improved modules
from improved_modules.losses import CombinedLoss, DepthWeightedLoss
from improved_modules.networks import create_improved_network, AttentionIRN
from improved_modules.optimizers import create_improved_optimizer
from improved_modules.evaluate_deep import evaluate_deep_layers, visualize_deep_improvement


class ImprovedIFWI(IFWI2D):
    """
    Extended IFWI2D with support for improved modules.
    """
    def __init__(self, *args, improved_config=None, **kwargs):
        # Store config before calling parent init
        self.improved_config = improved_config or {}

        # Initialize parent
        super().__init__(*args, **kwargs)

        # Replace velocity network if using improved version
        if improved_config and improved_config.get('model', {}).get('network_type') != 'vanilla':
            model_config = improved_config['model']
            model_config['neuron'] = kwargs.get('neuron', [2, 256, 256, 256, 256, 1])
            model_config['omega_0'] = kwargs.get('omega_0', 30)
            model_config['activation'] = kwargs.get('activation', 'sine')
            model_config['dropout'] = kwargs.get('dropout', False)
            model_config['prob'] = kwargs.get('prob', 0.2)

            self.vel_net = create_improved_network(model_config, device=kwargs['device'])

        # Create improved loss function
        loss_config = improved_config.get('loss', {})
        if loss_config.get('type') in ['combined', 'depth_weighted']:
            self.loss_fn = CombinedLoss(
                nz=kwargs['nz'],
                nx=kwargs['nx'],
                config=loss_config
            )
        else:
            self.loss_fn = None

        # Initialize gradient modifier
        self.grad_modifier = None

    def train_one_epoch(self, optimizer, vmodel=None, wavelet=None, shots=None, trade_off=0, option=0):
        """
        Override to use improved loss and gradient modification.
        """
        if self.netOpt == 'IRN':
            optimizer.zero_grad()
            vpred, _, _, _, _ = self.forward_process(None, None, None, None, option)
            loss = ((vpred - vmodel)**2).mean()
            loss.backward()
            optimizer.step()
            return vpred.detach(), [loss.item(), loss.item(), 0]

        # Standard IFWI training with improvements
        shots = shots.to(self.device)
        prev_state = torch.zeros([shots.shape[0], shots.shape[1], self.nz_pad, self.nx_pad],
                                 dtype=self.dtype).to(self.device)
        curr_state = torch.zeros([shots.shape[0], shots.shape[1], self.nz_pad, self.nx_pad],
                                 dtype=self.dtype).to(self.device)

        loss_total = 0
        loss_segAll = 0
        loss_regAll = 0

        from generator import gen_Segment2d

        for iseg, (segWavelet, segData) in enumerate(gen_Segment2d(wavelet, shots,
                                                                    segment_size=self.segment_size,
                                                                    option=option)):
            optimizer.zero_grad()
            vpred, vgrad, shot_segPred, prev_state, curr_state = self.forward_process(
                vmodel, segWavelet, prev_state, curr_state, option
            )

            from ifwi_modules import repackage_hidden
            prev_state = repackage_hidden(prev_state)
            curr_state = repackage_hidden(curr_state)

            # Use improved loss if available
            if self.loss_fn is not None:
                loss, loss_dict = self.loss_fn(
                    shot_segPred, segData, vpred,
                    prior_info={'deep_velocity_range': (1500, 5500)},
                    dt=0.0019
                )
                loss_Seg = loss_dict['data_loss']
                loss_Reg = loss_dict['prior_loss']
            else:
                # Standard loss
                loss_Seg = ((shot_segPred - segData)**2).sum() / (
                    shots.shape[0] * shots.shape[1] * shots.shape[-2] * shots.shape[-1]
                )
                if self.reg_op == "TV":
                    loss_Reg = (vgrad**2 + 1e-6).sqrt().mean()
                else:
                    loss_Reg = 0
                loss = loss_Seg + trade_off * loss_Reg

            loss.backward()

            # Apply gradient modification if available
            if self.grad_modifier is not None:
                for param in self.params:
                    if param.grad is not None:
                        param.grad.data = self.grad_modifier(param.grad.data)

            torch.nn.utils.clip_grad_norm_(self.params, self.clip)
            optimizer.step()

            loss_total += loss.detach().cpu().item()
            loss_segAll += loss_Seg.detach().cpu().item()
            loss_regAll += (loss_Reg.detach().cpu().item() if torch.is_tensor(loss_Reg)
                           else loss_Reg)

        return vpred.detach(), [loss_total, loss_segAll, loss_regAll / (iseg + 1)]


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment(config, args):
    """Setup experiment directories and logging."""
    # Create output directory
    exp_name = config['experiment_name']
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    print(f"Experiment: {exp_name}")
    print(f"Output directory: {output_dir}")

    return output_dir


def prepare_data(config, device):
    """Prepare training data."""
    data_config = config['data']

    # Load true velocity model
    data_path = Path(__file__).parent.parent / 'data' / data_config['model_file']
    truth = np.array(pd.read_csv(data_path, header=0))

    # Downsample
    downsample = data_config.get('downsample', 4)
    truth = truth[::downsample, ::downsample].astype(np.float32)

    vp = torch.from_numpy(truth[None]).to(device)
    nv, nz, nx = vp.shape

    # Setup acquisition geometry
    xs = torch.arange(20, nx-10, 20, dtype=torch.long).repeat(nv, 1)
    ns = xs.shape[1]
    xr = torch.arange(nx, dtype=torch.long).repeat(nv, ns, 1)
    zs = torch.full((nv, ns), 1, dtype=torch.long)  # source_depth_index
    zr = torch.full((nv, ns, nx), 2, dtype=torch.long)  # receiver_depth_index

    # Generate wavelet
    dt = data_config['dt']
    nt = data_config['nt']
    t = dt * torch.arange(nt, dtype=torch.float32)
    wavelet = wGenerator(t, 8).ricker().to(device)

    # Forward modeling to generate observed data
    print("Generating observed shots...")
    forward = rnn2D(
        nz=nz, nx=nx, zs=zs, xs=xs, zr=zr, xr=xr,
        dz=data_config['dz'], dt=dt,
        npad=15, order=2, vmax=vp.max(), log_para=1e-6,
        freeSurface=True, dtype=torch.float32, device=device
    ).to(device)

    with torch.no_grad():
        _, _, shots, _ = forward(vmodel=vp, segment_wavelet=wavelet)

    # Add noise if specified
    noise_level = data_config.get('noise_level', 0.0)
    if noise_level > 0:
        noise = torch.randn_like(shots) * (noise_level * shots.std())
        shots = shots + noise
        print(f"Added noise: {noise_level * 100:.1f}% of signal std")

    return {
        'vp_true': vp,
        'shots': shots,
        'wavelet': wavelet,
        'geometry': {'nz': nz, 'nx': nx, 'zs': zs, 'xs': xs, 'zr': zr, 'xr': xr},
        'params': {'dz': data_config['dz'], 'dt': dt, 'nt': nt}
    }


def train_model(model, data, config, output_dir, args):
    """Main training loop."""
    training_config = config['training']

    # Setup optimizer
    optimizer_config = config['optimizer']
    optimizer_config['nz'] = data['geometry']['nz']

    if optimizer_config['optimizer_type'] == 'depth_adaptive':
        optimizer, scheduler, grad_modifier = create_improved_optimizer(
            model.vel_net, optimizer_config
        )
    else:
        optimizer, scheduler, grad_modifier = create_improved_optimizer(
            model.vel_net.parameters(), optimizer_config
        )

    # Attach gradient modifier to model
    model.grad_modifier = grad_modifier

    # Training history
    history = []
    best_loss = float('inf')
    best_model = None

    max_iter = training_config['max_iterations']
    log_interval = training_config['log_interval']

    print(f"\nStarting training for {max_iter} iterations...")
    start_time = time.time()

    for epoch in range(max_iter):
        # Train one epoch
        _, losses = model.train_one_epoch(
            optimizer,
            vmodel=None,
            wavelet=data['wavelet'],
            shots=data['shots'],
            trade_off=training_config.get('alpha', 0),
            option=0
        )

        history.append(losses)

        # Update learning rate if scheduler exists
        if scheduler is not None:
            scheduler.step(epoch)

        # Log progress
        if epoch % log_interval == 0 or epoch == max_iter - 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:5d}/{max_iter} | "
                  f"Loss: {losses[0]:.4e} | "
                  f"Data: {losses[1]:.4e} | "
                  f"Reg: {losses[2]:.4e} | "
                  f"Time: {elapsed:.1f}s")

            # Save checkpoint
            if losses[0] < best_loss:
                best_loss = losses[0]
                best_model = {
                    'epoch': epoch,
                    'state_dict': model.vel_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': best_loss
                }

            torch.save({
                'epoch': epoch,
                'state_dict': model.vel_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_model': best_model,
                'history': history
            }, output_dir / f'checkpoint_epoch{epoch}.pth')

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")

    return history, best_model


def evaluate_model(model, data, config, output_dir):
    """Evaluate model on deep layer metrics."""
    eval_config = config['evaluation']

    # Get predictions
    model.vel_net.eval()
    with torch.no_grad():
        v_pred, _ = model.predict()

    v_pred = v_pred.squeeze().cpu().numpy()
    v_true = data['vp_true'].squeeze().cpu().numpy()

    # Compute metrics
    metrics = evaluate_deep_layers(
        v_true, v_pred,
        depth_threshold=eval_config['depth_threshold'],
        corner_size=eval_config['corner_size'],
        dz=data['params']['dz']
    )

    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in metrics.items()
        }
        json.dump(metrics_serializable, f, indent=2)

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Deep Layer Relative Error: {metrics['deep_relative_error']*100:.2f}%")
    print(f"Deep Layer RMSE: {metrics['deep_rmse']:.1f} m/s")
    print(f"Deep Layer SSIM: {metrics['deep_ssim']:.4f}")
    print(f"Bottom-Left Error: {metrics['bottom_left_error']*100:.2f}%")
    print(f"Bottom-Right Error: {metrics['bottom_right_error']*100:.2f}%")
    print(f"Deep Quality Score: {metrics['deep_quality_score']:.4f}")
    print("="*50)

    # Save velocity models
    np.save(output_dir / 'v_true.npy', v_true)
    np.save(output_dir / 'v_pred.npy', v_pred)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run IFWI experiment with improvements')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load configuration
    config = load_config(args.config)

    # Setup experiment
    output_dir = setup_experiment(config, args)

    # Prepare data
    data = prepare_data(config, args.device)

    # Create model
    geom = data['geometry']
    params = data['params']
    model_config = config['model']

    model = ImprovedIFWI(
        improved_config=config,
        mean=3.0,
        std=1.0,
        neuron=model_config['neuron'],
        omega_0=model_config['omega_0'],
        prob=model_config.get('prob', 0.2),
        activation=model_config['activation'],
        bias=True,
        dropout=model_config.get('dropout', False),
        outermost_linear=True,
        nz=geom['nz'],
        nx=geom['nx'],
        zs=geom['zs'],
        xs=geom['xs'],
        zr=geom['zr'],
        xr=geom['xr'],
        dz=params['dz'],
        dt=params['dt'],
        npad=15,
        order=2,
        vmax=data['vp_true'].max(),
        log_para=1e-6,
        segment_size=params['nt'],
        vpadding=None,
        freeSurface=True,
        regularization="TV",
        dtype=torch.float32,
        device=args.device,
        netOpt='IFWI'
    )

    # Train model
    history, best_model = train_model(model, data, config, output_dir, args)

    # Save training history
    history_array = np.array(history)
    np.savetxt(output_dir / 'loss_history.csv',
              history_array,
              delimiter=',',
              header='total_loss,data_loss,reg_loss',
              comments='')

    # Load best model for evaluation
    if best_model:
        model.vel_net.load_state_dict(best_model['state_dict'])

    # Evaluate
    metrics = evaluate_model(model, data, config, output_dir)

    print(f"\nResults saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
