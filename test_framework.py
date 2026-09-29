"""
Quick Test Script for IFWI Improvements Framework
-------------------------------------------------
Tests basic functionality without running full training.

Usage:
    python test_framework.py
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing IFWI Improvements Framework...")
print("=" * 50)

# Test 1: Import improved modules
print("\n[1/6] Testing imports...")
try:
    from experiments.improved_modules.losses import DepthWeightedLoss, CombinedLoss
    from experiments.improved_modules.networks import AttentionIRN, create_improved_network
    from experiments.improved_modules.optimizers import create_improved_optimizer
    from experiments.improved_modules.evaluate_deep import evaluate_deep_layers
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create mock data
print("\n[2/6] Creating test data...")
try:
    nz, nx = 94, 288
    v_true = torch.randn(nz, nx) * 500 + 3000  # Random velocity ~3000 m/s
    v_pred = v_true + torch.randn(nz, nx) * 200  # Add some error
    print(f"✓ Created test velocity models: {v_true.shape}")
except Exception as e:
    print(f"✗ Data creation failed: {e}")
    sys.exit(1)

# Test 3: Test loss functions
print("\n[3/6] Testing loss functions...")
try:
    # Depth weighted loss
    depth_loss = DepthWeightedLoss(nz, weight_type='piecewise')

    # Create fake shot data
    data_pred = torch.randn(1, 10, 100, nx)
    data_obs = data_pred + torch.randn_like(data_pred) * 0.1

    loss = depth_loss(data_pred, data_obs)
    print(f"✓ Depth-weighted loss computed: {loss.item():.4e}")

    # Combined loss
    loss_config = {
        'use_depth_weight': True,
        'use_prior': True,
        'depth_weight_params': {'weight_type': 'piecewise'},
        'prior_params': {'deep_start': 0.5},
        'lambda_prior': 0.1
    }
    combined_loss = CombinedLoss(nz, nx, loss_config)
    loss, loss_dict = combined_loss(data_pred, data_obs, v_true[None, :, :])
    print(f"✓ Combined loss computed: {loss.item():.4e}")

except Exception as e:
    print(f"✗ Loss function test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test network architectures
print("\n[4/6] Testing network architectures...")
try:
    # Test attention network
    config = {
        'network_type': 'attention',
        'neuron': [2, 64, 64, 1],
        'omega_0': 30,
        'activation': 'sine',
        'dropout': False,
        'use_attention': True,
        'attention_hidden': 32,
        'deep_bias': 2.0
    }

    net = create_improved_network(config, device='cpu')

    # Test forward pass
    coords = torch.randn(1, nz, nx, 2)
    output, _ = net(coords)
    print(f"✓ Attention network forward pass: input {coords.shape} -> output {output.shape}")

except Exception as e:
    print(f"✗ Network test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test optimizers
print("\n[5/6] Testing optimizers...")
try:
    optimizer_config = {
        'optimizer_type': 'depth_adaptive',
        'learning_rate': 1e-4,
        'lr_deep': 5e-4,
        'deep_threshold': 0.5,
        'base_optimizer': 'Adam',
        'use_scheduler': True,
        'scheduler_params': {
            'warmup_epochs': 10,
            'max_epochs': 100,
            'eta_min': 1e-6
        },
        'use_grad_modifier': False
    }

    optimizer, scheduler, grad_modifier = create_improved_optimizer(net, optimizer_config)
    print(f"✓ Created optimizer: {type(optimizer).__name__}")
    if scheduler:
        print(f"✓ Created scheduler")

except Exception as e:
    print(f"✗ Optimizer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test evaluation metrics
print("\n[6/6] Testing evaluation metrics...")
try:
    v_true_np = v_true.cpu().numpy()
    v_pred_np = v_pred.cpu().numpy()

    metrics = evaluate_deep_layers(v_true_np, v_pred_np, depth_threshold=0.5)

    print(f"✓ Computed metrics:")
    print(f"  - Deep relative error: {metrics['deep_relative_error']*100:.2f}%")
    print(f"  - Deep SSIM: {metrics['deep_ssim']:.4f}")
    print(f"  - Bottom-left error: {metrics['bottom_left_error']*100:.2f}%")
    print(f"  - Bottom-right error: {metrics['bottom_right_error']*100:.2f}%")

except Exception as e:
    print(f"✗ Evaluation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("✓ All tests passed!")
print("=" * 50)
print("\nFramework is ready to use.")
print("Next steps:")
print("  1. Run a full experiment: python experiments/run_experiment.py --config experiments/configs/baseline.yaml")
print("  2. Or run all experiments: bash experiments/run_all_experiments.sh")
