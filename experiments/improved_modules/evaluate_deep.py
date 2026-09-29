"""
Deep Layer Evaluation Metrics for IFWI
--------------------------------------
Specialized metrics focusing on deep layer inversion quality.
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


def evaluate_deep_layers(v_true, v_pred, depth_threshold=0.5,
                         corner_size=0.25, dz=15):
    """
    Comprehensive evaluation of deep layer inversion quality.

    Args:
        v_true: ground truth velocity [nz, nx] in m/s
        v_pred: predicted velocity [nz, nx] in m/s
        depth_threshold: fraction of depth where "deep" starts
        corner_size: fraction for corner regions
        dz: grid spacing in meters

    Returns:
        metrics: dict with detailed metrics
    """
    if torch.is_tensor(v_true):
        v_true = v_true.cpu().numpy()
    if torch.is_tensor(v_pred):
        v_pred = v_pred.cpu().numpy()

    nz, nx = v_true.shape
    metrics = {}

    # === 1. Deep Layer Metrics ===
    deep_start = int(nz * depth_threshold)
    v_true_deep = v_true[deep_start:, :]
    v_pred_deep = v_pred[deep_start:, :]

    # Relative error
    metrics['deep_relative_error'] = float(
        np.mean(np.abs(v_pred_deep - v_true_deep) / (v_true_deep + 1e-8))
    )

    # RMSE
    metrics['deep_rmse'] = float(
        np.sqrt(np.mean((v_pred_deep - v_true_deep) ** 2))
    )

    # SSIM (structural similarity)
    data_range = v_true_deep.max() - v_true_deep.min()
    metrics['deep_ssim'] = float(
        ssim(v_true_deep, v_pred_deep, data_range=data_range)
    )

    # Peak SNR
    mse = np.mean((v_true_deep - v_pred_deep) ** 2)
    if mse > 0:
        metrics['deep_psnr'] = float(
            20 * np.log10(v_true_deep.max() / np.sqrt(mse))
        )
    else:
        metrics['deep_psnr'] = float('inf')

    # === 2. Corner Region Metrics ===
    quarter_z = int(nz * corner_size)
    quarter_x = int(nx * corner_size)

    # Bottom-left corner
    bl_true = v_true[-quarter_z:, :quarter_x]
    bl_pred = v_pred[-quarter_z:, :quarter_x]
    metrics['bottom_left_error'] = float(
        np.mean(np.abs(bl_pred - bl_true) / (bl_true + 1e-8))
    )
    metrics['bottom_left_rmse'] = float(
        np.sqrt(np.mean((bl_pred - bl_true) ** 2))
    )

    # Bottom-right corner
    br_true = v_true[-quarter_z:, -quarter_x:]
    br_pred = v_pred[-quarter_z:, -quarter_x:]
    metrics['bottom_right_error'] = float(
        np.mean(np.abs(br_pred - br_true) / (br_true + 1e-8))
    )
    metrics['bottom_right_rmse'] = float(
        np.sqrt(np.mean((br_pred - br_true) ** 2))
    )

    # === 3. Depth-wise Profile ===
    # Error as a function of depth
    depth_errors = []
    for iz in range(nz):
        layer_error = np.mean(np.abs(v_pred[iz, :] - v_true[iz, :]) /
                             (v_true[iz, :] + 1e-8))
        depth_errors.append(layer_error)

    metrics['depth_profile'] = np.array(depth_errors)
    metrics['max_depth_error'] = float(np.max(depth_errors[deep_start:]))
    metrics['mean_depth_error'] = float(np.mean(depth_errors[deep_start:]))

    # === 4. Gradient Fidelity (Detail Recovery) ===
    # Vertical gradient (important for layer boundaries)
    grad_true_z = np.gradient(v_true_deep, axis=0)
    grad_pred_z = np.gradient(v_pred_deep, axis=0)

    grad_error = np.abs(grad_true_z - grad_pred_z) / (np.abs(grad_true_z) + 1e-8)
    metrics['deep_gradient_fidelity'] = float(1 - np.mean(grad_error))

    # === 5. Interface Detection ===
    # How well are sharp velocity contrasts recovered?
    threshold = 200  # m/s change defines an interface
    interfaces_true = np.abs(grad_true_z) > threshold
    interfaces_pred = np.abs(grad_pred_z) > threshold

    if interfaces_true.any():
        # Precision and recall for interface detection
        tp = np.sum(interfaces_true & interfaces_pred)
        fp = np.sum(~interfaces_true & interfaces_pred)
        fn = np.sum(interfaces_true & ~interfaces_pred)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        metrics['interface_precision'] = float(precision)
        metrics['interface_recall'] = float(recall)
        metrics['interface_f1'] = float(
            2 * precision * recall / (precision + recall + 1e-8)
        )
    else:
        metrics['interface_precision'] = 1.0
        metrics['interface_recall'] = 1.0
        metrics['interface_f1'] = 1.0

    # === 6. Overall Quality Score ===
    # Weighted combination emphasizing deep layers
    quality_score = (
        0.4 * (1 - metrics['deep_relative_error']) +
        0.2 * metrics['deep_ssim'] +
        0.2 * metrics['deep_gradient_fidelity'] +
        0.1 * (1 - metrics['bottom_left_error']) +
        0.1 * (1 - metrics['bottom_right_error'])
    )
    metrics['deep_quality_score'] = float(np.clip(quality_score, 0, 1))

    return metrics


def compare_experiments(results_dict, depth_threshold=0.5):
    """
    Compare multiple experiments focusing on deep layer performance.

    Args:
        results_dict: dict of {exp_name: metrics_dict}
        depth_threshold: depth fraction for "deep" definition

    Returns:
        comparison: formatted comparison table
    """
    import pandas as pd

    # Extract key metrics for comparison
    comparison_data = []

    for exp_name, metrics in results_dict.items():
        row = {
            'Experiment': exp_name,
            'Deep Error (%)': metrics['deep_relative_error'] * 100,
            'Deep RMSE (m/s)': metrics['deep_rmse'],
            'Deep SSIM': metrics['deep_ssim'],
            'BL Corner (%)': metrics['bottom_left_error'] * 100,
            'BR Corner (%)': metrics['bottom_right_error'] * 100,
            'Quality Score': metrics['deep_quality_score'],
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Sort by quality score
    df = df.sort_values('Quality Score', ascending=False)

    # Highlight best values
    best_indices = {}
    for col in df.columns[1:]:
        if 'Error' in col or 'RMSE' in col:
            best_indices[col] = df[col].idxmin()
        else:
            best_indices[col] = df[col].idxmax()

    return df, best_indices


def visualize_deep_improvement(v_true, v_baseline, v_improved,
                               depth_threshold=0.5, save_path=None):
    """
    Visualize improvement in deep layer inversion.

    Args:
        v_true: ground truth [nz, nx]
        v_baseline: baseline prediction [nz, nx]
        v_improved: improved prediction [nz, nx]
        depth_threshold: where deep layer starts
        save_path: path to save figure (optional)
    """
    import matplotlib.pyplot as plt

    if torch.is_tensor(v_true):
        v_true = v_true.cpu().numpy()
    if torch.is_tensor(v_baseline):
        v_baseline = v_baseline.cpu().numpy()
    if torch.is_tensor(v_improved):
        v_improved = v_improved.cpu().numpy()

    nz, nx = v_true.shape
    deep_start = int(nz * depth_threshold)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Common extent and colormap
    extent = [0, nx * 0.015, nz * 0.015, 0]  # Assuming dz=15m
    vmin, vmax = v_true.min() / 1000, v_true.max() / 1000

    # Row 1: Full models
    im0 = axes[0, 0].imshow(v_true / 1000, extent=extent, aspect='auto',
                            cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axhline(deep_start * 0.015, color='yellow',
                       linestyle='--', linewidth=2, label='Deep layer start')
    axes[0, 0].legend()

    axes[0, 1].imshow(v_baseline / 1000, extent=extent, aspect='auto',
                     cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Baseline')
    axes[0, 1].axhline(deep_start * 0.015, color='yellow',
                       linestyle='--', linewidth=2)

    axes[0, 2].imshow(v_improved / 1000, extent=extent, aspect='auto',
                     cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('Improved')
    axes[0, 2].axhline(deep_start * 0.015, color='yellow',
                       linestyle='--', linewidth=2)

    # Row 2: Error maps for deep layers only
    deep_true = v_true[deep_start:, :]
    deep_baseline = v_baseline[deep_start:, :]
    deep_improved = v_improved[deep_start:, :]

    error_baseline = np.abs(deep_baseline - deep_true)
    error_improved = np.abs(deep_improved - deep_true)

    extent_deep = [0, nx * 0.015, (nz - deep_start) * 0.015, 0]
    error_max = max(error_baseline.max(), error_improved.max())

    im1 = axes[1, 0].imshow(deep_true / 1000, extent=extent_deep,
                           aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Ground Truth (Deep Only)')

    im2 = axes[1, 1].imshow(error_baseline, extent=extent_deep, aspect='auto',
                           cmap='hot', vmin=0, vmax=error_max)
    axes[1, 1].set_title(f'Baseline Error (RMSE={np.sqrt(np.mean(error_baseline**2)):.1f} m/s)')

    im3 = axes[1, 2].imshow(error_improved, extent=extent_deep, aspect='auto',
                           cmap='hot', vmin=0, vmax=error_max)
    axes[1, 2].set_title(f'Improved Error (RMSE={np.sqrt(np.mean(error_improved**2)):.1f} m/s)')

    # Colorbars
    for ax in axes[0, :]:
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Depth (km)')
    for ax in axes[1, :]:
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Depth (km)')

    fig.colorbar(im0, ax=axes[0, :], label='Velocity (km/s)',
                 orientation='horizontal', pad=0.05)
    fig.colorbar(im2, ax=axes[1, 1:], label='Absolute Error (m/s)',
                 orientation='horizontal', pad=0.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_depth_profile_comparison(results_dict, save_path=None):
    """
    Plot depth-wise error profiles for multiple experiments.

    Args:
        results_dict: dict of {exp_name: metrics_dict} (must contain 'depth_profile')
        save_path: path to save figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_name, metrics in results_dict.items():
        if 'depth_profile' in metrics:
            profile = metrics['depth_profile']
            depths = np.arange(len(profile)) * 0.015  # Assuming dz=15m
            ax.plot(profile * 100, depths, label=exp_name, linewidth=2)

    ax.set_xlabel('Relative Error (%)', fontsize=12)
    ax.set_ylabel('Depth (km)', fontsize=12)
    ax.set_title('Error vs Depth Comparison', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Highlight deep region
    if 'depth_profile' in list(results_dict.values())[0]:
        nz = len(list(results_dict.values())[0]['depth_profile'])
        deep_start = int(nz * 0.5) * 0.015
        ax.axhline(deep_start, color='red', linestyle='--',
                   linewidth=2, alpha=0.5, label='Deep layer start')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved depth profile to {save_path}")

    return fig
