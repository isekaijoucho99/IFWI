"""
Compare Results from Multiple IFWI Experiments
----------------------------------------------
Analyze and compare deep layer inversion quality across experiments.

Usage:
    python compare_results.py --results-dir results
    python compare_results.py --results-dir results --baseline baseline_20240101_120000
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent))
from improved_modules.evaluate_deep import (
    compare_experiments,
    plot_depth_profile_comparison,
    visualize_deep_improvement
)


def find_experiment_dirs(results_dir):
    """Find all experiment result directories."""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return []

    # Find directories with metrics.json
    experiment_dirs = []
    for item in results_path.iterdir():
        if item.is_dir() and (item / 'metrics.json').exists():
            experiment_dirs.append(item)

    return sorted(experiment_dirs, key=lambda x: x.name)


def load_experiment_results(exp_dir):
    """Load results from an experiment directory."""
    results = {}

    # Load metrics
    metrics_file = exp_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            results['metrics'] = json.load(f)

    # Load config
    config_file = exp_dir / 'config.yaml'
    if config_file.exists():
        import yaml
        with open(config_file, 'r') as f:
            results['config'] = yaml.safe_load(f)

    # Load velocity models
    v_true_file = exp_dir / 'v_true.npy'
    v_pred_file = exp_dir / 'v_pred.npy'

    if v_true_file.exists() and v_pred_file.exists():
        results['v_true'] = np.load(v_true_file)
        results['v_pred'] = np.load(v_pred_file)

    # Load loss history
    loss_file = exp_dir / 'loss_history.csv'
    if loss_file.exists():
        results['loss_history'] = np.loadtxt(loss_file, delimiter=',', skiprows=1)

    return results


def create_comparison_table(all_results):
    """Create a comparison table of key metrics."""
    data = []

    for exp_name, results in all_results.items():
        metrics = results['metrics']

        row = {
            'Experiment': exp_name,
            'Deep Error (%)': metrics['deep_relative_error'] * 100,
            'Deep RMSE (m/s)': metrics['deep_rmse'],
            'Deep SSIM': metrics['deep_ssim'],
            'Bottom-Left (%)': metrics['bottom_left_error'] * 100,
            'Bottom-Right (%)': metrics['bottom_right_error'] * 100,
            'Quality Score': metrics['deep_quality_score'],
            'Gradient Fidelity': metrics['deep_gradient_fidelity'],
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Sort by quality score
    df = df.sort_values('Quality Score', ascending=False)

    return df


def plot_comparison_charts(all_results, output_dir):
    """Generate comparison visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Bar chart of key metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    exp_names = list(all_results.keys())
    metrics_to_plot = [
        ('deep_relative_error', 'Deep Relative Error (%)', True, axes[0, 0]),
        ('deep_rmse', 'Deep RMSE (m/s)', True, axes[0, 1]),
        ('deep_ssim', 'Deep SSIM', False, axes[0, 2]),
        ('bottom_left_error', 'Bottom-Left Error (%)', True, axes[1, 0]),
        ('bottom_right_error', 'Bottom-Right Error (%)', True, axes[1, 1]),
        ('deep_quality_score', 'Quality Score', False, axes[1, 2]),
    ]

    for metric_key, title, is_error, ax in metrics_to_plot:
        values = []
        for exp_name in exp_names:
            value = all_results[exp_name]['metrics'][metric_key]
            if 'error' in metric_key and metric_key != 'deep_quality_score':
                value *= 100  # Convert to percentage
            values.append(value)

        bars = ax.bar(range(len(exp_names)), values)

        # Color: green for best, red for worst (if error metric)
        if is_error:
            best_idx = np.argmin(values)
            bars[best_idx].set_color('green')
            if len(values) > 1:
                worst_idx = np.argmax(values)
                bars[worst_idx].set_color('red')
        else:
            best_idx = np.argmax(values)
            bars[best_idx].set_color('green')
            if len(values) > 1:
                worst_idx = np.argmin(values)
                bars[worst_idx].set_color('red')

        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels([name.split('_')[0] for name in exp_names],
                          rotation=45, ha='right')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'metrics_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'metrics_comparison.png'}")

    # 2. Depth profile comparison
    if all('depth_profile' in r['metrics'] for r in all_results.values()):
        metrics_dict = {name: r['metrics'] for name, r in all_results.items()}
        fig = plot_depth_profile_comparison(metrics_dict,
                                           save_path=output_path / 'depth_profiles.png')
        plt.close()
        print(f"Saved: {output_path / 'depth_profiles.png'}")

    # 3. Training loss comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_name, results in all_results.items():
        if 'loss_history' in results:
            history = results['loss_history']
            ax.semilogy(history[:, 1], label=exp_name.split('_')[0], linewidth=2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Data Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'loss_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'loss_comparison.png'}")


def generate_improvement_visualizations(all_results, baseline_name, output_dir):
    """Generate before/after visualizations against baseline."""
    baseline = all_results.get(baseline_name)

    if baseline is None:
        print(f"Warning: Baseline '{baseline_name}' not found. Skipping improvement plots.")
        return

    output_path = Path(output_dir)

    for exp_name, results in all_results.items():
        if exp_name == baseline_name:
            continue

        if 'v_true' not in baseline or 'v_pred' not in results:
            continue

        # Create improvement visualization
        save_path = output_path / f'improvement_{exp_name}_vs_{baseline_name}.png'
        fig = visualize_deep_improvement(
            baseline['v_true'],
            baseline['v_pred'],
            results['v_pred'],
            depth_threshold=0.5,
            save_path=save_path
        )
        plt.close()
        print(f"Saved: {save_path}")


def calculate_improvements(all_results, baseline_name):
    """Calculate percentage improvements over baseline."""
    baseline = all_results.get(baseline_name)

    if baseline is None:
        print(f"Warning: Baseline '{baseline_name}' not found.")
        return None

    baseline_metrics = baseline['metrics']
    improvements = []

    for exp_name, results in all_results.items():
        if exp_name == baseline_name:
            continue

        metrics = results['metrics']

        # Calculate improvements (negative for error metrics means improvement)
        deep_error_imp = ((baseline_metrics['deep_relative_error'] -
                          metrics['deep_relative_error']) /
                         baseline_metrics['deep_relative_error'] * 100)

        bl_error_imp = ((baseline_metrics['bottom_left_error'] -
                        metrics['bottom_left_error']) /
                       baseline_metrics['bottom_left_error'] * 100)

        br_error_imp = ((baseline_metrics['bottom_right_error'] -
                        metrics['bottom_right_error']) /
                       baseline_metrics['bottom_right_error'] * 100)

        ssim_imp = ((metrics['deep_ssim'] - baseline_metrics['deep_ssim']) /
                   baseline_metrics['deep_ssim'] * 100)

        improvements.append({
            'Experiment': exp_name,
            'Deep Error Reduction (%)': deep_error_imp,
            'BL Corner Reduction (%)': bl_error_imp,
            'BR Corner Reduction (%)': br_error_imp,
            'SSIM Improvement (%)': ssim_imp,
        })

    return pd.DataFrame(improvements).sort_values('Deep Error Reduction (%)',
                                                 ascending=False)


def main():
    parser = argparse.ArgumentParser(
        description='Compare IFWI experiment results'
    )
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Name of baseline experiment (default: auto-detect)')
    parser.add_argument('--output', type=str, default='comparison',
                       help='Output directory for comparison plots')

    args = parser.parse_args()

    # Find all experiment directories
    exp_dirs = find_experiment_dirs(args.results_dir)

    if not exp_dirs:
        print(f"No experiment results found in {args.results_dir}")
        return 1

    print(f"Found {len(exp_dirs)} experiments:")
    for exp_dir in exp_dirs:
        print(f"  - {exp_dir.name}")

    # Load all results
    all_results = {}
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        print(f"\nLoading: {exp_name}")
        results = load_experiment_results(exp_dir)
        all_results[exp_name] = results

    # Detect baseline if not specified
    if args.baseline is None:
        # Find experiment with 'baseline' in name
        baseline_candidates = [name for name in all_results.keys()
                              if 'baseline' in name.lower()]
        if baseline_candidates:
            args.baseline = baseline_candidates[0]
            print(f"\nAuto-detected baseline: {args.baseline}")
        else:
            # Use first experiment as baseline
            args.baseline = list(all_results.keys())[0]
            print(f"\nUsing first experiment as baseline: {args.baseline}")

    # Create output directory
    output_dir = Path(args.results_dir) / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    comparison_table = create_comparison_table(all_results)
    print(comparison_table.to_string(index=False))
    comparison_table.to_csv(output_dir / 'comparison_table.csv', index=False)
    print(f"\nSaved: {output_dir / 'comparison_table.csv'}")

    # Calculate improvements over baseline
    if args.baseline in all_results:
        print("\n" + "="*70)
        print(f"IMPROVEMENTS OVER BASELINE ({args.baseline})")
        print("="*70)
        improvements = calculate_improvements(all_results, args.baseline)
        if improvements is not None:
            print(improvements.to_string(index=False))
            improvements.to_csv(output_dir / 'improvements.csv', index=False)
            print(f"\nSaved: {output_dir / 'improvements.csv'}")

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    plot_comparison_charts(all_results, output_dir)

    # Generate improvement visualizations
    if args.baseline in all_results:
        generate_improvement_visualizations(all_results, args.baseline, output_dir)

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"All outputs saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
