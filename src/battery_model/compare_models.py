"""
Comparison script: RL model vs Mathematical (Pyomo) model
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from Battery_model.Optimization_Pyomo.pyomo_battery_model import optimize_battery_pyomo
from src.battery_model.evaluate import (
    DEFAULT_ALGORITHM,
    compute_total_cost,
    evaluate_model,
    load_rl_model,
    load_price_data,
    normalize_algorithm,
)


def run_pyomo_model(prices_window):
    """
    Run the Pyomo optimization model on given price window.
    Uses optimize_battery_pyomo from pyomo_battery_model.py to avoid code duplication.
    Returns cost and optimization time.
    """
    result = optimize_battery_pyomo(prices_window, verbose=False)
    return result


def run_rl_model(model, prices_window, price_scale):
    """
    Run the RL model on given price window.
    Uses evaluate_model from evaluate.py to avoid code duplication.
    Returns cost and results.
    """
    start_time = time.time()
    
    P_cmds, Ps, P_actuals, SOCs, rewards, prices_used = evaluate_model(model, prices_window, price_scale)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Match Pyomo's objective: grid-side power P. P_actual is used for SOC dynamics.
    cost = compute_total_cost(prices_used, Ps)
    
    return {
        'cost': cost,
        'time': elapsed_time,
        'u': P_cmds,
        'SOC': SOCs,
        'P': Ps,
        'P_actual': P_actuals,
        'battery_side_cost': compute_total_cost(prices_used, P_actuals),
        'rewards': rewards,
        'status': 'success'
    }


def _rmse(a, b):
    """Return RMSE over the overlapping part of two trajectories."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def compare_models(prices_full, global_price_scale, rl_model, window_len=24, n_comparisons=10):
    """
    Compare RL and Pyomo models on multiple random windows.
    """
    results = {
        'rl': {'costs': [], 'times': [], 'max_socs': [], 'min_socs': []},
        'pyomo': {'costs': [], 'times': [], 'max_socs': [], 'min_socs': []},
        'differences': [],
        'windows': [],
        'abs_cost_gaps': [],
        'soc_rmse': [],
        'power_rmse_kw': [],
    }
    
    print(f"\nRunning {n_comparisons} comparisons between RL and Pyomo models...")
    print("="*80)
    
    for i in range(n_comparisons):
        # Randomly select window
        max_start = len(prices_full) - window_len
        start_idx = np.random.randint(0, max_start + 1)
        prices_window = prices_full[start_idx:start_idx + window_len]
        
        print(f"\nComparison {i+1}/{n_comparisons}")
        print(f"  Window: [{start_idx}:{start_idx + window_len}]")
        
        # Run RL model
        rl_result = run_rl_model(rl_model, prices_window, global_price_scale)
        if rl_result['status'] == 'success':
            results['rl']['costs'].append(rl_result['cost'])
            results['rl']['times'].append(rl_result['time'])
            results['rl']['max_socs'].append(np.max(rl_result['SOC']))
            results['rl']['min_socs'].append(np.min(rl_result['SOC']))
            print(f"  RL Cost: {rl_result['cost']:.4f} EUR, Time: {rl_result['time']:.4f}s")
        else:
            print("  RL failed")
            continue
        
        # Run Pyomo model
        pyomo_result = run_pyomo_model(prices_window)
        if pyomo_result['status'] == 'success':
            results['pyomo']['costs'].append(pyomo_result['cost'])
            results['pyomo']['times'].append(pyomo_result['time'])
            results['pyomo']['max_socs'].append(np.max(pyomo_result['SOC']))
            results['pyomo']['min_socs'].append(np.min(pyomo_result['SOC']))
            print(f"  Pyomo Cost: {pyomo_result['cost']:.4f} EUR, Time: {pyomo_result['time']:.4f}s")
            
            diff = rl_result['cost'] - pyomo_result['cost']
            pct_diff = (diff / abs(pyomo_result['cost']) * 100) if pyomo_result['cost'] != 0 else 0
            results['differences'].append(pct_diff)
            results['windows'].append(f"{start_idx}-{start_idx + window_len}")
            results['abs_cost_gaps'].append(abs(diff))
            results['soc_rmse'].append(_rmse(rl_result['SOC'], pyomo_result['SOC']))
            results['power_rmse_kw'].append(_rmse(rl_result['P'], pyomo_result['P']) / 1000.0)
            print(f"  Difference: {diff:.4f} EUR ({pct_diff:+.2f}%)")
        else:
            for key in ('costs', 'times', 'max_socs', 'min_socs'):
                results['rl'][key].pop()
            print(f"  Pyomo failed: {pyomo_result.get('error', 'Unknown error')}")
    
    return results


def plot_comparison_distances(results, output_file=None, show=False):
    """
    Plot per-window RL vs Pyomo cost and absolute cost gap.
    """
    rl_costs = np.asarray(results['rl']['costs'], dtype=float)
    pyomo_costs = np.asarray(results['pyomo']['costs'], dtype=float)
    n = min(len(rl_costs), len(pyomo_costs))
    if n == 0:
        print("No successful comparisons available for plotting.")
        return None

    labels = results.get('windows') or [str(i + 1) for i in range(n)]
    labels = labels[:n]
    x = np.arange(n)

    def series(name, fallback):
        values = np.asarray(results.get(name, []), dtype=float)
        if len(values) < n:
            values = np.asarray(fallback, dtype=float)
        return values[:n]

    abs_cost_gaps = series('abs_cost_gaps', np.abs(rl_costs[:n] - pyomo_costs[:n]))

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(x, rl_costs[:n], marker='o', label='RL cost')
    axes[0].plot(x, pyomo_costs[:n], marker='o', label='Pyomo cost')
    axes[0].set_title("RL vs Pyomo distance per comparison")
    axes[0].set_ylabel('Cost [EUR]')
    axes[0].legend(loc='lower left')
    axes[0].grid(True, alpha=0.35)

    axes[1].bar(x, abs_cost_gaps, alpha=0.75)
    axes[1].set_ylabel('Absolute gap [EUR]')
    axes[1].grid(True, alpha=0.35)
    axes[1].set_xlabel('Comparison window')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')

    fig.tight_layout()

    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Distance plot saved to {output_file}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def print_comparison_stats(results, output_file=None):
    """Print or save comparison statistics."""
    lines = []
    lines.append("\n" + "="*80)
    lines.append("COMPARISON: RL Model vs Pyomo Mathematical Model")
    lines.append("="*80)
    
    if len(results['rl']['costs']) == 0:
        lines.append("\nNo successful comparisons completed.")
    else:
        n = len(results['rl']['costs'])
        lines.append(f"\nNumber of successful comparisons: {n}")
        
        # Cost comparison
        lines.append("\n" + "-"*80)
        lines.append("COST COMPARISON (EUR)")
        lines.append("-"*80)
        
        rl_costs = np.array(results['rl']['costs'])
        pyomo_costs = np.array(results['pyomo']['costs'])
        
        lines.append("RL Model:")
        lines.append(f"  Mean: {np.mean(rl_costs):.4f} +/- {np.std(rl_costs):.4f}")
        lines.append(f"  Range: [{np.min(rl_costs):.4f}, {np.max(rl_costs):.4f}]")
        
        lines.append("\nPyomo Model:")
        lines.append(f"  Mean: {np.mean(pyomo_costs):.4f} +/- {np.std(pyomo_costs):.4f}")
        lines.append(f"  Range: [{np.min(pyomo_costs):.4f}, {np.max(pyomo_costs):.4f}]")
        
        lines.append("\nRL - Pyomo (RL is better if negative):")
        diffs = rl_costs - pyomo_costs
        abs_gap = np.abs(diffs)
        rel_gap = abs_gap / np.maximum(np.abs(pyomo_costs), 1e-12) * 100.0
        lines.append(f"  Mean: {np.mean(diffs):.4f} +/- {np.std(diffs):.4f} EUR")
        lines.append(f"  Mean absolute gap: {np.mean(abs_gap):.4f} +/- {np.std(abs_gap):.4f} EUR")
        lines.append(f"  Mean relative absolute gap: {np.mean(rel_gap):.2f}% +/- {np.std(rel_gap):.2f}%")
        lines.append(f"  Signed percentage difference: {np.mean(results['differences']):.2f}% +/- {np.std(results['differences']):.2f}%")
        
        # Time comparison
        lines.append("\n" + "-"*80)
        lines.append("COMPUTATION TIME (seconds)")
        lines.append("-"*80)
        
        rl_times = np.array(results['rl']['times'])
        pyomo_times = np.array(results['pyomo']['times'])
        
        lines.append("RL Model:")
        lines.append(f"  Mean: {np.mean(rl_times):.4f} +/- {np.std(rl_times):.4f}s")
        lines.append(f"  Range: [{np.min(rl_times):.4f}, {np.max(rl_times):.4f}]s")
        
        lines.append("\nPyomo Model:")
        lines.append(f"  Mean: {np.mean(pyomo_times):.4f} +/- {np.std(pyomo_times):.4f}s")
        lines.append(f"  Range: [{np.min(pyomo_times):.4f}, {np.max(pyomo_times):.4f}]s")
        
        lines.append(f"\nSpeedup (Pyomo time / RL time): {np.mean(pyomo_times) / np.mean(rl_times):.2f}x")
        
        # SOC comparison
        lines.append("\n" + "-"*80)
        lines.append("SOC ANALYSIS")
        lines.append("-"*80)
        
        lines.append("RL Max SOC:")
        lines.append(f"  Mean: {np.mean(results['rl']['max_socs']):.4f} +/- {np.std(results['rl']['max_socs']):.4f}")
        
        lines.append("\nPyomo Max SOC:")
        lines.append(f"  Mean: {np.mean(results['pyomo']['max_socs']):.4f} +/- {np.std(results['pyomo']['max_socs']):.4f}")
        
        lines.append("\nRL Min SOC:")
        lines.append(f"  Mean: {np.mean(results['rl']['min_socs']):.4f} +/- {np.std(results['rl']['min_socs']):.4f}")
        
        lines.append("\nPyomo Min SOC:")
        lines.append(f"  Mean: {np.mean(results['pyomo']['min_socs']):.4f} +/- {np.std(results['pyomo']['min_socs']):.4f}")
    
    lines.append("\n" + "="*80 + "\n")
    
    output_text = "\n".join(lines)
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\nComparison results written to {output_file}")
    else:
        print(output_text)


def _parse_args():
    parser = argparse.ArgumentParser(description="Compare a Battery RL model against Pyomo")
    parser.add_argument(
        "--algorithm",
        type=normalize_algorithm,
        default=DEFAULT_ALGORITHM,
        choices=("SAC", "TD3"),
        help="RL algorithm used by the saved model",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to the saved RL model without .zip",
    )
    parser.add_argument(
        "--comparisons",
        type=int,
        default=1000,
        help="Number of random windows to compare",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    # Load price data and model
    print("Loading data and model...")
    prices_full, global_price_scale = load_price_data()
    
    rl_model = load_rl_model(args.algorithm, args.model_path)
    print(f"{args.algorithm} model loaded successfully!")
    
    # Run comparisons
    results = compare_models(
        prices_full, global_price_scale, rl_model,
        window_len=24, n_comparisons=args.comparisons
    )
    
    # Print results
    print_comparison_stats(results, output_file='src/battery_model/model_comparison.txt')
    plot_comparison_distances(results, output_file='src/battery_model/model_comparison_distances.png')
