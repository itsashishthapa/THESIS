"""
Comparison script: RL model vs Mathematical (Pyomo) model
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from Battery_model.Optimization_Pyomo.pyomo_battery_model_v2 import get_battery_model
from src.battery_model.evaluate import (
    DEFAULT_ALGORITHM,
    compute_total_cost,
    evaluate_model,
    load_price_data,
    load_rl_model,
    normalize_algorithm,
)
from src.battery_model.price_model import (
    DEFAULT_PRICE_MODEL_HOURS,
    load_price_model_data,
)

TERMINAL_SOC_MIN = 0.5
SOC_VIOLATION_TOLERANCE = 1e-6


def run_pyomo_model(prices_window):
    """
    Run the Pyomo optimization model on given price window.
    Uses get_battery_model from pyomo_battery_model_v2.py.
    Returns cost and optimization time.
    """
    result = get_battery_model(prices_window, verbose=False)
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


def _terminal_soc_shortfall(socs, terminal_soc_min=TERMINAL_SOC_MIN):
    """Return the final-SOC shortfall against the terminal requirement."""
    socs = np.asarray(socs, dtype=float)
    if socs.size == 0:
        return float("nan")
    return float(max(float(terminal_soc_min) - socs[-1], 0.0))


def compare_models(prices_full, global_price_scale, rl_model, window_len=24, n_comparisons=10):
    """
    Compare RL and Pyomo models on multiple random windows.
    """
    results = {
        'rl': {
            'costs': [], 'times': [], 'max_socs': [], 'min_socs': [],
            'final_socs': [], 'terminal_soc_shortfalls': [],
            'terminal_soc_violations': [],
        },
        'pyomo': {
            'costs': [], 'times': [], 'max_socs': [], 'min_socs': [],
            'final_socs': [], 'terminal_soc_shortfalls': [],
            'terminal_soc_violations': [],
        },
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
            rl_terminal_shortfall = _terminal_soc_shortfall(rl_result['SOC'])
            rl_terminal_violation = (
                rl_terminal_shortfall > SOC_VIOLATION_TOLERANCE
            )
            results['rl']['costs'].append(rl_result['cost'])
            results['rl']['times'].append(rl_result['time'])
            results['rl']['max_socs'].append(np.max(rl_result['SOC']))
            results['rl']['min_socs'].append(np.min(rl_result['SOC']))
            results['rl']['final_socs'].append(rl_result['SOC'][-1])
            results['rl']['terminal_soc_shortfalls'].append(rl_terminal_shortfall)
            results['rl']['terminal_soc_violations'].append(rl_terminal_violation)
            print(f"  RL Cost: {rl_result['cost']:.4f} EUR, Time: {rl_result['time']:.4f}s")
            print(
                f"  RL terminal SOC: {rl_result['SOC'][-1]:.4f} "
                f"({'VIOLATED' if rl_terminal_violation else 'satisfied'})"
            )
        else:
            print("  RL failed")
            continue
        
        # Run Pyomo model
        pyomo_result = run_pyomo_model(prices_window)
        if pyomo_result['status'] == 'success':
            pyomo_terminal_shortfall = _terminal_soc_shortfall(pyomo_result['SOC'])
            pyomo_terminal_violation = (
                pyomo_terminal_shortfall > SOC_VIOLATION_TOLERANCE
            )
            results['pyomo']['costs'].append(pyomo_result['cost'])
            results['pyomo']['times'].append(pyomo_result['time'])
            results['pyomo']['max_socs'].append(np.max(pyomo_result['SOC']))
            results['pyomo']['min_socs'].append(np.min(pyomo_result['SOC']))
            results['pyomo']['final_socs'].append(pyomo_result['SOC'][-1])
            results['pyomo']['terminal_soc_shortfalls'].append(
                pyomo_terminal_shortfall
            )
            results['pyomo']['terminal_soc_violations'].append(
                pyomo_terminal_violation
            )
            print(f"  Pyomo Cost: {pyomo_result['cost']:.4f} EUR, Time: {pyomo_result['time']:.4f}s")
            print(
                f"  Pyomo terminal SOC: {pyomo_result['SOC'][-1]:.4f} "
                f"({'VIOLATED' if pyomo_terminal_violation else 'satisfied'})"
            )
            
            diff = rl_result['cost'] - pyomo_result['cost']
            pct_diff = (diff / abs(pyomo_result['cost']) * 100) if pyomo_result['cost'] != 0 else 0
            results['differences'].append(pct_diff)
            results['windows'].append(f"{start_idx}-{start_idx + window_len}")
            results['abs_cost_gaps'].append(abs(diff))
            results['soc_rmse'].append(_rmse(rl_result['SOC'], pyomo_result['SOC']))
            results['power_rmse_kw'].append(_rmse(rl_result['P'], pyomo_result['P']))
            print(f"  Difference: {diff:.4f} EUR ({pct_diff:+.2f}%)")
        else:
            for values in results['rl'].values():
                if isinstance(values, list):
                    values.pop()
            print(f"  Pyomo failed: {pyomo_result.get('error', 'Unknown error')}")
    
    return results


def plot_comparison_distances(
    results,
    output_file=None,
    show=False,
    algorithm=DEFAULT_ALGORITHM,
):
    """Create a thesis-ready RL versus mathematical-optimization comparison plot."""
    rl_costs = np.asarray(results['rl']['costs'], dtype=float)
    pyomo_costs = np.asarray(results['pyomo']['costs'], dtype=float)
    n = min(len(rl_costs), len(pyomo_costs))
    if n == 0:
        print("No successful comparisons available for plotting.")
        return None

    windows = (results.get('windows') or [str(i + 1) for i in range(n)])[:n]
    labels = []
    for index, window in enumerate(windows, start=1):
        window_label = str(window).replace('-', '\u2013')
        labels.append(f"Window {index}\n({window_label})")
    x = np.arange(n)
    algorithm = normalize_algorithm(algorithm)

    def series(name, fallback):
        values = np.asarray(results.get(name, []), dtype=float)
        if len(values) < n:
            values = np.asarray(fallback, dtype=float)
        return values[:n]

    abs_cost_gaps = series('abs_cost_gaps', np.abs(rl_costs[:n] - pyomo_costs[:n]))

    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 10.5),
        gridspec_kw={'height_ratios': [1.15, 0.93]},
    )
    fig.subplots_adjust(hspace=0.42, bottom=0.16, top=0.93, left=0.08, right=0.97)

    sac_color = '#0b5bb5'
    optimum_color = '#ff7a00'
    bar_color = '#1f62b5'
    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(True, linestyle='--', linewidth=0.8, alpha=0.4)

    axes[0].plot(
        x,
        rl_costs[:n],
        marker='o',
        markersize=8,
        linewidth=2,
        color=sac_color,
        label=f'{algorithm} (RL)',
    )
    axes[0].plot(
        x,
        pyomo_costs[:n],
        marker='o',
        markersize=8,
        linewidth=2,
        color=optimum_color,
        label='Mathematical optimization',
    )
    axes[0].set_title(
        f'{algorithm} vs. Mathematical Optimization Across 24-Hour Comparison Windows',
        fontsize=20,
        fontweight='bold',
        pad=14,
    )
    axes[0].set_ylabel('Objective value [€]', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.95)

    bars = axes[1].bar(x, abs_cost_gaps, width=0.58, color=bar_color, alpha=0.85)
    mean_gap = float(np.mean(abs_cost_gaps))
    axes[1].axhline(mean_gap, color='#d7191c', linestyle='--', linewidth=1.4)
    axes[1].text(
        max(0, n / 2 - 0.2),
        mean_gap + max(0.04, 0.035 * max(abs_cost_gaps)),
        f'Mean gap = €{mean_gap:.2f}',
        color='#c41218',
        fontsize=13,
        fontweight='bold',
        ha='center',
    )
    for bar, gap in zip(bars, abs_cost_gaps):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(0.025, 0.025 * max(abs_cost_gaps)),
            f'{gap:.2f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
        )

    ranked = np.argsort(abs_cost_gaps)[::-1]
    for rank, index in enumerate(ranked[:2], start=1):
        label = 'Largest gap' if rank == 1 else '2nd largest gap'
        axes[1].annotate(
            f'{label}\n€{abs_cost_gaps[index]:.2f}',
            xy=(index, abs_cost_gaps[index]),
            xytext=(0, 26),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=10,
            bbox={'boxstyle': 'round,pad=0.45', 'fc': 'white', 'ec': '#e74c3c', 'lw': 0.8},
            arrowprops={'arrowstyle': '->', 'lw': 0.8, 'color': 'black'},
        )

    axes[1].set_title(
        f'Optimality Gap to Mathematical Optimization (|{algorithm} − Optimum|)',
        fontsize=18,
        fontweight='bold',
        pad=12,
    )
    axes[1].set_ylabel('Optimality gap [€]', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Window {i + 1}' for i in range(n)], fontweight='bold')
    axes[1].set_xlim(-0.55, n - 0.45)
    axes[1].set_ylim(0, max(abs_cost_gaps) * 1.25)
    fig.text(
        0.76,
        0.1,
        'Lower (more negative) objective value = higher profit',
        fontsize=11,
        ha='center',
        va='center',
        bbox={'boxstyle': 'round,pad=0.7', 'fc': '#eef4ff', 'ec': '#7aa0d5', 'lw': 1.0},
    )

    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
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

        # Terminal SOC feasibility comparison
        lines.append("\n" + "-"*80)
        lines.append("TERMINAL SOC REQUIREMENT")
        lines.append("-"*80)
        lines.append(
            f"Requirement: final SOC >= {TERMINAL_SOC_MIN:.3f} "
            f"(violation tolerance: {SOC_VIOLATION_TOLERANCE:g})"
        )
        for method, label in (("rl", "RL Model"), ("pyomo", "Pyomo Model")):
            violations = np.asarray(
                results[method]['terminal_soc_violations'], dtype=bool
            )
            shortfalls = np.asarray(
                results[method]['terminal_soc_shortfalls'], dtype=float
            )
            violation_count = int(np.count_nonzero(violations))
            lines.append(f"\n{label}:")
            lines.append(
                f"  Violated cases: {violation_count}/{n} "
                f"({violation_count / n:.2%})"
            )
            lines.append(f"  Mean terminal-SOC shortfall: {np.mean(shortfalls):.6f}")
            lines.append(f"  Largest terminal-SOC shortfall: {np.max(shortfalls):.6f}")
    
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
    parser.add_argument(
        "--use-price-model",
        action="store_true",
        help="Generate comparison prices with price_model instead of loading electricity_price.txt",
    )
    parser.add_argument(
        "--price-model-hours",
        type=int,
        default=DEFAULT_PRICE_MODEL_HOURS,
        help="Number of hourly prices to generate when --use-price-model is set",
    )
    parser.add_argument(
        "--price-model-seed",
        type=int,
        default=None,
        help="Random seed for price_model generation",
    )
    parser.add_argument(
        "--price-model-initial-price",
        type=float,
        default=None,
        help="Initial EUR/MWh price for price_model generation; defaults to price_seasonality(start_hour)",
    )
    parser.add_argument(
        "--price-model-start-hour",
        type=float,
        default=0.0,
        help="Starting hour used by price_model price seasonality",
    )
    parser.add_argument(
        "--price-model-dt-hours",
        type=float,
        default=1.0,
        help="Time step in hours for price_model generation",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    # Load price data and model
    print("Loading data and model...")
    if args.use_price_model:
        prices_full, global_price_scale = load_price_model_data(
            num_hours=args.price_model_hours,
            initial_price=args.price_model_initial_price,
            start_hour=args.price_model_start_hour,
            dt_hours=args.price_model_dt_hours,
            seed=args.price_model_seed,
        )
    else:
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
    plot_comparison_distances(
        results,
        output_file='src/battery_model/model_comparison_distances.png',
        algorithm=args.algorithm,
    )
