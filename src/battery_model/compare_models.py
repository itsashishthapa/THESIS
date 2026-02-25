"""
Comparison script: RL model vs Mathematical (Pyomo) model
"""

import os
import sys
import time

import numpy as np
from stable_baselines3 import SAC

# Add paths for imports
sys.path.insert(0, '../../Battery_model/Optimization_Pyomo')

# Import from evaluate module
# Import from Pyomo model
from Battery_model.Optimization_Pyomo.model_pyomo import optimize_battery_pyomo
from src.battery_model.evaluate import evaluate_model, load_price_data


def run_pyomo_model(prices_window):
    """
    Run the Pyomo optimization model on given price window.
    Uses optimize_battery_pyomo from model_pyomo.py to avoid code duplication.
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
    
    # Calculate cost
    cost = np.sum((np.array(prices_used) / 1e6) * np.array(Ps))
    
    return {
        'cost': cost,
        'time': elapsed_time,
        'u': P_cmds,
        'SOC': SOCs,
        'P': Ps,
        'P_actual': P_actuals,
        'rewards': rewards,
        'status': 'success'
    }


def compare_models(prices_full, global_price_scale, rl_model, window_len=24, n_comparisons=10):
    """
    Compare RL and Pyomo models on multiple random windows.
    """
    results = {
        'rl': {'costs': [], 'times': [], 'max_socs': [], 'min_socs': []},
        'pyomo': {'costs': [], 'times': [], 'max_socs': [], 'min_socs': []},
        'differences': []
    }
    
    print(f"\nRunning {n_comparisons} comparisons between RL and Pyomo models...")
    print("="*80)
    
    for i in range(n_comparisons):
        # Randomly select window
        max_start = len(prices_full) - window_len
        start_idx = np.random.randint(0, max_start)
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
            print(f"  RL Cost: {rl_result['cost']:.4f}€, Time: {rl_result['time']:.4f}s")
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
            print(f"  Pyomo Cost: {pyomo_result['cost']:.4f}€, Time: {pyomo_result['time']:.4f}s")
            
            diff = rl_result['cost'] - pyomo_result['cost']
            pct_diff = (diff / pyomo_result['cost'] * 100) if pyomo_result['cost'] != 0 else 0
            results['differences'].append(pct_diff)
            print(f"  Difference: {diff:.4f}€ ({pct_diff:+.2f}%)")
        else:
            print(f"  Pyomo failed: {pyomo_result.get('error', 'Unknown error')}")
    
    return results


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
        lines.append("COST COMPARISON (€)")
        lines.append("-"*80)
        
        rl_costs = np.array(results['rl']['costs'])
        pyomo_costs = np.array(results['pyomo']['costs'])
        
        lines.append("RL Model:")
        lines.append(f"  Mean: {np.mean(rl_costs):.4f} ± {np.std(rl_costs):.4f}")
        lines.append(f"  Range: [{np.min(rl_costs):.4f}, {np.max(rl_costs):.4f}]")
        
        lines.append("\nPyomo Model:")
        lines.append(f"  Mean: {np.mean(pyomo_costs):.4f} ± {np.std(pyomo_costs):.4f}")
        lines.append(f"  Range: [{np.min(pyomo_costs):.4f}, {np.max(pyomo_costs):.4f}]")
        
        lines.append("\nRL - Pyomo (RL is better if negative):")
        diffs = rl_costs - pyomo_costs
        lines.append(f"  Mean: {np.mean(diffs):.4f} ± {np.std(diffs):.4f}€")
        lines.append(f"  Percentage difference: {np.mean(results['differences']):.2f}% ± {np.std(results['differences']):.2f}%")
        
        # Time comparison
        lines.append("\n" + "-"*80)
        lines.append("COMPUTATION TIME (seconds)")
        lines.append("-"*80)
        
        rl_times = np.array(results['rl']['times'])
        pyomo_times = np.array(results['pyomo']['times'])
        
        lines.append("RL Model:")
        lines.append(f"  Mean: {np.mean(rl_times):.4f} ± {np.std(rl_times):.4f}s")
        lines.append(f"  Range: [{np.min(rl_times):.4f}, {np.max(rl_times):.4f}]s")
        
        lines.append("\nPyomo Model:")
        lines.append(f"  Mean: {np.mean(pyomo_times):.4f} ± {np.std(pyomo_times):.4f}s")
        lines.append(f"  Range: [{np.min(pyomo_times):.4f}, {np.max(pyomo_times):.4f}]s")
        
        lines.append(f"\nSpeedup (Pyomo time / RL time): {np.mean(pyomo_times) / np.mean(rl_times):.2f}x")
        
        # SOC comparison
        lines.append("\n" + "-"*80)
        lines.append("SOC ANALYSIS")
        lines.append("-"*80)
        
        lines.append("RL Max SOC:")
        lines.append(f"  Mean: {np.mean(results['rl']['max_socs']):.4f} ± {np.std(results['rl']['max_socs']):.4f}")
        
        lines.append("\nPyomo Max SOC:")
        lines.append(f"  Mean: {np.mean(results['pyomo']['max_socs']):.4f} ± {np.std(results['pyomo']['max_socs']):.4f}")
        
        lines.append("\nRL Min SOC:")
        lines.append(f"  Mean: {np.mean(results['rl']['min_socs']):.4f} ± {np.std(results['rl']['min_socs']):.4f}")
        
        lines.append("\nPyomo Min SOC:")
        lines.append(f"  Mean: {np.mean(results['pyomo']['min_socs']):.4f} ± {np.std(results['pyomo']['min_socs']):.4f}")
    
    lines.append("\n" + "="*80 + "\n")
    
    output_text = "\n".join(lines)
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\nComparison results written to {output_file}")
    else:
        print(output_text)


if __name__ == '__main__':
    # Load price data and model
    print("Loading data and model...")
    prices_full, global_price_scale = load_price_data()
    
    rl_model = SAC.load("src/battery_model/battery_sac_model")
    print("Model loaded successfully!")
    
    # Run comparisons
    results = compare_models(
        prices_full, global_price_scale, rl_model,
        window_len=24, n_comparisons=1000
    )
    
    # Print results
    print_comparison_stats(results, output_file='src/battery_model/model_comparison.txt')
