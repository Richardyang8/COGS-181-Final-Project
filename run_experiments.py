"""
Automated Experiment Runner for CharRNN
- Define experiment grid
- Run all configurations automatically
- Save results for visualization
"""

import os
import json
import time
import itertools
from train import train_model
from data_utils import TextDataset


def run_experiment_grid(dataset, experiments, save_dir="results", verbose=True):
    """
    Run a grid of experiments.

    Args:
        dataset: TextDataset instance
        experiments: list of config dicts
        save_dir: directory to save results
        verbose: print progress

    Returns:
        list of (config, history) tuples
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results = []

    for i, config in enumerate(experiments):
        exp_name = config.get('name', f"exp_{i}")
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(experiments)}: {exp_name}")
        print(f"Config: model={config['model_type']}, hidden={config['hidden_size']}, "
              f"layers={config['n_layers']}, lr={config['learning_rate']}, "
              f"dropout={config.get('dropout', 0)}, opt={config.get('optimizer_type', 'adam')}")
        print(f"{'='*60}")

        config['save_path'] = os.path.join(save_dir, f"{exp_name}.pt")

        try:
            history, model = train_model(config, dataset, verbose=verbose)
            result = {
                'name': exp_name,
                'config': config,
                'best_val_loss': history['best_val_loss'],
                'best_val_ppl': history['best_val_ppl'],
                'total_time': history['total_time'],
                'n_params': history['n_params'],
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'val_perplexity': history['val_perplexity'],
                'samples': history['samples'],
            }
            all_results.append(result)

            # Save individual result
            # (convert non-serializable items)
            save_result = {k: v for k, v in result.items()}
            save_result['config'] = {k: v for k, v in config.items() if k != 'save_path'}
            with open(os.path.join(save_dir, f"{exp_name}_result.json"), 'w') as f:
                json.dump(save_result, f, indent=2, default=str)

        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({'name': exp_name, 'config': config, 'error': str(e)})

    # Save summary
    summary = []
    for r in all_results:
        if 'error' not in r:
            summary.append({
                'name': r['name'],
                'model_type': r['config']['model_type'],
                'hidden_size': r['config']['hidden_size'],
                'n_layers': r['config']['n_layers'],
                'learning_rate': r['config']['learning_rate'],
                'dropout': r['config'].get('dropout', 0),
                'optimizer': r['config'].get('optimizer_type', 'adam'),
                'best_val_loss': r['best_val_loss'],
                'best_val_ppl': r['best_val_ppl'],
                'total_time': r['total_time'],
                'n_params': r['n_params'],
            })
    with open(os.path.join(save_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"\n{'Name':<30} {'Val Loss':<12} {'Val PPL':<12} {'Time':<10} {'Params':<12}")
    print("-" * 76)
    for r in all_results:
        if 'error' not in r:
            print(f"{r['name']:<30} {r['best_val_loss']:<12.4f} {r['best_val_ppl']:<12.2f} "
                  f"{r['total_time']:<10.1f} {r['n_params']:<12,}")

    return all_results


# ============================================================
# PRE-DEFINED EXPERIMENT SETS
# ============================================================

def get_core_experiments(n_epochs=2000, print_every=200):
    """
    Core experiment set covering all required comparisons.
    ~20 experiments, should take ~30-60 min on Colab T4.
    """
    base = {
        'n_epochs': n_epochs,
        'print_every': print_every,
        'chunk_len': 200,
        'batch_size': 64,
    }

    experiments = []

    # ---- Experiment Group 1: Model Type Comparison (RNN vs LSTM vs GRU) ----
    for model_type in ['rnn', 'lstm', 'gru']:
        exp = {**base,
               'name': f'model_{model_type}',
               'model_type': model_type,
               'hidden_size': 256,
               'n_layers': 2,
               'learning_rate': 0.002,
               'optimizer_type': 'adam',
               'dropout': 0.2}
        experiments.append(exp)

    # ---- Experiment Group 2: Hidden Size (LSTM) ----
    for hs in [128, 256, 512]:
        exp = {**base,
               'name': f'hidden_{hs}',
               'model_type': 'lstm',
               'hidden_size': hs,
               'n_layers': 2,
               'learning_rate': 0.002,
               'optimizer_type': 'adam',
               'dropout': 0.2}
        experiments.append(exp)

    # ---- Experiment Group 3: Number of Layers (LSTM) ----
    for nl in [1, 2, 3]:
        exp = {**base,
               'name': f'layers_{nl}',
               'model_type': 'lstm',
               'hidden_size': 256,
               'n_layers': nl,
               'learning_rate': 0.002,
               'optimizer_type': 'adam',
               'dropout': 0.2}
        experiments.append(exp)

    # ---- Experiment Group 4: Learning Rate (LSTM) ----
    for lr in [0.0005, 0.001, 0.002, 0.005]:
        exp = {**base,
               'name': f'lr_{lr}',
               'model_type': 'lstm',
               'hidden_size': 256,
               'n_layers': 2,
               'learning_rate': lr,
               'optimizer_type': 'adam',
               'dropout': 0.2}
        experiments.append(exp)

    # ---- Experiment Group 5: Optimizer Comparison ----
    for opt in ['adam', 'sgd', 'rmsprop']:
        exp = {**base,
               'name': f'opt_{opt}',
               'model_type': 'lstm',
               'hidden_size': 256,
               'n_layers': 2,
               'learning_rate': 0.002 if opt != 'sgd' else 0.01,
               'optimizer_type': opt,
               'dropout': 0.2}
        experiments.append(exp)

    # ---- Experiment Group 6: Dropout ----
    for dp in [0.0, 0.2, 0.5]:
        exp = {**base,
               'name': f'dropout_{dp}',
               'model_type': 'lstm',
               'hidden_size': 256,
               'n_layers': 2,
               'learning_rate': 0.002,
               'optimizer_type': 'adam',
               'dropout': dp}
        experiments.append(exp)

    # ---- Experiment Group 7: Chunk Length (Sequence Length) ----
    for cl in [100, 200, 300]:
        exp = {**base,
               'name': f'chunk_{cl}',
               'model_type': 'lstm',
               'hidden_size': 256,
               'n_layers': 2,
               'learning_rate': 0.002,
               'optimizer_type': 'adam',
               'dropout': 0.2,
               'chunk_len': cl}
        experiments.append(exp)

    # Remove duplicates (some configs overlap between groups)
    seen = set()
    unique_experiments = []
    for exp in experiments:
        key = (exp['model_type'], exp['hidden_size'], exp['n_layers'],
               exp['learning_rate'], exp.get('optimizer_type', 'adam'),
               exp.get('dropout', 0), exp.get('chunk_len', 200))
        if key not in seen:
            seen.add(key)
            unique_experiments.append(exp)

    print(f"Total unique experiments: {len(unique_experiments)}")
    return unique_experiments


def get_quick_test_experiments(n_epochs=500, print_every=100):
    """Quick test set to verify everything works ."""
    base = {
        'n_epochs': n_epochs,
        'print_every': print_every,
        'chunk_len': 200,
        'batch_size': 64,
    }
    return [
        {**base, 'name': 'quick_rnn', 'model_type': 'rnn',
         'hidden_size': 128, 'n_layers': 1, 'learning_rate': 0.002,
         'optimizer_type': 'adam', 'dropout': 0.0},
        {**base, 'name': 'quick_lstm', 'model_type': 'lstm',
         'hidden_size': 128, 'n_layers': 1, 'learning_rate': 0.002,
         'optimizer_type': 'adam', 'dropout': 0.0},
        {**base, 'name': 'quick_gru', 'model_type': 'gru',
         'hidden_size': 128, 'n_layers': 1, 'learning_rate': 0.002,
         'optimizer_type': 'adam', 'dropout': 0.0},
    ]
