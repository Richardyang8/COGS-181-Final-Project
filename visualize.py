"""
Visualization Module for CharRNN Experiments
- Loss curves comparison
- Perplexity plots
- Summary bar charts
- Temperature effect comparison
- Export publication-ready figures
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['font.size'] = 12


def plot_loss_curves(results, group_keys=None, title="Training Loss Curves",
                     save_path=None, metric='val_loss'):
    """
    Plot loss curves for a set of experiments.

    Args:
        results: list of result dicts from run_experiment_grid
        group_keys: list of experiment names to plot (None = plot all)
        title: plot title
        save_path: if set, save figure to this path
        metric: 'val_loss' or 'train_loss' or 'val_perplexity'
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        if 'error' in r:
            continue
        if group_keys and r['name'] not in group_keys:
            continue

        data = r[metric]
        epochs = [d[0] for d in data]
        values = [d[1] for d in data]
        ax.plot(epochs, values, label=r['name'], linewidth=2)

    ylabel = 'Loss' if 'loss' in metric else 'Perplexity'
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_group_comparison(results, group_prefix, title=None, save_path=None):
    """Plot all experiments matching a group prefix."""
    group_results = [r for r in results if r.get('name', '').startswith(group_prefix) and 'error' not in r]
    if not group_results:
        print(f"No results found for prefix: {group_prefix}")
        return

    if title is None:
        title = f"Comparison: {group_prefix}*"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for r in group_results:
        epochs = [d[0] for d in r['val_loss']]
        val_losses = [d[1] for d in r['val_loss']]
        val_ppls = [d[1] for d in r['val_perplexity']]
        ax1.plot(epochs, val_losses, label=r['name'], linewidth=2)
        ax2.plot(epochs, val_ppls, label=r['name'], linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Perplexity')
    ax2.set_title(f'{title} - Perplexity')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_summary_bar(results, metric='best_val_ppl', title="Best Validation Perplexity",
                     save_path=None):
    """Bar chart comparing final metric across all experiments."""
    valid = [r for r in results if 'error' not in r]
    valid.sort(key=lambda r: r.get(metric, float('inf')))

    names = [r['name'] for r in valid]
    values = [r.get(metric, 0) for r in valid]

    fig, ax = plt.subplots(figsize=(12, max(5, len(names) * 0.4)))
    bars = ax.barh(names, values, color=plt.cm.viridis(
        [v / max(values) for v in values]))
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_training_time(results, save_path=None):
    """Compare training time across experiments."""
    valid = [r for r in results if 'error' not in r]

    names = [r['name'] for r in valid]
    times = [r['total_time'] for r in valid]
    params = [r['n_params'] for r in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.barh(names, times, color='steelblue')
    ax1.set_xlabel('Training Time (seconds)')
    ax1.set_title('Training Time Comparison')

    ax2.barh(names, [p / 1000 for p in params], color='coral')
    ax2.set_xlabel('Parameters (thousands)')
    ax2.set_title('Model Size Comparison')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_temperature_samples(samples_dict, save_path=None):
    """Display text samples at different temperatures."""
    print("=" * 70)
    print("TEXT SAMPLES AT DIFFERENT TEMPERATURES")
    print("=" * 70)
    for temp, text in sorted(samples_dict.items()):
        print(f"\n--- Temperature = {temp} ---")
        print(text)
    print("=" * 70)

    # Also save to file if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write("TEXT SAMPLES AT DIFFERENT TEMPERATURES\n")
            f.write("=" * 70 + "\n")
            for temp, text in sorted(samples_dict.items()):
                f.write(f"\n--- Temperature = {temp} ---\n")
                f.write(text + "\n")
        print(f"Saved: {save_path}")


def generate_all_plots(results, save_dir="figures"):
    """Generate all standard plots for the report."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Model type comparison
    model_names = [r['name'] for r in results if r.get('name', '').startswith('model_')]
    if model_names:
        plot_group_comparison(results, 'model_',
                              title='Model Type (RNN vs LSTM vs GRU)',
                              save_path=os.path.join(save_dir, 'model_comparison.png'))

    # 2. Hidden size comparison
    hidden_names = [r['name'] for r in results if r.get('name', '').startswith('hidden_')]
    if hidden_names:
        plot_group_comparison(results, 'hidden_',
                              title='Hidden Size Comparison',
                              save_path=os.path.join(save_dir, 'hidden_size.png'))

    # 3. Layer comparison
    layer_names = [r['name'] for r in results if r.get('name', '').startswith('layers_')]
    if layer_names:
        plot_group_comparison(results, 'layers_',
                              title='Number of Layers Comparison',
                              save_path=os.path.join(save_dir, 'layers.png'))

    # 4. Learning rate comparison
    lr_names = [r['name'] for r in results if r.get('name', '').startswith('lr_')]
    if lr_names:
        plot_group_comparison(results, 'lr_',
                              title='Learning Rate Comparison',
                              save_path=os.path.join(save_dir, 'learning_rate.png'))

    # 5. Optimizer comparison
    opt_names = [r['name'] for r in results if r.get('name', '').startswith('opt_')]
    if opt_names:
        plot_group_comparison(results, 'opt_',
                              title='Optimizer Comparison',
                              save_path=os.path.join(save_dir, 'optimizer.png'))

    # 6. Dropout comparison
    dp_names = [r['name'] for r in results if r.get('name', '').startswith('dropout_')]
    if dp_names:
        plot_group_comparison(results, 'dropout_',
                              title='Dropout Comparison',
                              save_path=os.path.join(save_dir, 'dropout.png'))

    # 7. Chunk length comparison
    chunk_names = [r['name'] for r in results if r.get('name', '').startswith('chunk_')]
    if chunk_names:
        plot_group_comparison(results, 'chunk_',
                              title='Sequence Length Comparison',
                              save_path=os.path.join(save_dir, 'chunk_length.png'))

    # 8. Overall summary
    plot_summary_bar(results, metric='best_val_ppl',
                     title='Best Validation Perplexity (lower is better)',
                     save_path=os.path.join(save_dir, 'summary_ppl.png'))

    plot_summary_bar(results, metric='best_val_loss',
                     title='Best Validation Loss (lower is better)',
                     save_path=os.path.join(save_dir, 'summary_loss.png'))

    # 9. Training time
    plot_training_time(results, save_path=os.path.join(save_dir, 'training_time.png'))

    print(f"\nAll figures saved to {save_dir}/")
