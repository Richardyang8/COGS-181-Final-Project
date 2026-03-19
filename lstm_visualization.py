"""
LSTM Hidden State Visualization
- Record hidden/cell states during text generation
- Heatmap of hidden state activations over characters
- Identify neurons that track specific patterns (quotes, newlines, etc.)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11
import os


def record_hidden_states(model, dataset, text_input, device='cpu', layer=-1):
    """
    Feed a text string through the model and record hidden states at each step.

    Args:
        model: trained CharRNN (should be LSTM)
        dataset: TextDataset
        text_input: string to feed through the model
        device: 'cpu' or 'cuda'
        layer: which layer's hidden state to record (-1 = last layer)

    Returns:
        dict with:
            - hidden_states: numpy array (seq_len, hidden_size)
            - cell_states: numpy array (seq_len, hidden_size) [LSTM only]
            - chars: list of characters
            - predictions: list of predicted next characters
    """
    model.eval()
    hidden = model.init_hidden(1, device)

    indices = [dataset.char2idx.get(ch, 0) for ch in text_input]
    input_tensor = torch.tensor(indices, dtype=torch.long).to(device)

    hidden_states = []
    cell_states = []
    predictions = []

    with torch.no_grad():
        for i in range(len(text_input)):
            inp = input_tensor[i].unsqueeze(0)  # (1,)
            output, hidden = model(inp, hidden)

            # Extract hidden state for the specified layer
            if model.model_type == 'lstm':
                h = hidden[0][layer].squeeze(0).cpu().numpy()  # (hidden_size,)
                c = hidden[1][layer].squeeze(0).cpu().numpy()
                cell_states.append(c)
            else:
                h = hidden[layer].squeeze(0).cpu().numpy()

            hidden_states.append(h)

            # Get prediction
            probs = torch.softmax(output.view(-1), dim=0)
            pred_idx = probs.argmax().item()
            predictions.append(dataset.idx2char.get(pred_idx, '?'))

    result = {
        'hidden_states': np.array(hidden_states),
        'chars': list(text_input),
        'predictions': predictions,
    }
    if cell_states:
        result['cell_states'] = np.array(cell_states)

    return result


def plot_hidden_state_heatmap(record, n_neurons=20, state_type='hidden',
                               title=None, save_path=None, figsize=None):
    """
    Plot heatmap of hidden state activations over characters.

    Args:
        record: output from record_hidden_states
        n_neurons: number of neurons to display (picks most variable ones)
        state_type: 'hidden' or 'cell'
        title: plot title
        save_path: save figure path
    """
    if state_type == 'cell' and 'cell_states' in record:
        states = record['cell_states']
        default_title = 'LSTM Cell State Activations'
    else:
        states = record['hidden_states']
        default_title = 'Hidden State Activations'

    chars = record['chars']

    # Select most variable neurons (most interesting to visualize)
    variance = np.var(states, axis=0)
    top_indices = np.argsort(variance)[-n_neurons:][::-1]
    selected_states = states[:, top_indices].T  # (n_neurons, seq_len)

    # Determine figure size
    seq_len = len(chars)
    if figsize is None:
        width = max(14, min(seq_len * 0.15, 40))
        height = max(6, n_neurons * 0.35)
        figsize = (width, height)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(selected_states, aspect='auto', cmap='RdBu_r', interpolation='nearest')

    # X-axis: characters
    ax.set_xticks(range(len(chars)))
    ax.set_xticklabels(chars, fontsize=8, fontfamily='monospace')
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Y-axis: neuron indices
    ax.set_yticks(range(n_neurons))
    ax.set_yticklabels([f'n{i}' for i in top_indices], fontsize=8)

    ax.set_xlabel('Input Character')
    ax.set_ylabel('Neuron (sorted by variance)')
    ax.set_title(title or default_title)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def find_pattern_neurons(record, state_type='cell'):
    """
    Identify neurons that respond to specific patterns in text.
    Analyzes correlation between neuron activations and character types.

    Returns dict mapping pattern names to (neuron_index, correlation) pairs.
    """
    if state_type == 'cell' and 'cell_states' in record:
        states = record['cell_states']
    else:
        states = record['hidden_states']

    chars = record['chars']
    n_neurons = states.shape[1]

    # Define patterns to look for
    patterns = {
        'newline': [1.0 if ch == '\n' else 0.0 for ch in chars],
        'space': [1.0 if ch == ' ' else 0.0 for ch in chars],
        'uppercase': [1.0 if ch.isupper() else 0.0 for ch in chars],
        'lowercase': [1.0 if ch.islower() else 0.0 for ch in chars],
        'punctuation': [1.0 if ch in '.,;:!?\'"()-' else 0.0 for ch in chars],
        'digit': [1.0 if ch.isdigit() else 0.0 for ch in chars],
        'quote': [1.0 if ch in '\'"' else 0.0 for ch in chars],
        'colon': [1.0 if ch == ':' else 0.0 for ch in chars],
        'period': [1.0 if ch == '.' else 0.0 for ch in chars],
    }

    results = {}
    for pattern_name, pattern_vec in patterns.items():
        if sum(pattern_vec) == 0:
            continue
        pattern_arr = np.array(pattern_vec)
        best_corr = 0
        best_neuron = 0
        for n in range(n_neurons):
            neuron_vals = states[:, n]
            # Pearson correlation
            if np.std(neuron_vals) > 1e-8 and np.std(pattern_arr) > 1e-8:
                corr = np.corrcoef(neuron_vals, pattern_arr)[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_neuron = n
        results[pattern_name] = (best_neuron, best_corr)

    return results


def plot_pattern_neurons(record, save_path=None):
    """
    Find and plot neurons that track specific patterns.
    Creates a multi-panel figure showing top pattern-tracking neurons.
    """
    state_type = 'cell' if 'cell_states' in record else 'hidden'
    if state_type == 'cell':
        states = record['cell_states']
    else:
        states = record['hidden_states']

    chars = record['chars']
    pattern_results = find_pattern_neurons(record, state_type=state_type)

    # Filter to patterns with |correlation| > 0.2
    significant = {k: v for k, v in pattern_results.items() if abs(v[1]) > 0.2}

    if not significant:
        print("No strongly pattern-tracking neurons found.")
        return

    # Sort by absolute correlation
    sorted_patterns = sorted(significant.items(), key=lambda x: abs(x[1][1]), reverse=True)
    n_plots = min(len(sorted_patterns), 6)
    sorted_patterns = sorted_patterns[:n_plots]

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    x = range(len(chars))

    for idx, (pattern_name, (neuron_idx, corr)) in enumerate(sorted_patterns):
        ax = axes[idx]
        neuron_vals = states[:, neuron_idx]

        ax.plot(x, neuron_vals, 'b-', linewidth=1, alpha=0.8)
        ax.set_ylabel(f'Neuron {neuron_idx}')
        ax.set_title(f'Pattern: {pattern_name} (neuron {neuron_idx}, corr={corr:.3f})',
                     fontsize=11)

        # Highlight positions matching the pattern
        if pattern_name == 'newline':
            positions = [i for i, ch in enumerate(chars) if ch == '\n']
        elif pattern_name == 'space':
            positions = [i for i, ch in enumerate(chars) if ch == ' ']
        elif pattern_name == 'uppercase':
            positions = [i for i, ch in enumerate(chars) if ch.isupper()]
        elif pattern_name == 'punctuation':
            positions = [i for i, ch in enumerate(chars) if ch in '.,;:!?\'"()-']
        elif pattern_name == 'quote':
            positions = [i for i, ch in enumerate(chars) if ch in '\'"']
        elif pattern_name == 'colon':
            positions = [i for i, ch in enumerate(chars) if ch == ':']
        elif pattern_name == 'period':
            positions = [i for i, ch in enumerate(chars) if ch == '.']
        else:
            positions = []

        for pos in positions:
            ax.axvline(x=pos, color='red', alpha=0.3, linewidth=1)

        ax.grid(True, alpha=0.2)

    # Bottom x-axis labels
    axes[-1].set_xticks(range(len(chars)))
    axes[-1].set_xticklabels(chars, fontsize=7, fontfamily='monospace')
    axes[-1].set_xlabel('Input Character')

    plt.suptitle(f'LSTM {state_type.title()} State - Pattern Tracking Neurons',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

    # Print summary
    print("\nPattern Neuron Summary:")
    print(f"{'Pattern':<15} {'Neuron':<10} {'Correlation':<12}")
    print("-" * 37)
    for pattern_name, (neuron_idx, corr) in sorted_patterns:
        print(f"{pattern_name:<15} {neuron_idx:<10} {corr:<12.4f}")


def run_full_visualization(model, dataset, device, text_sample=None,
                           save_dir='figures', prefix=''):
    """
    Run the complete visualization pipeline.

    Args:
        model: trained LSTM model
        dataset: TextDataset
        device: torch device
        text_sample: text to feed through model (if None, uses first 150 chars of dataset)
        save_dir: directory to save figures
        prefix: prefix for saved file names
    """
    os.makedirs(save_dir, exist_ok=True)

    if text_sample is None:
        text_sample = dataset.raw_text[:150]

    print(f"Analyzing text ({len(text_sample)} chars): {text_sample[:60]}...")
    record = record_hidden_states(model, dataset, text_sample, device)

    # 1. Hidden state heatmap
    print("\n1. Hidden State Heatmap")
    plot_hidden_state_heatmap(
        record, n_neurons=20, state_type='hidden',
        title='Hidden State Activations (Top 20 Neurons by Variance)',
        save_path=os.path.join(save_dir, f'{prefix}hidden_heatmap.png')
    )

    # 2. Cell state heatmap (LSTM only)
    if 'cell_states' in record:
        print("\n2. Cell State Heatmap")
        plot_hidden_state_heatmap(
            record, n_neurons=20, state_type='cell',
            title='LSTM Cell State Activations (Top 20 Neurons by Variance)',
            save_path=os.path.join(save_dir, f'{prefix}cell_heatmap.png')
        )

    # 3. Pattern-tracking neurons
    print("\n3. Pattern-Tracking Neurons")
    plot_pattern_neurons(
        record,
        save_path=os.path.join(save_dir, f'{prefix}pattern_neurons.png')
    )

    print(f"\nVisualization complete! Figures saved to {save_dir}/")
