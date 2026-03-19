"""
Advanced Sampling Strategies for CharRNN
- Temperature sampling (already exists)
- Top-k sampling (already exists)
- Nucleus (top-p) sampling
- Greedy sampling
- Comparison and evaluation utilities
"""

import torch
import math
from model import CharRNN


def generate_with_strategy(model, dataset, prime_str='The ', predict_len=300,
                           strategy='temperature', temperature=0.8, top_k=None,
                           top_p=None, device='cpu'):
    """
    Unified generation function supporting multiple sampling strategies.

    Strategies:
        'greedy'      - always pick the most likely character
        'temperature' - standard temperature sampling
        'top_k'       - sample from top-k most likely characters
        'top_p'       - nucleus sampling (sample from smallest set whose cumulative prob >= p)
        'top_k_p'     - combined: first apply top-k, then top-p within that set
    """
    model.eval()
    hidden = model.init_hidden(1, device)

    prime_input = torch.tensor(
        [dataset.char2idx.get(ch, 0) for ch in prime_str], dtype=torch.long
    ).unsqueeze(0).to(device)

    predicted = prime_str

    with torch.no_grad():
        # Feed prime string
        for p in range(len(prime_str) - 1):
            _, hidden = model(prime_input[:, p], hidden)

        inp = prime_input[:, -1]

        for _ in range(predict_len):
            output, hidden = model(inp, hidden)
            logits = output.data.view(-1)

            if strategy == 'greedy':
                top_i = logits.argmax().item()

            elif strategy == 'temperature':
                probs = torch.softmax(logits / temperature, dim=0)
                top_i = torch.multinomial(probs, 1).item()

            elif strategy == 'top_k':
                k = top_k or 10
                topk_logits, topk_idx = torch.topk(logits / temperature, k)
                probs = torch.softmax(topk_logits, dim=0)
                sampled = torch.multinomial(probs, 1)
                top_i = topk_idx[sampled].item()

            elif strategy == 'top_p':
                p = top_p or 0.9
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=0)

                # Sort by probability descending
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)

                # Find cutoff: smallest set with cumulative prob >= p
                cutoff_mask = cumulative_probs - sorted_probs < p
                filtered_probs = sorted_probs * cutoff_mask.float()
                filtered_probs = filtered_probs / filtered_probs.sum()

                sampled = torch.multinomial(filtered_probs, 1)
                top_i = sorted_indices[sampled].item()

            elif strategy == 'top_k_p':
                k = top_k or 40
                p = top_p or 0.9
                # First apply top-k
                topk_logits, topk_idx = torch.topk(logits / temperature, k)
                probs = torch.softmax(topk_logits, dim=0)

                # Then apply top-p within top-k
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                cutoff_mask = cumulative_probs - sorted_probs < p
                filtered_probs = sorted_probs * cutoff_mask.float()
                filtered_probs = filtered_probs / filtered_probs.sum()

                sampled = torch.multinomial(filtered_probs, 1)
                local_idx = sorted_indices[sampled]
                top_i = topk_idx[local_idx].item()

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            predicted += dataset.idx2char.get(top_i, '?')
            inp = torch.tensor([top_i], dtype=torch.long).to(device)

    return predicted


def compute_text_diversity(text):
    """
    Compute diversity metrics for generated text.
    Returns dict with:
        - unique_chars: number of unique characters
        - unique_ratio: unique chars / total chars
        - repetition_score: fraction of bigrams that are repeated
        - avg_word_length: average word length (for English text)
    """
    unique_chars = len(set(text))
    unique_ratio = unique_chars / max(len(text), 1)

    # Bigram repetition
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    unique_bigrams = len(set(bigrams))
    repetition_score = 1.0 - (unique_bigrams / max(len(bigrams), 1))

    # Word-level stats (for space-separated text)
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1) if words else 0
    unique_words = len(set(words))
    word_diversity = unique_words / max(len(words), 1) if words else 0

    return {
        'unique_chars': unique_chars,
        'char_diversity': unique_ratio,
        'bigram_repetition': repetition_score,
        'avg_word_length': avg_word_len,
        'word_diversity': word_diversity,
        'total_length': len(text),
    }


def compare_sampling_strategies(model, dataset, device, prime_str='The ',
                                 predict_len=500, temperature=0.8):
    """
    Run all sampling strategies and compare outputs + diversity metrics.
    """
    strategies = [
        {'name': 'Greedy', 'strategy': 'greedy'},
        {'name': 'Temperature (0.5)', 'strategy': 'temperature', 'temperature': 0.5},
        {'name': 'Temperature (0.8)', 'strategy': 'temperature', 'temperature': 0.8},
        {'name': 'Temperature (1.0)', 'strategy': 'temperature', 'temperature': 1.0},
        {'name': 'Top-k (k=5)', 'strategy': 'top_k', 'top_k': 5},
        {'name': 'Top-k (k=20)', 'strategy': 'top_k', 'top_k': 20},
        {'name': 'Top-k (k=40)', 'strategy': 'top_k', 'top_k': 40},
        {'name': 'Nucleus (p=0.5)', 'strategy': 'top_p', 'top_p': 0.5},
        {'name': 'Nucleus (p=0.9)', 'strategy': 'top_p', 'top_p': 0.9},
        {'name': 'Nucleus (p=0.95)', 'strategy': 'top_p', 'top_p': 0.95},
        {'name': 'Top-k+p (k=40,p=0.9)', 'strategy': 'top_k_p', 'top_k': 40, 'top_p': 0.9},
    ]

    results = []
    for s in strategies:
        text = generate_with_strategy(
            model, dataset, prime_str=prime_str, predict_len=predict_len,
            strategy=s['strategy'], temperature=s.get('temperature', temperature),
            top_k=s.get('top_k'), top_p=s.get('top_p'), device=device
        )
        metrics = compute_text_diversity(text)
        results.append({
            'name': s['name'],
            'text': text,
            'metrics': metrics,
        })

    return results


def print_sampling_comparison(results, save_path=None):
    """Print comparison table and text samples."""
    output_lines = []

    # Metrics table
    output_lines.append("=" * 90)
    output_lines.append("SAMPLING STRATEGY COMPARISON - DIVERSITY METRICS")
    output_lines.append("=" * 90)
    header = (f"{'Strategy':<25} {'Char Div':<10} {'Bigram Rep':<12} "
              f"{'Word Div':<10} {'Avg Word Len':<12}")
    output_lines.append(header)
    output_lines.append("-" * 90)

    for r in results:
        m = r['metrics']
        line = (f"{r['name']:<25} {m['char_diversity']:<10.4f} {m['bigram_repetition']:<12.4f} "
                f"{m['word_diversity']:<10.4f} {m['avg_word_length']:<12.2f}")
        output_lines.append(line)

    output_lines.append("=" * 90)

    # Text samples
    output_lines.append("\nTEXT SAMPLES")
    output_lines.append("=" * 90)
    for r in results:
        output_lines.append(f"\n--- {r['name']} ---")
        output_lines.append(r['text'][:300])

    output_lines.append("=" * 90)

    # Print
    full_output = "\n".join(output_lines)
    print(full_output)

    # Save
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(full_output)
        print(f"\nSaved to {save_path}")

    return results
