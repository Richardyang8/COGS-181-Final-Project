"""
Text Generation Module for CharRNN
- Temperature-based sampling
- Top-k sampling
- Generate from priming string
"""

import torch


def generate_text(model, dataset, prime_str='A', predict_len=200,
                  temperature=0.8, device='cpu', top_k=None):
    """
    Generate text using trained model.

    Args:
        model: trained CharRNN model
        dataset: TextDataset (for char mappings)
        prime_str: string to prime/seed the generation
        predict_len: number of characters to generate
        temperature: controls randomness (lower = more conservative)
        device: 'cpu' or 'cuda'
        top_k: if set, only sample from top-k most likely characters

    Returns:
        generated string
    """
    model.eval()
    hidden = model.init_hidden(1, device)

    # Encode prime string
    prime_input = torch.tensor(
        [dataset.char2idx.get(ch, 0) for ch in prime_str], dtype=torch.long
    ).unsqueeze(0).to(device)  # (1, len(prime_str))

    predicted = prime_str

    # Feed prime string to build up hidden state
    with torch.no_grad():
        for p in range(len(prime_str) - 1):
            _, hidden = model(prime_input[:, p], hidden)

        # Start generating from last character of prime
        inp = prime_input[:, -1]

        for _ in range(predict_len):
            output, hidden = model(inp, hidden)

            # Apply temperature
            output_dist = output.data.view(-1).div(temperature)

            if top_k is not None:
                # Top-k sampling
                topk_vals, topk_idx = torch.topk(output_dist, top_k)
                probs = torch.softmax(topk_vals, dim=0)
                sampled = torch.multinomial(probs, 1)
                top_i = topk_idx[sampled].item()
            else:
                # Standard temperature sampling
                probs = torch.softmax(output_dist, dim=0)
                top_i = torch.multinomial(probs, 1).item()

            # Add predicted character
            predicted_char = dataset.idx2char.get(top_i, '?')
            predicted += predicted_char

            # Next input
            inp = torch.tensor([top_i], dtype=torch.long).to(device)

    return predicted


def generate_samples_at_temperatures(model, dataset, device,
                                      prime_str="The ", predict_len=300,
                                      temperatures=[0.2, 0.5, 0.8, 1.0, 1.5]):
    """Generate text at multiple temperatures for comparison."""
    samples = {}
    for temp in temperatures:
        text = generate_text(model, dataset, prime_str=prime_str,
                            predict_len=predict_len, temperature=temp, device=device)
        samples[temp] = text
    return samples
