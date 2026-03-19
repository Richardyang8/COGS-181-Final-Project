# From Characters to Creativity: CharRNN Text Generation

**COGS 181 Final Project — UC San Diego**

A comprehensive study of character-level recurrent neural networks for text generation, comparing RNN/LSTM/GRU architectures, sampling strategies, and LSTM interpretability across three diverse datasets. Based on [spro/char-rnn.pytorch](https://github.com/spro/char-rnn.pytorch).

## Key Results

| Dataset | Best Model | Val Loss | Perplexity |
|---------|-----------|----------|------------|
| Tiny Shakespeare | LSTM (hidden 512) | 1.5276 | 4.61 |
| Sherlock Holmes | GRU | 1.1810 | 3.26 |
| Chinese Poetry (Tang Dynasty) | GRU | 4.7677 | 117.64 |

## Project Structure

```
├── model.py                  # CharRNN model (RNN/LSTM/GRU + Dropout)
├── data_utils.py             # Data loading, encoding, batching
├── train.py                  # Training loop with validation & perplexity
├── generate.py               # Text generation with 6 sampling strategies
├── run_experiments.py        # Automated experiment grid runner
├── visualize.py              # Publication-ready plots
├── lstm_visualization.py     # LSTM cell state analysis & pattern neurons
├── chinese_poetry_data.py    # Tang Dynasty poetry dataset download
├── main.py                   # CLI entry point
├── CharRNN_Project.ipynb     # Colab notebook (recommended)
├── report/                   # NeurIPS-format LaTeX report
│   ├── main.tex
│   └── neurips_2024.sty
└── data/                     # Downloaded datasets
```

## Quick Start (Google Colab)

1. Upload all `.py` files and the notebook to Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run cells sequentially

## Quick Start (Local/CLI)

```bash
# Quick test 
python main.py --mode quick

# Full experiments 
python main.py --mode full

# Generate text from saved model
python main.py --mode generate
```

## Experiments

### 1. Hyperparameter Study (16 configurations on Shakespeare)

| Hyperparameter | Variations |
|----------------|------------|
| Model Type | RNN, LSTM, GRU |
| Hidden Size | 128, 256, 512 |
| Num Layers | 1, 2, 3 |
| Learning Rate | 0.0005, 0.001, 0.002, 0.005 |
| Optimizer | Adam, SGD, RMSProp |
| Dropout | 0.0, 0.2, 0.5 |
| Sequence Length | 100, 200, 300 |

### 2. Cross-Dataset Comparison
- **Tiny Shakespeare**: 1.1M chars, 65 unique chars (English drama)
- **Sherlock Holmes**: 3.4M chars, 97 unique chars (English prose)
- **Chinese Classical Poetry**: 853K chars, 6,114 unique chars (10,008 Tang Dynasty poems)

### 3. Sampling Strategy Analysis (11 configurations)
Greedy, Temperature (0.5/0.8/1.0), Top-k (k=5/20/40), Nucleus (p=0.5/0.9/0.95), Combined Top-k+p — with quantitative diversity metrics (character diversity, word diversity, bigram repetition).

### 4. LSTM Interpretability
Cell state visualization identifying 6 pattern-tracking neurons:
- Lowercase (neuron 399, r=+0.675), Space (317, r=−0.637), Newline (255, r=−0.598)
- Punctuation (372, r=+0.477), Colon (267, r=−0.420), Uppercase (480, r=−0.396)

## Key Improvements over spro/char-rnn.pytorch

- Added vanilla RNN for three-way architecture comparison
- Dropout regularization between recurrent layers
- Validation split with perplexity tracking
- 6 sampling strategies (greedy, temperature, top-k, nucleus, combined)
- LSTM cell state visualization and interpretability analysis
- Chinese Classical Poetry dataset support (6,114-character vocabulary)
- Automated experiment grid (no manual re-running)
- All plots auto-generated for report
- Modern PyTorch 2.2.1 (no deprecated Variable/autograd)

## Hardware

All experiments trained on NVIDIA A30 GPU with PyTorch 2.2.1.

## Report

The full report is in NeurIPS 2024 format. See `report/main.tex` or the compiled PDF.
