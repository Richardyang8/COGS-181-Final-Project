"""
=============================================================
CharRNN Final Project - Main Pipeline
=============================================================
Run this script or use it as a reference for your Colab notebook.

Usage:
    python main.py --mode quick     # Quick test (~5 min)
    python main.py --mode full      # Full experiments (~30-60 min)
    python main.py --mode generate  # Generate text from saved model
=============================================================
"""

import argparse
import os
import torch

from data_utils import TextDataset, download_shakespeare, download_sherlock
from model import CharRNN
from train import train_model
from generate import generate_text, generate_samples_at_temperatures
from run_experiments import (run_experiment_grid,
                              get_core_experiments,
                              get_quick_test_experiments)
from visualize import (generate_all_plots, plot_temperature_samples,
                        plot_group_comparison, plot_summary_bar)


def main():
    parser = argparse.ArgumentParser(description='CharRNN Final Project')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['quick', 'full', 'generate', 'dataset2'],
                        help='Run mode')
    parser.add_argument('--dataset', type=str, default='shakespeare',
                        choices=['shakespeare', 'sherlock'],
                        help='Dataset to use')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ========================================
    # Step 1: Download and load dataset
    # ========================================
    if args.dataset == 'shakespeare':
        filepath = download_shakespeare()
    else:
        filepath = download_sherlock()

    dataset = TextDataset(filepath, val_fraction=0.1)

    # ========================================
    # Step 2: Run experiments
    # ========================================
    if args.mode == 'quick':
        print("\n>>> Running QUICK test experiments...")
        experiments = get_quick_test_experiments(n_epochs=500, print_every=100)
        results = run_experiment_grid(dataset, experiments, save_dir="results_quick")
        generate_all_plots(results, save_dir="figures_quick")

    elif args.mode == 'full':
        print("\n>>> Running FULL experiment grid...")
        experiments = get_core_experiments(n_epochs=2000, print_every=200)
        results = run_experiment_grid(dataset, experiments, save_dir="results")
        generate_all_plots(results, save_dir="figures")

        # Generate temperature comparison with best model
        best = min([r for r in results if 'error' not in r],
                   key=lambda r: r['best_val_loss'])
        print(f"\nBest model: {best['name']}")

        # Load best model for generation
        checkpoint = torch.load(best['config']['save_path'], map_location=device)
        model = CharRNN(
            input_size=dataset.n_characters,
            hidden_size=best['config']['hidden_size'],
            output_size=dataset.n_characters,
            model_type=best['config']['model_type'],
            n_layers=best['config']['n_layers'],
            dropout=best['config'].get('dropout', 0.0)
        ).to(device)
        model.load_state_dict(checkpoint['model_state'])

        samples = generate_samples_at_temperatures(
            model, dataset, device,
            prime_str="ROMEO: ",
            predict_len=500,
            temperatures=[0.2, 0.5, 0.8, 1.0, 1.5]
        )
        plot_temperature_samples(samples, save_path="figures/temperature_samples.txt")

    elif args.mode == 'generate':
        # Load a saved model and generate text
        model_path = "results/model_lstm.pt"
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}. Run training first.")
            return

        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']
        model = CharRNN(
            input_size=dataset.n_characters,
            hidden_size=config['hidden_size'],
            output_size=dataset.n_characters,
            model_type=config['model_type'],
            n_layers=config['n_layers'],
            dropout=config.get('dropout', 0.0)
        ).to(device)
        model.load_state_dict(checkpoint['model_state'])

        print("\nGenerating text...")
        for temp in [0.2, 0.5, 0.8, 1.0, 1.5]:
            print(f"\n--- Temperature = {temp} ---")
            text = generate_text(model, dataset, prime_str="The ",
                                predict_len=500, temperature=temp, device=device)
            print(text)

    elif args.mode == 'dataset2':
        # Run experiments on second dataset for comparison
        print("\n>>> Running experiments on Sherlock Holmes dataset...")
        filepath2 = download_sherlock()
        dataset2 = TextDataset(filepath2, val_fraction=0.1)
        experiments = get_quick_test_experiments(n_epochs=2000, print_every=200)
        # Only run model type comparison
        experiments = [e for e in get_core_experiments(n_epochs=2000, print_every=200)
                       if e['name'].startswith('model_')]
        results = run_experiment_grid(dataset2, experiments, save_dir="results_sherlock")
        generate_all_plots(results, save_dir="figures_sherlock")


if __name__ == '__main__':
    main()
