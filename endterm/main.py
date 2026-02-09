"""
Multi-Horizon Financial Predictor
Deep Learning End-Term Project (Track 2: Predictive Intelligence for Time-Series)
Integrates: Sequences (LSTM/GRU), Transformers, and Regularization.
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import DEVICE, DATA_CONFIG, LSTM_CONFIG, GRU_CONFIG, TRANSFORMER_CONFIG, \
    TRAINING_CONFIG, ENSEMBLE_CONFIG, ABLATION_CONFIG, RANDOM_SEED, get_mode_config
from data.data_loader import download_stock_data, engineer_features, get_dataloaders
from models.lstm_model import MultiHorizonLSTM
from models.transformer_model import MultiHorizonTransformer
from training.trainer import Trainer
from training.ensemble import EnsemblePredictor
from evaluation.metrics import compute_all_metrics, format_metrics_table
from evaluation.ablation import AblationRunner
from visualization.plots import (
    plot_training_curves, plot_predictions, plot_attention_heatmap,
    plot_ablation_results, plot_feature_importance, plot_model_comparison,
    plot_residuals,
)

BASE_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
RESULTS_DIR = BASE_RESULTS_DIR  # Will be updated when running experiments


def get_timestamped_results_dir(mode):
    """Create and return a timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_subdir = f"results_{mode}_{timestamp}"
    path = os.path.join(BASE_RESULTS_DIR, results_subdir)
    os.makedirs(path, exist_ok=True)
    return path


def set_seed(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_train_config():
    """Merge training config with mode-dependent epochs and patience."""
    mode = get_mode_config()
    tc = dict(TRAINING_CONFIG)
    tc['epochs'] = mode['epochs']
    tc['patience'] = mode['patience']
    return tc


# ─── State holders ───────────────────────────────────────────────────────────
raw_data = {}
data_loaders = {}         # ticker -> (train, val, test, scaler, feature_names)
trained_models = {}       # model_name -> (model, history)
eval_results = {}         # model_name -> {horizon: metrics}
ensemble_results = {}     # ticker -> ensemble output
ablation_df = None


def menu_download_data():
    """1. Download & Prepare Data"""
    global raw_data, data_loaders
    mode = get_mode_config()
    tickers = mode['tickers']
    print(f"\nDownloading data for: {tickers}")
    raw_data = download_stock_data(tickers, DATA_CONFIG['start_date'], DATA_CONFIG['end_date'])

    loader_config = dict(DATA_CONFIG)
    loader_config['batch_size'] = TRAINING_CONFIG['batch_size']

    for ticker in tickers:
        print(f"\nPreparing {ticker}...")
        loaders = get_dataloaders(raw_data[ticker], loader_config)
        data_loaders[ticker] = loaders

    print(f"\nData ready for {len(data_loaders)} ticker(s).")


def _ensure_data():
    if not data_loaders:
        print("No data loaded. Running data download first...")
        menu_download_data()


def _get_primary_ticker():
    return list(data_loaders.keys())[0]


def _get_input_dim():
    ticker = _get_primary_ticker()
    feature_names = data_loaders[ticker][4]
    return len(feature_names)


def _train_model(model_name, model):
    """Train a model and store results."""
    _ensure_data()
    ticker = _get_primary_ticker()
    train_loader, val_loader, test_loader, scaler, feature_names = data_loaders[ticker]

    tc = get_train_config()
    horizons = DATA_CONFIG['horizons']

    print(f"\nTraining {model_name} on {ticker} ({tc['epochs']} epochs, device={DEVICE})...")
    set_seed()

    trainer = Trainer(model, tc, DEVICE, horizons)
    history = trainer.train(train_loader, val_loader)

    # Evaluate
    print(f"\nEvaluating {model_name}...")
    results = trainer.evaluate(test_loader)
    model_metrics = {}
    for h in horizons:
        key = f'h{h}'
        metrics = compute_all_metrics(results[key]['targets'], results[key]['predictions'])
        model_metrics[key] = metrics
        print(f"  {key}: RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}, "
              f"Dir_Acc={metrics['Dir_Acc']:.2f}%, R²={metrics['R2']:.4f}")

    trained_models[model_name] = (model, history)
    eval_results[model_name] = model_metrics

    # Store raw predictions for plotting
    trained_models[model_name] = (model, history, results)

    return model, history, results


def menu_train_lstm():
    """2. Train LSTM Model"""
    _ensure_data()
    input_dim = _get_input_dim()
    model = MultiHorizonLSTM(
        input_dim=input_dim,
        hidden_dim=LSTM_CONFIG['hidden_dim'],
        num_layers=LSTM_CONFIG['num_layers'],
        horizons=DATA_CONFIG['horizons'],
        dropout=LSTM_CONFIG['dropout'],
        use_batch_norm=LSTM_CONFIG['use_batch_norm'],
        cell_type='LSTM',
    )
    _train_model('LSTM', model)


def menu_train_gru():
    """3. Train GRU Model"""
    _ensure_data()
    input_dim = _get_input_dim()
    model = MultiHorizonLSTM(
        input_dim=input_dim,
        hidden_dim=GRU_CONFIG['hidden_dim'],
        num_layers=GRU_CONFIG['num_layers'],
        horizons=DATA_CONFIG['horizons'],
        dropout=GRU_CONFIG['dropout'],
        use_batch_norm=GRU_CONFIG['use_batch_norm'],
        cell_type='GRU',
    )
    _train_model('GRU', model)


def menu_train_transformer():
    """4. Train Transformer Model"""
    _ensure_data()
    input_dim = _get_input_dim()
    model = MultiHorizonTransformer(
        input_dim=input_dim,
        d_model=TRANSFORMER_CONFIG['d_model'],
        nhead=TRANSFORMER_CONFIG['nhead'],
        num_layers=TRANSFORMER_CONFIG['num_layers'],
        dim_ff=TRANSFORMER_CONFIG['dim_feedforward'],
        horizons=DATA_CONFIG['horizons'],
        dropout=TRANSFORMER_CONFIG['dropout'],
        use_batch_norm=TRANSFORMER_CONFIG['use_batch_norm'],
    )
    _train_model('Transformer', model)


def menu_ensemble():
    """5. Run Ensemble Prediction"""
    global ensemble_results
    _ensure_data()
    ticker = _get_primary_ticker()
    train_loader, val_loader, test_loader, scaler, feature_names = data_loaders[ticker]
    input_dim = len(feature_names)
    horizons = DATA_CONFIG['horizons']
    mode = get_mode_config()

    tc = get_train_config()

    print(f"\nTraining ensemble ({mode['num_seeds']} models) on {ticker}...")

    model_kwargs = {
        'input_dim': input_dim,
        'hidden_dim': LSTM_CONFIG['hidden_dim'],
        'num_layers': LSTM_CONFIG['num_layers'],
        'horizons': horizons,
        'dropout': LSTM_CONFIG['dropout'],
        'use_batch_norm': LSTM_CONFIG['use_batch_norm'],
        'cell_type': 'LSTM',
    }

    ensemble = EnsemblePredictor(
        model_class=MultiHorizonLSTM,
        model_kwargs=model_kwargs,
        train_config=tc,
        device=DEVICE,
        num_models=mode['num_seeds'],
        base_seed=ENSEMBLE_CONFIG['base_seed'],
        horizons=horizons,
    )

    ensemble.train_ensemble(train_loader, val_loader)
    results = ensemble.predict_with_uncertainty(test_loader)
    ensemble_results[ticker] = results

    print("\nEnsemble Results:")
    for h in horizons:
        key = f'h{h}'
        metrics = compute_all_metrics(results[key]['targets'], results[key]['predictions'])
        avg_ci = np.mean(results[key]['upper_bound'] - results[key]['lower_bound'])
        print(f"  {key}: RMSE={metrics['RMSE']:.6f}, Dir_Acc={metrics['Dir_Acc']:.2f}%, "
              f"Avg CI Width={avg_ci:.6f}")


def menu_ablation():
    """6. Run Ablation Studies"""
    global ablation_df, RESULTS_DIR
    _ensure_data()
    ticker = _get_primary_ticker()
    train_loader, val_loader, test_loader, scaler, feature_names = data_loaders[ticker]
    input_dim = len(feature_names)
    mode = get_mode_config()

    # Create timestamped results directory
    RESULTS_DIR = get_timestamped_results_dir(config.CURRENT_MODE)

    ablation_config = {
        'lr': TRAINING_CONFIG['lr'],
        'weight_decay': TRAINING_CONFIG['weight_decay'],
        'ablation_epochs': mode['ablation_epochs'],
        'patience': mode['patience'],
        'ablation': ABLATION_CONFIG,
    }

    print(f"\nRunning ablation studies (epochs={mode['ablation_epochs']})...")
    runner = AblationRunner(ablation_config, DEVICE, DATA_CONFIG['horizons'])
    ablation_df = runner.run_all(train_loader, val_loader, test_loader, input_dim)

    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, 'ablation_results.csv')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ablation_df.to_csv(csv_path, index=False)
    print(f"\nAblation results saved to {csv_path}")
    print(ablation_df.to_string(index=False))


def menu_visualize():
    """7. Generate All Visualizations"""
    global RESULTS_DIR
    # Create timestamped results directory if not already set by ablation
    if RESULTS_DIR == BASE_RESULTS_DIR:
        RESULTS_DIR = get_timestamped_results_dir(config.CURRENT_MODE)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    horizons = DATA_CONFIG['horizons']

    # Training curves
    if trained_models:
        names = list(trained_models.keys())
        histories = [trained_models[n][1] for n in names]
        plot_training_curves(histories, names,
                             os.path.join(RESULTS_DIR, 'training_curves.png'))

    # Model comparison
    if len(eval_results) > 1:
        plot_model_comparison(eval_results,
                              os.path.join(RESULTS_DIR, 'model_comparison.png'))

    # Per-model prediction plots & residuals
    for name in trained_models:
        model, history, results = trained_models[name]
        for h in horizons:
            key = f'h{h}'
            y_true = results[key]['targets']
            y_pred = results[key]['predictions']

            # Check if ensemble CIs exist
            ticker = _get_primary_ticker()
            ci_lower = ci_upper = None
            if ticker in ensemble_results and key in ensemble_results[ticker]:
                ci_lower = ensemble_results[ticker][key]['lower_bound']
                ci_upper = ensemble_results[ticker][key]['upper_bound']

            plot_predictions(y_true, y_pred, ci_lower, ci_upper, key,
                             os.path.join(RESULTS_DIR, f'{name}_predictions_{key}.png'))
            plot_residuals(y_true, y_pred, key,
                           os.path.join(RESULTS_DIR, f'{name}_residuals_{key}.png'))

    # Ensemble prediction plots with CI
    for ticker in ensemble_results:
        for h in horizons:
            key = f'h{h}'
            r = ensemble_results[ticker][key]
            plot_predictions(
                r['targets'], r['predictions'],
                r['lower_bound'], r['upper_bound'],
                f'{key} (Ensemble)',
                os.path.join(RESULTS_DIR, f'ensemble_{ticker}_{key}.png'),
            )

    # Attention heatmap (if Transformer was trained)
    if 'Transformer' in trained_models:
        model = trained_models['Transformer'][0]
        _ensure_data()
        ticker = _get_primary_ticker()
        test_loader = data_loaders[ticker][2]
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            attn = model.get_attention_weights(X_batch[:1])
            if hasattr(attn, 'detach'):
                attn = attn.detach().cpu().numpy()
            plot_attention_heatmap(attn,
                                   os.path.join(RESULTS_DIR, 'attention_heatmap.png'))
            break

    # Feature importance (using first available model)
    if trained_models:
        name = list(trained_models.keys())[0]
        model = trained_models[name][0]
        ticker = _get_primary_ticker()
        test_loader = data_loaders[ticker][2]
        feature_names = data_loaders[ticker][4]
        plot_feature_importance(model, feature_names, test_loader, DEVICE,
                                os.path.join(RESULTS_DIR, 'feature_importance.png'))

    # Ablation plots
    if ablation_df is not None:
        plot_ablation_results(ablation_df,
                              os.path.join(RESULTS_DIR, 'ablation_results.png'))

    print(f"\nAll visualizations saved to {RESULTS_DIR}/")


def menu_full_pipeline():
    """8. Run Full Pipeline"""
    global RESULTS_DIR
    print("\n" + "=" * 60)
    print("  FULL PIPELINE")
    print(f"  Mode: {config.CURRENT_MODE}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    print("\n[Step 1/7] Download & Prepare Data")
    menu_download_data()

    print("\n[Step 2/7] Train LSTM")
    menu_train_lstm()

    print("\n[Step 3/7] Train GRU")
    menu_train_gru()

    print("\n[Step 4/7] Train Transformer")
    menu_train_transformer()

    print("\n[Step 5/7] Ensemble Prediction")
    menu_ensemble()

    print("\n[Step 6/7] Ablation Studies")
    menu_ablation()

    print("\n[Step 7/7] Generate Visualizations")
    menu_visualize()

    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    if eval_results:
        print(format_metrics_table(eval_results))
    print(f"\nAll outputs saved to: {RESULTS_DIR}/")
    print("Pipeline complete!")


def mode_selection_menu():
    """Menu to select execution mode."""
    while True:
        print(f"\n{'=' * 50}")
        print(f"  Select Execution Mode")
        print(f"{'=' * 50}")
        print("  1. QUICK  - Fast testing (30 epochs, 1 ticker, 2 seeds)")
        print("  2. HYBRID - Moderate (100 epochs, 3 tickers, 3 seeds)")
        print("  3. FULL   - Full experiments (200 epochs, 5 tickers, 5 seeds)")
        print("  0. Cancel")
        print(f"{'=' * 50}")

        choice = input("  Select mode: ").strip()

        if choice == '1':
            config.CURRENT_MODE = 'QUICK'
            print(f"  Mode set to: QUICK")
            return
        elif choice == '2':
            config.CURRENT_MODE = 'HYBRID'
            print(f"  Mode set to: HYBRID")
            return
        elif choice == '3':
            config.CURRENT_MODE = 'FULL'
            print(f"  Mode set to: FULL")
            return
        elif choice == '0':
            print("  Cancelled.")
            return
        else:
            print("  Invalid option, try again.")


def interactive_menu():
    """Interactive menu for running experiments."""
    while True:
        print(f"\n{'=' * 50}")
        print(f"  Multi-Horizon Financial Predictor")
        print(f"  Mode: {config.CURRENT_MODE} | Device: {DEVICE}")
        print(f"{'=' * 50}")
        print("  1. Download & Prepare Data")
        print("  2. Train LSTM Model")
        print("  3. Train GRU Model")
        print("  4. Train Transformer Model")
        print("  5. Run Ensemble Prediction (with uncertainty)")
        print("  6. Run Ablation Studies")
        print("  7. Generate All Visualizations")
        print("  8. Run Full Pipeline (1-7)")
        print("  9. Change Execution Mode")
        print("  0. Exit")
        print(f"{'=' * 50}")

        choice = input("  Select option: ").strip()

        if choice == '1':
            menu_download_data()
        elif choice == '2':
            menu_train_lstm()
        elif choice == '3':
            menu_train_gru()
        elif choice == '4':
            menu_train_transformer()
        elif choice == '5':
            menu_ensemble()
        elif choice == '6':
            menu_ablation()
        elif choice == '7':
            menu_visualize()
        elif choice == '8':
            menu_full_pipeline()
        elif choice == '9':
            mode_selection_menu()
        elif choice == '0':
            print("Goodbye!")
            break
        else:
            print("Invalid option, try again.")


def main():
    parser = argparse.ArgumentParser(description='Multi-Horizon Financial Predictor')
    parser.add_argument('--mode', choices=['QUICK', 'HYBRID', 'FULL'],
                        default=None, help='Execution mode')
    parser.add_argument('--run', type=int, default=None,
                        help='Menu option to run non-interactively (e.g. --run 8)')
    args = parser.parse_args()

    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")

    if args.run is not None:
        # Non-interactive mode with explicit mode
        if args.mode is None:
            args.mode = 'QUICK'
        config.CURRENT_MODE = args.mode
        print(f"Mode: {config.CURRENT_MODE}")
        actions = {
            1: menu_download_data,
            2: menu_train_lstm,
            3: menu_train_gru,
            4: menu_train_transformer,
            5: menu_ensemble,
            6: menu_ablation,
            7: menu_visualize,
            8: menu_full_pipeline,
        }
        if args.run in actions:
            actions[args.run]()
        else:
            print(f"Invalid run option: {args.run}")
    else:
        # Interactive mode: show mode selection menu first if not specified
        if args.mode is not None:
            config.CURRENT_MODE = args.mode
        else:
            mode_selection_menu()
        print(f"Mode: {config.CURRENT_MODE}")
        interactive_menu()


if __name__ == '__main__':
    main()
