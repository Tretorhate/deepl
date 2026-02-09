import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 100})


def _save(fig, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved: {save_path}")


def plot_training_curves(histories, model_names, save_path):
    """Plot training & validation loss curves for multiple models.

    Args:
        histories: list of history dicts (each has 'train_loss', 'val_loss')
        model_names: list of model name strings
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for hist, name in zip(histories, model_names):
        axes[0].plot(hist['train_loss'], label=f'{name} Train')
        axes[1].plot(hist['val_loss'], label=f'{name} Val')

    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    fig.suptitle('Training Curves', fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)


def plot_predictions(y_true, y_pred, ci_lower, ci_upper, horizon, save_path):
    """Plot predictions vs actuals with confidence bands.

    Args:
        y_true: array of true values
        y_pred: array of predicted values
        ci_lower: lower bound of 95% CI (or None)
        ci_upper: upper bound of 95% CI (or None)
        horizon: horizon label string (e.g. 'h1')
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    t = np.arange(len(y_true))

    ax.plot(t, y_true, label='Actual', color='#2196F3', alpha=0.8, linewidth=1.2)
    ax.plot(t, y_pred, label='Predicted', color='#FF5722', alpha=0.8, linewidth=1.2)

    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(t, ci_lower, ci_upper, color='#FF5722', alpha=0.15,
                        label='95% CI')

    ax.set_title(f'Predictions vs Actuals — Horizon {horizon}', fontsize=13)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Predicted Return')
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)


def plot_attention_heatmap(attention_weights, save_path, max_len=60):
    """Plot attention weight heatmap.

    Args:
        attention_weights: (nhead, seq_len, seq_len) or (batch, nhead, seq_len, seq_len)
        save_path: path to save figure
        max_len: truncate sequence length for readability
    """
    if attention_weights.ndim == 4:
        attn = attention_weights[0]  # first sample
    else:
        attn = attention_weights

    nhead = attn.shape[0]
    cols = min(nhead, 4)
    rows = (nhead + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if nhead == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i in range(nhead):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[0, c]
        data = attn[i, :max_len, :max_len]
        if hasattr(data, 'detach'):
            data = data.detach().cpu().numpy()
        sns.heatmap(data, ax=ax, cmap='viridis', cbar=True, square=True)
        ax.set_title(f'Head {i+1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')

    # Hide unused subplots
    for i in range(nhead, rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[0, c]
        ax.set_visible(False)

    fig.suptitle('Transformer Attention Heatmap', fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)


def plot_ablation_results(ablation_df, save_path):
    """Plot ablation study results as grouped bar charts.

    Args:
        ablation_df: DataFrame with columns [experiment, variant, horizon, MSE, RMSE, ...]
        save_path: path to save figure
    """
    experiments = ablation_df['experiment'].unique()
    n_exp = len(experiments)
    fig, axes = plt.subplots(n_exp, 1, figsize=(12, 4 * n_exp))
    if n_exp == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        subset = ablation_df[ablation_df['experiment'] == exp]
        pivot = subset.pivot_table(index='variant', columns='horizon', values='RMSE')
        pivot.plot(kind='bar', ax=ax, rot=0)
        ax.set_title(f'Ablation: {exp}')
        ax.set_ylabel('RMSE')
        ax.legend(title='Horizon')

    fig.suptitle('Ablation Study Results', fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)


def plot_feature_importance(model, feature_names, test_loader, device, save_path):
    """Gradient-based feature importance analysis.

    Args:
        model: trained model
        feature_names: list of feature names
        test_loader: DataLoader for test data
        device: torch.device
        save_path: path to save figure
    """
    model.eval()
    importances = np.zeros(len(feature_names))
    count = 0

    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(device).requires_grad_(True)
        preds = model(X_batch)
        # Sum all horizon predictions
        total = sum(preds[k].sum() for k in preds)
        total.backward()
        grad = X_batch.grad.abs().mean(dim=(0, 1)).cpu().numpy()  # avg over batch & time
        importances += grad
        count += 1
        if count >= 5:  # sample a few batches
            break

    importances /= count
    # Sort by importance
    idx = np.argsort(importances)[::-1][:20]  # top 20

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [feature_names[i] for i in idx]
    values = importances[idx]
    ax.barh(range(len(idx)), values[::-1], color='#4CAF50')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel('Mean |Gradient|')
    ax.set_title('Top 20 Features by Gradient Importance')
    fig.tight_layout()
    _save(fig, save_path)


def plot_model_comparison(results_dict, save_path):
    """Compare LSTM vs GRU vs Transformer across horizons.

    Args:
        results_dict: {model_name: {horizon: metrics_dict}}
        save_path: path to save figure
    """
    metrics_to_plot = ['RMSE', 'MAE', 'Dir_Acc', 'R2']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics_to_plot):
        models = list(results_dict.keys())
        horizons = list(list(results_dict.values())[0].keys())
        x = np.arange(len(horizons))
        width = 0.8 / len(models)

        for i, model_name in enumerate(models):
            values = [results_dict[model_name][h][metric] for h in horizons]
            ax.bar(x + i * width, values, width, label=model_name, alpha=0.85)

        ax.set_xlabel('Horizon')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Model & Horizon')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(horizons)
        ax.legend()

    fig.suptitle('Model Comparison', fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)


def plot_residuals(y_true, y_pred, horizon, save_path):
    """Residual analysis plot.

    Args:
        y_true, y_pred: arrays
        horizon: horizon label
        save_path: path to save figure
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Residuals over time
    axes[0].plot(residuals, alpha=0.6, linewidth=0.8)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_title(f'Residuals Over Time ({horizon})')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Residual')

    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='#4CAF50')
    axes[1].set_title(f'Residual Distribution ({horizon})')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Count')

    # Q-Q style: predicted vs actual
    axes[2].scatter(y_pred, y_true, alpha=0.3, s=10, color='#2196F3')
    lims = [min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())]
    axes[2].plot(lims, lims, 'r--', alpha=0.5)
    axes[2].set_title(f'Predicted vs Actual ({horizon})')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')

    fig.tight_layout()
    _save(fig, save_path)
