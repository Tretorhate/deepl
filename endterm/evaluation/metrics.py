import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-10
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def directional_accuracy(y_true, y_pred):
    """Percentage of predictions that correctly predict the direction of change."""
    correct = np.sign(y_true) == np.sign(y_pred)
    return np.mean(correct) * 100


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return 1 - (ss_res / ss_tot)


def compute_all_metrics(y_true, y_pred):
    """Compute all metrics and return as dict."""
    return {
        'MSE': mse(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'Dir_Acc': directional_accuracy(y_true, y_pred),
        'R2': r_squared(y_true, y_pred),
    }


def format_metrics_table(results_dict):
    """Format a dict of {model_name: {horizon: metrics}} into a printable table."""
    lines = []
    header = f"{'Model':<20} {'Horizon':<10} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'MAPE%':<10} {'Dir%':<10} {'R²':<10}"
    lines.append(header)
    lines.append('-' * len(header))

    for model_name, horizons in results_dict.items():
        for horizon, metrics in horizons.items():
            line = (f"{model_name:<20} {horizon:<10} "
                    f"{metrics['MSE']:<12.6f} {metrics['RMSE']:<12.6f} "
                    f"{metrics['MAE']:<12.6f} {metrics['MAPE']:<10.2f} "
                    f"{metrics['Dir_Acc']:<10.2f} {metrics['R2']:<10.4f}")
            lines.append(line)

    return '\n'.join(lines)
