import torch
import numpy as np
from training.trainer import Trainer


class EnsemblePredictor:
    """Multi-seed ensemble with uncertainty estimation.

    Args:
        model_class: class to instantiate (e.g. MultiHorizonLSTM)
        model_kwargs: dict of kwargs for model_class
        train_config: training config dict
        device: torch.device
        num_models: number of ensemble members
        base_seed: starting seed
        horizons: list of horizon ints
    """

    def __init__(self, model_class, model_kwargs, train_config, device,
                 num_models=5, base_seed=42, horizons=None):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.train_config = train_config
        self.device = device
        self.num_models = num_models
        self.base_seed = base_seed
        self.horizons = horizons or [1, 5, 20]
        self.models = []
        self.histories = []

    def train_ensemble(self, train_loader, val_loader):
        """Train N models with different random seeds.

        Returns:
            list of training histories
        """
        self.models = []
        self.histories = []

        for i in range(self.num_models):
            seed = self.base_seed + i
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            print(f"\n  --- Ensemble member {i+1}/{self.num_models} (seed={seed}) ---")
            model = self.model_class(**self.model_kwargs)
            trainer = Trainer(model, self.train_config, self.device, self.horizons)
            history = trainer.train(train_loader, val_loader)

            self.models.append(model)
            self.histories.append(history)

        return self.histories

    def predict_with_uncertainty(self, test_loader):
        """Generate predictions with uncertainty estimates.

        Returns:
            dict per horizon: predictions (mean), std, lower_bound, upper_bound, targets
        """
        # Collect predictions from each model
        all_model_preds = {f'h{h}': [] for h in self.horizons}
        targets_collected = {f'h{h}': None for h in self.horizons}

        for model in self.models:
            model.eval()
            model.to(self.device)
            preds_per_h = {f'h{h}': [] for h in self.horizons}

            with torch.no_grad():
                for X_batch, Y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    preds = model(X_batch)
                    for i, h in enumerate(self.horizons):
                        preds_per_h[f'h{h}'].append(preds[f'h{h}'].cpu().numpy())
                        if targets_collected[f'h{h}'] is None:
                            pass  # collect once below

            for h in self.horizons:
                all_model_preds[f'h{h}'].append(np.concatenate(preds_per_h[f'h{h}']))

        # Collect targets once
        for X_batch, Y_batch in test_loader:
            for i, h in enumerate(self.horizons):
                if targets_collected[f'h{h}'] is None:
                    targets_collected[f'h{h}'] = []
                targets_collected[f'h{h}'].append(Y_batch[:, i].numpy())
        for h in self.horizons:
            targets_collected[f'h{h}'] = np.concatenate(targets_collected[f'h{h}'])

        # Aggregate
        results = {}
        for h in self.horizons:
            key = f'h{h}'
            stacked = np.stack(all_model_preds[key], axis=0)  # (num_models, N)
            mean_pred = stacked.mean(axis=0)
            std_pred = stacked.std(axis=0)
            results[key] = {
                'predictions': mean_pred,
                'std': std_pred,
                'lower_bound': mean_pred - 1.96 * std_pred,
                'upper_bound': mean_pred + 1.96 * std_pred,
                'targets': targets_collected[key],
            }

        return results
