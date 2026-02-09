import copy
import torch
import torch.nn as nn
import numpy as np


class Trainer:
    """Training engine with early stopping, LR scheduling, and checkpointing.

    Args:
        model: nn.Module with multi-horizon output
        config: dict with lr, weight_decay, epochs, patience, scheduler params
        device: torch.device
        horizons: list of horizon ints (e.g. [1, 5, 20])
    """

    def __init__(self, model, config, device, horizons=None):
        if horizons is None:
            horizons = [1, 5, 20]
        self.model = model.to(device)
        self.device = device
        self.horizons = horizons
        self.config = config

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 5),
        )

        self.epochs = config.get('epochs', 100)
        self.patience = config.get('patience', 15)
        # Don't start early stopping until warmup is done — gives the
        # model time to stabilize before we judge val loss improvements.
        self.warmup_epochs = config.get('warmup_epochs', max(10, self.epochs // 5))

    def _compute_loss(self, preds, targets):
        """Compute MSE loss summed across all horizons."""
        total_loss = 0.0
        for i, h in enumerate(self.horizons):
            total_loss += self.criterion(preds[f'h{h}'], targets[:, i])
        return total_loss

    def train(self, train_loader, val_loader):
        """Train the model with early stopping.

        Returns:
            history: dict with train_loss, val_loss, and per-horizon val metrics
        """
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        for h in self.horizons:
            history[f'val_mse_h{h}'] = []

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # ── Train ──
            self.model.train()
            train_losses = []
            for X_batch, Y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self._compute_loss(preds, Y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # ── Validate ──
            self.model.eval()
            val_losses = []
            horizon_mses = {h: [] for h in self.horizons}
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)
                    preds = self.model(X_batch)
                    loss = self._compute_loss(preds, Y_batch)
                    val_losses.append(loss.item())
                    for i, h in enumerate(self.horizons):
                        mse_h = self.criterion(preds[f'h{h}'], Y_batch[:, i]).item()
                        horizon_mses[h].append(mse_h)

            avg_val_loss = np.mean(val_losses)
            self.scheduler.step(avg_val_loss)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            for h in self.horizons:
                history[f'val_mse_h{h}'].append(np.mean(horizon_mses[h]))

            # ── Early stopping (only after warmup) ──
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            elif epoch > self.warmup_epochs:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]['lr']
                warmup_str = " (warmup)" if epoch <= self.warmup_epochs else ""
                print(f"    Epoch {epoch:3d}/{self.epochs} | "
                      f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
                      f"LR: {lr:.2e} | Patience: {patience_counter}/{self.patience}{warmup_str}")

            if patience_counter >= self.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return history

    def evaluate(self, test_loader):
        """Evaluate model on test set.

        Returns:
            results: dict with per-horizon predictions and targets
        """
        self.model.eval()
        all_preds = {f'h{h}': [] for h in self.horizons}
        all_targets = {f'h{h}': [] for h in self.horizons}

        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                preds = self.model(X_batch)
                for i, h in enumerate(self.horizons):
                    all_preds[f'h{h}'].append(preds[f'h{h}'].cpu().numpy())
                    all_targets[f'h{h}'].append(Y_batch[:, i].cpu().numpy())

        results = {}
        for h in self.horizons:
            key = f'h{h}'
            results[key] = {
                'predictions': np.concatenate(all_preds[key]),
                'targets': np.concatenate(all_targets[key]),
            }

        return results
