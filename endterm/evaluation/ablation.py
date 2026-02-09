import copy
import torch
import numpy as np
import pandas as pd
from models.lstm_model import MultiHorizonLSTM
from models.transformer_model import MultiHorizonTransformer
from training.trainer import Trainer
from evaluation.metrics import compute_all_metrics


class AblationRunner:
    """Run ablation studies varying one hyperparameter at a time.

    Args:
        config: base config dict (merged DATA_CONFIG + TRAINING_CONFIG etc.)
        device: torch.device
        horizons: list of horizon ints
    """

    def __init__(self, config, device, horizons=None):
        self.config = config
        self.device = device
        self.horizons = horizons or [1, 5, 20]
        self.results = []

    def _train_and_evaluate(self, model, train_loader, val_loader, test_loader,
                            train_config, experiment_name, variant_name):
        """Train a model and record metrics."""
        trainer = Trainer(model, train_config, self.device, self.horizons)
        trainer.train(train_loader, val_loader)
        eval_results = trainer.evaluate(test_loader)

        for h in self.horizons:
            key = f'h{h}'
            metrics = compute_all_metrics(eval_results[key]['targets'],
                                          eval_results[key]['predictions'])
            row = {
                'experiment': experiment_name,
                'variant': variant_name,
                'horizon': key,
                **metrics,
            }
            self.results.append(row)

    def _build_model(self, model_type, input_dim, dropout=0.2,
                     use_batch_norm=True, **kwargs):
        """Instantiate a model by type."""
        if model_type in ('LSTM', 'GRU'):
            return MultiHorizonLSTM(
                input_dim=input_dim,
                hidden_dim=kwargs.get('hidden_dim', 128),
                num_layers=kwargs.get('num_layers', 2),
                horizons=self.horizons,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                cell_type=model_type,
            )
        else:  # Transformer
            return MultiHorizonTransformer(
                input_dim=input_dim,
                d_model=kwargs.get('d_model', 64),
                nhead=kwargs.get('nhead', 4),
                num_layers=kwargs.get('num_layers', 2),
                dim_ff=kwargs.get('dim_feedforward', 256),
                horizons=self.horizons,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
            )

    def run_all(self, train_loader, val_loader, test_loader, input_dim):
        """Run all ablation experiments.

        Returns:
            pd.DataFrame with all results
        """
        self.results = []
        ablation_config = self.config.get('ablation', {})
        base_train_config = {
            'lr': self.config.get('lr', 1e-3),
            'weight_decay': self.config.get('weight_decay', 1e-4),
            'epochs': self.config.get('ablation_epochs', 30),
            'patience': self.config.get('patience', 15),
            'scheduler_factor': 0.5,
            'scheduler_patience': 10,
        }

        # 1. Model type comparison: LSTM vs GRU vs Transformer
        print("\n[Ablation 1/7] Model type comparison...")
        for mtype in ablation_config.get('model_types', ['LSTM', 'GRU', 'Transformer']):
            print(f"  Training {mtype}...")
            torch.manual_seed(42)
            model = self._build_model(mtype, input_dim)
            self._train_and_evaluate(model, train_loader, val_loader, test_loader,
                                     base_train_config, 'model_type', mtype)

        # 2. Dropout rates
        print("\n[Ablation 2/7] Dropout rates...")
        for dr in ablation_config.get('dropout_rates', [0.0, 0.1, 0.2, 0.3, 0.5]):
            print(f"  Dropout={dr}...")
            torch.manual_seed(42)
            model = self._build_model('LSTM', input_dim, dropout=dr)
            self._train_and_evaluate(model, train_loader, val_loader, test_loader,
                                     base_train_config, 'dropout', str(dr))

        # 3. Batch normalization toggle
        print("\n[Ablation 3/7] Batch normalization...")
        for bn in ablation_config.get('batch_norm_toggle', [True, False]):
            print(f"  BatchNorm={bn}...")
            torch.manual_seed(42)
            model = self._build_model('LSTM', input_dim, use_batch_norm=bn)
            self._train_and_evaluate(model, train_loader, val_loader, test_loader,
                                     base_train_config, 'batch_norm', str(bn))

        # 4. Attention toggle (Transformer only — with/without extra layers)
        print("\n[Ablation 4/7] Attention (Transformer layers)...")
        for n_layers in ablation_config.get('attention_toggle_layers', [1, 2]):
            label = f"layers={n_layers}"
            print(f"  Transformer {label}...")
            torch.manual_seed(42)
            model = self._build_model('Transformer', input_dim, num_layers=n_layers)
            self._train_and_evaluate(model, train_loader, val_loader, test_loader,
                                     base_train_config, 'attention_depth', label)

        # 5. Weight decay
        print("\n[Ablation 5/7] Weight decay...")
        for wd in ablation_config.get('weight_decays', [0, 1e-5, 1e-4, 1e-3]):
            print(f"  weight_decay={wd}...")
            torch.manual_seed(42)
            model = self._build_model('LSTM', input_dim)
            tc = copy.deepcopy(base_train_config)
            tc['weight_decay'] = wd
            self._train_and_evaluate(model, train_loader, val_loader, test_loader,
                                     tc, 'weight_decay', str(wd))

        # 6. Single vs multi-horizon
        print("\n[Ablation 6/7] Single vs multi-horizon...")
        # Multi-horizon (default)
        torch.manual_seed(42)
        model = self._build_model('LSTM', input_dim)
        self._train_and_evaluate(model, train_loader, val_loader, test_loader,
                                 base_train_config, 'horizon_mode', 'multi')
        # Single horizon (h1 only)
        torch.manual_seed(42)
        model_single = MultiHorizonLSTM(
            input_dim=input_dim, hidden_dim=128, num_layers=2,
            horizons=[1], dropout=0.2, use_batch_norm=True, cell_type='LSTM',
        )
        single_trainer = Trainer(model_single, base_train_config, self.device, horizons=[1])
        single_trainer.train(train_loader, val_loader)
        eval_res = single_trainer.evaluate(test_loader)
        metrics = compute_all_metrics(eval_res['h1']['targets'], eval_res['h1']['predictions'])
        self.results.append({
            'experiment': 'horizon_mode', 'variant': 'single_h1',
            'horizon': 'h1', **metrics,
        })

        # 7. Window sizes (note: this requires re-creating dataloaders,
        #    so we skip if not provided — handled in main.py)
        print("\n[Ablation 7/7] Window sizes — skipped (requires data re-creation)")
        # Window size ablation is handled separately in main.py where
        # we can rebuild dataloaders with different seq_len

        df = pd.DataFrame(self.results)
        return df
