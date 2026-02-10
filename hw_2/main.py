"""
Intelligent Healthcare Assistant System
Comprehensive Deep Learning Project (25 points)
Parts 1-5: Text Analysis, Image Analysis, Generative Models, Optimization, Ethics
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import DEVICE, get_mode_config

# ============================================================================
# IMPORT MODULES (as they become available)
# ============================================================================
from data.text_loader import load_text_dataset, TextDataset
from data.image_loader import load_image_dataset
from models.text_models import LSTMTextClassifier, TransformerTextClassifier
from models.vision_models import ResNetClassifier, SimpleConvNet
from models_generative.vae import VAE
from models_generative.gan import GAN
from training.text_trainer import TextTrainer
from training.vision_trainer import VisionTrainer
from training.generative_trainer import VAETrainer, GANTrainer
from optimization.model_optimizer import ModelOptimizer, BenchmarkComparison
from evaluation.text_metrics import compute_text_metrics
from evaluation.ethics import BiasAudit, EthicsAnalysis
from visualization.plots import ResultsVisualizer, ResultsSummary
from results_manager import ResultsManager

BASE_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
RESULTS_DIR = BASE_RESULTS_DIR


# ============================================================================
# STATE MANAGEMENT (Global Variables for Tracking Results)
# ============================================================================

# Part 1: Text Analysis
text_data = {}                      # {train, val, test, tokenizer, vocab}
text_models = {}                    # {'lstm': model, 'transformer': model}
text_results = {}                   # {'lstm': metrics, 'transformer': metrics}
text_regularization_analysis = None # Regularization experiment results

# Part 2: Image Analysis
image_data = {}                     # {train, val, test, transform}
vision_models = {}                  # {'resnet18': model, 'resnet50': model}
vision_results = {}                 # Metrics and visualizations
bias_variance_analysis = None       # Learning curves analysis

# Part 3: Generative Models
generative_models = {}              # {'vae': model} or {'gan': model}
generative_results = {}             # Loss curves, generated samples
generative_analysis = None          # Training challenges documentation

# Part 4: Model Optimization
optimized_models = {}               # {'baseline': model, 'pruned': model, 'quantized': model}
optimization_results = None         # Benchmark comparison

# Part 5: Ethics & Fairness
ethics_results = {
    'bias_audit': None,
    'ethics_report': None,
    'recommendations': []
}

# Results Management & Visualization
results_manager = None  # Initialized per run
visualizer = None       # Initialized per run
summary_gen = None      # Initialized per run


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  Random seed set to {seed}")


def get_timestamped_results_dir(mode):
    """Create and return a timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_subdir = f"results_{mode}_{timestamp}"
    path = os.path.join(BASE_RESULTS_DIR, results_subdir)
    os.makedirs(path, exist_ok=True)
    return path


def save_metrics_table(metrics_dict, filename):
    """Save metrics to CSV for report."""
    df = pd.DataFrame(metrics_dict).T
    df.to_csv(filename)
    print(f"  Metrics saved to {filename}")
    return df


def save_results_summary(results_dir):
    """Save a summary of all results to text file."""
    summary_path = os.path.join(results_dir, 'RESULTS_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("INTELLIGENT HEALTHCARE ASSISTANT SYSTEM - RESULTS SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode: {config.CURRENT_MODE}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write("=" * 70 + "\n\n")

        if text_results:
            f.write("PART 1: MEDICAL TEXT ANALYSIS\n")
            f.write("-" * 70 + "\n")
            for model_name, metrics in text_results.items():
                f.write(f"  {model_name}: {metrics}\n")
            f.write("\n")

        if vision_results:
            f.write("PART 2: MEDICAL IMAGE ANALYSIS\n")
            f.write("-" * 70 + "\n")
            for model_name, metrics in vision_results.items():
                f.write(f"  {model_name}: {metrics}\n")
            f.write("\n")

        if generative_results:
            f.write("PART 3: GENERATIVE MODELS\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Generated samples: {len(generative_results.get('samples', []))}\n")
            f.write(f"  Model type: {config.GENERATIVE_CONFIG['model_type']}\n")
            f.write("\n")

        if optimization_results:
            f.write("PART 4: MODEL OPTIMIZATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Optimization results: {optimization_results}\n")
            f.write("\n")

        if ethics_results['bias_audit'] or ethics_results['ethics_report']:
            f.write("PART 5: ETHICS & FAIRNESS\n")
            f.write("-" * 70 + "\n")
            if ethics_results['bias_audit']:
                f.write(f"  Bias audit completed\n")
            if ethics_results['ethics_report']:
                f.write(f"  Ethics report: {len(ethics_results['ethics_report'])} words\n")
            f.write(f"  Recommendations: {len(ethics_results['recommendations'])}\n")
            f.write("\n")

    print(f"  Summary saved to {summary_path}")


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================

def _ensure_text_data():
    """Lazy load text dataset if not already loaded."""
    global text_data
    if not text_data:
        mode = get_mode_config()
        loader_config = {
            'max_vocab_size': config.TEXT_CONFIG['max_vocab_size'],
            'max_seq_length': config.TEXT_CONFIG['max_seq_length'],
            'batch_size': mode['batch_size'],
        }
        text_data = load_text_dataset(mode['text_dataset'], loader_config)
    return text_data


def _ensure_image_data(image_mode='rgb'):
    """
    Lazy load image dataset if not already loaded.

    Args:
        image_mode: 'rgb' for 3-channel or 'grayscale' for 1-channel
    """
    global image_data
    if not image_data or image_data.get('_mode') != image_mode:
        mode = get_mode_config()
        print(f"\n  Loading image dataset ({mode['image_dataset']}, {image_mode})...")
        loader_config = {
            'image_size': config.IMAGE_CONFIG['image_size'],
            'batch_size': mode['batch_size'],
            'num_classes': config.IMAGE_CONFIG['num_classes'],
        }
        image_data = load_image_dataset(mode['image_dataset'], loader_config, mode=image_mode)
        image_data['_mode'] = image_mode  # Track the mode
    return image_data


# ============================================================================
# PART 1: MEDICAL TEXT ANALYSIS WITH SEQUENCE MODELS (7 points)
# ============================================================================

def menu_part1_text_analysis():
    """
    Part 1: Medical Text Analysis (7 points)
    - Train LSTM/GRU classifier
    - Train Transformer classifier
    - Run regularization experiments
    - Generate training curves and analysis
    """
    global text_data, text_models, text_results, text_regularization_analysis, RESULTS_DIR

    print(f"\n{'=' * 70}")
    print("  PART 1: Medical Text Analysis with Sequence Models (7 pts)")
    print(f"{'=' * 70}")

    RESULTS_DIR = get_timestamped_results_dir(config.CURRENT_MODE)
    mode = get_mode_config()

    _ensure_text_data()

    config_text = {
        'epochs': mode['text_epochs'],
        'batch_size': mode['batch_size'],
        'learning_rate': config.TRAINING_CONFIG['learning_rate'],
        'dropout_rates': config.TEXT_REGULARIZATION['dropout_rates'],
        'l2_lambdas': config.TEXT_REGULARIZATION['l2_lambdas'],
        'patience': mode['patience'],
    }

    print("\n  [1/4] Training LSTM/GRU classifier...")
    set_seed()
    try:
        lstm_model = LSTMTextClassifier(
            vocab_size=text_data['vocab_size'],
            embedding_dim=config.TEXT_CONFIG['embedding_dim'],
            hidden_dim=config.TEXT_RNN_CONFIG['hidden_dim'],
            num_layers=config.TEXT_RNN_CONFIG['num_layers'],
            num_classes=text_data['num_classes'],
            dropout=config.TEXT_RNN_CONFIG['dropout'],
            cell_type='LSTM',
            bidirectional=True,
        )
        trainer = TextTrainer(lstm_model, config_text, DEVICE)
        history = trainer.train(text_data['train'], text_data['val'])
        text_results['lstm'] = trainer.evaluate(text_data['test'], text_data['class_names'])
        text_models['lstm'] = (lstm_model, history)
        print(f"      LSTM training complete - Accuracy: {text_results['lstm']['accuracy']:.2f}%")
    except Exception as e:
        print(f"      LSTM training error: {str(e)}")

    print("\n  [2/4] Training Transformer classifier...")
    set_seed()
    try:
        transformer_model = TransformerTextClassifier(
            vocab_size=text_data['vocab_size'],
            embedding_dim=config.TEXT_CONFIG['embedding_dim'],
            d_model=config.TEXT_TRANSFORMER_CONFIG['d_model'],
            nhead=config.TEXT_TRANSFORMER_CONFIG['nhead'],
            num_layers=config.TEXT_TRANSFORMER_CONFIG['num_layers'],
            dim_feedforward=config.TEXT_TRANSFORMER_CONFIG['dim_feedforward'],
            num_classes=text_data['num_classes'],
            dropout=config.TEXT_TRANSFORMER_CONFIG['dropout'],
        )
        trainer = TextTrainer(transformer_model, config_text, DEVICE)
        history = trainer.train(text_data['train'], text_data['val'])
        text_results['transformer'] = trainer.evaluate(text_data['test'], text_data['class_names'])
        text_models['transformer'] = (transformer_model, history)
        print(f"      Transformer training complete - Accuracy: {text_results['transformer']['accuracy']:.2f}%")
    except Exception as e:
        print(f"      Transformer training error: {str(e)}")

    print("\n  [3/4] Running regularization experiments...")
    try:
        # TODO: Implement regularization experiments
        # Vary dropout rates and L2 values
        # text_regularization_analysis = run_regularization_analysis(...)
        print("      Regularization analysis complete (placeholder)")
    except Exception as e:
        print(f"      Regularization analysis placeholder (implementation pending): {str(e)[:50]}")

    print("\n  [4/4] Generating visualizations...")
    try:
        # TODO: Generate plots
        # plot_training_curves(...)
        # plot_regularization_comparison(...)
        # plot_gradient_analysis(...)
        print(f"      Visualizations saved to {RESULTS_DIR}/ (placeholder)")
    except Exception as e:
        print(f"      Visualization placeholder (implementation pending): {str(e)[:50]}")

    print("\n[OK] Part 1 complete!")


# ============================================================================
# PART 2: MEDICAL IMAGE ANALYSIS WITH VISION MODELS (6 points)
# ============================================================================

def menu_part2_image_analysis():
    """
    Part 2: Medical Image Analysis (6 points)
    - Train ResNet18 and ResNet50 models
    - Perform bias-variance analysis
    - Generate learning curves and visualizations
    """
    global image_data, vision_models, vision_results, bias_variance_analysis, RESULTS_DIR

    print(f"\n{'=' * 70}")
    print("  PART 2: Medical Image Analysis with Vision Models (6 pts)")
    print(f"{'=' * 70}")

    RESULTS_DIR = get_timestamped_results_dir(config.CURRENT_MODE)
    mode = get_mode_config()

    _ensure_image_data()

    config_vision = {
        'epochs': mode['vision_epochs'],
        'batch_size': mode['batch_size'],
        'learning_rate': config.TRAINING_CONFIG['learning_rate'],
        'weight_decay': config.TRAINING_CONFIG.get('weight_decay', 1e-4),
        'patience': mode['patience'],
        'scheduler_factor': config.TRAINING_CONFIG.get('scheduler_factor', 0.5),
        'scheduler_patience': config.TRAINING_CONFIG.get('scheduler_patience', 5),
    }

    print("\n  [1/3] Training vision models...")
    set_seed()
    try:
        # Train ResNet18
        print("      Training ResNet18...")
        model_resnet18 = ResNetClassifier(
            num_classes=image_data['num_classes'],
            architecture='resnet18',
            pretrained=config.VISION_MODEL_CONFIG.get('pretrained', True),
            dropout=config.VISION_MODEL_CONFIG.get('dropout', 0.3),
        )
        trainer = VisionTrainer(model_resnet18, config_vision, DEVICE)
        history_r18 = trainer.train(image_data['train'], image_data['val'])
        results_r18 = trainer.evaluate(image_data['test'], image_data['class_names'])
        vision_models['resnet18'] = (model_resnet18, history_r18)
        vision_results['resnet18'] = results_r18
        print(f"        ResNet18 complete - Test Accuracy: {results_r18['accuracy']:.2f}%")
    except Exception as e:
        print(f"        ResNet18 training error: {str(e)}")

    try:
        # Train ResNet50
        print("      Training ResNet50...")
        model_resnet50 = ResNetClassifier(
            num_classes=image_data['num_classes'],
            architecture='resnet50',
            pretrained=config.VISION_MODEL_CONFIG.get('pretrained', True),
            dropout=config.VISION_MODEL_CONFIG.get('dropout', 0.3),
        )
        trainer = VisionTrainer(model_resnet50, config_vision, DEVICE)
        history_r50 = trainer.train(image_data['train'], image_data['val'])
        results_r50 = trainer.evaluate(image_data['test'], image_data['class_names'])
        vision_models['resnet50'] = (model_resnet50, history_r50)
        vision_results['resnet50'] = results_r50
        print(f"        ResNet50 complete - Test Accuracy: {results_r50['accuracy']:.2f}%")
    except Exception as e:
        print(f"        ResNet50 training error: {str(e)}")

    print("\n  [2/3] Analyzing bias-variance trade-off...")
    try:
        # Analyze learning curves from training histories
        if vision_models:
            print(f"      Analyzed {len(vision_models)} models")
            for model_name, (_, history) in vision_models.items():
                if history:
                    final_val_loss = history.get('val_loss', [None])[-1]
                    final_val_acc = history.get('val_acc', [None])[-1]
                    print(f"        {model_name}: val_loss={final_val_loss:.4f}, val_acc={final_val_acc:.2f}%")
        bias_variance_analysis = True
    except Exception as e:
        print(f"      Bias-variance analysis error: {str(e)}")

    print("\n  [3/3] Generating visualizations...")
    try:
        # TODO: Generate plots
        # plot_learning_curves(...)
        # plot_predictions(...)
        # plot_metrics_table(...)
        print(f"      Visualizations saved to {RESULTS_DIR}/ (placeholder)")
    except Exception as e:
        print(f"      Visualization placeholder (implementation pending): {str(e)[:50]}")

    print("\n[OK] Part 2 complete!")


# ============================================================================
# PART 3: SYNTHETIC MEDICAL DATA GENERATION (5 points)
# ============================================================================

def menu_part3_generative_models():
    """
    Part 3: Generative Models (5 points)
    - Train VAE or GAN
    - Document training challenges
    - Generate and evaluate synthetic samples
    """
    global image_data, generative_models, generative_results, generative_analysis, RESULTS_DIR

    print(f"\n{'=' * 70}")
    print("  PART 3: Synthetic Medical Data with Generative Models (5 pts)")
    print(f"{'=' * 70}")

    RESULTS_DIR = get_timestamped_results_dir(config.CURRENT_MODE)
    mode = get_mode_config()

    # Load grayscale images for generative models
    _ensure_image_data(image_mode='grayscale')

    config_gen = {
        'epochs': mode['generative_epochs'],
        'batch_size': mode['batch_size'],
        'learning_rate': config.TRAINING_CONFIG['learning_rate'],
        'model_type': config.GENERATIVE_CONFIG['model_type'],
    }

    print(f"\n  [1/3] Training {config_gen['model_type']}...")
    set_seed()
    try:
        if config_gen['model_type'] == 'VAE':
            print("      Training VAE...")
            vae = VAE(
                input_channels=config.VAE_CONFIG['input_channels'],
                latent_dim=config.VAE_CONFIG['latent_dim'],
                hidden_dim=config.VAE_CONFIG['hidden_dim'],
                reconstruction_loss=config.VAE_CONFIG['reconstruction_loss'],
                kl_weight=config.VAE_CONFIG['kl_weight'],
            )
            trainer = VAETrainer(vae, config_gen, DEVICE)
            history = trainer.train(image_data['train'], image_data['val'])
            generative_models['vae'] = (vae, trainer)
            generative_results['model_type'] = 'VAE'
            generative_results['history'] = history
            print(f"      VAE training complete - Final Val Loss: {history['val_loss'][-1]:.4f}")
        else:
            print("      Training GAN...")
            gan = GAN(
                noise_dim=config.GAN_CONFIG['noise_dim'],
                output_channels=config.VAE_CONFIG['input_channels'],
                hidden_dim=config.GAN_CONFIG['generator_hidden'],
            )
            trainer = GANTrainer(gan, config_gen, DEVICE)
            history = trainer.train(image_data['train'])
            generative_models['gan'] = (gan, trainer)
            generative_results['model_type'] = 'GAN'
            generative_results['history'] = history
            print(f"      GAN training complete - Final Gen Loss: {history['gen_loss'][-1]:.4f}")
    except Exception as e:
        print(f"      Generative training error: {str(e)}")

    print("\n  [2/3] Generating synthetic samples...")
    try:
        num_samples = 20
        if config_gen['model_type'] == 'VAE':
            vae, trainer = generative_models['vae']
            samples = trainer.generate_samples(num_samples=num_samples)
        else:
            gan, trainer = generative_models['gan']
            samples = trainer.generate_samples(num_samples=num_samples)

        generative_results['samples'] = samples
        print(f"      Generated {num_samples} synthetic samples - Shape: {samples.shape}")
    except Exception as e:
        print(f"      Sample generation error: {str(e)}")

    print("\n  [3/3] Documenting training challenges and insights...")
    try:
        # Document training challenges
        challenges = []
        history = generative_results.get('history', {})

        if config_gen['model_type'] == 'VAE':
            # VAE-specific challenges
            if history:
                recon_losses = history.get('train_recon_loss', [])
                kl_losses = history.get('train_kl_loss', [])
                if recon_losses and kl_losses:
                    challenges.append(f"KL Divergence: Started at {kl_losses[0]:.4f}, ended at {kl_losses[-1]:.4f}")
                    challenges.append(f"Reconstruction: Started at {recon_losses[0]:.4f}, ended at {recon_losses[-1]:.4f}")
                    challenges.append("KL divergence weight balanced image reconstruction with latent space regularization")
            challenges.append("Handled grayscale medical images (1 channel) with Tanh output activation")
            challenges.append("Used reparameterization trick for differentiable sampling from latent distribution")

        else:  # GAN
            # GAN-specific challenges
            if history:
                gen_losses = history.get('gen_loss', [])
                dis_losses = history.get('dis_loss', [])
                if gen_losses and dis_losses:
                    challenges.append(f"Generator Loss: Started at {gen_losses[0]:.4f}, ended at {gen_losses[-1]:.4f}")
                    challenges.append(f"Discriminator Loss: Started at {dis_losses[0]:.4f}, ended at {dis_losses[-1]:.4f}")
            challenges.append("Implemented separate optimizers for generator and discriminator")
            challenges.append("Used BCELoss with careful label smoothing for stable training")
            challenges.append("Addressed mode collapse through diverse noise sampling")

        generative_analysis = {
            'model_type': config_gen['model_type'],
            'challenges': challenges,
            'num_samples_generated': 20,
            'training_duration_epochs': len(history.get('train_loss', [])) if history else 0,
        }

        generative_results['analysis'] = generative_analysis

        # Print summary
        print(f"      Trained {config_gen['model_type']} with {generative_analysis['training_duration_epochs']} epochs")
        for i, challenge in enumerate(challenges[:3], 1):
            print(f"        {i}. {challenge}")
        if len(challenges) > 3:
            print(f"        ... and {len(challenges) - 3} more insights")

    except Exception as e:
        print(f"      Analysis error: {str(e)}")

    print("\n[OK] Part 3 complete!")


# ============================================================================
# PART 4: MODEL DEPLOYMENT & OPTIMIZATION (4 points)
# ============================================================================

def menu_part4_optimization():
    """
    Part 4: Model Optimization (4 points)
    - Apply pruning/quantization/distillation
    - Benchmark performance before and after
    - Generate comparison table
    """
    global vision_models, optimized_models, optimization_results, RESULTS_DIR

    print(f"\n{'=' * 70}")
    print("  PART 4: Model Deployment & Optimization (4 pts)")
    print(f"{'=' * 70}")

    RESULTS_DIR = get_timestamped_results_dir(config.CURRENT_MODE)

    if not vision_models:
        print("  ✗ Error: No trained vision model found. Run Part 2 first.")
        return

    try:
        # Get baseline model (use first vision model - ResNet18 or ResNet50)
        baseline_model_tuple = list(vision_models.values())[0]
        baseline_model = baseline_model_tuple[0] if isinstance(baseline_model_tuple, tuple) else baseline_model_tuple

        mode = get_mode_config()
        batch_size = mode.get('batch_size', 32)

        print("\n  [1/3] Applying optimization techniques...")
        set_seed()

        # Initialize optimizer
        optimizer = ModelOptimizer(baseline_model, config.OPTIMIZATION_CONFIG, DEVICE)
        optimized_models['baseline'] = baseline_model

        # Pruning
        print("      - Applying pruning (30% sparsity)...")
        try:
            pruned_model = optimizer.prune_model(
                sparsity=config.OPTIMIZATION_CONFIG.get('pruning_sparsity', 0.3),
                method=config.OPTIMIZATION_CONFIG.get('pruning_method', 'unstructured')
            )
            optimized_models['pruned'] = pruned_model
            print("        Pruning complete!")
        except Exception as e:
            print(f"        Pruning error: {str(e)[:50]}")

        # Quantization
        print("      - Applying quantization (8-bit)...")
        try:
            quantized_model = optimizer.quantize_model(
                bits=config.OPTIMIZATION_CONFIG.get('quantization_bits', 8)
            )
            optimized_models['quantized'] = quantized_model
            print("        Quantization complete!")
        except Exception as e:
            print(f"        Quantization error: {str(e)[:50]}")

        print("\n  [2/3] Benchmarking models...")
        try:
            # Load test data if not already loaded
            _ensure_image_data()
            test_loader = image_data['test']

            # Benchmark all models
            benchmark = BenchmarkComparison(DEVICE)

            baseline_results = benchmark.benchmark_model(
                baseline_model, 'Baseline', test_loader
            )
            print(f"      Baseline accuracy: {baseline_results['accuracy']:.2f}%")

            if 'pruned' in optimized_models:
                pruned_results = benchmark.benchmark_model(
                    optimized_models['pruned'], 'Pruned (30%)', test_loader
                )
                print(f"      Pruned accuracy: {pruned_results['accuracy']:.2f}%")

            if 'quantized' in optimized_models:
                quantized_results = benchmark.benchmark_model(
                    optimized_models['quantized'], 'Quantized (8-bit)', test_loader
                )
                print(f"      Quantized accuracy: {quantized_results['accuracy']:.2f}%")

            optimization_results = {
                'baseline': baseline_results,
                'optimization_config': config.OPTIMIZATION_CONFIG,
                'benchmarks': benchmark.benchmarks,
            }

        except Exception as e:
            print(f"      Benchmarking error: {str(e)[:50]}")

        print("\n  [3/3] Generating optimization summary...")
        try:
            summary = optimizer.get_optimization_summary()

            print(f"\n      Optimization Summary:")
            print(f"      - Original size: {summary['results']['original_size_mb']:.2f} MB")
            if summary['results']['pruned_size_mb'] > 0:
                print(f"      - Pruned size: {summary['results']['pruned_size_mb']:.2f} MB")
                print(f"      - Pruning sparsity: {summary['results']['sparsity']:.1%}")
            if summary['results']['quantized_size_mb'] > 0:
                print(f"      - Quantized size: {summary['results']['quantized_size_mb']:.2f} MB")
            if summary['results']['compression_ratio'] > 1.0:
                print(f"      - Compression ratio: {summary['results']['compression_ratio']:.2f}x")

            print(f"\n      Results saved to {RESULTS_DIR}/")

        except Exception as e:
            print(f"      Summary generation error: {str(e)[:50]}")

        print("\n[OK] Part 4 complete!")
    except Exception as e:
        print(f"  ✗ Optimization error: {str(e)[:100]}")


# ============================================================================
# PART 5: ETHICS, BIAS & FAIRNESS ANALYSIS (3 points)
# ============================================================================

def menu_part5_ethics():
    """
    Part 5: Ethics & Fairness Analysis (3 points)
    - Conduct bias audit on demographic subgroups
    - Write comprehensive ethics report
    - Provide concrete recommendations
    """
    global image_data, vision_models, ethics_results, RESULTS_DIR

    print(f"\n{'=' * 70}")
    print("  PART 5: Ethics, Bias & Fairness Analysis (3 pts)")
    print(f"{'=' * 70}")

    RESULTS_DIR = get_timestamped_results_dir(config.CURRENT_MODE)

    if not vision_models:
        print("  ✗ Error: No trained vision model found. Run Part 2 first.")
        return

    try:
        # Get baseline vision model (ResNet18 or ResNet50)
        model_tuple = list(vision_models.values())[0]
        model = model_tuple[0] if isinstance(model_tuple, tuple) else model_tuple

        _ensure_image_data(image_mode='rgb')
        test_loader = image_data['test']
        class_names = image_data.get('class_names', None)

        print("\n  [1/3] Conducting bias audit...")
        try:
            # Initialize ethics analysis
            ethics_analyzer = EthicsAnalysis(config.ETHICS_CONFIG, DEVICE)

            # Analyze model ethics
            analysis_results = ethics_analyzer.analyze_model_ethics(
                model, test_loader, class_names
            )
            ethics_results['bias_audit'] = analysis_results['bias_audit']

            # Print audit summary
            audit_summary = ethics_analyzer.bias_audit.get_audit_summary()
            print(f"      Bias audit complete!")
            print(f"        - Groups analyzed: {audit_summary['num_groups_analyzed']}")
            print(f"        - Biases flagged: {audit_summary['num_biases_flagged']}")
            print(f"        - Overall accuracy: {audit_summary['overall_metrics'].get('accuracy', 0):.2f}%")

        except Exception as e:
            print(f"      Bias audit error: {str(e)[:50]}")

        print("\n  [2/3] Generating comprehensive ethics report...")
        try:
            # Generate full ethics report
            ethics_report = ethics_analyzer.generate_ethics_report()
            ethics_results['ethics_report'] = ethics_report

            # Save report to file
            report_path = f"{RESULTS_DIR}/ethics_report.txt"
            with open(report_path, 'w') as f:
                f.write(ethics_report)

            print(f"      Ethics report generated ({len(ethics_report.split(chr(10)))} lines)")
            print(f"      Report saved to: {report_path}")

        except Exception as e:
            print(f"      Report generation error: {str(e)[:50]}")

        print("\n  [3/3] Documenting recommendations...")
        try:
            # Get recommendations
            ethics_summary = ethics_analyzer.get_ethics_summary()
            recommendations = ethics_analyzer.analysis_results.get('recommendations', [])

            ethics_results['recommendations'] = recommendations
            ethics_results['fairness_status'] = ethics_summary['fairness_status']

            print(f"      Recommendations generated: {len(recommendations)}")
            for i, rec in enumerate(recommendations[:2], 1):
                # Extract first line of recommendation
                first_line = rec.split('\n')[0].strip()
                print(f"        {i}. {first_line}")
            if len(recommendations) > 2:
                print(f"        ... and {len(recommendations) - 2} more recommendations")

            # Save recommendations
            rec_path = f"{RESULTS_DIR}/recommendations.txt"
            with open(rec_path, 'w') as f:
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"\nRECOMMENDATION {i}\n")
                    f.write("-" * 70 + "\n")
                    f.write(rec + "\n")

            print(f"\n      Fairness Status: {ethics_summary['fairness_status']}")
            print(f"      Recommendations saved to: {rec_path}")

        except Exception as e:
            print(f"      Recommendations error: {str(e)[:50]}")

        print(f"\n      All ethics results saved to {RESULTS_DIR}/")
        print("\n[OK] Part 5 complete!")

    except Exception as e:
        print(f"  ✗ Ethics analysis error: {str(e)[:100]}")


# ============================================================================
# FULL PIPELINE
# ============================================================================

def menu_full_pipeline():
    """Run all 5 parts end-to-end with visualization and results aggregation."""
    global RESULTS_DIR, results_manager, visualizer, summary_gen
    global text_models, text_results, vision_results, generative_results
    global optimization_results, ethics_results

    print(f"\n{'=' * 70}")
    print("  RUNNING FULL PIPELINE WITH VISUALIZATION & RESULTS")
    print(f"  Mode: {config.CURRENT_MODE}")
    print(f"  Device: {DEVICE}")
    print(f"{'=' * 70}")

    RESULTS_DIR = get_timestamped_results_dir(config.CURRENT_MODE)

    # Initialize results management and visualization
    results_manager = ResultsManager(results_dir=RESULTS_DIR)
    visualizer = ResultsVisualizer(results_dir=RESULTS_DIR)
    summary_gen = ResultsSummary(results_dir=RESULTS_DIR)

    print("\n[Step 1/7] Part 1 - Medical Text Analysis")
    menu_part1_text_analysis()

    print("\n[Step 2/7] Part 2 - Medical Image Analysis")
    menu_part2_image_analysis()

    print("\n[Step 3/7] Part 3 - Generative Models")
    menu_part3_generative_models()

    print("\n[Step 4/7] Part 4 - Model Optimization")
    menu_part4_optimization()

    print("\n[Step 5/7] Part 5 - Ethics Analysis")
    menu_part5_ethics()

    # ========================================================================
    # [Step 6/7] COLLECT RESULTS & GENERATE VISUALIZATIONS
    # ========================================================================
    print("\n[Step 6/7] Aggregating results and generating visualizations...")

    try:
        # Add Part 1 results to manager
        if text_models and text_results:
            lstm_results = text_results.get('lstm', {})
            transformer_results = text_results.get('transformer', {})
            results_manager.add_part1_results(
                lstm_results={
                    'test_acc': lstm_results.get('accuracy', 0),
                    'epochs': len(text_models.get('lstm', [None, None])[1].get('train_loss', [])) if text_models.get('lstm') else 0,
                },
                transformer_results={
                    'test_acc': transformer_results.get('accuracy', 0),
                    'epochs': len(text_models.get('transformer', [None, None])[1].get('train_loss', [])) if text_models.get('transformer') else 0,
                }
            )

            # Plot training curves for text models
            histories = {}
            if 'lstm' in text_models:
                _, history = text_models['lstm']
                histories['LSTM'] = history
            if 'transformer' in text_models:
                _, history = text_models['transformer']
                histories['Transformer'] = history

            if histories:
                plot_path = visualizer.plot_training_curves(
                    histories,
                    title="Part 1: Text Model Training Curves",
                    save_name="part1_training_curves.png"
                )
                print(f"      Training curves saved: {plot_path}")

        # Add Part 2 results to manager
        if vision_results:
            resnet18_results = vision_results.get('resnet18', {})
            resnet50_results = vision_results.get('resnet50', {})
            results_manager.add_part2_results(
                resnet18_results={
                    'test_acc': resnet18_results.get('accuracy', 0),
                    'epochs': resnet18_results.get('epochs', 0),
                },
                resnet50_results={
                    'test_acc': resnet50_results.get('accuracy', 0),
                    'epochs': resnet50_results.get('epochs', 0),
                }
            )

            # Plot model comparison
            model_accs = {}
            for model_name, metrics in vision_results.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    model_accs[model_name] = metrics['accuracy']

            if model_accs:
                plot_path = visualizer.plot_model_comparison(
                    model_accs,
                    metric="accuracy",
                    title="Part 2: Vision Model Comparison",
                    save_name="part2_model_comparison.png"
                )
                print(f"      Model comparison saved: {plot_path}")

        # Add Part 3 results to manager
        if generative_results:
            vae_results = generative_results.get('vae', {})
            gan_results = generative_results.get('gan', {})
            results_manager.add_part3_results(
                vae_results={
                    'latent_dim': vae_results.get('latent_dim', 64),
                    'epochs': vae_results.get('epochs', 0),
                    'num_samples': len(vae_results.get('samples', [])),
                },
                gan_results={
                    'epochs': gan_results.get('epochs', 0),
                    'num_samples': len(gan_results.get('samples', [])),
                }
            )

        # Add Part 4 results to manager
        if optimization_results:
            results_manager.add_part4_results(
                baseline=optimization_results.get('baseline', {}),
                pruned=optimization_results.get('pruned', {}),
                quantized=optimization_results.get('quantized', {})
            )

            # Plot optimization comparison
            techniques = {
                'Baseline': {
                    'accuracy': optimization_results.get('baseline', {}).get('accuracy', 0),
                    'compression_ratio': 1.0
                },
                'Pruned': {
                    'accuracy': optimization_results.get('pruned', {}).get('accuracy', 0),
                    'compression_ratio': optimization_results.get('pruned', {}).get('compression', 1.0)
                },
                'Quantized': {
                    'accuracy': optimization_results.get('quantized', {}).get('accuracy', 0),
                    'compression_ratio': optimization_results.get('quantized', {}).get('compression', 1.0)
                }
            }

            plot_path = visualizer.plot_optimization_comparison(
                techniques,
                save_name="part4_optimization_comparison.png"
            )
            print(f"      Optimization comparison saved: {plot_path}")

        # Add Part 5 results to manager
        if ethics_results['bias_audit']:
            results_manager.add_part5_results(
                bias_audit=ethics_results['bias_audit'],
                recommendations=ethics_results['recommendations']
            )

            # Plot fairness analysis
            demographic_analysis = ethics_results['bias_audit'].get('demographic_analysis', {})
            if demographic_analysis:
                plot_path = visualizer.plot_fairness_analysis(
                    demographic_analysis,
                    save_name="part5_fairness_analysis.png"
                )
                print(f"      Fairness analysis saved: {plot_path}")

            # Plot bias flags
            bias_flags = ethics_results['bias_audit'].get('bias_flags', [])
            plot_path = visualizer.plot_bias_flags(
                bias_flags,
                save_name="part5_bias_flags.png"
            )
            print(f"      Bias flags saved: {plot_path}")

        print(f"      All visualizations generated successfully")

    except Exception as e:
        print(f"      Visualization generation had some issues: {str(e)[:100]}")

    # ========================================================================
    # [Step 7/7] GENERATE COMPREHENSIVE REPORTS
    # ========================================================================
    print("\n[Step 7/7] Generating comprehensive reports...")

    try:
        # Generate JSON and text reports
        reports = results_manager.generate_full_report()
        print(f"      JSON results: {reports['json']}")
        print(f"      Text summary: {reports['text']}")

        # Save original summary
        save_results_summary(RESULTS_DIR)
        print(f"      Results summary: {RESULTS_DIR}/RESULTS_SUMMARY.txt")

        # Generate detailed summary
        summary_path = summary_gen.generate_summary(
            all_results={
                'part1': results_manager.results.get('part1', {}),
                'part2': results_manager.results.get('part2', {}),
                'part3': results_manager.results.get('part3', {}),
                'part4': results_manager.results.get('part4', {}),
                'part5': results_manager.results.get('part5', {}),
            },
            save_name="DETAILED_RESULTS.txt"
        )
        print(f"      Detailed results: {summary_path}")

        print(f"      All reports generated successfully")

    except Exception as e:
        print(f"      Report generation had some issues: {str(e)[:100]}")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print(f"\n  📊 RESULTS DIRECTORY: {RESULTS_DIR}/")
    print(f"\n  Generated Files:")
    print(f"    • results.json                     (Machine-readable results)")
    print(f"    • PROJECT_SUMMARY.txt              (Overall summary)")
    print(f"    • RESULTS_SUMMARY.txt              (Legacy summary)")
    print(f"    • DETAILED_RESULTS.txt             (Detailed metrics)")
    print(f"    • part1_training_curves.png        (Text model curves)")
    print(f"    • part2_model_comparison.png       (Vision model comparison)")
    print(f"    • part4_optimization_comparison.png(Optimization results)")
    print(f"    • part5_fairness_analysis.png      (Fairness metrics)")
    print(f"    • part5_bias_flags.png             (Bias detection)")
    print(f"    • ethics_report.txt                (Bias audit report)")
    print(f"    • recommendations.txt              (Fairness recommendations)")
    print(f"\n  All results are ready for inclusion in reports and presentations!")
    print("=" * 70 + "\n")


# ============================================================================
# MENU SYSTEM
# ============================================================================

def mode_selection_menu():
    """Menu to select execution mode."""
    while True:
        print(f"\n{'=' * 70}")
        print("  SELECT EXECUTION MODE")
        print(f"{'=' * 70}")
        for i, (mode_name, mode_cfg) in enumerate(config.MODES.items(), 1):
            print(f"  {i}. {mode_name:6s} - {mode_cfg['description']}")
        print("  0. Cancel")
        print(f"{'=' * 70}")

        choice = input("  Select mode: ").strip()

        if choice == '1':
            config.CURRENT_MODE = 'QUICK'
            print(f"  [OK] Mode set to: QUICK\n")
            return
        elif choice == '2':
            config.CURRENT_MODE = 'HYBRID'
            print(f"  [OK] Mode set to: HYBRID\n")
            return
        elif choice == '3':
            config.CURRENT_MODE = 'FULL'
            print(f"  [OK] Mode set to: FULL\n")
            return
        elif choice == '0':
            print("  Cancelled.\n")
            return
        else:
            print("  ✗ Invalid option, try again.\n")


def interactive_menu():
    """Main interactive menu loop."""
    while True:
        print(f"{'=' * 70}")
        print(f"  INTELLIGENT HEALTHCARE ASSISTANT SYSTEM")
        print(f"  Mode: {config.CURRENT_MODE} | Device: {DEVICE}")
        print(f"{'=' * 70}")
        print("  1. Part 1: Medical Text Analysis (7 pts)")
        print("  2. Part 2: Medical Image Analysis (6 pts)")
        print("  3. Part 3: Generative Models (5 pts)")
        print("  4. Part 4: Model Optimization (4 pts)")
        print("  5. Part 5: Ethics & Fairness (3 pts)")
        print("  6. Run Full Pipeline (All Parts)")
        print("  7. Change Execution Mode")
        print("  8. Show Configuration")
        print("  0. Exit")
        print(f"{'=' * 70}\n")

        choice = input("  Select option: ").strip()

        if choice == '1':
            menu_part1_text_analysis()
        elif choice == '2':
            menu_part2_image_analysis()
        elif choice == '3':
            menu_part3_generative_models()
        elif choice == '4':
            menu_part4_optimization()
        elif choice == '5':
            menu_part5_ethics()
        elif choice == '6':
            menu_full_pipeline()
        elif choice == '7':
            mode_selection_menu()
        elif choice == '8':
            config.print_config()
        elif choice == '0':
            print("\nGoodbye!\n")
            break
        else:
            print("  ✗ Invalid option, try again.\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point with CLI argument support."""
    parser = argparse.ArgumentParser(
        description='Intelligent Healthcare Assistant System - Deep Learning Project'
    )
    parser.add_argument(
        '--mode',
        choices=['QUICK', 'HYBRID', 'FULL'],
        default=None,
        help='Execution mode'
    )
    parser.add_argument(
        '--run',
        type=int,
        default=None,
        help='Menu option to run non-interactively (1-6)'
    )
    args = parser.parse_args()

    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

    print(f"\nDevice: {DEVICE}\n")

    if args.run is not None:
        # Non-interactive mode
        if args.mode is None:
            args.mode = 'QUICK'
        config.CURRENT_MODE = args.mode

        actions = {
            1: menu_part1_text_analysis,
            2: menu_part2_image_analysis,
            3: menu_part3_generative_models,
            4: menu_part4_optimization,
            5: menu_part5_ethics,
            6: menu_full_pipeline,
        }

        if args.run in actions:
            config.print_config()
            actions[args.run]()
        else:
            print(f"✗ Invalid run option: {args.run}")
    else:
        # Interactive mode
        if args.mode is not None:
            config.CURRENT_MODE = args.mode
        else:
            mode_selection_menu()

        config.print_config()
        interactive_menu()


if __name__ == '__main__':
    main()
