"""
Configuration for Intelligent Healthcare Assistant System
Execution modes: QUICK, HYBRID, FULL
"""

import torch

RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# EXECUTION MODES
# ============================================================================

MODES = {
    'QUICK': {
        'dataset_size': 'small',
        'text_dataset': 'MedQuAD (subset)',
        'image_dataset': 'HAM10000 (subset - 100 samples)',
        'text_epochs': 15,
        'vision_epochs': 15,
        'generative_epochs': 15,
        'batch_size': 16,
        'num_workers': 0,
        'patience': 8,
        'description': 'Fast testing mode',
    },
    'HYBRID': {
        'dataset_size': 'medium',
        'text_dataset': 'PubMed abstracts (5000 samples)',
        'image_dataset': 'ChestX-ray14 (subset - 1000 samples)',
        'text_epochs': 50,
        'vision_epochs': 50,
        'generative_epochs': 50,
        'batch_size': 32,
        'num_workers': 2,
        'patience': 15,
        'description': 'Balanced mode',
    },
    'FULL': {
        'dataset_size': 'large',
        'text_dataset': 'PubMed abstracts (20000+ samples)',
        'image_dataset': 'ChestX-ray14 or HAM10000 (full)',
        'text_epochs': 100,
        'vision_epochs': 100,
        'generative_epochs': 100,
        'batch_size': 64,
        'num_workers': 4,
        'patience': 20,
        'description': 'Full experiments',
    },
}

CURRENT_MODE = 'QUICK'  # Will be overridden by CLI args or user selection


def get_mode_config():
    """Return configuration for current execution mode."""
    return MODES[CURRENT_MODE]


# ============================================================================
# PART 1: MEDICAL TEXT ANALYSIS
# ============================================================================

TEXT_CONFIG = {
    'embedding_dim': 100,
    'max_vocab_size': 5000,
    'max_seq_length': 128,
    'num_classes': 2,
}

# LSTM/GRU Configuration
TEXT_RNN_CONFIG = {
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': True,
}

# Transformer Configuration
TEXT_TRANSFORMER_CONFIG = {
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'dropout': 0.2,
}

# Regularization experiments for Part 1
TEXT_REGULARIZATION = {
    'dropout_rates': [0.1, 0.3, 0.5],
    'l2_lambdas': [1e-5, 1e-4, 1e-3],
}


# ============================================================================
# PART 2: MEDICAL IMAGE ANALYSIS
# ============================================================================

IMAGE_CONFIG = {
    'image_size': 224,
    'num_classes': 2,  # Binary classification (normal vs abnormal)
}

# Vision Model Configurations
VISION_MODEL_CONFIG = {
    'ResNet18': {
        'pretrained': True,
        'num_classes': IMAGE_CONFIG['num_classes'],
    },
    'ResNet50': {
        'pretrained': True,
        'num_classes': IMAGE_CONFIG['num_classes'],
    },
}

# Image augmentation
IMAGE_AUGMENTATION = {
    'rotation_range': 20,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'brightness_range': [0.8, 1.2],
}


# ============================================================================
# PART 3: GENERATIVE MODELS
# ============================================================================

GENERATIVE_CONFIG = {
    'model_type': 'VAE',  # 'VAE' or 'GAN'
    'latent_dim': 32,
}

# VAE Configuration
VAE_CONFIG = {
    'input_channels': 1,  # Grayscale X-rays
    'latent_dim': 32,
    'hidden_dim': 128,
    'reconstruction_loss': 'mse',  # 'mse' or 'bce'
    'kl_weight': 1.0,
}

# GAN Configuration
GAN_CONFIG = {
    'noise_dim': 100,
    'generator_hidden': 256,
    'discriminator_hidden': 256,
    'generator_lr': 2e-4,
    'discriminator_lr': 2e-4,
    'beta1': 0.5,
}


# ============================================================================
# PART 4: OPTIMIZATION
# ============================================================================

OPTIMIZATION_CONFIG = {
    'pruning_sparsity': 0.3,  # Remove 30% of weights
    'pruning_method': 'unstructured',  # 'structured' or 'unstructured'
    'quantization_bits': 8,
    'knowledge_distillation_temp': 4.0,
    'distillation_alpha': 0.7,
}


# ============================================================================
# PART 5: ETHICS & FAIRNESS
# ============================================================================

ETHICS_CONFIG = {
    'demographic_groups': [
        'age_group',
        'gender',
        'skin_tone',
    ],
    'fairness_metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
    ],
    'bias_threshold': 0.1,  # Flag if >10% disparity
}


# ============================================================================
# TRAINING HYPERPARAMETERS (Shared across all parts)
# ============================================================================

TRAINING_CONFIG = {
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_factor': 0.5,
    'scheduler_patience': 5,
    'gradient_clip': 1.0,
    'warmup_epochs': 5,
}


# ============================================================================
# DATA PATHS
# ============================================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATHS = {
    'text_dataset': os.path.join(BASE_DIR, 'data', 'text_data'),
    'image_dataset': os.path.join(BASE_DIR, 'data', 'image_data'),
    'results': os.path.join(BASE_DIR, 'results'),
}

# Create directories if they don't exist
for path in DATA_PATHS.values():
    os.makedirs(path, exist_ok=True)


# ============================================================================
# UTILITY FUNCTION
# ============================================================================

def print_config():
    """Print current configuration."""
    mode = MODES[CURRENT_MODE]
    print(f"\n{'=' * 60}")
    print(f"  Mode: {CURRENT_MODE}")
    print(f"  {mode['description']}")
    print(f"{'=' * 60}")
    print(f"  Text Dataset: {mode['text_dataset']}")
    print(f"  Image Dataset: {mode['image_dataset']}")
    print(f"  Text Epochs: {mode['text_epochs']}")
    print(f"  Vision Epochs: {mode['vision_epochs']}")
    print(f"  Generative Epochs: {mode['generative_epochs']}")
    print(f"  Batch Size: {mode['batch_size']}")
    print(f"  Patience: {mode['patience']}")
    print(f"  Device: {DEVICE}")
    print(f"{'=' * 60}\n")
