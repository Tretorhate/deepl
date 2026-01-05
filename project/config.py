"""
Centralized configuration and hyperparameters for the Deep Learning project.
"""

import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Section configurations are set dynamically based on per-section modes below
# Initial definitions removed - now handled by get_section_config()

# Device configuration
# Note: We check torch.cuda.is_available() at runtime in the actual code
# This is just a default - actual device selection happens in section4/resnet.py
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'  # Fallback if torch not installed

# Random seed for reproducibility
RANDOM_SEED = 42

# Per-section execution mode options:
# Each section can have its own mode: 'quick', 'hybrid', or 'full'
# Modes can be set via command-line or interactive menu
# Default modes (can be overridden):
SECTION1_MODE = 'full'  # 'quick', 'hybrid', or 'full'
SECTION2_MODE = 'full'  # 'quick', 'hybrid', or 'full'
SECTION3_MODE = 'full'  # 'quick', 'hybrid', or 'full'
SECTION4_MODE = 'full'  # 'quick', 'hybrid', or 'full'

# Legacy global modes (for backward compatibility, but per-section modes take precedence)
QUICK_MODE = False
HYBRID_MODE = False

def get_section_mode(section_num):
    """Get execution mode for a specific section."""
    mode_map = {
        1: SECTION1_MODE,
        2: SECTION2_MODE,
        3: SECTION3_MODE,
        4: SECTION4_MODE,
    }
    return mode_map.get(section_num, 'full')

def is_quick_mode(section_num=None):
    """Check if quick mode is enabled for a section."""
    if section_num:
        return get_section_mode(section_num) == 'quick'
    return QUICK_MODE

def is_hybrid_mode(section_num=None):
    """Check if hybrid mode is enabled for a section."""
    if section_num:
        return get_section_mode(section_num) == 'hybrid'
    return HYBRID_MODE

def is_full_mode(section_num=None):
    """Check if full mode is enabled for a section."""
    if section_num:
        return get_section_mode(section_num) == 'full'
    return not QUICK_MODE and not HYBRID_MODE

# Mode-specific configurations
# These are applied based on per-section modes
def get_section_config(section_num, mode):
    """Get configuration for a section based on its mode."""
    if mode == 'quick':
        return get_quick_config(section_num)
    elif mode == 'hybrid':
        return get_hybrid_config(section_num)
    else:  # full
        return get_full_config(section_num)

def get_full_config(section_num):
    """Get full mode configuration for a section."""
    configs = {
        1: {
            'hidden_sizes': [128, 64],
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 20,
            'optimizer': 'adam',
            'loss': 'cross_entropy',
        },
        2: {
            'learning_rates': [0.001, 0.01, 0.1],
            'optimizers': ['sgd', 'adam', 'rmsprop'],
            'epochs': 20,
            'batch_size': 64,
            'weight_decay': 0.0001,
        },
        3: {
            'num_filters': [32, 64, 128],
            'kernel_sizes': [3, 5],
            'pool_sizes': [2],
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam',
        },
        4: {
            'num_layers': [18, 34, 50],
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'pretrained': True,
        },
    }
    return configs.get(section_num, {})

def get_hybrid_config(section_num):
    """Get hybrid mode configuration for a section."""
    configs = {
        1: {
            'hidden_sizes': [128, 64],
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 128,
            'epochs': 3,
            'optimizer': 'adam',
            'loss': 'cross_entropy',
        },
        2: {
            'learning_rates': [0.001, 0.01, 0.1],
            'optimizers': ['sgd', 'adam', 'rmsprop'],
            'epochs': 3,
            'batch_size': 128,
            'weight_decay': 0.0001,
        },
        3: {
            'num_filters': [32, 64, 128],
            'kernel_sizes': [3, 5],
            'pool_sizes': [2],
            'epochs': 3,
            'batch_size': 64,
            'learning_rate': 0.001,
            'optimizer': 'adam',
        },
        4: {
            'num_layers': [18, 34, 50],
            'epochs': 2,
            'batch_size': 64,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'pretrained': True,
        },
    }
    return configs.get(section_num, {})

def get_quick_config(section_num):
    """Get quick mode configuration for a section."""
    configs = {
        1: {
            'hidden_sizes': [64, 32],
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 128,
            'epochs': 3,
            'optimizer': 'adam',
            'loss': 'cross_entropy',
        },
        2: {
            'learning_rates': [0.001, 0.01, 0.1],
            'optimizers': ['sgd', 'adam', 'rmsprop'],
            'epochs': 3,
            'batch_size': 128,
            'weight_decay': 0.0001,
        },
        3: {
            'num_filters': [32, 64, 128],
            'kernel_sizes': [3, 5],
            'pool_sizes': [2],
            'epochs': 3,
            'batch_size': 64,
            'learning_rate': 0.001,
            'optimizer': 'adam',
        },
        4: {
            'num_layers': [18, 34, 50],
            'epochs': 2,
            'batch_size': 128,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'pretrained': True,
        },
    }
    return configs.get(section_num, {})

# Apply configurations based on per-section modes
# Initialize with default modes, can be overridden at runtime
SECTION1_CONFIG = get_section_config(1, SECTION1_MODE)
SECTION2_CONFIG = get_section_config(2, SECTION2_MODE)
SECTION3_CONFIG = get_section_config(3, SECTION3_MODE)
SECTION4_CONFIG = get_section_config(4, SECTION4_MODE)
