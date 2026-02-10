# Intelligent Healthcare Assistant System

**Comprehensive Deep Learning Project** (25 points) | Due: Feb 12

A complete ML engineering workflow implementing an intelligent healthcare AI system across 5 major components.

---

## Project Overview

This project simulates a realistic ML engineering workflow by building an intelligent healthcare assistant system that:

1. **Analyzes medical text** with sequence models (LSTM/GRU vs Transformer)
2. **Classifies medical images** with vision models (ResNet, object detection)
3. **Generates synthetic medical data** with generative models (VAE/GAN)
4. **Deploys optimized models** for resource-constrained devices
5. **Evaluates ethics & fairness** with bias audits and comprehensive analysis

---

## Project Structure

```
hw_2/
├── config.py                       # Configuration (QUICK/HYBRID/FULL modes)
├── main.py                         # Main entry point with interactive menu
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── PLAN.md                         # Detailed requirements (25 points breakdown)
├── ARCHITECTURE.md                 # Design patterns and architecture
├── IMPLEMENTATION_GUIDE.md         # Implementation roadmap
├── main_template.py                # Reference template
├── config_template.py              # Reference template
│
├── data/                           # Data loading modules
│   ├── __init__.py
│   ├── text_loader.py             # Medical text dataset loader
│   └── image_loader.py            # Medical image dataset loader
│
├── models/                         # Model implementations
│   ├── __init__.py
│   ├── text_models.py             # LSTM/GRU + Transformer for text
│   └── vision_models.py           # ResNet for images
│
├── models_generative/              # Generative models
│   ├── __init__.py
│   ├── vae.py                     # Variational Autoencoder
│   └── gan.py                     # Generative Adversarial Network
│
├── training/                       # Training modules
│   ├── __init__.py
│   ├── text_trainer.py            # Text model trainer
│   ├── vision_trainer.py          # Vision model trainer
│   └── generative_trainer.py      # Generative model trainer
│
├── optimization/                   # Model optimization
│   ├── __init__.py
│   └── model_optimizer.py         # Pruning, quantization, distillation
│
├── evaluation/                     # Evaluation metrics & analysis
│   ├── __init__.py
│   ├── text_metrics.py            # Text classification metrics
│   ├── vision_metrics.py          # Vision task metrics
│   └── ethics.py                  # Bias audit & ethics analysis
│
├── visualization/                  # Plotting functions
│   ├── __init__.py
│   └── plots.py                   # All visualization functions
│
├── notebooks/                      # Jupyter notebooks (optional)
│   ├── part1_analysis.ipynb
│   ├── part2_analysis.ipynb
│   └── ...
│
└── results/                        # Auto-created output directory
    └── results_{mode}_{timestamp}/ # Timestamped results
```

---

## Quick Start

### 1. Installation

```bash
# Navigate to project
cd D:\allcode\deeplearning\deepl\hw_2

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Interactive Mode

```bash
# Start interactive menu
python main.py

# Select mode (QUICK/HYBRID/FULL)
# Choose what to run (Part 1-5 or full pipeline)
```

### 3. Run Non-Interactive Mode

```bash
# Run specific part in QUICK mode
python main.py --mode QUICK --run 1

# Run full pipeline in FULL mode
python main.py --mode FULL --run 6
```

### 4. Check Configuration

```bash
# View current configuration
python main.py --mode QUICK --run 8
```

---

## Execution Modes

| Mode | Use Case | Dataset Size | Epochs | Time |
|------|----------|--------------|--------|------|
| **QUICK** | Fast testing, debugging | Small | 15-20 | ~10-30 min |
| **HYBRID** | Balanced experiments | Medium | 50 | ~1-2 hours |
| **FULL** | Complete experiments | Large | 100 | ~4-8 hours |

Select mode when starting the program or via CLI: `--mode QUICK/HYBRID/FULL`

---

## Part Descriptions

### Part 1: Medical Text Analysis (7 points)
**Objective**: Build and compare text classifiers

- **Models**: LSTM/GRU vs Transformer
- **Task**: Classify medical documents or extract clinical information
- **Regularization**: Test dropout (0.1, 0.3, 0.5) and L2 (1e-5, 1e-4, 1e-3)
- **Deliverables**:
  - Training curves for both models
  - Comparison table (parameters, training time, accuracy, overfitting)
  - Gradient analysis plots
  - Written explanation (150-250 words)

### Part 2: Medical Image Analysis (6 points)
**Objective**: Build and optimize vision model with bias-variance analysis

- **Task**: Classification, detection, or segmentation of medical images
- **Models**: ResNet-18 vs ResNet-50 (or other architectures)
- **Analysis**: Bias-variance trade-off, learning curves
- **Deliverables**:
  - 5-10 prediction visualizations
  - Learning curves (train vs validation)
  - Metrics: precision, recall, F1, AUC-ROC
  - Written analysis (200-300 words)

### Part 3: Generative Models (5 points)
**Objective**: Generate synthetic medical data with VAE or GAN

- **Options**: VAE for medical images, GAN for medical images, or VAE/GAN for tabular data
- **Documentation**: Training challenges (mode collapse, posterior collapse)
- **Deliverables**:
  - 10-20 generated samples
  - Loss curves (reconstruction + KL or discriminator + generator)
  - Qualitative evaluation
  - Written reflection (150-200 words)

### Part 4: Model Optimization (4 points)
**Objective**: Optimize model for deployment on resource-constrained devices

- **Techniques**: Choose 2+ (pruning, quantization, knowledge distillation, ONNX)
- **Benchmarking**: Size, inference time, accuracy trade-offs
- **Deliverables**:
  - Comparison table (before/after)
  - Written explanation (100-150 words)

### Part 5: Ethics & Fairness (3 points)
**Objective**: Evaluate ethical implications and fairness

- **Bias Audit**: Test on demographic subgroups
- **Ethics Report**: 400-600 words covering privacy, fairness, transparency, clinical validation, dual-use risks
- **Recommendations**: At least 2 concrete actions
- **Deliverables**:
  - Bias audit results
  - Comprehensive ethics report
  - Actionable recommendations

---

## Configuration

All hyperparameters are managed in `config.py`. Key sections:

- **Execution Modes**: QUICK/HYBRID/FULL with different settings
- **Part 1 (Text)**: Text RNN and Transformer configs
- **Part 2 (Vision)**: Image size, model architectures, augmentation
- **Part 3 (Generative)**: VAE and GAN configs
- **Part 4 (Optimization)**: Pruning sparsity, quantization bits
- **Part 5 (Ethics)**: Demographic groups, fairness metrics
- **Training**: Learning rate, weight decay, scheduler params

---

## Implementation Status

### Foundation (Complete)
- [x] Project structure and directories
- [x] `config.py` with 3 execution modes
- [x] `main.py` with interactive menu system
- [x] Requirements.txt with dependencies
- [x] Skeleton modules for all components

### To Implement
- [ ] Data loaders (text and image)
- [ ] Text models (LSTM/GRU, Transformer)
- [ ] Vision models (ResNet wrappers)
- [ ] Generative models (VAE, GAN)
- [ ] Trainers (text, vision, generative)
- [ ] Evaluation metrics (all parts)
- [ ] Optimization techniques
- [ ] Ethics analysis
- [ ] Visualization functions

---

## Usage Example: Full Pipeline

```python
# Interactive mode - follow menu prompts
$ python main.py
Device: cuda

Select Execution Mode:
  1. QUICK   - Fast testing
  2. HYBRID  - Balanced
  3. FULL    - Full experiments
Select mode: 1
Mode set to: QUICK

Intelligent Healthcare Assistant System
Mode: QUICK | Device: cuda
==================================================
  1. Part 1: Medical Text Analysis (7 pts)
  2. Part 2: Medical Image Analysis (6 pts)
  3. Part 3: Generative Models (5 pts)
  4. Part 4: Model Optimization (4 pts)
  5. Part 5: Ethics & Fairness (3 pts)
  6. Run Full Pipeline (All Parts)
  7. Change Execution Mode
  8. Show Configuration
  0. Exit
Select option: 6

RUNNING FULL PIPELINE
Mode: QUICK
Device: cuda
==================================================

[Step 1/5] Part 1 - Medical Text Analysis
...
[Step 5/5] Part 5 - Ethics Analysis
...

PIPELINE COMPLETE!
All results saved to: results_QUICK_20260212_101530/
```

---

## Output Structure

Results are automatically organized in timestamped directories:

```
results_QUICK_20260212_101530/
├── RESULTS_SUMMARY.txt
├── part1_training_curves.png
├── part1_regularization.csv
├── part2_learning_curves.png
├── part2_predictions.png
├── part3_generated_samples.png
├── part3_loss_curves.png
├── part4_optimization_comparison.csv
├── part5_bias_audit.csv
└── part5_ethics_report.txt
```

---

## Key Design Patterns

This project follows proven architectural patterns from the endterm project:

1. **State Management**: Global dictionaries for tracking models and results
2. **Lazy Loading**: Data only loads when needed
3. **Helper Functions**: Reduce code duplication across menu functions
4. **Timestamped Results**: Auto-organize outputs by date/time
5. **Mode-Based Config**: Different scales for QUICK/HYBRID/FULL
6. **Dual Execution**: Interactive menu or command-line automation
7. **Modular Structure**: Each component is independent but coordinated

---

## Development Workflow

1. **Start with Part 1**: Get text analysis working end-to-end
2. **Add Part 2**: Implement image analysis
3. **Add Parts 3-5**: Build generative, optimization, and ethics components
4. **Test Full Pipeline**: Run all parts together
5. **Polish & Package**: Add visualizations, finalize report

---

## Common Tasks

### Change execution mode
```bash
python main.py --mode HYBRID
# Then select from menu, or run directly with --run
```

### Run only Part 2 (Vision)
```bash
python main.py --mode FULL --run 2
```

### Run full pipeline non-interactively
```bash
python main.py --mode FULL --run 6
```

### Check what's configured
```bash
python main.py --mode HYBRID --run 8
```

---

## References

- **PLAN.md**: Detailed breakdown of all 25 points and requirements
- **ARCHITECTURE.md**: Design patterns and architecture decisions
- **IMPLEMENTATION_GUIDE.md**: Step-by-step implementation roadmap
- **Endterm Project**: Reference implementation of similar architecture

---

## Submission Checklist

- [ ] All code implemented and tested
- [ ] Config.py properly configured
- [ ] Main.py menu system working
- [ ] All 5 parts functional
- [ ] Full pipeline runs end-to-end
- [ ] Results directory with all outputs
- [ ] Report PDF generated
- [ ] Package as ZIP file: `LastName_FirstName_DL_Project.zip`
- [ ] Submit before Feb 12 deadline

---

## Help & Troubleshooting

### Import errors
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### CUDA not available
- Model will fall back to CPU automatically
- Slower but still functional
- Check: `python main.py --mode QUICK --run 8`

### Out of memory
- Reduce batch size in config.py
- Use QUICK mode with smaller datasets
- Reduce model hidden dimensions

### Data not loading
- Check data paths in config.py
- Implement actual dataset loading in data/ modules
- See comments in data_loader.py for guidance

---

## Contact & Questions

For questions about the project structure or implementation:
- Check IMPLEMENTATION_GUIDE.md
- Review reference templates (main_template.py, config_template.py)
- Examine endterm project for similar patterns

---

**Good luck with your implementation! Remember: thoughtful application of deep learning principles to a complex, multi-faceted problem is the goal.**
