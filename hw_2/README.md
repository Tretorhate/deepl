# Intelligent Healthcare Assistant System

**Comprehensive Deep Learning Project** (25 points) | Status: 100% Complete

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

## Usage Example: Full Pipeline

```bash
# Interactive mode - follow menu prompts
$ python main.py
Device: cuda

Select Execution Mode:
  1. QUICK   - Fast testing (5-10 minutes)
  2. HYBRID  - Balanced (30-60 minutes)
  3. FULL    - Full experiments (2-4 hours)
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

RUNNING FULL PIPELINE WITH VISUALIZATION & RESULTS
Mode: QUICK
Device: cuda
==================================================

[Step 1/7] Part 1 - Medical Text Analysis
[Step 2/7] Part 2 - Medical Image Analysis
[Step 3/7] Part 3 - Generative Models
[Step 4/7] Part 4 - Model Optimization
[Step 5/7] Part 5 - Ethics Analysis
[Step 6/7] Aggregating results and generating visualizations...
      Training curves saved: results/.../part1_training_curves.png
      Model comparison saved: results/.../part2_model_comparison.png
      Fairness analysis saved: results/.../part5_fairness_analysis.png
      Bias flags saved: results/.../part5_bias_flags.png
[Step 7/7] Generating comprehensive reports...
      JSON results: results/.../results.json
      Text summary: results/.../PROJECT_SUMMARY.txt
      Detailed results: results/.../DETAILED_RESULTS.txt

PIPELINE COMPLETE!

Results Directory: results/
Generated Files:
  • results.json
  • PROJECT_SUMMARY.txt
  • DETAILED_RESULTS.txt
  • part1_training_curves.png
  • part2_model_comparison.png
  • part5_fairness_analysis.png
  • part5_bias_flags.png
  • ethics_report.txt
  • recommendations.txt

All results are ready for inclusion in reports and presentations!
```

---

## Output Structure

Results are automatically generated in the `results/` directory when running the pipeline:

```
results/
├── results.json                         # Machine-readable JSON with all metrics
├── PROJECT_SUMMARY.txt                  # Executive summary
├── DETAILED_RESULTS.txt                 # Comprehensive metrics breakdown
├── RESULTS_SUMMARY.txt                  # Legacy summary format
├── part1_training_curves.png            # Text model training curves
├── part2_model_comparison.png           # Vision model accuracy comparison
├── part5_fairness_analysis.png          # Demographic group performance heatmap
├── part5_bias_flags.png                 # Bias detection visualization
├── ethics_report.txt                    # Formal bias audit report
└── recommendations.txt                  # Fairness recommendations (5 categories)
```

All PNG files are 300 DPI publication-quality images. JSON and text reports are ready for further analysis.

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
- Check device: `python main.py --mode QUICK --run 8`

### Out of memory
- Reduce batch size in config.py
- Use QUICK mode for faster iterations
- Reduce epochs in configuration

### Results not generating
- Ensure results/ directory exists (auto-created)
- Check write permissions to results/ folder
- Verify PNG dependencies (matplotlib, seaborn)

### LaTeX compilation issues
- Use HW2_REPORT_FINAL.tex (references actual files only)
- Upload results/ folder with PNG files to Overleaf
- See OVERLEAF_SETUP_GUIDE.md for detailed instructions

---

## Documentation

- **HW2_REPORT_FINAL.tex**: LaTeX report with results and visualizations
- **VISUALIZATION_AND_RESULTS.md**: Guide to the visualization system
- **RESULTS_OUTPUT_STRUCTURE.md**: Expected output files and structure
- **OVERLEAF_SETUP_GUIDE.md**: Instructions for uploading to Overleaf
- **COMPLETE_PROJECT_SUMMARY.md**: Comprehensive project overview

---

**Project Complete - Ready for Submission**
