# Deep Learning Practice Session: Weeks 5-6 - Complete Implementation

## Overview

Complete implementation of all 9 deep learning tasks covering regularization techniques and sequence modeling with RNNs. All experiments are fully implemented, optimized for speed, and save results automatically to the `results/` folder.

**Status**: 100% Complete (9/9 tasks)
**Runtime**: ~1 hour (optimized for speed)

## Quick Start

Run the interactive menu system:

```bash
python main.py
```

## Menu Options

### PART 1: REGULARIZATION & GENERALIZATION

1. **Task 1.1**: Polynomial Regression (Overfitting Analysis)
   - Demonstrates overfitting, underfitting, and well-fitting with polynomial degrees 1, 3, 15
   - Generates plots with train/test MSE comparison
   - Saves analysis explaining each case

2. **Task 1.2**: Dropout Regularization (0.0 vs 0.5)
   - 3-layer neural network (512-256-128) on MNIST
   - Compares models with/without dropout
   - Shows generalization gap and test accuracy comparison

3. **Task 1.3**: L2 Regularization Comparison
   - Tests 4 lambda values: [0, 0.0001, 0.001, 0.01]
   - Plots validation accuracy vs lambda
   - Visualizes weight magnitude distributions with histograms
   - Identifies optimal regularization strength

4. **Task 1.4**: Early Stopping Demonstration
   - Compares fixed 30 epochs vs early stopping with patience=7
   - Shows loss curves with stopping point marked
   - Reports time savings and accuracy difference

5. **Task 1.5**: Bias-Variance Tradeoff Analysis
   - Trains polynomials with degrees 1-20
   - Plots U-shaped error curve showing optimal complexity
   - Visualizes underfit, optimal, and overfit models
   - Includes comprehensive 1-page analysis

### PART 2: SEQUENCE MODELING & RNNs

6. **Task 2.1**: RNN Fundamentals (Manual BPTT Training)
   - NumPy implementation with forward pass and BPTT
   - Trains on Echo task with 300 iterations
   - Plots training loss curve
   - Visualizes hidden state evolution as heatmap

7. **Task 2.2**: Vanishing Gradients Analysis
   - Tests gradient decay across 3 sequence lengths (20, 50, 100)
   - Plots gradient magnitude on log scale
   - Shows how gradients vanish over time steps

8. **Task 2.2b**: Spectral Radius Analysis
   - Calculates spectral radius of RNN weight matrix
   - Plots eigenvalues in complex plane
   - Analyzes relationship to vanishing gradients

9. **Task 2.3**: RNN vs LSTM vs GRU Comparison
   - Basic convergence comparison on sequence sum task
   - Trains all 3 architectures with identical hidden dimensions

10. **Task 2.3b**: Ablation Study (Sequence Lengths)
    - Tests RNN/LSTM/GRU on 3 sequence lengths: 15, 30, 50
    - Creates 3x3 grid showing convergence at each length
    - Demonstrates how each architecture handles longer sequences

11. **Task 2.3c**: Comprehensive Model Comparison
    - Compares training time, final loss, parameter counts
    - Provides architectural recommendations
    - Includes detailed analysis table

12. **Task 2.4**: Time-Series Prediction Application
    - Trains RNN/LSTM/GRU on synthetic sine wave data
    - Compares against Moving Average baseline
    - Reports MSE/MAE metrics
    - Plots predictions vs actual values

## File Structure

```
practice_s_3/
├── main.py                      # Interactive menu system
├── models.py                    # Neural network definitions
├── part1_poly_analysis.py       # Tasks 1.1 & 1.5
├── part1_nn_regularization.py   # Tasks 1.2, 1.3, 1.4
├── part2_rnn_fundamentals.py    # Tasks 2.1 & 2.2
├── part2_comparison.py          # Task 2.3 (all variants)
├── part2_application.py         # Task 2.4 (time-series)
├── data/                        # Downloaded datasets
└── results/                     # All output plots and analyses
```

## Results Saved to `results/` Folder

### Part 1 Results

- `polynomial_regression_analysis.png` - Task 1.1 plots
- `polynomial_regression_analysis.txt` - Analysis text
- `bias_variance_complexity_error_plot.png` - Task 1.5 complexity plot
- `bias_variance_tradeoff_analysis.txt` - Comprehensive analysis
- `dropout_comparison_curves.png` - Task 1.2 comparison
- `l2_regularization_comparison.png` - Task 1.3 plots (accuracy + histograms)
- `early_stopping_comparison.png` - Task 1.4 loss curves

### Part 2 Results

- `manual_rnn_training_loss.png` - Task 2.1 training loss
- `manual_rnn_hidden_states_heatmap.png` - Hidden state visualization
- `vanishing_gradients_multi_length.png` - Task 2.2 gradient analysis
- `spectral_radius_analysis.png` - Task 2.2b eigenvalue plots
- `rnn_convergence_comparison.png` - Task 2.3 basic comparison
- `ablation_study_sequence_lengths.png` - Task 2.3b 3x3 grid
- `time_series_predictions.png` - Task 2.4 predictions plot
- `time_series_error_comparison.png` - Task 2.4 error bars

## Requirements

```
Python 3.8+
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
```

## Installation

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

## Usage Tips

- **Run all tasks**: Execute options 1-12 sequentially from menu
- **Individual tasks**: Run any option independently
- **Results**: Check `results/` folder after each run for plots and analyses
- **Data**: Datasets auto-download on first run to `data/` folder
- **Menu**: Press `0` to exit, `Enter` to return to menu

## Implementation Highlights

- **BPTT from scratch**: Manual backpropagation through time in NumPy
- **Complete RNN architectures**: Vanilla RNN, LSTM, GRU implementations
- **Regularization techniques**: Dropout, L2 weight decay, early stopping
- **Analysis visualizations**: Heatmaps, histograms, error curves, eigenvalue plots
- **Comparison tables**: MSE, accuracy, parameters, training time
- **Fast execution**: Optimized hyperparameters for 1-hour total runtime
- **High-quality plots**: 300 DPI, clear labels, legends, grid

## Key Insights Generated

1. How regularization prevents overfitting
2. Bias-variance tradeoff and optimal model complexity
3. Vanishing gradients in vanilla RNNs
4. LSTM/GRU advantages over vanilla RNN
5. Effect of sequence length on model performance
6. Time-series prediction with deep learning

## Educational Value

Each task includes:

- Clear visualizations of key concepts
- Comparison tables with multiple metrics
- Detailed written analysis
- Code demonstrating both manual and framework implementations
- Practical recommendations for model selection

## Notes

- All experiments use smaller datasets/fewer epochs for speed (not accuracy maximization)
- Results demonstrate concepts clearly even with reduced scale
- For production: increase epochs, dataset sizes, and perform proper hyperparameter tuning
- All random seeds fixed for reproducibility

## Completion Status

```
Part 1: REGULARIZATION & GENERALIZATION
├──   1.1 Overfitting/Underfitting
├──   1.2 Dropout Regularization
├──   1.3 L2 Regularization
├──   1.4 Early Stopping
└──   1.5 Bias-Variance Tradeoff

Part 2: SEQUENCE MODELING & RNNs
├──   2.1 RNN Fundamentals (BPTT)
├──   2.2 Vanishing Gradients
├──   2.3 Model Comparison (3 variants)
└──   2.4 Time-Series Application

OVERALL: 9/9 tasks completed
```
