# Deep Learning Homework 1
## Setup

```bash
pip install -r requirements.txt
```

## Running

Run all problems at once:

```bash
python run_all.py
```

Or run individually:

```bash
python problem1_mlp.py      # MLPs, activations, XOR
python problem2_optimization.py  # Optimizers, gradients, regularization
python problem3_cnn.py      # CNNs on CIFAR-10
python bonus_transfer_learning.py  # Transfer learning
```

## Files

- `problem1_mlp.py` - Part 1: MLP implementation, activation comparison, XOR
- `problem2_optimization.py` - Part 2: Optimizer comparison, gradient analysis, regularization
- `problem3_cnn.py` - Part 3: CNN on CIFAR-10, architecture experiments, visualizations
- `bonus_transfer_learning.py` - Bonus: ResNet18 fine-tuning

## Outputs

All plots saved to `results/` folder:

- `part1b_loss_comparison.png` - Activation function comparison
- `part1c_xor_boundary.png` - XOR decision boundary
- `part2a_optimizer_comparison.png` - Optimizer comparison
- `part2b_gradient_analysis.png` - Vanishing gradients
- `part2c_regularization.png` - Regularization effects
- `part3a_training_curves.png` - CNN training
- `part3a_confusion_matrix.png` - CIFAR-10 confusion matrix
- `part3b_architecture_comparison.png` - CNN architectures
- `part3c_filters.png` - Conv layer filters
- `part3c_activations.png` - Activation maps
- `bonus_transfer_learning.png` - Transfer learning results

## Notes

- Uses PyTorch
- Data downloaded automatically to `./data`
- GPU used if available

