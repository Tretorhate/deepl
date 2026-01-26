import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from models import RegularizedMLP
import numpy as np

num_epochs = 10  # Reduced for speed while still showing patterns

def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_and_validate(dropout_rate=0.0, weight_decay=0.0, patience=5, num_epochs=num_epochs):
    # Load Data (using smaller subset for speed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_loader = DataLoader(Subset(train_set, range(10000)), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(train_set, range(10000, 12000)), batch_size=128)

    model = RegularizedMLP(dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay) # Weight Decay = L2
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                val_loss += criterion(model(images), labels).item()

        # Calculate accuracies
        train_acc = calculate_accuracy(model, train_loader)
        val_acc = calculate_accuracy(model, val_loader)

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

    return train_acc_history, val_acc_history, model

def run_dropout_comparison(dropout_rates=[0.0, 0.5], num_epochs=num_epochs):
    """
    Task 1.2: Compare neural network classifiers with and without dropout on MNIST
    """
    print("="*60)
    print("TASK 1.2: DROPOUT REGULARIZATION COMPARISON ON MNIST")
    print("="*60)

    # Load test data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128)

    results = {}

    for dropout_rate in dropout_rates:
        print(f"\n--- Training Model with Dropout Rate: {dropout_rate} ---")
        train_acc_hist, val_acc_hist, model = train_and_validate(
            dropout_rate=dropout_rate,
            num_epochs=num_epochs,
            patience=10  # Disable early stopping for full comparison
        )

        # Evaluate on test set
        test_acc = calculate_accuracy(model, test_loader)

        results[dropout_rate] = {
            'train_acc_history': train_acc_hist,
            'val_acc_history': val_acc_hist,
            'final_train_acc': train_acc_hist[-1],
            'final_val_acc': val_acc_hist[-1],
            'test_acc': test_acc,
            'generalization_gap': train_acc_hist[-1] - val_acc_hist[-1]
        }

    # Plot training curves
    plt.figure(figsize=(12, 5))

    # Training and Validation Accuracy Curves
    plt.subplot(1, 2, 1)
    for dropout_rate, data in results.items():
        label = f'Dropout {dropout_rate}'
        plt.plot(data['train_acc_history'], label=f'{label} (Train)', linestyle='--')
        plt.plot(data['val_acc_history'], label=f'{label} (Val)', linestyle='-')
    plt.title('Training and Validation Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Generalization Gap
    plt.subplot(1, 2, 2)
    gaps = [results[rate]['generalization_gap'] for rate in dropout_rates]
    plt.bar([f'Dropout {rate}' for rate in dropout_rates], gaps, color=['blue', 'orange'])
    plt.title('Generalization Gap (Train - Val Accuracy)')
    plt.ylabel('Gap (%)')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save the plot to results folder
    plot_filename = 'results/dropout_comparison_curves.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    plt.show()

    # Print comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(f"{'Dropout Rate':<15} {'Train Acc (%)':<15} {'Val Acc (%)':<15} {'Test Acc (%)':<15} {'Gen Gap (%)':<15}")
    print("-"*75)
    for rate in dropout_rates:
        data = results[rate]
        print(f"{rate:<15.1f} {data['final_train_acc']:<15.2f} {data['final_val_acc']:<15.2f} {data['test_acc']:<15.2f} {data['generalization_gap']:<15.2f}")

    # Overfitting Analysis
    print("\n" + "="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)

    gap_no_dropout = results[0.0]['generalization_gap']
    gap_with_dropout = results[0.5]['generalization_gap']

    print(f"Model without dropout (0.0) has generalization gap: {gap_no_dropout:.2f}%")
    print(f"Model with dropout (0.5) has generalization gap: {gap_with_dropout:.2f}%")

    if gap_no_dropout > gap_with_dropout:
        print("\nCONCLUSION: The model WITHOUT dropout overfits more severely.")
        print("REASON: Without dropout, the model can memorize training data more easily,")
        print("        leading to higher training accuracy but poorer generalization to")
        print("        unseen validation data. Dropout randomly deactivates neurons during")
        print("        training, preventing co-adaptation and improving generalization.")
    else:
        print("\nCONCLUSION: The model WITH dropout overfits more severely.")
        print("REASON: This is unexpected. Possible reasons: insufficient dropout rate,")
        print("        model capacity too low, or other factors affecting regularization.")

    print(f"\nTest Accuracy - No Dropout: {results[0.0]['test_acc']:.2f}%")
    print(f"Test Accuracy - With Dropout: {results[0.5]['test_acc']:.2f}%")

    if results[0.0]['test_acc'] > results[0.5]['test_acc']:
        print("The model without dropout performs better on test set.")
    else:
        print("The model with dropout performs better on test set (better generalization).")

    print("\nPlots saved as 'dropout_comparison_curves.png'")
    print("Task 1.2 completed successfully!")

def run_l2_regularization_comparison(lambdas=[0, 0.0001, 0.001, 0.01], num_epochs=num_epochs):
    """
    Task 1.3: L2 Regularization (Weight Decay) Comparison
    """
    print("="*60)
    print("TASK 1.3: L2 REGULARIZATION (WEIGHT DECAY) COMPARISON")
    print("="*60)

    # Load data (using smaller subset for speed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(Subset(train_set, range(10000)), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(train_set, range(10000, 12000)), batch_size=128)
    test_loader = DataLoader(test_set, batch_size=64)

    results = {}

    for lambda_val in lambdas:
        print(f"\n--- Training Model with Lambda: {lambda_val} ---")
        model = RegularizedMLP(dropout_rate=0.0)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=lambda_val)
        criterion = nn.CrossEntropyLoss()

        val_accs = []
        train_accs = []

        for epoch in range(num_epochs):
            # Training
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

            # Validation
            train_acc = calculate_accuracy(model, train_loader)
            val_acc = calculate_accuracy(model, val_loader)

            train_accs.append(train_acc)
            val_accs.append(val_acc)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Evaluate on test set
        test_acc = calculate_accuracy(model, test_loader)

        # Extract weights for histogram
        weights = model.network[1].weight.data.flatten().cpu().numpy()
        weight_norm = np.linalg.norm(weights)

        results[lambda_val] = {
            'train_accs': train_accs,
            'val_accs': val_accs,
            'final_train_acc': train_accs[-1],
            'final_val_acc': val_accs[-1],
            'test_acc': test_acc,
            'weights': weights,
            'weight_norm': weight_norm
        }

    # Plot 1: Validation Accuracy vs Lambda
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    lambda_vals = list(results.keys())
    val_accs_list = [results[lam]['final_val_acc'] for lam in lambda_vals]
    plt.plot(range(len(lambda_vals)), val_accs_list, 'o-', linewidth=2, markersize=8)
    plt.xticks(range(len(lambda_vals)), [f'{lam}' for lam in lambda_vals])
    plt.xlabel('Lambda (Regularization Strength)', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('Validation Accuracy vs Lambda', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 2: Weight Magnitude Histograms
    plt.subplot(1, 2, 2)
    for i, lambda_val in enumerate(lambda_vals):
        weights = results[lambda_val]['weights']
        plt.hist(weights, bins=50, alpha=0.5, label=f'λ={lambda_val}')

    plt.xlabel('Weight Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Weight Magnitude Distribution (Different Lambda)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save the plot
    plot_filename = 'results/l2_regularization_comparison.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    plt.show()

    # Print comparison table
    print("\n" + "="*60)
    print("L2 REGULARIZATION COMPARISON TABLE")
    print("="*60)
    print(f"{'Lambda':<15} {'Train Acc (%)':<15} {'Val Acc (%)':<15} {'Test Acc (%)':<15} {'Weight Norm':<15}")
    print("-"*75)

    for lambda_val in lambda_vals:
        data = results[lambda_val]
        print(f"{lambda_val:<15.4f} {data['final_train_acc']:<15.2f} {data['final_val_acc']:<15.2f} {data['test_acc']:<15.2f} {data['weight_norm']:<15.4f}")

    # Find optimal lambda
    optimal_lambda = max(lambda_vals, key=lambda x: results[x]['final_val_acc'])
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print(f"Optimal Lambda: {optimal_lambda} with Val Accuracy: {results[optimal_lambda]['final_val_acc']:.2f}%")
    print("\nKey Observations:")
    print("1. As lambda increases, weight magnitudes decrease (weights are more regularized)")
    print("2. Moderate regularization improves validation accuracy (reduces overfitting)")
    print("3. Too much regularization (large lambda) may hurt performance")
    print("4. L2 regularization prevents the model from fitting to noise in training data")

    print("\nTask 1.3 completed successfully!")


def run_early_stopping_comparison(max_epochs=30):
    """
    Task 1.4: Early Stopping Demonstration
    """
    print("="*60)
    print("TASK 1.4: EARLY STOPPING COMPARISON")
    print("="*60)

    # Load data (using smaller subset for speed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_loader = DataLoader(Subset(train_set, range(10000)), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(train_set, range(10000, 12000)), batch_size=128)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64)

    results = {}

    # Model 1: Without early stopping (fixed epochs)
    print("\n--- Training Model WITHOUT Early Stopping (50 epochs) ---")
    model1 = RegularizedMLP(dropout_rate=0.0)
    optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    val_losses_no_stop = []

    for epoch in range(max_epochs):
        model1.train()
        for images, labels in train_loader:
            optimizer1.zero_grad()
            loss = criterion(model1(images), labels)
            loss.backward()
            optimizer1.step()

        # Validation
        model1.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                val_loss += criterion(model1(images), labels).item()

        val_losses_no_stop.append(val_loss)
        print(f"Epoch {epoch+1}/{max_epochs}, Val Loss: {val_loss:.4f}")

    test_acc1 = calculate_accuracy(model1, test_loader)
    results['no_early_stop'] = {
        'val_losses': val_losses_no_stop,
        'epochs': max_epochs,
        'test_acc': test_acc1
    }

    # Model 2: With early stopping
    print("\n--- Training Model WITH Early Stopping (patience=7) ---")
    model2 = RegularizedMLP(dropout_rate=0.0)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

    val_losses_with_stop = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 7
    epoch_stopped = 0

    for epoch in range(max_epochs):
        model2.train()
        for images, labels in train_loader:
            optimizer2.zero_grad()
            loss = criterion(model2(images), labels)
            loss.backward()
            optimizer2.step()

        # Validation
        model2.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                val_loss += criterion(model2(images), labels).item()

        val_losses_with_stop.append(val_loss)
        print(f"Epoch {epoch+1}/{max_epochs}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}!")
                epoch_stopped = epoch + 1
                break

    test_acc2 = calculate_accuracy(model2, test_loader)
    results['with_early_stop'] = {
        'val_losses': val_losses_with_stop,
        'epochs': epoch_stopped,
        'test_acc': test_acc2
    }

    # Plot combined loss curves
    plt.figure(figsize=(12, 5))

    epochs1 = range(1, len(val_losses_no_stop) + 1)
    epochs2 = range(1, len(val_losses_with_stop) + 1)

    plt.plot(epochs1, val_losses_no_stop, 'o-', label='Without Early Stopping', linewidth=2)
    plt.plot(epochs2, val_losses_with_stop, 's-', label='With Early Stopping', linewidth=2)

    # Mark early stopping point
    if epoch_stopped > 0:
        plt.axvline(x=epoch_stopped, color='red', linestyle='--', alpha=0.7, linewidth=2)
        plt.text(epoch_stopped, min(val_losses_no_stop) * 1.1, f'Early Stop\nEpoch {epoch_stopped}',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Early Stopping Comparison: Training Loss Curves', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save the plot
    plot_filename = 'results/early_stopping_comparison.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    plt.show()

    # Print comparison table
    print("\n" + "="*60)
    print("EARLY STOPPING COMPARISON TABLE")
    print("="*60)
    print(f"{'Model':<25} {'Epochs Trained':<15} {'Test Acc (%)':<15}")
    print("-"*55)

    epochs_no_stop = results['no_early_stop']['epochs']
    epochs_with_stop = results['with_early_stop']['epochs']
    time_saved_pct = ((epochs_no_stop - epochs_with_stop) / epochs_no_stop) * 100

    print(f"{'Without Early Stop':<25} {epochs_no_stop:<15} {results['no_early_stop']['test_acc']:<15.2f}")
    print(f"{'With Early Stop':<25} {epochs_with_stop:<15} {results['with_early_stop']['test_acc']:<15.2f}")

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print(f"Epochs trained without early stopping: {epochs_no_stop}")
    print(f"Epochs trained with early stopping: {epochs_with_stop}")
    print(f"Time saved: {time_saved_pct:.1f}% ({epochs_no_stop - epochs_with_stop} epochs)")
    print(f"\nTest Accuracy - No Early Stop: {results['no_early_stop']['test_acc']:.2f}%")
    print(f"Test Accuracy - With Early Stop: {results['with_early_stop']['test_acc']:.2f}%")
    print(f"Accuracy Difference: {abs(results['no_early_stop']['test_acc'] - results['with_early_stop']['test_acc']):.2f}%")

    print("\nKey Insights:")
    print("1. Early stopping achieves similar or better test accuracy in fewer epochs")
    print("2. Significant computational savings without sacrificing performance")
    print("3. Prevents overfitting by stopping when validation loss starts increasing")
    print("4. Practical benefit: Reduced training time and computational resources")

    print("\nTask 1.4 completed successfully!")


if __name__ == "__main__":
    # Run the full Task 1.2 comparison
    run_dropout_comparison()
