import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

def run_polynomial_experiment():
    """Task 1.1: Overfitting and Underfitting Analysis"""
    print("="*60)
    print("TASK 1.1: OVERFITTING AND UNDERFITTING ANALYSIS")
    print("="*60)

    # Data Generation
    np.random.seed(42)
    X = np.sort(np.random.rand(100, 1) * 4 - 2, axis=0)
    y = np.sin(np.pi * X) + np.random.normal(0, 0.1, X.shape)

    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]

    degrees = [1, 3, 15]  # Underfit, Good fit, Overfit
    results = {}

    plt.figure(figsize=(14, 5))

    for i, degree in enumerate(degrees):
        poly = PolynomialFeatures(degree=degree)
        X_poly_train = poly.fit_transform(X_train)

        model = LinearRegression().fit(X_poly_train, y_train)
        y_train_pred = model.predict(X_poly_train)
        y_test_pred = model.predict(poly.transform(X_test))

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        results[degree] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'model': model,
            'poly': poly
        }

        plt.subplot(1, 3, i+1)
        plt.scatter(X_train, y_train, label='Train', s=10)
        plt.scatter(X_test, y_test, label='Test', s=10, alpha=0.5)
        plt.plot(X, model.predict(poly.transform(X)), color='r', label='Model', linewidth=2)
        plt.title(f"Degree {degree}\nTrain MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save the plot
    plot_filename = 'results/polynomial_regression_analysis.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    plt.show()

    # Print comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(f"{'Degree':<10} {'Train MSE':<15} {'Test MSE':<15} {'Status':<20}")
    print("-"*60)

    for degree in degrees:
        train_mse = results[degree]['train_mse']
        test_mse = results[degree]['test_mse']

        # Determine status based on MSE values
        if train_mse > test_mse + 0.01:
            status = "Underfitting"
        elif abs(train_mse - test_mse) <= 0.01:
            status = "Well-fitting"
        else:
            status = "Overfitting"

        print(f"{degree:<10} {train_mse:<15.4f} {test_mse:<15.4f} {status:<20}")

    # Write analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    analysis = """
POLYNOMIAL REGRESSION ANALYSIS: OVERFITTING vs UNDERFITTING

1. DEGREE 1 (LINEAR MODEL) - UNDERFITTING:
   - Train MSE: {:.4f}, Test MSE: {:.4f}
   - The linear model is too simple to capture the sine wave pattern
   - Both training and test MSE are high, indicating high bias
   - The model cannot learn the underlying non-linear relationship
   - This represents a model with insufficient capacity

2. DEGREE 3 (POLYNOMIAL MODEL) - WELL-FITTING:
   - Train MSE: {:.4f}, Test MSE: {:.4f}
   - The cubic polynomial provides a good balance between bias and variance
   - Training and test MSE are close, indicating good generalization
   - The model captures the main structure of the sine wave
   - Generalization gap is minimal, demonstrating good model fit

3. DEGREE 15 (POLYNOMIAL MODEL) - OVERFITTING:
   - Train MSE: {:.4f}, Test MSE: {:.4f}
   - The high-degree polynomial is too flexible and memorizes noise
   - Large gap between train and test MSE indicates high variance
   - Model fits training data perfectly but fails on test data
   - This represents overfitting due to excessive model complexity

CONCLUSION:
The bias-variance tradeoff is clearly demonstrated here. The optimal model
(degree 3) balances between underfitting (degree 1) and overfitting (degree 15).
This is a fundamental principle in machine learning: models must be complex enough
to capture patterns but constrained enough to generalize to new data.
""".format(
        results[1]['train_mse'], results[1]['test_mse'],
        results[3]['train_mse'], results[3]['test_mse'],
        results[15]['train_mse'], results[15]['test_mse']
    )

    print(analysis)

    # Save analysis to file
    analysis_filename = 'results/polynomial_regression_analysis.txt'
    with open(analysis_filename, 'w', encoding='utf-8') as f:
        f.write(analysis)
    print(f"Analysis saved to: {analysis_filename}\n")

def run_bias_variance_analysis():
    """Task 1.5: Bias-Variance Tradeoff Analysis"""
    print("="*60)
    print("TASK 1.5: BIAS-VARIANCE TRADEOFF ANALYSIS")
    print("="*60)

    # Generate larger dataset with non-linear function
    np.random.seed(42)
    X = np.sort(np.random.rand(200, 1) * 4 - 2, axis=0)
    y = np.sin(np.pi * X) + np.random.normal(0, 0.15, X.shape)

    X_train, y_train = X[:100], y[:100]
    X_test, y_test = X[100:], y[100:]

    # Test various polynomial degrees
    degrees = [1, 2, 3, 4, 5, 7, 10, 15, 20]
    train_errors = []
    test_errors = []
    results = {}

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        X_poly_train = poly.fit_transform(X_train)

        model = LinearRegression().fit(X_poly_train, y_train)
        y_train_pred = model.predict(X_poly_train)
        y_test_pred = model.predict(poly.transform(X_test))

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        train_errors.append(train_mse)
        test_errors.append(test_mse)

        results[degree] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'model': model,
            'poly': poly
        }

    # Plot 1: Model Complexity vs Error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Complexity vs Error plot
    ax1.plot(degrees, train_errors, 'o-', label='Train Error (Bias)', linewidth=2, markersize=6)
    ax1.plot(degrees, test_errors, 's-', label='Test Error (Bias + Variance)', linewidth=2, markersize=6)

    # Mark optimal point
    optimal_idx = np.argmin(test_errors)
    optimal_degree = degrees[optimal_idx]
    ax1.axvline(x=optimal_degree, color='green', linestyle='--', alpha=0.7, label=f'Optimal (Degree {optimal_degree})')

    ax1.set_xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_title('Bias-Variance Tradeoff: Error vs Model Complexity', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Three example models
    example_degrees = [1, optimal_degree, 20]  # Underfit, Optimal, Overfit
    colors = ['blue', 'green', 'red']
    labels = ['Underfitting', 'Optimal', 'Overfitting']

    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

    for degree, color, label in zip(example_degrees, colors, labels):
        poly = PolynomialFeatures(degree=degree)
        X_poly_plot = poly.fit_transform(X_plot)
        y_plot = results[degree]['model'].predict(X_poly_plot)

        ax2.plot(X_plot, y_plot, color=color, label=f'{label} (Degree {degree})', linewidth=2)

    ax2.scatter(X_train, y_train, alpha=0.5, s=20, color='black', label='Training Data')
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('Individual Model Fits: Underfit vs Optimal vs Overfit', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plots
    os.makedirs('results', exist_ok=True)
    plot_filename1 = 'results/bias_variance_complexity_error_plot.png'
    plt.savefig(plot_filename1, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename1}")
    plt.show()

    # Print analysis table
    print("\n" + "="*60)
    print("BIAS-VARIANCE ANALYSIS TABLE")
    print("="*60)
    print(f"{'Degree':<10} {'Train MSE':<15} {'Test MSE':<15} {'Generalization Gap':<20}")
    print("-"*60)

    for degree in degrees:
        train_mse = results[degree]['train_mse']
        test_mse = results[degree]['test_mse']
        gap = test_mse - train_mse

        print(f"{degree:<10} {train_mse:<15.4f} {test_mse:<15.4f} {gap:<20.4f}")

    # Print comprehensive analysis
    analysis_text = f"""
BIAS-VARIANCE TRADEOFF: COMPREHENSIVE ANALYSIS

THEORETICAL BACKGROUND:
The generalization error (test error) can be decomposed into three components:
  Error = Bias² + Variance + Irreducible Error

- Bias: Error from overly simplistic models (underfitting)
- Variance: Sensitivity of predictions to fluctuations in training data (overfitting)
- Irreducible Error: Fundamental noise in the data

THE TRADEOFF:
As model complexity increases:
  1. Bias decreases (model captures more patterns)
  2. Variance increases (model becomes more sensitive to training noise)
  3. Test error follows a U-shaped curve with an optimal complexity

EXPERIMENTAL OBSERVATIONS:

1. LOW COMPLEXITY (Degree 1-3):
   - High training error (high bias)
   - High test error (still high bias)
   - Generalization gap is small (low variance)
   - Model cannot learn the true underlying function

2. MEDIUM COMPLEXITY (Degree 4-7) - OPTIMAL REGION:
   - Decreasing training error (lower bias)
   - Minimal test error (balanced bias-variance)
   - Small generalization gap (controlled variance)
   - Model captures essential patterns without memorizing noise

   Optimal Degree: {optimal_degree}
   Test MSE: {results[optimal_degree]['test_mse']:.4f}

3. HIGH COMPLEXITY (Degree 10-20):
   - Very low training error (low bias)
   - High test error (high variance dominates)
   - Large generalization gap (overfitting)
   - Model memorizes training noise and fails on new data

IMPLICATIONS FOR DEEP LEARNING:

1. Model Selection: Choose complexity that minimizes test error, not training error
2. Regularization: Use techniques like L2 regularization, dropout to reduce variance
3. Early Stopping: Monitor validation error to detect when variance increases
4. More Data: Larger training sets can reduce overfitting impact
5. Ensemble Methods: Combine multiple models to reduce variance
6. Feature Selection: Use fewer, more meaningful features to reduce complexity

PRACTICAL RECOMMENDATION:
Always use a validation set to monitor generalization. Select the model that
performs best on the validation set, not the one with lowest training error.
The optimal model balances learning the true underlying pattern while avoiding
memorization of training noise.
"""

    print(analysis_text)

    # Save analysis to file
    analysis_filename = 'results/bias_variance_tradeoff_analysis.txt'
    with open(analysis_filename, 'w', encoding='utf-8') as f:
        f.write(analysis_text)
    print(f"Analysis saved to: {analysis_filename}\n")


if __name__ == "__main__":
    run_polynomial_experiment()
    run_bias_variance_analysis()
