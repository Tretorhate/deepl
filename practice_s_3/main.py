import sys
import os

try:
    from part1_poly_analysis import run_polynomial_experiment, run_bias_variance_analysis
    from part1_nn_regularization import train_and_validate, run_dropout_comparison, run_l2_regularization_comparison, run_early_stopping_comparison
    from part2_rnn_fundamentals import test_manual_rnn, train_manual_rnn, analyze_vanishing_gradients, analyze_vanishing_gradients_multi_length, calculate_spectral_radius
    from part2_comparison import run_model_comparison, run_ablation_study, create_comprehensive_comparison
    from part2_application import run_time_series_prediction
except ImportError as e:
    print(f"Error: Could not find one of the required files. {e}")
    sys.exit(1)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    while True:
        clear_screen()
        print("="*60)
        print("  DEEP LEARNING PRACTICE SESSION: WEEKS 5-6")
        print("="*60)
        print("PART 1: REGULARIZATION & GENERALIZATION")
        print("  1. Task 1.1: Polynomial Regression (Overfitting Analysis)")
        print("  2. Task 1.2: Dropout Regularization Test")
        print("  3. Task 1.3: L2 Regularization Comparison")
        print("  4. Task 1.4: Early Stopping Demonstration")
        print("  5. Task 1.5: Bias-Variance Tradeoff Analysis")
        print("\nPART 2: SEQUENCE MODELING & RNNs")
        print("  6. Task 2.1: RNN Fundamentals (Manual BPTT Training)")
        print("  7. Task 2.2: Vanishing Gradients Analysis")
        print("  8. Task 2.2b: Spectral Radius Analysis")
        print("  9. Task 2.3: RNN vs LSTM vs GRU Comparison")
        print("  10. Task 2.3b: Ablation Study (Sequence Lengths)")
        print("  11. Task 2.3c: Comprehensive Model Comparison")
        print("  12. Task 2.4: Time-Series Prediction Application")
        print("\n  0. Exit")
        print("="*60)

        choice = input("\nSelect a task to run (0-12): ")

        if choice == '1':
            print("\n--- Running Task 1.1: Polynomial Regression Analysis ---")
            run_polynomial_experiment()
            input("\nPress Enter to return to menu...")

        elif choice == '2':
            print("\n--- Running Task 1.2: Dropout Comparison (0.0 vs 0.5) ---")
            run_dropout_comparison()
            input("\nPress Enter to return to menu...")

        elif choice == '3':
            print("\n--- Running Task 1.3: L2 Regularization Comparison ---")
            run_l2_regularization_comparison()
            input("\nPress Enter to return to menu...")

        elif choice == '4':
            print("\n--- Running Task 1.4: Early Stopping Demonstration ---")
            run_early_stopping_comparison()
            input("\nPress Enter to return to menu...")

        elif choice == '5':
            print("\n--- Running Task 1.5: Bias-Variance Tradeoff Analysis ---")
            run_bias_variance_analysis()
            input("\nPress Enter to return to menu...")

        elif choice == '6':
            print("\n--- Running Task 2.1: RNN Fundamentals (Manual BPTT) ---")
            train_manual_rnn(num_iterations=1000, seq_len=10)
            input("\nPress Enter to return to menu...")

        elif choice == '7':
            print("\n--- Running Task 2.2: Vanishing Gradients (Multi-Length) ---")
            analyze_vanishing_gradients_multi_length()
            input("\nPress Enter to return to menu...")

        elif choice == '8':
            print("\n--- Running Task 2.2b: Spectral Radius Analysis ---")
            calculate_spectral_radius()
            input("\nPress Enter to return to menu...")

        elif choice == '9':
            print("\n--- Running Task 2.3: Model Comparison ---")
            run_model_comparison()
            input("\nPress Enter to return to menu...")

        elif choice == '10':
            print("\n--- Running Task 2.3b: Ablation Study ---")
            run_ablation_study()
            input("\nPress Enter to return to menu...")

        elif choice == '11':
            print("\n--- Running Task 2.3c: Comprehensive Comparison ---")
            create_comprehensive_comparison()
            input("\nPress Enter to return to menu...")

        elif choice == '12':
            print("\n--- Running Task 2.4: Time-Series Prediction ---")
            run_time_series_prediction()
            input("\nPress Enter to return to menu...")

        elif choice == '0':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter...")

if __name__ == "__main__":
    main_menu()
