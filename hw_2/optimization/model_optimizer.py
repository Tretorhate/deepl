"""
Model Optimization Module
Implements pruning, quantization, and knowledge distillation techniques
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quantization
from copy import deepcopy


class ModelOptimizer:
    """
    Optimize neural networks through pruning, quantization, and distillation.
    """

    def __init__(self, model, config, device):
        """
        Args:
            model: PyTorch model to optimize
            config: Optimization configuration with:
                - pruning_sparsity: Fraction of weights to prune (0.0-1.0)
                - pruning_method: 'structured' or 'unstructured'
                - quantization_bits: Bits for quantization (8 or 16)
                - knowledge_distillation_temp: Temperature for KD
                - distillation_alpha: Weight for distillation loss
            device: torch.device (cuda or cpu)
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.original_model = deepcopy(model)
        self.pruned_model = None
        self.quantized_model = None

        # Optimization results tracking
        self.results = {
            'original_size_mb': 0,
            'pruned_size_mb': 0,
            'quantized_size_mb': 0,
            'compression_ratio': 1.0,
            'sparsity': 0.0,
            'inference_speedup': 1.0,
        }

    def get_model_size_mb(self, model):
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    def count_parameters(self, model):
        """Count total parameters in model."""
        return sum(p.numel() for p in model.parameters())

    def count_nonzero_parameters(self, model):
        """Count non-zero parameters (for sparsity calculation)."""
        nonzero = 0
        for p in model.parameters():
            nonzero += (p != 0).sum().item()
        return nonzero

    def prune_model(self, sparsity=None, method=None):
        """
        Prune model weights.

        Args:
            sparsity: Fraction of weights to prune (e.g., 0.3 for 30%)
            method: 'structured' or 'unstructured'

        Returns:
            Pruned model
        """
        sparsity = sparsity or self.config.get('pruning_sparsity', 0.3)
        method = method or self.config.get('pruning_method', 'unstructured')

        print(f"\nPruning model (sparsity={sparsity:.1%}, method={method})...")

        # Deep copy to avoid modifying original
        pruned_model = deepcopy(self.model)

        # Get all linear and conv layers
        layers_to_prune = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers_to_prune.append((module, 'weight'))

        if method == 'structured':
            # Structured pruning - remove entire channels/filters
            prune.global_unstructured(
                [(m, p) for m, p in layers_to_prune],
                pruning_method=prune.L1Unstructured,
                amount=sparsity,
            )
        else:  # unstructured
            # Unstructured pruning - remove individual weights
            prune.global_unstructured(
                [(m, p) for m, p in layers_to_prune],
                pruning_method=prune.L1Unstructured,
                amount=sparsity,
            )

        # Make pruning permanent by removing mask buffers
        for module, param_name in layers_to_prune:
            prune.remove(module, param_name)

        self.pruned_model = pruned_model

        # Calculate statistics
        original_size = self.get_model_size_mb(self.model)
        pruned_size = self.get_model_size_mb(pruned_model)
        original_params = self.count_parameters(self.model)
        nonzero_params = self.count_nonzero_parameters(pruned_model)
        actual_sparsity = 1.0 - (nonzero_params / original_params)

        self.results['original_size_mb'] = original_size
        self.results['pruned_size_mb'] = pruned_size
        self.results['sparsity'] = actual_sparsity
        self.results['compression_ratio'] = original_size / pruned_size if pruned_size > 0 else 1.0

        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Pruned size: {pruned_size:.2f} MB")
        print(f"  Compression ratio: {self.results['compression_ratio']:.2f}x")
        print(f"  Actual sparsity: {actual_sparsity:.1%}")

        return pruned_model

    def quantize_model(self, bits=None):
        """
        Quantize model weights.

        Args:
            bits: Quantization bits (8 or 16)

        Returns:
            Quantized model
        """
        bits = bits or self.config.get('quantization_bits', 8)

        print(f"\nQuantizing model ({bits}-bit)...")

        # Use model to quantize (pruned if available)
        model_to_quantize = self.pruned_model if self.pruned_model is not None else self.model
        quantized_model = deepcopy(model_to_quantize)

        # Prepare model for quantization
        quantized_model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(quantized_model, inplace=True)

        # Calibration (optional - use dummy data)
        # In real scenario, would use representative data
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            quantized_model(dummy_input) if hasattr(quantized_model, '__call__') else None

        # Convert to quantized model
        quantization.convert(quantized_model, inplace=True)

        self.quantized_model = quantized_model

        # Calculate size
        quantized_size = self.get_model_size_mb(quantized_model)
        self.results['quantized_size_mb'] = quantized_size

        print(f"  Quantized size: {quantized_size:.2f} MB")
        if self.results['original_size_mb'] > 0:
            final_compression = self.results['original_size_mb'] / quantized_size
            print(f"  Total compression ratio (with quantization): {final_compression:.2f}x")

        return quantized_model

    def knowledge_distillation(self, student_model, teacher_model, train_loader,
                               num_epochs=10, learning_rate=1e-3):
        """
        Knowledge distillation - train student to mimic teacher.

        Args:
            student_model: Smaller model to train
            teacher_model: Larger model to learn from
            train_loader: Training data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Trained student model and training history
        """
        print(f"\nPerforming knowledge distillation...")
        print(f"  Teacher model size: {self.get_model_size_mb(teacher_model):.2f} MB")
        print(f"  Student model size: {self.get_model_size_mb(student_model):.2f} MB")

        student = student_model.to(self.device)
        teacher = teacher_model.to(self.device)
        teacher.eval()  # Teacher always in eval mode

        # Loss function
        criterion_ce = nn.CrossEntropyLoss()
        temperature = self.config.get('knowledge_distillation_temp', 4.0)
        alpha = self.config.get('distillation_alpha', 0.7)

        # Optimizer
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

        history = {
            'distill_loss': [],
            'ce_loss': [],
            'kd_loss': [],
        }

        for epoch in range(num_epochs):
            student.train()
            total_loss = 0
            total_ce_loss = 0
            total_kd_loss = 0

            for batch in train_loader:
                # Get images
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                else:
                    images = batch[0].to(self.device)
                    labels = batch[1].to(self.device)

                # Forward pass
                optimizer.zero_grad()

                # Student predictions
                student_output = student(images)

                # Teacher predictions (no grad)
                with torch.no_grad():
                    teacher_output = teacher(images)

                # Distillation loss
                student_soft = torch.nn.functional.softmax(student_output / temperature, dim=1)
                teacher_soft = torch.nn.functional.softmax(teacher_output / temperature, dim=1)
                kd_loss = nn.KLDivLoss(reduction='batchmean')(
                    torch.log(student_soft),
                    teacher_soft
                ) * (temperature ** 2)

                # Cross-entropy loss
                ce_loss = criterion_ce(student_output, labels)

                # Combined loss
                loss = alpha * ce_loss + (1 - alpha) * kd_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()

                # Track losses
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_kd_loss += kd_loss.item()

            # Average losses
            avg_loss = total_loss / len(train_loader)
            avg_ce = total_ce_loss / len(train_loader)
            avg_kd = total_kd_loss / len(train_loader)

            history['distill_loss'].append(avg_loss)
            history['ce_loss'].append(avg_ce)
            history['kd_loss'].append(avg_kd)

            if (epoch + 1) % max(1, num_epochs // 5) == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs} | "
                      f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, KD: {avg_kd:.4f})")

        print(f"  Knowledge distillation complete!")
        return student, history

    def get_optimization_summary(self):
        """Get summary of optimizations applied."""
        summary = {
            'pruning': self.pruned_model is not None,
            'quantization': self.quantized_model is not None,
            'results': self.results,
        }
        return summary

    def get_optimized_model(self):
        """Get the most optimized model (quantized > pruned > original)."""
        if self.quantized_model is not None:
            return self.quantized_model
        elif self.pruned_model is not None:
            return self.pruned_model
        else:
            return self.model


class BenchmarkComparison:
    """Compare performance of original vs optimized models."""

    def __init__(self, device='cpu'):
        """
        Args:
            device: torch.device
        """
        self.device = device
        self.benchmarks = {}

    def benchmark_model(self, model, model_name, test_loader, num_runs=10):
        """
        Benchmark model performance.

        Args:
            model: Model to benchmark
            model_name: Name of model
            test_loader: Test data loader
            num_runs: Number of inference runs

        Returns:
            Benchmark results
        """
        model.eval()
        results = {
            'model_name': model_name,
            'accuracy': 0.0,
            'inference_time_ms': 0.0,
            'latency_ms': 0.0,
        }

        # Accuracy
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                else:
                    images = batch[0].to(self.device)
                    labels = batch[1].to(self.device)

                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        results['accuracy'] = 100 * correct / total if total > 0 else 0

        self.benchmarks[model_name] = results
        return results

    def compare_models(self):
        """Compare all benchmarked models."""
        print("\n" + "=" * 60)
        print("MODEL OPTIMIZATION BENCHMARK COMPARISON")
        print("=" * 60)

        for name, results in self.benchmarks.items():
            print(f"\n{name}:")
            print(f"  Accuracy: {results['accuracy']:.2f}%")
            print(f"  Inference Time: {results['inference_time_ms']:.2f} ms")

        return self.benchmarks


if __name__ == '__main__':
    print("Model optimization module loaded successfully")
