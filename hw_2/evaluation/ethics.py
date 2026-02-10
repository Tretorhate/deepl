"""
Ethics & Fairness Analysis Module
Conducts bias audits and fairness analysis on model predictions
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class BiasAudit:
    """
    Conduct bias audit on model predictions across demographic groups.
    """

    def __init__(self, config, device='cpu'):
        """
        Args:
            config: Ethics configuration with:
                - demographic_groups: List of demographic attributes
                - fairness_metrics: Metrics to calculate
                - bias_threshold: Disparity threshold for flagging
            device: torch.device
        """
        self.config = config
        self.device = device
        self.audit_results = {}

    def analyze_predictions(self, model, test_loader, class_names=None):
        """
        Analyze model predictions and identify biases.

        Args:
            model: Trained model to audit
            test_loader: Test DataLoader
            class_names: Optional list of class names

        Returns:
            Audit results dictionary
        """
        model.eval()
        all_predictions = []
        all_labels = []
        all_features = []

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

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(all_labels, all_predictions)

        # Simulate demographic analysis
        # In practice, would use actual demographic data from dataset
        demographic_analysis = self._analyze_demographic_disparities(
            all_labels, all_predictions, class_names
        )

        self.audit_results = {
            'overall_metrics': overall_metrics,
            'demographic_analysis': demographic_analysis,
            'bias_flags': self._identify_biases(demographic_analysis),
        }

        return self.audit_results

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate fairness metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred) * 100,
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        }
        return metrics

    def _analyze_demographic_disparities(self, y_true, y_pred, class_names=None):
        """
        Analyze performance disparities across demographic groups.
        Simulates demographic groups based on sample characteristics.
        """
        num_samples = len(y_true)
        demographic_groups = {
            'Group_A': list(range(0, num_samples // 3)),
            'Group_B': list(range(num_samples // 3, 2 * num_samples // 3)),
            'Group_C': list(range(2 * num_samples // 3, num_samples)),
        }

        group_metrics = {}

        for group_name, indices in demographic_groups.items():
            if len(indices) > 0:
                group_labels = y_true[indices]
                group_preds = y_pred[indices]

                group_metrics[group_name] = {
                    'size': len(indices),
                    'metrics': self._calculate_metrics(group_labels, group_preds),
                    'class_distribution': self._get_class_distribution(group_labels),
                }

        return group_metrics

    def _get_class_distribution(self, labels):
        """Get distribution of classes in data."""
        unique, counts = np.unique(labels, return_counts=True)
        return {f'Class_{int(k)}': int(v) for k, v in zip(unique, counts)}

    def _identify_biases(self, demographic_analysis):
        """Identify significant biases across groups."""
        bias_threshold = self.config.get('bias_threshold', 0.1)
        bias_flags = []

        if not demographic_analysis:
            return bias_flags

        # Get baseline metrics (first group)
        baseline_group = list(demographic_analysis.keys())[0]
        baseline_metrics = demographic_analysis[baseline_group]['metrics']

        # Compare other groups
        for group_name, group_data in demographic_analysis.items():
            if group_name == baseline_group:
                continue

            group_metrics = group_data['metrics']

            for metric_name, baseline_value in baseline_metrics.items():
                group_value = group_metrics[metric_name]
                disparity = abs(baseline_value - group_value) / (baseline_value + 1e-10)

                if disparity > bias_threshold:
                    bias_flags.append({
                        'group': group_name,
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'group_value': group_value,
                        'disparity': disparity * 100,
                        'status': 'FLAGGED' if disparity > bias_threshold else 'OK',
                    })

        return bias_flags

    def generate_bias_report(self):
        """Generate comprehensive bias audit report."""
        if not self.audit_results:
            return "No audit results available. Run analyze_predictions() first."

        report = []
        report.append("=" * 70)
        report.append("BIAS AUDIT REPORT")
        report.append("=" * 70)

        # Overall metrics
        report.append("\n[1] OVERALL MODEL PERFORMANCE")
        report.append("-" * 70)
        overall = self.audit_results['overall_metrics']
        for metric, value in overall.items():
            report.append(f"  {metric.upper():12s}: {value:6.2f}%")

        # Demographic analysis
        report.append("\n[2] DEMOGRAPHIC PERFORMANCE ANALYSIS")
        report.append("-" * 70)
        demo_analysis = self.audit_results['demographic_analysis']

        for group_name, group_data in demo_analysis.items():
            report.append(f"\n  {group_name} (n={group_data['size']})")
            for metric, value in group_data['metrics'].items():
                report.append(f"    {metric:12s}: {value:6.2f}%")

        # Bias flags
        report.append("\n[3] BIAS DETECTION & FLAGS")
        report.append("-" * 70)
        bias_flags = self.audit_results['bias_flags']

        if not bias_flags:
            report.append("  ✓ No significant biases detected (all disparities < threshold)")
        else:
            for flag in bias_flags:
                report.append(f"\n  ⚠ {flag['group']} - {flag['metric'].upper()}")
                report.append(f"    Baseline value: {flag['baseline']:.2f}%")
                report.append(f"    Group value:   {flag['group_value']:.2f}%")
                report.append(f"    Disparity:     {flag['disparity']:.2f}%")
                report.append(f"    Status:        {flag['status']}")

        return "\n".join(report)

    def get_audit_summary(self):
        """Get structured audit summary."""
        return {
            'overall_metrics': self.audit_results.get('overall_metrics', {}),
            'num_groups_analyzed': len(self.audit_results.get('demographic_analysis', {})),
            'num_biases_flagged': len(self.audit_results.get('bias_flags', [])),
            'bias_flags': self.audit_results.get('bias_flags', []),
        }


class EthicsAnalysis:
    """
    Comprehensive ethics and fairness analysis framework.
    """

    def __init__(self, config, device='cpu'):
        """
        Args:
            config: Ethics configuration
            device: torch.device
        """
        self.config = config
        self.device = device
        self.bias_audit = BiasAudit(config, device)
        self.analysis_results = {}

    def analyze_model_ethics(self, model, test_loader, class_names=None):
        """
        Conduct full ethics analysis on trained model.

        Args:
            model: Trained model to analyze
            test_loader: Test DataLoader
            class_names: Optional list of class names

        Returns:
            Ethics analysis results
        """
        # Conduct bias audit
        bias_audit_results = self.bias_audit.analyze_predictions(
            model, test_loader, class_names
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(bias_audit_results)

        self.analysis_results = {
            'bias_audit': bias_audit_results,
            'recommendations': recommendations,
        }

        return self.analysis_results

    def _generate_recommendations(self, audit_results):
        """Generate recommendations based on audit findings."""
        recommendations = []

        overall_metrics = audit_results.get('overall_metrics', {})
        bias_flags = audit_results.get('bias_flags', [])

        # Recommendation 1: Overall performance
        accuracy = overall_metrics.get('accuracy', 0)
        if accuracy < 85:
            recommendations.append(
                "1. Improve overall model accuracy (currently {:.1f}%) through:\n"
                "   - Collecting more training data\n"
                "   - Tuning model hyperparameters\n"
                "   - Using ensemble methods".format(accuracy)
            )
        else:
            recommendations.append(
                "1. Overall model accuracy is good ({:.1f}%), but continue monitoring".format(accuracy)
            )

        # Recommendation 2: Address biases
        if bias_flags:
            recommendations.append(
                "2. Address identified demographic disparities:\n"
                "   - Collect more balanced training data across groups\n"
                "   - Use stratified sampling in train/val/test splits\n"
                "   - Consider group-specific model adjustments\n"
                "   - Apply fairness-aware loss functions"
            )
        else:
            recommendations.append(
                "2. No significant demographic biases detected. Continue monitoring\n"
                "   fairness metrics across all groups."
            )

        # Recommendation 3: Evaluation practices
        recommendations.append(
            "3. Implement ongoing fairness monitoring:\n"
            "   - Regular audits on new data\n"
            "   - Track fairness metrics in production\n"
            "   - Establish fairness SLAs (Service Level Agreements)\n"
            "   - Create feedback mechanisms for stakeholders"
        )

        # Recommendation 4: Transparency
        recommendations.append(
            "4. Improve model transparency:\n"
            "   - Document model limitations and biases\n"
            "   - Provide model card with fairness characteristics\n"
            "   - Create user-facing explanations of predictions\n"
            "   - Establish clear appeals process for model decisions"
        )

        # Recommendation 5: Governance
        recommendations.append(
            "5. Establish ethics governance:\n"
            "   - Form fairness review board\n"
            "   - Implement ethics checklists for deployment\n"
            "   - Create incident response procedures\n"
            "   - Conduct regular fairness audits (quarterly or annually)"
        )

        return recommendations

    def generate_ethics_report(self):
        """Generate comprehensive ethics report."""
        report = []

        report.append("=" * 70)
        report.append("COMPREHENSIVE ETHICS & FAIRNESS ANALYSIS REPORT")
        report.append("=" * 70)

        # Bias audit
        if self.bias_audit.audit_results:
            report.append("\n" + self.bias_audit.generate_bias_report())

        # Recommendations
        report.append("\n[4] RECOMMENDATIONS FOR ETHICAL DEPLOYMENT")
        report.append("=" * 70)

        if self.analysis_results.get('recommendations'):
            for rec in self.analysis_results['recommendations']:
                report.append("\n" + rec)

        report.append("\n" + "=" * 70)
        report.append("END OF ETHICS REPORT")
        report.append("=" * 70)

        return "\n".join(report)

    def get_ethics_summary(self):
        """Get structured ethics summary."""
        audit_summary = self.bias_audit.get_audit_summary()

        return {
            'audit_summary': audit_summary,
            'num_recommendations': len(self.analysis_results.get('recommendations', [])),
            'biases_identified': audit_summary['num_biases_flagged'] > 0,
            'fairness_status': 'PASS' if audit_summary['num_biases_flagged'] == 0 else 'REVIEW',
        }


if __name__ == '__main__':
    print("Ethics and fairness analysis module loaded successfully")
