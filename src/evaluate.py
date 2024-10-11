import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, log_loss,
    confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, ConfusionMatrixDisplay
)

class ModelEvaluator:
    def __init__(self, model=None):
        self.model = model
        self.metrics = {}

    def set_model(self, model):
        """
        Set the model to evaluate
        """
        self.model = model

    def compute_metrics(self, X_test, y_test):
        """
        Compute all evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model set. Use set_model first.")

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'log_loss': log_loss(y_test, y_prob)
        }

        return self.metrics

    def plot_confusion_matrix(self, X_test, y_test):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(
            self.model, X_test, y_test, cmap='Blues'
        )
        plt.title("Confusion Matrix")
        plt.show()

    def plot_roc_curve(self, X_test, y_test):
        """
        Plot ROC curve
        """
        y_prob = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self, X_test, y_test):
        """
        Plot precision-recall curve
        """
        y_prob = self.model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        average_precision = average_precision_score(y_test, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='b', lw=2,
                label=f'AP = {average_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()

    def plot_train_test_accuracy(self, X_train, y_train, X_test, y_test):
        """
        Plot train vs test accuracy comparison
        """
        train_accuracy = accuracy_score(
            y_train, self.model.predict(X_train)
        )
        test_accuracy = accuracy_score(
            y_test, self.model.predict(X_test)
        )

        plt.figure(figsize=(6, 4))
        plt.bar(['Train Accuracy', 'Test Accuracy'],
                [train_accuracy * 100, test_accuracy * 100],
                color=['green', 'blue'])
        plt.title('Train vs Test Accuracy')
        plt.ylabel('Accuracy (%)')
        plt.show()

    def generate_evaluation_report(self, X_train, y_train, X_test, y_test):
        """
        Generate complete evaluation report with all metrics and plots
        """
        # Compute metrics
        metrics = self.compute_metrics(X_test, y_test)
        
        # Print metrics
        print("=== Model Evaluation Report ===")
        print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall: {metrics['recall']:.2f}")
        print(f"F1 Score: {metrics['f1']:.2f}")
        print(f"AUC-ROC: {metrics['roc_auc']:.2f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.2f}")
        print(f"Log Loss: {metrics['log_loss']:.2f}")
        
        # Generate all plots
        self.plot_confusion_matrix(X_test, y_test)
        self.plot_roc_curve(X_test, y_test)
        self.plot_precision_recall_curve(X_test, y_test)
        self.plot_train_test_accuracy(X_train, y_train, X_test, y_test)