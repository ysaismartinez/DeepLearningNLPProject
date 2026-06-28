from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from .config import ID_TO_LABEL


def compute_metrics(y_true, y_pred) -> dict:
    """Return common classification metrics for urgent-ticket detection."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_urgent": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_urgent": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_urgent": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion_matrix(y_true, y_pred, output_path: Path, title: str) -> None:
    """Save a confusion matrix visualization."""
    labels = [0, 1]
    display_labels = [ID_TO_LABEL[i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Persist metrics to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def get_classification_report(y_true, y_pred) -> str:
    """Return a readable classification report."""
    return classification_report(y_true, y_pred, target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]], zero_division=0)
