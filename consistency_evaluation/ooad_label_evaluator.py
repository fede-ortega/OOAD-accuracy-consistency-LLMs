import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Iterable

import numpy as np
import matplotlib.pyplot as plt


SUPPORTED_LABELS = ["DIP", "ISP", "LSP", "OCP", "SRP"]


@dataclass
class LabeledExample:
    index: int
    gold_label: str
    predicted_label: str


@dataclass
class PerLabelMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int  # number of gold instances for this label


@dataclass
class EvaluationResult:
    accuracy: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_label: List[PerLabelMetrics]
    confusion_matrix: Dict[str, Dict[str, int]]  # confusion[gold][pred]


class LabelSelectionStrategy(ABC):
    """Strategy for choosing one label from the list in the LLM output."""

    @abstractmethod
    def select_label(self, labels: Iterable[str]) -> str:
        pass


class MajorityLabelSelection(LabelSelectionStrategy):
    """Selects the label that appears most often (majority vote)."""

    def select_label(self, labels: Iterable[str]) -> str:
        counts: Dict[str, int] = {}
        order: List[str] = []

        for label in labels:
            if label not in counts:
                counts[label] = 0
                order.append(label)
            counts[label] += 1

        if not counts:
            raise ValueError("Cannot select a label from an empty sequence")

        max_count = max(counts.values())
        for label in order:
            if counts[label] == max_count:
                return label

        raise RuntimeError("Majority selection failed unexpectedly")


class GoldLoader:
    """Loads the gold-standard labels from file."""

    def load(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Gold file must contain a JSON list.")

        gold_labels: List[str] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict) or "labels" not in item:
                raise ValueError(f"Gold item at index {i} is malformed: {item}")
            label = item["labels"]
            if not isinstance(label, str):
                raise ValueError(f"Gold 'labels' must be a string at index {i}: {item}")
            gold_labels.append(label)

        return gold_labels


class PredictionLoader:
    """Loads predicted labels from LLM output file."""

    def __init__(self, selection_strategy: LabelSelectionStrategy):
        self._selection_strategy = selection_strategy

    def load(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Prediction file must contain a JSON list.")

        preds: List[str] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Prediction item at index {i} is malformed: {item}")
            if "labels" not in item:
                raise ValueError(f"Prediction item at index {i} missing 'labels': {item}")

            labels_list = item["labels"]
            if not isinstance(labels_list, list) or not labels_list:
                raise ValueError(
                    f"'labels' must be a non-empty list at index {i}: {item}"
                )

            selected = self._selection_strategy.select_label(labels_list)
            preds.append(selected)

        return preds


class ClassificationEvaluator:
    """Computes metrics and confusion matrix for multi-class labels."""

    def __init__(self, labels: List[str]):
        self.labels = labels

    def evaluate(self, examples: List[LabeledExample]) -> EvaluationResult:
        if not examples:
            raise ValueError("No examples provided for evaluation.")

        label_set = set(self.labels)

        # Validate labels
        for ex in examples:
            if ex.gold_label not in label_set:
                raise ValueError(f"Unknown gold label '{ex.gold_label}' in example {ex}")
            if ex.predicted_label not in label_set:
                raise ValueError(f"Unknown predicted label '{ex.predicted_label}' in example {ex}")

        # confusion[gold][pred]
        confusion: Dict[str, Dict[str, int]] = {
            g: {p: 0 for p in self.labels} for g in self.labels
        }

        total = len(examples)
        correct = 0

        for ex in examples:
            confusion[ex.gold_label][ex.predicted_label] += 1
            if ex.gold_label == ex.predicted_label:
                correct += 1

        accuracy = correct / total

        per_label_metrics: List[PerLabelMetrics] = []
        micro_tp = micro_fp = micro_fn = 0

        for label in self.labels:
            tp = confusion[label][label]
            fp = sum(confusion[g][label] for g in self.labels if g != label)
            fn = sum(confusion[label][p] for p in self.labels if p != label)
            support = sum(confusion[label][p] for p in self.labels)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_label_metrics.append(
                PerLabelMetrics(
                    label=label,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    support=support,
                )
            )

            micro_tp += tp
            micro_fp += fp
            micro_fn += fn

        micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
        micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        macro_precision = sum(m.precision for m in per_label_metrics) / len(self.labels)
        macro_recall = sum(m.recall for m in per_label_metrics) / len(self.labels)
        macro_f1 = sum(m.f1 for m in per_label_metrics) / len(self.labels)

        return EvaluationResult(
            accuracy=accuracy,
            micro_precision=micro_precision,
            micro_recall=micro_recall,
            micro_f1=micro_f1,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            per_label=per_label_metrics,
            confusion_matrix=confusion,
        )


class ConfusionMatrixPlotter:
    """
    Responsible only for visualizing a confusion matrix as a heatmap and
    saving it as a PNG file.
    """

    def __init__(self, labels: List[str]):
        self.labels = labels

    def plot(self, confusion: Dict[str, Dict[str, int]], output_path: str,
             title: str | None = None) -> None:
        """
        confusion: confusion[gold][pred]
        Figure axes:
            x-axis: Human Annotations (gold labels)
            y-axis: LLM's Annotations (predicted labels)
        """
        n = len(self.labels)
        matrix = np.zeros((n, n), dtype=int)

        # row = predicted, col = gold  (to match "LLM vs Human" visual)
        for gi, g in enumerate(self.labels):
            for pi, p in enumerate(self.labels):
                matrix[pi, gi] = confusion[g][p]

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(matrix, interpolation="nearest")  # default colormap

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("count")

        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(self.labels)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(self.labels)

        ax.set_xlabel("Human Annotations (Gold)")
        ax.set_ylabel("LLM's Annotations")

        # annotate cells with counts
        max_val = matrix.max() if matrix.size > 0 else 0
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if val == 0:
                    continue
                text_color = "white" if max_val > 0 and val > max_val / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=text_color)

        if title:
            ax.set_title(title)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def build_examples(
    gold_labels: List[str],
    predicted_labels: List[str],
) -> List[LabeledExample]:
    if len(gold_labels) != len(predicted_labels):
        raise ValueError(
            f"Gold and prediction files have different lengths: "
            f"{len(gold_labels)} vs {len(predicted_labels)}"
        )

    return [
        LabeledExample(index=i, gold_label=g, predicted_label=p)
        for i, (g, p) in enumerate(zip(gold_labels, predicted_labels))
    ]


def print_metrics(result: EvaluationResult) -> None:
    print("=== Overall Metrics ===")
    print(f"Accuracy         : {result.accuracy:.4f}")
    print(f"Micro Precision  : {result.micro_precision:.4f}")
    print(f"Micro Recall     : {result.micro_recall:.4f}")
    print(f"Micro F1         : {result.micro_f1:.4f}")
    print(f"Macro Precision  : {result.macro_precision:.4f}")
    print(f"Macro Recall     : {result.macro_recall:.4f}")
    print(f"Macro F1         : {result.macro_f1:.4f}")
    print()

    print("=== Per-Label Metrics ===")
    header = f"{'Label':<5} {'Support':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for m in result.per_label:
        print(
            f"{m.label:<5} {m.support:>8d} "
            f"{m.precision:>8.4f} {m.recall:>8.4f} {m.f1:>8.4f}"
        )
    print()


def print_confusion_matrix(confusion: Dict[str, Dict[str, int]], labels: List[str]) -> None:
    print("=== Confusion Matrix (gold rows x predicted columns) ===")
    header = "Gold \\ Pred".ljust(12) + " ".join(l.rjust(6) for l in labels)
    print(header)
    print("-" * len(header))
    for g in labels:
        row = g.ljust(12)
        for p in labels:
            row += f"{confusion[g][p]:>6d}"
        print(row)
    print()


# CLI

def main(argv: List[str]) -> None:
    if len(argv) not in (3, 4):
        print(
            "Usage:\n"
            "  python ooad_label_evaluator.py <gold_file.txt> <llm_file.txt> [confusion.png]"
        )
        sys.exit(1)

    gold_path = argv[1]
    pred_path = argv[2]
    png_path = argv[3] if len(argv) == 4 else None

    gold_loader = GoldLoader()
    selection_strategy = MajorityLabelSelection()
    pred_loader = PredictionLoader(selection_strategy)

    gold_labels = gold_loader.load(gold_path)
    pred_labels = pred_loader.load(pred_path)

    examples = build_examples(gold_labels, pred_labels)

    evaluator = ClassificationEvaluator(SUPPORTED_LABELS)
    result = evaluator.evaluate(examples)

    print_metrics(result)
    print_confusion_matrix(result.confusion_matrix, SUPPORTED_LABELS)

    if png_path is not None:
        plotter = ConfusionMatrixPlotter(SUPPORTED_LABELS)
        title = f"Agreement Matrix\nAccuracy={result.accuracy:.2f}  Macro-F1={result.macro_f1:.2f}"
        plotter.plot(result.confusion_matrix, png_path, title=title)
        print(f"\nConfusion matrix figure saved to: {png_path}")


if __name__ == "__main__":
    main(sys.argv)