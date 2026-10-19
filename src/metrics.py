"""Shared evaluation metrics for task preparation and training."""


def compute_accuracy(preds, labels) -> float:
    if len(preds) != len(labels):
        raise ValueError("预测数量和标签数量必须一致")
    if len(labels) == 0:
        return 0.0
    return sum(p == label for p, label in zip(preds, labels)) / len(labels)
