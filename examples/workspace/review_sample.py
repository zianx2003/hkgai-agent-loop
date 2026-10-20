"""Intentionally flawed source used only as static code-review input."""


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def add_label(label: str, labels: list[str] = []) -> list[str]:
    labels.append(label)
    return labels
