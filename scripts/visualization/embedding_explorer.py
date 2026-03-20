"""Embedding-space visualization helpers for contrastive honesty learning."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def _to_matrix(embeddings: Sequence[Any]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for item in embeddings:
        if hasattr(item, "tolist"):
            item = item.tolist()
        matrix.append([float(value) for value in item])
    return matrix


def reduce_embeddings(embeddings: Sequence[Any], method: str = "pca") -> list[list[float]]:
    """Reduce embeddings to 2D using UMAP, t-SNE, or a PCA fallback."""
    matrix = _to_matrix(embeddings)
    if not matrix:
        return []
    method = str(method or "pca").lower()
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - numpy should usually exist
        raise ImportError("numpy is required for embedding reduction") from exc

    data = np.asarray(matrix, dtype=float)
    if method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(data).tolist()
        except Exception:
            method = "pca"
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE  # type: ignore

            reducer = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)
            return reducer.fit_transform(data).tolist()
        except Exception:
            method = "pca"
    centered = data - data.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    reduced = centered @ components.T
    return reduced.tolist()


def plot_embeddings(
    embeddings: Sequence[Any],
    labels: Sequence[Any],
    output_path: str | Path,
    *,
    method: str = "pca",
    title: str = "Contrastive Honesty Embeddings",
) -> Path:
    """Save a 2D embedding plot to disk."""
    points = reduce_embeddings(embeddings, method=method)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        raise ImportError("matplotlib is required to plot embeddings") from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(xs, ys, c=list(range(len(labels))), cmap="viridis", alpha=0.8)
    for point, label in zip(points, labels):
        plt.annotate(str(label), (point[0], point[1]), fontsize=8, alpha=0.8)
    plt.title(title)
    plt.xlabel("dim_1")
    plt.ylabel("dim_2")
    plt.colorbar(scatter, label="sample index")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return output


__all__ = ["plot_embeddings", "reduce_embeddings"]
