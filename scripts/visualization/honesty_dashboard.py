"""Visualization helpers for honesty and aggregation telemetry."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

try:  # Optional heavy deps for environments that have them installed
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
except ImportError:  # pragma: no cover - gracefully degrade when libs missing
    pd = None  # type: ignore
    plt = None  # type: ignore
    sns = None  # type: ignore

DEFAULT_LOG = Path("data/processed/honesty_logs/multi_teacher_aggregation.jsonl")
DEFAULT_LOG_DIR = DEFAULT_LOG.parent
DEFAULT_OUTPUT = Path("experiments/visualizations")
SEARCH_LOG = Path("data/processed/honesty_logs/grok_search_evaluator.jsonl")


def _require_plotting() -> bool:
    if pd is None or plt is None or sns is None:
        print("pandas/matplotlib/seaborn not available; skipping visualization")
        return False
    return True


def load_aggregation_data(log_file: Path = DEFAULT_LOG):
    """Load aggregation JSONL into a DataFrame (empty when unavailable)."""
    if pd is None:
        return None
    records = []
    try:
        with Path(log_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_individual_logs(log_dir: Path = DEFAULT_LOG_DIR):
    """Aggregate per-teacher JSONL logs into a DataFrame."""
    if pd is None:
        return None
    entries = []
    for candidate in Path(log_dir).glob("*.jsonl"):
        try:
            teacher = candidate.stem
            with candidate.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        row["teacher"] = teacher
                        entries.append(row)
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"Failed to read {candidate}: {exc}")
    return pd.DataFrame(entries)


def _ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_score_distribution(df, output_dir: Path = DEFAULT_OUTPUT):
    if not _require_plotting() or df is None or df.empty:
        return
    _ensure_output_dir(output_dir)
    plt.figure(figsize=(12, 6))
    if "teacher" in df.columns:
        sns.boxplot(x="teacher", y="score", data=df)
        plt.title("Score Distribution by Teacher Model")
        plt.xticks(rotation=45, ha="right")
    elif "aggregated_score" in df.columns:
        sns.histplot(df["aggregated_score"], bins=20, kde=True)
        plt.title("Distribution of Aggregated Scores")
    plt.xlabel("Score (-2 to +2)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png")
    plt.close()


def plot_disagreement_over_time(df, output_dir: Path = DEFAULT_OUTPUT):
    if not _require_plotting() or df is None or df.empty or "disagreement" not in df.columns:
        return
    _ensure_output_dir(output_dir)
    plt.figure(figsize=(12, 6))
    if "timestamp" not in df.columns:
        df = df.copy()
        df["timestamp"] = range(len(df))
    plt.plot(df["timestamp"], df["disagreement"], marker="o")
    threshold = df.get("disagreement_threshold", pd.Series([1.5])).iloc[0]
    plt.axhline(y=threshold, color="r", linestyle="--", label="Disagreement Threshold")
    plt.xlabel("Iteration")
    plt.ylabel("Disagreement")
    plt.title("Teacher Disagreement Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "disagreement_over_time.png")
    plt.close()


def plot_teacher_agreement(df, output_dir: Path = DEFAULT_OUTPUT):
    if not _require_plotting() or df is None or df.empty or "teacher" not in df.columns:
        return
    _ensure_output_dir(output_dir)
    pivot_df = df.pivot_table(index="prompt", columns="teacher", values="score", aggfunc="mean")
    corr_matrix = pivot_df.corr().fillna(0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Teacher Agreement Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "teacher_agreement_heatmap.png")
    plt.close()


def plot_score_trends(df, output_dir: Path = DEFAULT_OUTPUT, window: int = 10):
    if not _require_plotting() or df is None or df.empty:
        return
    _ensure_output_dir(output_dir)
    plt.figure(figsize=(12, 6))
    frame = df.copy()
    if "timestamp" not in frame.columns:
        frame["timestamp"] = range(len(frame))
    if "teacher" in frame.columns:
        for teacher in frame["teacher"].unique():
            t_df = frame[frame["teacher"] == teacher]
            rolling = t_df["score"].rolling(window=window, min_periods=1).mean()
            plt.plot(t_df["timestamp"], rolling, label=teacher)
        plt.legend()
        plt.title("Rolling Average Scores by Teacher")
    elif "aggregated_score" in frame.columns:
        rolling = frame["aggregated_score"].rolling(window=window, min_periods=1).mean()
        plt.plot(frame["timestamp"], rolling, label="Aggregated")
        plt.legend()
        plt.title("Rolling Average of Aggregated Scores")
    plt.xlabel("Iteration")
    plt.ylabel("Score (-2 to +2)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "score_trends.png")
    plt.close()


def generate_summary_report(df, output_dir: Path = DEFAULT_OUTPUT):
    if df is None or df.empty:
        return
    _ensure_output_dir(output_dir)
    lines = ["# Honesty Metrics Summary Report", f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    if "aggregated_score" in df.columns:
        avg_score = df["aggregated_score"].mean()
        avg_disagreement = df["disagreement"].mean()
        high_disagreement_pct = df["high_disagreement"].mean() * 100
        lines.extend([
            "## Aggregated Metrics",
            f"- Average Score: {avg_score:.2f}",
            f"- Average Disagreement: {avg_disagreement:.2f}",
            f"- High Disagreement Instances: {high_disagreement_pct:.1f}%",
            "",
        ])
    if "teacher" in df.columns and "score" in df.columns:
        lines.append("## Individual Teacher Performance")
        for teacher, subset in df.groupby("teacher"):
            lines.append(f"- {teacher}: Average Score {subset['score'].mean():.2f}")
        lines.append("")
    summary_path = output_dir / "honesty_summary_report.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def load_search_delta(log_path: Path = SEARCH_LOG):
    if pd is None:
        return None
    if not log_path.exists():
        return pd.DataFrame()
    rows = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def plot_search_delta(agg_df, search_df, output_dir: Path = DEFAULT_OUTPUT):
    if not _require_plotting() or search_df is None or search_df.empty:
        return
    _ensure_output_dir(output_dir)
    search_df = search_df.copy()
    search_df["mode"] = search_df["search_source"].apply(lambda src: "search" if src and "api" in src else "static")
    grouped = search_df.groupby("mode")["reward"].mean().reset_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(data=grouped, x="mode", y="reward", palette="viridis")
    plt.title("Average Reward: Search vs Static")
    plt.ylabel("Average Reward")
    plt.xlabel("Mode")
    plt.tight_layout()
    plt.savefig(output_dir / "search_delta.png")
    plt.close()
    table_path = output_dir / "search_delta.md"
    lines = ["# Search vs Static Delta", "", "| Mode | Avg Reward |", "| --- | --- |"]
    for _, row in grouped.iterrows():
        lines.append(f"| {row['mode']} | {row['reward']:.2f} |")
    table_path.write_text("\n".join(lines), encoding="utf-8")


def create_dashboard(output_dir: Optional[Path] = None):
    """Generate dashboard assets if dependencies/data are available."""
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    agg_df = load_aggregation_data()
    indiv_df = load_individual_logs()
    search_df = load_search_delta()
    plot_score_distribution(agg_df, out_dir)
    plot_disagreement_over_time(agg_df, out_dir)
    plot_teacher_agreement(indiv_df, out_dir)
    plot_score_trends(agg_df, out_dir)
    plot_search_delta(agg_df, search_df, out_dir)
    generate_summary_report(agg_df, out_dir)
    print(f"Dashboard artifacts (if any) saved under {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    create_dashboard()
