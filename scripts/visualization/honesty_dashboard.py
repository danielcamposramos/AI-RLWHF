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
    """Checks if required plotting libraries are installed.

    Returns:
        True if pandas, matplotlib, and seaborn are available, False otherwise.
    """
    if pd is None or plt is None or sns is None:
        print("pandas/matplotlib/seaborn not available; skipping visualization")
        return False
    return True


def load_aggregation_data(log_file: Path = DEFAULT_LOG):
    """Loads aggregation data from a JSONL log file into a pandas DataFrame.

    Args:
        log_file: The path to the aggregation log file.

    Returns:
        A pandas DataFrame containing the log data, or an empty DataFrame
        if the file is not found. Returns None if pandas is not installed.
    """
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
    """Aggregates individual teacher logs from a directory into a single DataFrame.

    Args:
        log_dir: The directory containing the individual teacher log files.

    Returns:
        A pandas DataFrame containing the combined log data, or None if
        pandas is not installed.
    """
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
    """Ensures that the output directory exists.

    Args:
        output_dir: The path to the output directory.

    Returns:
        The path to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_score_distribution(df, output_dir: Path = DEFAULT_OUTPUT):
    """Plots the distribution of scores.

    This function can create a boxplot for individual teacher scores or a
    histogram for aggregated scores.

    Args:
        df: DataFrame containing the score data.
        output_dir: The directory to save the plot image.
    """
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
    """Plots the teacher disagreement score over time.

    Args:
        df: DataFrame containing disagreement data with timestamps.
        output_dir: The directory to save the plot image.
    """
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
    """Plots a heatmap of the correlation between teacher scores.

    Args:
        df: DataFrame containing individual teacher scores.
        output_dir: The directory to save the plot image.
    """
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
    """Plots the rolling average of scores over time.

    Args:
        df: DataFrame containing score data.
        output_dir: The directory to save the plot image.
        window: The window size for the rolling average.
    """
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
    """Generates a markdown summary report of honesty metrics.

    Args:
        df: DataFrame containing the data to summarize.
        output_dir: The directory to save the markdown report.
    """
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
    """Loads search evaluation data from a JSONL log file.

    Args:
        log_path: The path to the search evaluation log file.

    Returns:
        A pandas DataFrame containing the log data, or an empty DataFrame
        if the file is not found. Returns None if pandas is not installed.
    """
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
    """Plots and summarizes the difference between search and static evaluation.

    Args:
        agg_df: DataFrame with aggregated data (currently unused but kept for API consistency).
        search_df: DataFrame with search evaluation data.
        output_dir: The directory to save the plot and markdown table.
    """
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
    """Generates all dashboard assets.

    This function orchestrates the loading of data and the creation of all
    plots and summary reports.

    Args:
        output_dir: The directory to save all dashboard assets.
    """
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
