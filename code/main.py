#!/usr/bin/env python3
"""Reproduce the downstream emotional-bias analysis from archived model outputs."""

from __future__ import annotations

import argparse
import ast
import json
import math
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.stem import SnowballStemmer
from scipy.stats import ttest_1samp


MODELS = ("deepseek", "gemini", "gpt", "llama", "qwen")
DIMENSIONS = ("valence", "arousal", "dominance")
COLORS = {
    "deepseek": "#3498db",
    "gemini": "#9b59b6",
    "gpt": "#e74c3c",
    "llama": "#2ecc71",
    "qwen": "#f39c12",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map predictions to NRC VAD, calculate EBS statistics, and create figures."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument(
        "--raw-results-dir",
        type=Path,
        default=None,
        help="Directory containing the five archived {model}-results.csv files",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("/results"))
    parser.add_argument("--alpha", type=float, default=0.001)
    return parser.parse_args()


def required_inputs(data_dir: Path, raw_results_dir: Path | None) -> dict[str, Path]:
    raw_results_dir = raw_results_dir or data_dir / "raw-results"
    paths = {
        "vad": data_dir / "NRC-VAD-Lexicon.txt",
        "topics": data_dir / "express-single-mask.csv",
        "emotion_lexicon": data_dir / "emotion-lexicons.pkl",
    }
    paths.update(
        {model: raw_results_dir / f"{model}-results.csv" for model in MODELS}
    )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required capsule input(s):\n- " + "\n- ".join(missing))
    return paths


def load_vad(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["word", "valence", "arousal", "dominance"],
    )
    frame["word"] = frame["word"].astype(str).str.lower().str.strip()
    numeric_columns = ["valence", "arousal", "dominance"]
    corrections = 0
    for column in numeric_columns:
        original = frame[column].astype(str)
        parsed = pd.to_numeric(original, errors="coerce")
        recovered = pd.to_numeric(
            original.str.extract(r"([-+]?(?:\d+\.?\d*|\.\d+))", expand=False),
            errors="coerce",
        )
        corrections += int(parsed.isna().sum() - recovered.isna().sum())
        frame[column] = parsed.fillna(recovered)
    if corrections:
        print(f"Warning: normalized {corrections} malformed NRC-VAD numeric value(s)", flush=True)
    malformed = frame[numeric_columns].isna().any(axis=1)
    if malformed.any():
        bad_words = ", ".join(frame.loc[malformed, "word"].head(10))
        print(
            f"Warning: dropping {int(malformed.sum())} malformed NRC-VAD row(s): {bad_words}",
            flush=True,
        )
        frame = frame.loc[~malformed].copy()
    return {
        word: np.asarray(values, dtype=float)
        for word, values in zip(
            frame["word"], frame[["valence", "arousal", "dominance"]].to_numpy()
        )
    }


def normalize_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
            text = str(parsed[0])
    except (ValueError, SyntaxError):
        pass
    text = text.lower().strip().strip("[]'\"").strip()
    return text or None


def normalize_output(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).lower().strip().strip("'\"").strip()
    return text or None


def vector_text(vector: np.ndarray) -> str:
    return json.dumps([round(float(value), 3) for value in vector])


def process_model(
    model: str,
    raw_path: Path,
    topics: pd.DataFrame,
    vad: dict[str, np.ndarray],
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(raw_path, low_memory=False)
    input_rows = len(frame)
    for column in ("topic_id", "topic_name"):
        if column in frame.columns:
            frame = frame.drop(columns=column)
    if "index" not in frame.columns:
        raise KeyError(f"{raw_path} does not contain the required 'index' column")
    frame = frame.merge(topics, on="index", how="left", validate="many_to_one")

    frame["labels"] = frame["labels"].map(normalize_label)
    frame["output"] = frame["output"].map(normalize_output)
    frame["probability"] = pd.to_numeric(frame.get("probability"), errors="coerce")
    frame = frame.dropna(subset=["labels", "output"])
    complete_rows = len(frame)
    exact_all_outputs = float((frame["labels"] == frame["output"]).mean()) if len(frame) else math.nan
    valid = frame["labels"].isin(vad) & frame["output"].isin(vad)
    frame = frame.loc[valid].copy().reset_index(drop=True)

    label_vectors = np.vstack(frame["labels"].map(vad).to_numpy())
    output_vectors = np.vstack(frame["output"].map(vad).to_numpy())
    differences = output_vectors - label_vectors
    weighted = differences * frame["probability"].to_numpy(dtype=float)[:, None]

    frame["vad_label"] = [vector_text(row) for row in label_vectors]
    frame["vad_output"] = [vector_text(row) for row in output_vectors]
    frame["vad_diff"] = [vector_text(row) for row in differences]
    frame["vad_weighted_diff"] = [vector_text(row) for row in weighted]
    frame["model"] = model
    frame.to_csv(output_path, index=False)

    exact = float((frame["labels"] == frame["output"]).mean()) if len(frame) else math.nan
    vector_match = np.all(label_vectors == output_vectors, axis=1) if len(frame) else np.array([])
    summary = {
        "model": model,
        "input_rows": input_rows,
        "complete_prediction_rows": complete_rows,
        "vad_covered_rows": len(frame),
        "vad_coverage_percent": round(100 * len(frame) / input_rows, 3),
        "exact_match_accuracy_all_outputs": exact_all_outputs,
        "exact_match_accuracy_vad_covered": exact,
        "vad_vector_accuracy": float(vector_match.mean()) if len(vector_match) else math.nan,
    }
    return frame, summary


def ttest_stats(values: np.ndarray, alternative: str, alpha: float) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {key: math.nan for key in ("mean", "std", "t", "p", "cohen_d", "ci_low", "ci_high")} | {"n": 0, "significant": False}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else math.nan
    if n > 1:
        t_stat, p_two = ttest_1samp(values, 0)
        if alternative == "greater":
            p_value = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        else:
            p_value = p_two
        cohen_d = mean / std if std and np.isfinite(std) else math.nan
        half_width = 1.96 * std / math.sqrt(n)
        ci_low, ci_high = mean - half_width, mean + half_width
    else:
        t_stat = p_value = cohen_d = math.nan
        ci_low = ci_high = mean
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "t": float(t_stat),
        "p": float(p_value),
        "cohen_d": float(cohen_d),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant": bool(np.isfinite(p_value) and p_value < alpha),
    }


def expand_weighted(frame: pd.DataFrame) -> pd.DataFrame:
    vectors = np.asarray(frame["vad_weighted_diff"].map(json.loads).tolist(), dtype=float)
    expanded = frame[["model", "topic_name"]].copy()
    for index, dimension in enumerate(DIMENSIONS):
        expanded[dimension] = vectors[:, index]
    return expanded


def overall_tests(expanded: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [("all_models", expanded), *list(expanded.groupby("model", sort=True))]
    for model, group in groups:
        for dimension in DIMENSIONS:
            signed = group[dimension].to_numpy(dtype=float)
            tests = {
                "signed_bias": (signed, "two-sided"),
                "positive_bias": (np.maximum(signed, 0), "greater"),
                "negative_bias": (np.maximum(-signed, 0), "greater"),
                "absolute_bias": (np.abs(signed), "greater"),
            }
            for bias_type, (values, alternative) in tests.items():
                rows.append(
                    {
                        "model": model,
                        "dimension": dimension,
                        "bias_type": bias_type,
                        "alternative": alternative,
                        **ttest_stats(values, alternative, alpha),
                    }
                )
    return pd.DataFrame(rows)


def topic_tests(expanded: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    usable = expanded.dropna(subset=["topic_name"])
    for topic, group in usable.groupby("topic_name", sort=True):
        row: dict[str, object] = {"topic_name": topic}
        for dimension in DIMENSIONS:
            stats = ttest_stats(group[dimension].to_numpy(), "two-sided", alpha)
            row.update({f"{dimension}_{key}": value for key, value in stats.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def emotion_term_coverage(
    full: pd.DataFrame, lexicon_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure exact and stem-based coverage of predicted emotion terms."""
    with lexicon_path.open("rb") as handle:
        raw_lexicon = pickle.load(handle)
    if not isinstance(raw_lexicon, (set, list, tuple)):
        raise TypeError("emotion-lexicons.pkl must contain a set or sequence of emotion terms")
    vocabulary = {str(term).lower().strip() for term in raw_lexicon}
    stemmer = SnowballStemmer("english")

    def tokens(text: object) -> list[str]:
        return re.findall(r"[a-z]+", str(text).lower())

    vocabulary_stems = {stemmer.stem(token) for term in vocabulary for token in tokens(term)}

    checked = full[["model", "labels", "output"]].copy()
    checked["exact_lexicon_match"] = checked["output"].isin(vocabulary)
    checked["stem_lexicon_match"] = checked["output"].map(
        lambda value: any(stemmer.stem(token) in vocabulary_stems for token in tokens(value))
    )
    checked["covered"] = checked["exact_lexicon_match"] | checked["stem_lexicon_match"]

    rows = []
    for model, group in checked.groupby("model", sort=True):
        rows.append(
            {
                "model": model,
                "rows": len(group),
                "exact_matches": int(group["exact_lexicon_match"].sum()),
                "exact_coverage_percent": 100 * float(group["exact_lexicon_match"].mean()),
                "exact_or_stem_matches": int(group["covered"].sum()),
                "exact_or_stem_coverage_percent": 100 * float(group["covered"].mean()),
            }
        )
    unmatched = (
        checked.loc[~checked["covered"]]
        .groupby(["model", "labels", "output"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["model", "count"], ascending=[True, False])
    )
    return pd.DataFrame(rows), unmatched


def error_pair_summary(full: pd.DataFrame, expanded: pd.DataFrame) -> pd.DataFrame:
    """Create the three thresholded prediction-error summaries used in review."""
    values = expanded.copy()
    values[["labels", "output"]] = full[["labels", "output"]]
    subsets = {
        "small_positive_arousal": values[values["arousal"].gt(0) & values["arousal"].le(0.15)],
        "large_negative_dominance": values[values["dominance"].le(-0.25)],
        "large_positive_dominance": values[values["dominance"].ge(0.25)],
    }
    summaries = []
    for error_type, subset in subsets.items():
        summary = (
            subset.groupby(["model", "labels", "output"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        summary["error_type"] = error_type
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True).sort_values(
        ["error_type", "model", "count"], ascending=[True, True, False]
    )


def create_figures(expanded: pd.DataFrame, full: pd.DataFrame, figure_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, dimension in zip(axes, DIMENSIONS):
        values = expanded[dimension].dropna()
        axis.hist(values, bins=50, color="#4c78a8", alpha=0.85)
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set(title=dimension.title(), xlabel="Probability-weighted difference", ylabel="Count")
    fig.tight_layout()
    fig.savefig(figure_dir / "EBS-errors.png", dpi=200, bbox_inches="tight")
    fig.savefig(figure_dir / "EBS-errors.pdf", bbox_inches="tight")
    plt.close(fig)

    means = expanded.groupby("model")[list(DIMENSIONS)].mean().reindex(MODELS)
    fig, axis = plt.subplots(figsize=(10, 5))
    means.plot(kind="bar", ax=axis, color=["#4c78a8", "#f58518", "#54a24b"])
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Model", ylabel="Mean probability-weighted difference", title="VAD bias by model")
    axis.legend(title="Dimension")
    fig.tight_layout()
    fig.savefig(figure_dir / "EBS-by-model.png", dpi=200, bbox_inches="tight")
    fig.savefig(figure_dir / "EBS-by-model.pdf", bbox_inches="tight")
    plt.close(fig)

    topic_means = expanded.groupby("topic_name")[list(DIMENSIONS)].mean()
    topic_means = topic_means.loc[topic_means.abs().max(axis=1).nlargest(20).index]
    topic_means = topic_means.sort_values("valence")
    fig, axis = plt.subplots(figsize=(11, 8))
    topic_means.plot(kind="barh", ax=axis, color=["#4c78a8", "#f58518", "#54a24b"])
    axis.axvline(0, color="black", linewidth=1)
    axis.set(xlabel="Mean probability-weighted difference", ylabel="Topic", title="Topics with largest VAD bias")
    fig.tight_layout()
    fig.savefig(figure_dir / "EBS-topics.png", dpi=200, bbox_inches="tight")
    fig.savefig(figure_dir / "EBS-topics.pdf", bbox_inches="tight")
    plt.close(fig)

    top_words = full["output"].value_counts().head(20).index
    counts = (
        full.loc[full["output"].isin(top_words)]
        .groupby(["output", "model"])
        .size()
        .unstack(fill_value=0)
        .reindex(top_words)
    )
    fig, axis = plt.subplots(figsize=(12, 6))
    counts.plot(kind="bar", ax=axis, color=[COLORS.get(column, "gray") for column in counts.columns])
    axis.set(xlabel="Predicted emotion", ylabel="Count", title="Most frequent predicted emotions")
    axis.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(figure_dir / "model_emotion_distribution.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "model_emotion_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    label_vectors = np.asarray(full["vad_label"].map(json.loads).tolist(), dtype=float)
    output_vectors = np.asarray(full["vad_output"].map(json.loads).tolist(), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bins = np.linspace(0, 1, 41)
    for index, (axis, dimension) in enumerate(zip(axes, DIMENSIONS)):
        axis.hist(label_vectors[:, index], bins=bins, histtype="step", color="black", linewidth=2, label="Labels")
        axis.hist(output_vectors[:, index], bins=bins, histtype="step", color="#4c78a8", linewidth=1.5, label="Predictions")
        axis.set(title=dimension.title(), xlabel="NRC-VAD value", ylabel="Count")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "vad_value_distributions.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "vad_value_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = required_inputs(args.data_dir, args.raw_results_dir)
    processed_dir = args.results_dir / "processed-results"
    figure_dir = args.results_dir / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    topics = pd.read_csv(paths["topics"], usecols=["index", "topic_id", "topic_name"])
    topics = topics.drop_duplicates("index")
    vad = load_vad(paths["vad"])

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for model in MODELS:
        print(f"Processing {model}...", flush=True)
        frame, summary = process_model(
            model,
            paths[model],
            topics,
            vad,
            processed_dir / f"{model}-results.csv",
        )
        frames.append(frame)
        summaries.append(summary)

    full = pd.concat(frames, ignore_index=True)
    full.to_csv(processed_dir / "all-models-results.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.results_dir / "model_summary.csv", index=False)

    expanded = expand_weighted(full)
    overall_tests(expanded, args.alpha).to_csv(
        args.results_dir / "vad_bias_ttests.csv", index=False
    )
    topic_tests(expanded, args.alpha).to_csv(
        args.results_dir / "vad_ttests_by_topic.csv", index=False
    )
    coverage, unmatched = emotion_term_coverage(full, paths["emotion_lexicon"])
    coverage.to_csv(args.results_dir / "emotion_term_coverage.csv", index=False)
    unmatched.to_csv(args.results_dir / "unmatched_emotion_terms.csv", index=False)
    error_pair_summary(full, expanded).to_csv(
        args.results_dir / "error_pair_summary.csv", index=False
    )
    create_figures(expanded, full, figure_dir)
    print(f"Complete. Results written to {args.results_dir}", flush=True)


if __name__ == "__main__":
    main()
