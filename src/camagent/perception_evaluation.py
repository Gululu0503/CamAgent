"""Held-out species-classification evaluation for the CamAgent perception layer."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


MISSING_PREDICTION = "<missing>"


def _label(value: Any, *, missing_allowed: bool) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        if missing_allowed:
            return MISSING_PREDICTION
        raise ValueError("Ground-truth labels must not be missing.")
    text = " ".join(str(value).strip().split())
    if not text:
        if missing_allowed:
            return MISSING_PREDICTION
        raise ValueError("Ground-truth labels must not be blank.")
    return text.casefold()


def _confidence(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        return None
    return parsed


def _bootstrap_accuracy(
    frame: pd.DataFrame,
    *,
    group_column: str | None,
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    if iterations < 0:
        raise ValueError("bootstrap_iters must be non-negative.")
    unit = "record"
    group_values: list[Any] = []
    group_indices: dict[Any, np.ndarray] = {}
    if group_column and group_column in frame and frame[group_column].notna().any():
        unit = group_column
        normalized_groups = frame[group_column].fillna("<missing-group>").astype(str)
        group_values = normalized_groups.drop_duplicates().tolist()
        group_indices = {
            group: np.flatnonzero(normalized_groups.to_numpy() == group)
            for group in group_values
        }

    if iterations == 0:
        return {
            "bootstrap_iters": 0,
            "bootstrap_unit": unit,
            "accuracy_ci_low": None,
            "accuracy_ci_high": None,
        }

    rng = np.random.default_rng(random_seed)
    correct = frame["correct"].to_numpy(dtype=float)
    estimates = []
    if group_values:
        for _ in range(iterations):
            sampled_groups = rng.choice(group_values, size=len(group_values), replace=True)
            indices = np.concatenate([group_indices[group] for group in sampled_groups])
            estimates.append(float(correct[indices].mean()))
    else:
        for _ in range(iterations):
            indices = rng.integers(0, len(frame), size=len(frame))
            estimates.append(float(correct[indices].mean()))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "bootstrap_iters": iterations,
        "bootstrap_unit": unit,
        "accuracy_ci_low": round(float(low), 6),
        "accuracy_ci_high": round(float(high), 6),
    }


def _reliability_rows(frame: pd.DataFrame, bin_count: int) -> tuple[list[dict[str, Any]], float | None]:
    if bin_count < 2:
        raise ValueError("calibration_bins must be at least 2.")
    scored = frame.loc[frame["confidence"].notna()].copy()
    if scored.empty:
        return [], None
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    bin_ids = np.minimum(np.digitize(scored["confidence"], edges[1:-1], right=False), bin_count - 1)
    rows = []
    ece = 0.0
    for bin_index in range(bin_count):
        selected = scored.loc[bin_ids == bin_index]
        if selected.empty:
            continue
        mean_confidence = float(selected["confidence"].mean())
        empirical_accuracy = float(selected["correct"].mean())
        weight = len(selected) / len(scored)
        ece += weight * abs(empirical_accuracy - mean_confidence)
        rows.append({
            "bin_index": bin_index + 1,
            "lower_bound": round(float(edges[bin_index]), 6),
            "upper_bound": round(float(edges[bin_index + 1]), 6),
            "sample_count": len(selected),
            "mean_confidence": round(mean_confidence, 6),
            "empirical_accuracy": round(empirical_accuracy, 6),
            "absolute_calibration_gap": round(abs(empirical_accuracy - mean_confidence), 6),
        })
    return rows, round(float(ece), 6)


def evaluate_perception_records(
    records: Iterable[dict[str, Any]],
    *,
    true_label_column: str = "true_label",
    predicted_label_column: str = "predicted_label",
    confidence_column: str = "confidence",
    group_column: str | None = "camera_id",
    sample_id_column: str | None = "sample_id",
    confidence_threshold: float = 0.0,
    calibration_bins: int = 10,
    bootstrap_iters: int = 1000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate top-label taxonomic predictions without conflating classification and detection."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    frame = pd.DataFrame(list(records))
    if frame.empty:
        raise ValueError("At least one held-out annotated record is required.")
    for column in (true_label_column, predicted_label_column):
        if column not in frame:
            raise ValueError(f"Required column is missing: {column}")

    scored = frame.copy()
    scored["true_label_normalized"] = [
        _label(value, missing_allowed=False) for value in scored[true_label_column]
    ]
    scored["predicted_label_normalized"] = [
        _label(value, missing_allowed=True) for value in scored[predicted_label_column]
    ]
    if confidence_column in scored:
        scored["confidence"] = [_confidence(value) for value in scored[confidence_column]]
    else:
        scored["confidence"] = None
    scored["correct"] = scored["true_label_normalized"] == scored["predicted_label_normalized"]

    labels = sorted(set(scored["true_label_normalized"]) | set(scored["predicted_label_normalized"]))
    confusion = pd.crosstab(
        scored["true_label_normalized"],
        scored["predicted_label_normalized"],
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)
    per_class = []
    for label in labels:
        true_positive = int(confusion.loc[label, label])
        false_positive = int(confusion[label].sum() - true_positive)
        false_negative = int(confusion.loc[label].sum() - true_positive)
        support = int(confusion.loc[label].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if recall is not None and precision + recall
            else 0.0 if recall is not None else None
        )
        per_class.append({
            "label": label,
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": round(precision, 6),
            "recall": round(recall, 6) if recall is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
        })

    supported = [row for row in per_class if row["support"] > 0]
    accuracy = float(scored["correct"].mean())
    valid_confidence = scored["confidence"].notna()
    threshold_selected = valid_confidence & (scored["confidence"] >= confidence_threshold)
    reliability, expected_calibration_error = _reliability_rows(scored, calibration_bins)
    duplicate_sample_ids = None
    sample_id_available = bool(
        sample_id_column and sample_id_column in scored
    )
    if sample_id_column and sample_id_column in scored:
        duplicate_sample_ids = int(scored[sample_id_column].astype(str).duplicated(keep=False).sum())
    group_available = bool(
        group_column
        and group_column in scored
        and scored[group_column].notna().any()
    )
    bootstrap = _bootstrap_accuracy(
        scored,
        group_column=group_column,
        iterations=bootstrap_iters,
        random_seed=random_seed,
    )
    summary = {
        "evaluation_scope": "top_label_species_classification",
        "sample_count": len(scored),
        "class_count_with_ground_truth": len(supported),
        "correct_count": int(scored["correct"].sum()),
        "accuracy": round(accuracy, 6),
        "macro_precision": round(float(np.mean([row["precision"] for row in supported])), 6),
        "macro_recall": round(float(np.mean([row["recall"] for row in supported])), 6),
        "macro_f1": round(float(np.mean([row["f1"] for row in supported])), 6),
        "missing_prediction_count": int((scored["predicted_label_normalized"] == MISSING_PREDICTION).sum()),
        "valid_confidence_count": int(valid_confidence.sum()),
        "invalid_or_missing_confidence_count": int((~valid_confidence).sum()),
        "confidence_threshold": confidence_threshold,
        "threshold_coverage": round(float(threshold_selected.mean()), 6),
        "threshold_accuracy": (
            round(float(scored.loc[threshold_selected, "correct"].mean()), 6)
            if threshold_selected.any() else None
        ),
        "expected_calibration_error": expected_calibration_error,
        "top_label_brier_score": (
            round(float(np.mean((scored.loc[valid_confidence, "confidence"] - scored.loc[valid_confidence, "correct"].astype(float)) ** 2)), 6)
            if valid_confidence.any() else None
        ),
        "duplicate_sample_record_count": duplicate_sample_ids,
        "sample_id_column_available": sample_id_available,
        "camera_group_column_available": group_available,
        "camera_group_count": (
            int(scored[group_column].dropna().astype(str).nunique())
            if group_available
            else 0
        ),
        "random_seed": random_seed,
        **bootstrap,
    }
    confusion_rows = [
        {"true_label": true_label, "predicted_label": predicted_label, "count": int(confusion.loc[true_label, predicted_label])}
        for true_label in labels
        for predicted_label in labels
    ]
    return {
        "schema_version": "1.0",
        "summary": summary,
        "per_class": per_class,
        "confusion_matrix": confusion_rows,
        "reliability": reliability,
        "scored_records": scored.to_dict(orient="records"),
        "limitations": [
            "This evaluation measures top-label taxonomic classification, not animal-detection localization or counting accuracy.",
            "Held-out labels must be human reviewed and separated from model development data before manuscript reporting.",
            "Camera-cluster bootstrap intervals address within-camera dependence only when a valid camera/group column is supplied.",
        ],
        "method_references": [
            (
                "Norouzzadeh et al. (2018) Proceedings of the National "
                "Academy of Sciences 115:E5716-E5725; "
                "doi:10.1073/pnas.1719367115"
            ),
            (
                "Beery, Van Horn and Perona (2018) ECCV:456-473; "
                "camera-location-disjoint recognition benchmark"
            ),
        ],
    }


def evaluate_perception_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    camera_disjoint_verified: bool = False,
    human_review_status: str = "unverified",
    split_note: str | None = None,
    development_camera_ids_path: str | Path | None = None,
    annotation_manifest_path: str | Path | None = None,
    annotation_input_path: str | Path | None = None,
    minimum_class_support: int = 20,
    **kwargs,
) -> dict[str, Any]:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if minimum_class_support < 1:
        raise ValueError("minimum_class_support must be at least 1.")
    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        records = payload.get("records", payload) if isinstance(payload, dict) else payload
    else:
        records = pd.read_csv(input_path).to_dict(orient="records")
    if not isinstance(records, list):
        raise ValueError("Input must contain a list of annotated prediction records.")
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    annotation_source_path = (
        Path(annotation_input_path).resolve()
        if annotation_input_path
        else input_path
    )
    annotation_input_sha256 = hashlib.sha256(
        annotation_source_path.read_bytes()
    ).hexdigest()
    result = evaluate_perception_records(records, **kwargs)
    annotation_provenance = None
    annotation_manifest_valid = False
    if annotation_manifest_path:
        annotation_path = Path(annotation_manifest_path).resolve()
        annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(annotation_payload, dict):
            raise ValueError("Annotation manifest must be a JSON object.")
        manifest_review_status = str(
            annotation_payload.get("review_status", "unverified")
        ).casefold()
        reviewer_roles = annotation_payload.get("reviewer_roles", [])
        reviewed_at_utc = annotation_payload.get("reviewed_at_utc")
        annotation_protocol = annotation_payload.get("annotation_protocol")
        declared_input_sha256 = annotation_payload.get("held_out_input_sha256")
        annotation_manifest_valid = bool(
            manifest_review_status
            in {"human_reviewed", "adjudicated", "expert_consensus"}
            and isinstance(reviewer_roles, list)
            and any(str(role).strip() for role in reviewer_roles)
            and reviewed_at_utc
            and annotation_protocol
            and declared_input_sha256 == annotation_input_sha256
        )
        annotation_provenance = {
            "file": annotation_path.name,
            "sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
            "review_status": manifest_review_status,
            "reviewer_roles": reviewer_roles,
            "reviewed_at_utc": reviewed_at_utc,
            "annotation_protocol": annotation_protocol,
            "declared_input_sha256": declared_input_sha256,
            "annotation_input_file": annotation_source_path.name,
            "input_hash_matches": (
                declared_input_sha256 == annotation_input_sha256
            ),
            "manifest_valid": annotation_manifest_valid,
        }
        human_review_status = manifest_review_status
    development_camera_provenance = None
    camera_overlap_count = None
    if development_camera_ids_path:
        development_path = Path(development_camera_ids_path).resolve()
        if development_path.suffix.lower() == ".json":
            development_payload = json.loads(development_path.read_text(encoding="utf-8"))
            if isinstance(development_payload, dict):
                development_payload = (
                    development_payload.get("camera_ids")
                    or development_payload.get("development_camera_ids")
                    or []
                )
            development_ids = {str(value).strip() for value in development_payload if str(value).strip()}
        else:
            development_frame = pd.read_csv(development_path, dtype=str)
            if development_frame.shape[1] != 1:
                raise ValueError("Development-camera file must contain exactly one camera-ID column.")
            development_ids = {
                str(value).strip()
                for value in development_frame.iloc[:, 0].dropna()
                if str(value).strip()
            }
        group_column = kwargs.get("group_column", "camera_id")
        test_frame = pd.DataFrame(records)
        if not group_column or group_column not in test_frame:
            raise ValueError(
                "A development-camera file requires the configured group/camera column in held-out records."
            )
        test_ids = {
            str(value).strip()
            for value in test_frame[group_column].dropna()
            if str(value).strip()
        }
        camera_overlap_count = len(test_ids & development_ids)
        camera_disjoint_verified = camera_overlap_count == 0
        development_camera_provenance = {
            "file": development_path.name,
            "sha256": hashlib.sha256(development_path.read_bytes()).hexdigest(),
            "development_camera_count": len(development_ids),
            "held_out_camera_count": len(test_ids),
            "overlap_count": camera_overlap_count,
        }
    eligibility = [
        {
            "criterion": "camera_disjoint_split_machine_verified",
            "passed": bool(
                camera_disjoint_verified
                and development_camera_provenance
                and camera_overlap_count == 0
            ),
            "evidence": development_camera_provenance,
        },
        {
            "criterion": "human_annotation_manifest_valid",
            "passed": annotation_manifest_valid,
            "evidence": annotation_provenance,
        },
        {
            "criterion": "unique_sample_ids",
            "passed": bool(
                result["summary"].get("sample_id_column_available")
                and (
                    result["summary"].get(
                        "duplicate_sample_record_count"
                    )
                    or 0
                )
                == 0
            ),
            "evidence": {
                "sample_id_column_available": result["summary"].get(
                    "sample_id_column_available"
                ),
                "duplicate_sample_record_count": result["summary"].get(
                    "duplicate_sample_record_count"
                )
            },
        },
        {
            "criterion": "camera_cluster_bootstrap_available",
            "passed": result["summary"].get("bootstrap_unit") != "record",
            "evidence": {"bootstrap_unit": result["summary"].get("bootstrap_unit")},
        },
    ]
    diagnostic_reasons = []
    if not camera_disjoint_verified:
        diagnostic_reasons.append("camera_disjoint_split_not_verified")
    elif not development_camera_provenance:
        diagnostic_reasons.append(
            "camera_disjoint_claim_lacks_machine_verified_provenance"
        )
    if human_review_status.casefold() not in {
        "human_reviewed",
        "adjudicated",
        "expert_consensus",
    }:
        diagnostic_reasons.append("ground_truth_review_not_verified")
    elif not annotation_manifest_valid:
        diagnostic_reasons.append("human_review_claim_lacks_valid_annotation_manifest")
    if not result["summary"].get("sample_id_column_available"):
        diagnostic_reasons.append("unique_sample_id_column_unavailable")
    if (result["summary"].get("duplicate_sample_record_count") or 0) > 0:
        diagnostic_reasons.append("duplicate_sample_ids")
    if result["summary"].get("bootstrap_unit") == "record":
        diagnostic_reasons.append("camera_cluster_bootstrap_unavailable")
    supported_rows = [
        row for row in result["per_class"]
        if int(row.get("support") or 0) > 0
    ]
    if supported_rows and min(int(row["support"]) for row in supported_rows) < minimum_class_support:
        diagnostic_reasons.append("one_or_more_classes_below_minimum_support")
    eligibility.append({
        "criterion": "minimum_class_support",
        "passed": bool(
            supported_rows
            and min(int(row["support"]) for row in supported_rows)
            >= minimum_class_support
        ),
        "evidence": {
            "minimum_required": int(minimum_class_support),
            "minimum_observed": (
                min(int(row["support"]) for row in supported_rows)
                if supported_rows
                else 0
            ),
        },
    })
    provenance = {
        "input_file": input_path.name,
        "input_sha256": input_sha256,
        "annotation_input_file": annotation_source_path.name,
        "annotation_input_sha256": annotation_input_sha256,
        "camera_disjoint_verified": bool(camera_disjoint_verified),
        "human_review_status": human_review_status,
        "split_note": split_note,
        "development_camera_provenance": development_camera_provenance,
        "annotation_provenance": annotation_provenance,
        "camera_overlap_count": camera_overlap_count,
        "minimum_class_support": int(minimum_class_support),
    }
    manifest = {
        "schema_version": "1.0",
        "experiment_id": "PERCEPTION_BENCHMARK",
        "experiment_name": "Perception Validation Benchmark",
        "score_status": "diagnostic_only" if diagnostic_reasons else "held_out_scored_records",
        "diagnostic_reasons": diagnostic_reasons,
        "eligibility_criteria": eligibility,
        "provenance": provenance,
        "evaluation_scope": result["summary"].get("evaluation_scope"),
        "interpretation_boundary": (
            "This output is diagnostic only until the camera-disjoint split and human-reviewed labels are documented."
            if diagnostic_reasons else
            "These scores apply only to top-label taxonomic classification on the documented held-out set."
        ),
    }
    result["provenance"] = provenance

    (output_dir / "perception_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    pd.DataFrame([result["summary"]]).to_csv(output_dir / "perception_summary.csv", index=False)
    pd.DataFrame(result["per_class"]).to_csv(output_dir / "perception_per_class.csv", index=False)
    pd.DataFrame(result["confusion_matrix"]).to_csv(output_dir / "perception_confusion_matrix.csv", index=False)
    pd.DataFrame(result["reliability"]).to_csv(output_dir / "perception_reliability.csv", index=False)
    (output_dir / "perception_evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "camagent-matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in result["per_class"]]
    matrix = pd.DataFrame(result["confusion_matrix"]).pivot(
        index="true_label", columns="predicted_label", values="count"
    ).reindex(index=labels, columns=labels, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    image = axes[0].imshow(matrix.to_numpy(), cmap="Blues")
    axes[0].set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("Human-reviewed label")
    axes[0].set_title("Held-out taxonomic confusion matrix")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    reliability = pd.DataFrame(result["reliability"])
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Perfect calibration")
    if not reliability.empty:
        axes[1].plot(
            reliability["mean_confidence"],
            reliability["empirical_accuracy"],
            marker="o",
            label="Observed",
        )
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Empirical accuracy")
    axes[1].set_title("Top-label confidence calibration")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    for suffix, options in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(output_dir / f"fig_perception_evaluation.{suffix}", bbox_inches="tight", facecolor="white", **options)
    plt.close(fig)
    return result


def _load_benchmark_annotations(
    annotation_path: str | Path,
) -> list[dict[str, Any]]:
    annotation_path = Path(annotation_path).resolve()
    suffix = annotation_path.suffix.casefold()
    if suffix == ".json":
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        records = (
            payload.get("records", payload)
            if isinstance(payload, dict)
            else payload
        )
        if not isinstance(records, list):
            raise ValueError(
                "JSON annotations must contain a list or a `records` list."
            )
        return [dict(row) for row in records]
    if suffix in {".jsonl", ".ndjson"}:
        return [
            json.loads(line)
            for line in annotation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(annotation_path, sep="\t").to_dict(
            orient="records"
        )
    return pd.read_csv(annotation_path).to_dict(orient="records")


def _parse_label_aliases(
    label_aliases: dict[str, str] | str | Path | None,
) -> dict[str, str]:
    if not label_aliases:
        return {}
    if isinstance(label_aliases, dict):
        payload = label_aliases
    else:
        text = str(label_aliases).strip()
        candidate = Path(text).expanduser()
        if candidate.exists():
            payload = json.loads(
                candidate.resolve().read_text(encoding="utf-8")
            )
        else:
            payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("label_aliases must be a JSON object or dictionary.")
    return {
        " ".join(str(key).strip().split()).casefold(): str(value).strip()
        for key, value in payload.items()
        if str(key).strip() and str(value).strip()
    }


def _aggregate_video_top_label(
    predictions: list[dict[str, Any]],
) -> tuple[str | None, float | None]:
    """Aggregate sampled-frame labels into one auditable clip-level top label."""
    by_label: dict[str, list[float]] = {}
    display_labels: dict[str, str] = {}
    for prediction in predictions:
        label = str(
            prediction.get("species")
            or prediction.get("scientific_name")
            or ""
        ).strip()
        if not label:
            continue
        key = " ".join(label.split()).casefold()
        score = _confidence(
            prediction.get("confidence")
            if prediction.get("confidence") is not None
            else prediction.get("prediction_score")
        )
        by_label.setdefault(key, []).append(
            score if score is not None else 0.0
        )
        display_labels.setdefault(key, label)
    if not by_label:
        return None, None
    ranked = sorted(
        by_label,
        key=lambda key: (
            -sum(by_label[key]) / max(len(by_label[key]), 1),
            -len(by_label[key]),
            key,
        ),
    )
    winner = ranked[0]
    return (
        display_labels[winner],
        float(sum(by_label[winner]) / max(len(by_label[winner]), 1)),
    )


def run_perception_benchmark(
    annotation_path: str | Path,
    output_dir: str | Path,
    *,
    media_root: str | Path | None = None,
    media_path_column: str = "media_path",
    true_label_column: str = "true_label",
    predicted_label_column: str = "predicted_label",
    confidence_column: str = "confidence",
    group_column: str | None = "camera_id",
    sample_id_column: str = "sample_id",
    run_inference: bool = True,
    country: str | None = None,
    admin1_region: str | None = None,
    model: str | None = None,
    video_frame_interval_seconds: float = 1.0,
    max_video_frames: int = 12,
    max_samples: int | None = None,
    label_aliases: dict[str, str] | str | Path | None = None,
    confidence_threshold: float = 0.0,
    calibration_bins: int = 10,
    bootstrap_iters: int = 1000,
    random_seed: int = 42,
    camera_disjoint_verified: bool = False,
    human_review_status: str = "unverified",
    split_note: str | None = None,
    development_camera_ids_path: str | Path | None = None,
    annotation_manifest_path: str | Path | None = None,
    minimum_class_support: int = 20,
    resume: bool = True,
) -> dict[str, Any]:
    """
    Run a batch, media-linked perception benchmark and score the predictions.

    The benchmark is dataset-agnostic: labels and media paths come from the
    supplied annotation manifest.  The function never infers ground truth from
    a folder name or ecological query.  Images are classified directly and
    videos are classified from bounded, traceable sampled frames.
    """
    from .detection_inference import (
        run_speciesnet_on_image,
        run_speciesnet_on_video,
    )

    annotation_path = Path(annotation_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    media_root_path = (
        Path(media_root).expanduser().resolve()
        if media_root
        else annotation_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_dir = output_dir / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "perception_scored_records.csv"
    records = _load_benchmark_annotations(annotation_path)
    if not records:
        raise ValueError("The annotation table contains no records.")
    for required in (media_path_column, true_label_column):
        if required not in records[0]:
            raise ValueError(
                f"Required annotation column is missing: {required}"
            )
    if max_samples is not None:
        max_samples = max(int(max_samples), 1)
        records = records[:max_samples]

    cached: dict[str, dict[str, Any]] = {}
    if resume and checkpoint_path.exists():
        cached_frame = pd.read_csv(checkpoint_path)
        if sample_id_column in cached_frame:
            cached = {
                str(row[sample_id_column]): row.to_dict()
                for _, row in cached_frame.iterrows()
            }

    aliases = _parse_label_aliases(label_aliases)
    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    }
    video_extensions = {
        ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"
    }
    scored_records = []
    failures = []
    for index, source in enumerate(records, start=1):
        row = dict(source)
        sample_id = str(
            row.get(sample_id_column)
            or row.get(media_path_column)
            or f"sample_{index:06d}"
        ).strip()
        sample_token = (
            f"{index:06d}_"
            + hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]
        )
        row[sample_id_column] = sample_id
        true_label = str(row.get(true_label_column, "")).strip()
        alias_key = " ".join(true_label.split()).casefold()
        row[true_label_column] = aliases.get(alias_key, true_label)
        if sample_id in cached:
            scored_records.append(dict(cached[sample_id]))
            continue

        if run_inference:
            media_path = Path(str(row.get(media_path_column, "")).strip())
            if not media_path.is_absolute():
                media_path = media_root_path / media_path
            media_path = media_path.expanduser().resolve()
            row[media_path_column] = str(media_path)
            try:
                if media_path.suffix.casefold() in image_extensions:
                    result = run_speciesnet_on_image(
                        media_path,
                        output_json=(
                            inference_dir
                            / f"{sample_token}_speciesnet.json"
                        ),
                        country=country,
                        admin1_region=admin1_region,
                        model=model,
                    )
                    predicted_label = result.get("species")
                    confidence = result.get("confidence")
                    provenance_path = result.get("predictions_json")
                    sampled_frame_count = 1
                elif media_path.suffix.casefold() in video_extensions:
                    result = run_speciesnet_on_video(
                        media_path,
                        output_dir=inference_dir / sample_token,
                        frame_interval_seconds=video_frame_interval_seconds,
                        max_frames=max_video_frames,
                        country=country,
                        admin1_region=admin1_region,
                        model=model,
                    )
                    predicted_label, confidence = (
                        _aggregate_video_top_label(
                            result.get("predictions", [])
                        )
                    )
                    provenance_path = result.get("manifest_json")
                    sampled_frame_count = len(
                        result.get("predictions", [])
                    )
                else:
                    raise ValueError(
                        f"Unsupported media extension: {media_path.suffix}"
                    )
                row[predicted_label_column] = predicted_label
                row[confidence_column] = confidence
                row["prediction_provenance"] = provenance_path
                row["sampled_frame_count"] = sampled_frame_count
                row["inference_status"] = "success"
            except Exception as exc:
                row[predicted_label_column] = None
                row[confidence_column] = None
                row["prediction_provenance"] = None
                row["sampled_frame_count"] = 0
                row["inference_status"] = "failed"
                row["inference_error"] = f"{type(exc).__name__}: {exc}"
                failures.append({
                    "sample_id": sample_id,
                    "media_path": str(row.get(media_path_column, "")),
                    "error": row["inference_error"],
                })
        elif predicted_label_column not in row:
            raise ValueError(
                f"`{predicted_label_column}` is required when "
                "run_inference=False."
            )

        predicted_text = str(
            row.get(predicted_label_column) or ""
        ).strip()
        predicted_key = " ".join(predicted_text.split()).casefold()
        if predicted_text:
            row[predicted_label_column] = aliases.get(
                predicted_key,
                predicted_text,
            )
        scored_records.append(row)
        pd.DataFrame(scored_records).to_csv(
            checkpoint_path,
            index=False,
        )

    pd.DataFrame(scored_records).to_csv(checkpoint_path, index=False)
    result = evaluate_perception_file(
        checkpoint_path,
        output_dir,
        true_label_column=true_label_column,
        predicted_label_column=predicted_label_column,
        confidence_column=confidence_column,
        group_column=group_column,
        sample_id_column=sample_id_column,
        confidence_threshold=confidence_threshold,
        calibration_bins=calibration_bins,
        bootstrap_iters=bootstrap_iters,
        random_seed=random_seed,
        camera_disjoint_verified=camera_disjoint_verified,
        human_review_status=human_review_status,
        split_note=split_note,
        development_camera_ids_path=development_camera_ids_path,
        annotation_manifest_path=annotation_manifest_path,
        annotation_input_path=annotation_path,
        minimum_class_support=minimum_class_support,
    )
    benchmark_run = {
        "annotation_path": str(annotation_path),
        "annotation_sha256": hashlib.sha256(
            annotation_path.read_bytes()
        ).hexdigest(),
        "media_root": str(media_root_path),
        "run_inference": bool(run_inference),
        "model": model or "speciesnet_default",
        "requested_sample_count": len(records),
        "scored_sample_count": len(scored_records),
        "inference_failure_count": len(failures),
        "video_frame_interval_seconds": float(
            video_frame_interval_seconds
        ),
        "max_video_frames": int(max_video_frames),
        "resume": bool(resume),
        "scored_records_path": str(checkpoint_path),
        "figure_pdf_path": str(
            output_dir / "fig_perception_evaluation.pdf"
        ),
        "figure_png_path": str(
            output_dir / "fig_perception_evaluation.png"
        ),
    }
    result["benchmark_run"] = benchmark_run
    result["inference_failures"] = failures
    (output_dir / "perception_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    manifest_path = output_dir / "perception_evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["batch_inference"] = benchmark_run
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "status": "success",
        "summary": result["summary"],
        "per_class": result["per_class"],
        "method_references": result.get("method_references", []),
        "provenance": result["provenance"],
        "benchmark_run": benchmark_run,
        "inference_failures": failures,
        "figures": [{
            "figure_id": "perception_evaluation",
            "label": (
                "Held-out perception confusion matrix and calibration"
            ),
            "pdf_path": benchmark_run["figure_pdf_path"],
            "png_path": benchmark_run["figure_png_path"],
        }],
        "tables": [
            str(output_dir / "perception_summary.csv"),
            str(output_dir / "perception_per_class.csv"),
            str(output_dir / "perception_confusion_matrix.csv"),
            str(output_dir / "perception_reliability.csv"),
            str(checkpoint_path),
        ],
        "manifest_path": str(manifest_path),
    }
