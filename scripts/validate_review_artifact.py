#!/usr/bin/env python3
"""Validate the versioned, privacy-safe anonymous review artifact."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUERY_DIR = ROOT / "results" / "paper_query_experiments"
RAW_MEDIA_TRACE = (
    ROOT / "results" / "raw_media_validation" / "raw_media_public_trace.json"
)
EXPECTED_EXPERIMENTS = ("QUAL", "TEMP", "SPAT", "COMM", "MOVE", "INTEG")
REDACTED_LOCATION = "[REDACTED_SENSITIVE_LOCATION]"
SENSITIVE_LOCATION_KEYS = {
    "camera_id",
    "cameraid",
    "centercameraid",
    "center_camera_id",
    "camera_token",
    "from_camera",
    "to_camera",
    "camera_a",
    "camera_b",
    "latitude",
    "longitude",
    "location",
    "coordinates",
}
TEXT_SUFFIXES = {
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
RAW_DATA_SUFFIXES = {
    ".avi",
    ".bmp",
    ".db",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp4",
    ".parquet",
    ".sqlite",
    ".tif",
    ".tiff",
}
FORBIDDEN_TEXT_PATTERNS = {
    "macOS absolute user path": re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    "Linux absolute user path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "Windows absolute user path": re.compile(
        r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+\\\\"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}


# Local environments and caches created while following the README.
LOCAL_ENVIRONMENT_PARTS = {
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "tmp",
    "workspace",
}


def _is_local_environment(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    return any(
        part in LOCAL_ENVIRONMENT_PARTS or part.endswith(".egg-info") for part in parts
    )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def walk_sensitive_locations(value, path: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}.{key}" if path else key
            if key.casefold() in SENSITIVE_LOCATION_KEYS:
                yield item_path, item
            yield from walk_sensitive_locations(item, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_sensitive_locations(item, f"{path}[{index}]")


def validate_questions_and_traces() -> None:
    config = load_json(ROOT / "configs" / "paper_query_experiments.json")
    manifest = load_json(QUERY_DIR / "paper_query_experiment_manifest.json")
    configured = {item["experiment_id"]: item for item in config["experiments"]}
    recorded = {item["experiment_id"]: item for item in manifest["experiments"]}
    require(tuple(configured) == EXPECTED_EXPERIMENTS, "Unexpected configured experiment order")
    require(tuple(recorded) == EXPECTED_EXPERIMENTS, "Unexpected manifest experiment order")
    require(manifest["experiment_count"] == 6, "Manifest must declare six experiments")
    require(
        manifest.get("manifest_assembly_mode") in {
            "live_agent_execution",
            "assembled_from_completed_agent_traces",
        },
        "Manifest must state whether traces were executed or assembled",
    )
    for item in manifest["code"]["files"]:
        path = ROOT / item["path"]
        require(path.is_file(), f"Missing code-manifest file: {item['path']}")
        require(sha256(path) == item["sha256"], f"Code hash differs: {item['path']}")

    for experiment_id in EXPECTED_EXPERIMENTS:
        expected = configured[experiment_id]
        item = recorded[experiment_id]
        trace_path = ROOT / item["trace_path"]
        trace = load_json(trace_path)
        require(item["question"] == expected["question"], f"{experiment_id}: manifest question differs")
        require(trace["question"] == expected["question"], f"{experiment_id}: trace question differs")
        require(trace["experiment_id"] == experiment_id, f"{experiment_id}: trace ID differs")
        require(sha256(trace_path) == item["trace_sha256"], f"{experiment_id}: trace hash differs")
        source_trace_sha256 = str(item.get("source_trace_sha256") or "")
        require(
            re.fullmatch(r"[0-9a-f]{64}", source_trace_sha256) is not None,
            f"{experiment_id}: source trace hash is missing or invalid",
        )
        postprocessed = source_trace_sha256 != item["trace_sha256"]
        require(
            bool(item.get("trace_postprocessing_applied")) == postprocessed,
            f"{experiment_id}: trace postprocessing provenance differs",
        )
        if postprocessed:
            require(
                manifest.get("review_artifact_postprocessing", {}).get("applied") is True,
                f"{experiment_id}: postprocessed trace lacks an audit declaration",
            )
        require(
            item.get("trace_started_at_utc") == trace.get("started_at_utc")
            and item.get("trace_completed_at_utc") == trace.get("completed_at_utc")
            and item.get("trace_duration_seconds") == trace.get("duration_seconds"),
            f"{experiment_id}: trace timing provenance differs",
        )
        for field_path, value in walk_sensitive_locations(trace):
            require(
                value == REDACTED_LOCATION,
                f"{experiment_id}: unredacted location value at {field_path}",
            )


def validate_raw_media_trace() -> None:
    require(RAW_MEDIA_TRACE.is_file(), "Missing public raw-media Agent trace")
    trace = load_json(RAW_MEDIA_TRACE)
    require(
        trace.get("experiment_id") == "RAW_MEDIA_PIPELINE",
        "Raw-media trace has the wrong experiment ID",
    )
    require(
        isinstance(trace.get("question"), str) and trace["question"].strip(),
        "Raw-media trace must retain the natural-language question",
    )
    require(
        "prepare_camera_trap_dataset" in trace.get("selected_tools", []),
        "The LLM plan did not select prepare_camera_trap_dataset",
    )
    planning = trace.get("trace", {}).get("planning", {})
    normalized_tasks = planning.get("normalized_tasks", [])
    require(
        any(
            item.get("tool_name") == "prepare_camera_trap_dataset"
            for item in normalized_tasks
            if isinstance(item, dict)
        ),
        "The public planning record does not retain the preparation task",
    )
    executions = [
        item for item in trace.get("executed_results", [])
        if item.get("tool_name") == "prepare_camera_trap_dataset"
    ]
    require(len(executions) == 1, "Expected one raw-media preparation call")
    execution = executions[0]
    require(execution.get("status") == "success", "Raw-media preparation failed")
    result = execution.get("result", {})
    evidence = result.get("perception_evidence", {})
    model_versions = evidence.get("model_versions", {})
    require(
        "4.0.2a" in model_versions,
        "SpeciesNet model version 4.0.2a is missing",
    )
    sampling = evidence.get("video_sampling", {})
    require(
        sampling.get("frame_interval_seconds") == 1.0
        and sampling.get("max_frames") == 1,
        "Raw-media frame-sampling provenance differs",
    )
    predictions = evidence.get("sample_predictions", [])
    require(len(predictions) >= 1, "Raw-media trace has no prediction evidence")
    prediction = predictions[0]
    require(
        "tiger" in str(prediction.get("species", "")).casefold(),
        "Expected tiger prediction is missing",
    )
    require(
        abs(float(prediction.get("confidence")) - 0.9989) < 1e-12,
        "Tiger confidence differs from 0.9989",
    )
    require(
        prediction.get("timestamp") == "2024-09-22 16:56:28"
        and prediction.get("timestamp_source") == "paddleocr_video_timestamp",
        "PaddleOCR timestamp evidence differs",
    )
    require(
        result.get("camtrap_validation_error_count") == 0
        and result.get("camtrap_validation_warning_count") == 0,
        "CamtrapDP validation must report 0 errors and 0 warnings",
    )
    require(
        result.get("active_backend", {}).get("activated") is True
        and result.get("analysis_readiness", {}).get(
            "active_for_downstream_tools"
        ) is True,
        "The prepared CamtrapDP backend was not activated",
    )
    reflection = trace.get("trace", {}).get("replanning", {})
    require(
        isinstance(reflection.get("reason"), str)
        and reflection["reason"].strip(),
        "Raw-media trace must retain the reflection decision",
    )
    require(
        isinstance(trace.get("final_answer"), str)
        and trace["final_answer"].strip(),
        "Raw-media trace must retain the final answer",
    )
    for field_path, value in walk_sensitive_locations(trace):
        require(
            value == REDACTED_LOCATION,
            f"Raw-media trace: unredacted location value at {field_path}",
        )
    public_text = RAW_MEDIA_TRACE.read_text(encoding="utf-8")
    for forbidden in (
        "/Users/",
        "/home/",
        "190790",
        "2024-09-22_190790_0",
    ):
        require(
            forbidden not in public_text,
            f"Raw-media trace leaks a private value: {forbidden}",
        )


def validate_synthetic_example() -> None:
    package_dir = ROOT / "examples" / "privacy_safe_camtrap_dp"
    package = load_json(package_dir / "datapackage.json")
    require("synthetic" in package["name"], "Example package must be explicitly synthetic")
    for resource in package["resources"]:
        require((package_dir / resource["path"]).is_file(), f"Missing example resource: {resource['path']}")
    deployments = (package_dir / "deployments.csv").read_text(encoding="utf-8")
    require("Synthetic deployment; invented coordinate." in deployments, "Synthetic coordinate notice missing")
    example_config = load_json(ROOT / "configs" / "example_pipeline.json")
    require(
        example_config["input"]["datapackage"]
        == "examples/privacy_safe_camtrap_dp/datapackage.json",
        "Synthetic quickstart must use the included privacy-safe package",
    )


def validate_distribution_boundary() -> None:
    require(not (ROOT / "data").exists(), "The public artifact must not contain a data directory")
    require(not (ROOT / "artifacts").exists(), "Generated artifact packages must not be distributed")
    require(
        not (ROOT / "results" / "figure_sources").exists(),
        "Generated figure-source results must not be distributed",
    )
    allowed_scripts = {
        "run_agent_stress_matrix.py",
        "run_paper_query_experiments.py",
        "run_research_pipeline.py",
        "validate_review_artifact.py",
    }
    actual_scripts = {
        path.name for path in (ROOT / "scripts").iterdir() if path.is_file()
    }
    require(
        actual_scripts == allowed_scripts,
        "The public scripts directory contains an unapproved helper script",
    )
    allowed_example_files = {
        "agent_independent_events.csv",
        "conversion_report.json",
        "datapackage.json",
        "deployments.csv",
        "media.csv",
        "observations.csv",
    }
    example_dir = ROOT / "examples" / "privacy_safe_camtrap_dp"
    actual_example_files = {
        path.name for path in example_dir.iterdir() if path.is_file()
    }
    require(
        actual_example_files == allowed_example_files,
        "The public example must contain only the six approved synthetic files",
    )
    for path in ROOT.rglob("*"):
        if path.is_file() and not _is_local_environment(path):
            require(
                path.suffix.casefold() not in RAW_DATA_SUFFIXES,
                f"Raw data or media file found: {path.relative_to(ROOT)}",
            )


def validate_text_privacy() -> None:
    ignored_parts = LOCAL_ENVIRONMENT_PARTS
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        relative_parts = path.relative_to(ROOT).parts
        if any(part in ignored_parts or part.endswith(".egg-info") for part in relative_parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN_TEXT_PATTERNS.items():
            require(not pattern.search(text), f"{label} found in {path.relative_to(ROOT)}")


def main() -> None:
    validate_questions_and_traces()
    validate_raw_media_trace()
    validate_synthetic_example()
    validate_distribution_boundary()
    validate_text_privacy()
    print(
        "Review artifact validation: PASS "
        "(6 questions, 6 ecological-query traces, 1 raw-media Agent trace, "
        "synthetic example)"
    )


if __name__ == "__main__":
    main()
