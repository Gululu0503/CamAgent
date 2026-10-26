"""Deterministic checks for the Agent's post-LLM validation and privacy layer."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm_agent import WildlifeAgent


STRESS_EXPERIMENT_ID = "AGENT_STRESS_TEST"
STRESS_EXPERIMENT_NAME = "Agent Stress Test"


def build_agent_contract_stress_cases() -> list[dict[str, Any]]:
    """Return 60 deterministic cases spanning plan pass-through and privacy."""
    species = [
        "Siberian tiger",
        "Amur leopard",
        "Sika deer",
        "Roe deer",
        "Black bear",
    ]
    comparisons = [
        "Sika deer",
        "Roe deer",
        "Black bear",
        "Amur leopard",
        "Siberian tiger",
    ]
    cases: list[dict[str, Any]] = []

    for index in range(10):
        name = species[index % len(species)]
        camera_id = str(310000 + index)
        limit = 5 + index
        cases.append({
            "case_id": f"bounded_filter_{index + 1:02d}",
            "category": "bounded_filter",
            "question": (
                f"Return at most {limit} {name} records from camera {camera_id}, "
                "sorted chronologically."
            ),
            "species": name,
            "raw_plan": {
                "species": name,
                "tasks": [{
                    "tool_name": "filter_records",
                    "args": {
                        "species": name,
                        "camera_id": "[REDACTED]",
                        "limit": limit,
                        "sort_by": "timestamp_asc",
                    },
                }],
            },
            "expected_tools": ["filter_records"],
            "expected_args": {
                "filter_records": {
                    "species": name,
                    "camera_id": camera_id,
                    "limit": limit,
                    "sort_by": "timestamp_asc",
                },
            },
            "forbidden_tools": ["infer_transition_corridors"],
        })

    for index in range(10):
        name = species[index % len(species)]
        camera_id = str(320000 + index)
        radius = 5 * (index + 1)
        cases.append({
            "case_id": f"local_spatial_{index + 1:02d}",
            "category": "local_spatial",
            "question": (
                f"Summarize {name} detections within {radius} km of camera {camera_id} "
                "and rank local hotspots; do not infer movement corridors."
            ),
            "species": name,
            "raw_plan": {
                "species": name,
                "tasks": [{
                    "tool_name": "analyze_spatial_distribution",
                    "args": {
                        "species": name,
                        "center_camera_id": "[REDACTED]",
                        "radius_km": radius,
                    },
                }],
            },
            "expected_tools": ["analyze_spatial_distribution"],
            "expected_args": {
                "analyze_spatial_distribution": {
                    "species": name,
                    "center_camera_id": camera_id,
                    "radius_km": radius,
                },
            },
            "forbidden_tools": ["infer_transition_corridors"],
        })

    for index in range(10):
        name = species[index % len(species)]
        cases.append({
            "case_id": f"quality_contract_{index + 1:02d}",
            "category": "quality_audit",
            "question": (
                f"Audit missing fields, invalid timestamps, label noise, and duplicate burden "
                f"for {name} records."
            ),
            "species": name,
            "raw_plan": {
                "species": name,
                "tasks": [{
                    "tool_name": "analyze_data_quality",
                    "args": {"species": name},
                }],
            },
            "expected_tools": ["analyze_data_quality"],
            "expected_args": {"analyze_data_quality": {"species": name}},
            "forbidden_tools": ["fit_single_season_occupancy_model"],
        })

    for index in range(10):
        name = species[index % len(species)]
        max_gap = 12 * (index + 1)
        max_distance = 20 + 5 * index
        cases.append({
            "case_id": f"movement_guardrail_{index + 1:02d}",
            "category": "movement_guardrail",
            "question": (
                f"List temporally adjacent camera-transition hypotheses for {name} only as "
                "descriptive steps; do not infer corridors or individual tracks."
            ),
            "species": name,
            "raw_plan": {
                "species": name,
                "tasks": [{
                    "tool_name": "analyze_movement_steps",
                    "args": {
                        "species": name,
                        "max_time_gap_hours": max_gap,
                        "max_distance_km": max_distance,
                    },
                }],
            },
            "expected_tools": ["analyze_movement_steps"],
            "expected_args": {
                "analyze_movement_steps": {
                    "species": name,
                    "max_time_gap_hours": max_gap,
                    "max_distance_km": max_distance,
                },
            },
            "forbidden_tools": ["infer_transition_corridors"],
        })

    gaps = [15, 30, 60, 20, 45]
    for index in range(10):
        focal = species[index % len(species)]
        comparison = comparisons[index % len(comparisons)]
        if comparison == focal:
            comparison = comparisons[(index + 1) % len(comparisons)]
        gap = gaps[index % len(gaps)]
        cases.append({
            "case_id": f"temporal_contract_{index + 1:02d}",
            "category": "temporal_overlap",
            "question": (
                f"Compare diel activity overlap between {focal} and {comparison} using a "
                f"{gap}-minute independent-event gap, first verify timestamp provenance, "
                "and report Delta without a plot or direct-interaction claim."
            ),
            "species": focal,
            "raw_plan": {
                "species": focal,
                "tasks": [
                    {
                        "tool_name": "check_timestamp_quality",
                        "args": {"species": focal},
                    },
                    {
                        "tool_name": "build_independent_events",
                        "args": {
                            "species": focal,
                            "min_gap_minutes": gap,
                        },
                    },
                    {
                        "tool_name": "build_hourly_activity_distribution",
                        "args": {
                            "species": focal,
                            "min_gap_minutes": gap,
                        },
                    },
                    {
                        "tool_name": "estimate_pairwise_activity_overlap",
                        "args": {
                            "species_a": focal,
                            "species_list": [comparison],
                            "min_gap_minutes": gap,
                        },
                    },
                ],
            },
            "expected_tools": [
                "check_timestamp_quality",
                "build_independent_events",
                "build_hourly_activity_distribution",
                "estimate_pairwise_activity_overlap",
            ],
            "expected_args": {
                "build_independent_events": {
                    "species": focal,
                    "min_gap_minutes": float(gap),
                },
                "estimate_pairwise_activity_overlap": {
                    "species_a": focal,
                    "species_list": [comparison],
                    "min_gap_minutes": float(gap),
                },
            },
            "forbidden_tools": [
                "plot_hourly_activity_curve",
                "plot_pairwise_activity_overlap_panels",
                "infer_transition_corridors",
            ],
        })

    for index in range(10):
        camera_id = str(330000 + index)
        cases.append({
            "case_id": f"public_trace_privacy_{index + 1:02d}",
            "category": "public_trace_privacy",
            "question": f"Show records near camera {camera_id}.",
            "camera_id": camera_id,
            "latitude": 42.5 + index / 100,
            "longitude": 130.0 + index / 100,
        })

    if len(cases) != 60:
        raise AssertionError(f"Expected 60 stress cases, built {len(cases)}.")
    return cases


def _normalized(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().casefold()
    if isinstance(value, list):
        return sorted(_normalized(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in sorted(value.items())}
    return value


def _sha256_text(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _code_manifest() -> list[dict[str, str]]:
    paths = [
        Path(__file__),
        Path(__file__).with_name("llm_agent.py"),
        Path(__file__).with_name("tool_registry.py"),
    ]
    return [
        {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]


def _score_planning_case(agent: WildlifeAgent, case: dict[str, Any]) -> dict[str, Any]:
    plan = agent._normalize_plan(case["question"], case["raw_plan"], {})
    tasks = plan.get("tasks", [])
    validation = plan.get("_planner_validation", {})
    selected_tools = [task.get("tool_name") for task in tasks]
    forbidden = sorted(set(selected_tools) & set(case.get("forbidden_tools", [])))
    arg_failures = []
    for tool_name, expected in case.get("expected_args", {}).items():
        candidates = [
            task.get("args", {})
            for task in tasks
            if task.get("tool_name") == tool_name
        ]
        for key, expected_value in expected.items():
            if not any(
                _normalized(candidate.get(key)) == _normalized(expected_value)
                for candidate in candidates
            ):
                arg_failures.append({
                    "tool_name": tool_name,
                    "argument": key,
                    "expected": expected_value,
                    "actual_candidates": [candidate.get(key) for candidate in candidates],
                })
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "passed": bool(
            selected_tools == case["expected_tools"]
            and not forbidden
            and not arg_failures
            and validation.get("validation_only") is True
            and validation.get("local_tool_rewrite") is False
            and validation.get("fallback_used") is False
        ),
        "selected_tools": selected_tools,
        "expected_tools": case["expected_tools"],
        "forbidden_tool_violations": forbidden,
        "argument_failures": arg_failures,
        "validation_only": validation.get("validation_only"),
        "local_tool_rewrite": validation.get("local_tool_rewrite"),
        "fallback_used": validation.get("fallback_used"),
    }


def _score_privacy_case(agent: WildlifeAgent, case: dict[str, Any]) -> dict[str, Any]:
    record = {
        "question": case["question"],
        "plan": {
            "tasks": [{
                "tool_name": "filter_records",
                "args": {"camera_id": case["camera_id"]},
            }],
        },
        "executed_results": [{
            "tool_name": "filter_records",
            "args": {"camera_id": case["camera_id"]},
            "status": "success",
            "result": {
                "camera_id": case["camera_id"],
                "latitude": case["latitude"],
                "longitude": case["longitude"],
                "records": [{
                    "cameraID": case["camera_id"],
                    "latitude": case["latitude"],
                    "longitude": case["longitude"],
                }],
            },
        }],
    }
    exported = agent.exportable_trace(record, detail="public")
    serialized = json.dumps(exported, ensure_ascii=False, sort_keys=True)
    leaked_fields = []
    if case["camera_id"] in serialized:
        leaked_fields.append("camera_id")
    if str(case["latitude"]) in serialized:
        leaked_fields.append("latitude")
    if str(case["longitude"]) in serialized:
        leaked_fields.append("longitude")
    trace_export = exported.get("trace_export", {})
    metadata_passed = bool(
        trace_export.get("raw_tool_results_included") is False
        and trace_export.get("raw_model_responses_included") is False
        and trace_export.get("sensitive_camera_locations_redacted") is True
    )
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "passed": bool(not leaked_fields and metadata_passed),
        "selected_tools": ["filter_records"],
        "expected_tools": ["filter_records"],
        "forbidden_tool_violations": [],
        "argument_failures": [],
        "leaked_fields": leaked_fields,
        "trace_export_metadata_passed": metadata_passed,
    }


def run_agent_contract_stress_matrix(
    agent: WildlifeAgent,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the Agent stress matrix and optionally write auditable outputs."""
    started = datetime.now(timezone.utc)
    records = []
    cases = build_agent_contract_stress_cases()
    for case in cases:
        scorer = _score_privacy_case if case["category"] == "public_trace_privacy" else _score_planning_case
        records.append(scorer(agent, case))
    completed = datetime.now(timezone.utc)
    category_summary = []
    for category in sorted({record["category"] for record in records}):
        subset = [record for record in records if record["category"] == category]
        passed = sum(bool(record["passed"]) for record in subset)
        category_summary.append({
            "category": category,
            "case_count": len(subset),
            "passed_case_count": passed,
            "pass_rate": passed / len(subset),
        })
    passed_count = sum(bool(record["passed"]) for record in records)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": STRESS_EXPERIMENT_ID,
        "experiment_name": STRESS_EXPERIMENT_NAME,
        "status": "completed" if passed_count == len(records) else "completed_with_failures",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "case_count": len(records),
        "passed_case_count": passed_count,
        "pass_rate": passed_count / len(records),
        "case_definition_sha256": _sha256_text(cases),
        "records_sha256": _sha256_text(records),
        "code_manifest": _code_manifest(),
        "category_summary": category_summary,
        "interpretation_boundary": (
            "The Agent stress test deterministically checks post-LLM schema validation without local tool rewriting, "
            "local sensitive-argument rehydration, and public-trace privacy. It is not an independently reviewed benchmark "
            "of end-to-end language-model reasoning or ecological correctness."
        ),
    }
    result = {"manifest": manifest, "records": records}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "agent_contract_stress_records.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        fieldnames = [
            "case_id",
            "category",
            "passed",
            "selected_tools",
            "expected_tools",
            "forbidden_tool_violations",
            "argument_failures",
            "leaked_fields",
            "trace_export_metadata_passed",
            "validation_only",
            "local_tool_rewrite",
            "fallback_used",
        ]
        with (output_dir / "agent_contract_stress_records.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow({
                    key: (
                        json.dumps(record.get(key), ensure_ascii=False)
                        if isinstance(record.get(key), (list, dict))
                        else record.get(key)
                    )
                    for key in fieldnames
                })
        (output_dir / "agent_contract_stress_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result
