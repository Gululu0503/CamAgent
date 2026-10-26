#!/usr/bin/env python3
"""Run the six prespecified CamAgent paper queries through the real LLM agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from camagent.llm_agent import AgentRunPolicy, WildlifeAgent
from run_agent import build_llm, configure_runtime


EXPECTED_IDS = ("QUAL", "TEMP", "SPAT", "COMM", "MOVE", "INTEG")
CODE_PATHS = (
    "run_agent.py",
    "scripts/run_paper_query_experiments.py",
    "src/camagent/base_tools.py",
    "src/camagent/llm_agent.py",
    "src/camagent/paths.py",
    "src/camagent/tool_registry.py",
    "src/camagent/agent_router.py",
    "src/camagent/quality_tools.py",
    "src/camagent/temporal_tools.py",
    "src/camagent/spatial_tools.py",
    "src/camagent/movement_tools.py",
    "src/camagent/visualization_tools.py",
)
UNSAFE_PUBLIC_SPATIAL_FIGURE_TOOLS = {
    "plot_species_distribution_map",
    "plot_local_hotspot_map",
    "plot_multi_species_ecology_map",
    "plot_transition_corridor_map",
}
OCCUPANCY_TOOLS = {
    "build_occupancy_summary",
    "build_detection_history_matrix",
    "fit_single_season_occupancy_model",
    "plot_occupancy_model_forest",
    "plot_occupancy_bootstrap_density",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _configured_endpoint_host() -> str | None:
    provider = os.getenv("LLM_PROVIDER", "").strip().casefold()
    if provider == "deepseek":
        endpoint = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    elif provider in {"openai", "openai_compatible"}:
        endpoint = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com"
        )
    else:
        endpoint = (
            os.getenv("AZURE_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or "https://<your-azure-endpoint>"
        )
    return urlparse(endpoint).hostname


def _code_manifest() -> dict:
    files = []
    combined = hashlib.sha256()
    for relative in CODE_PATHS:
        path = ROOT / relative
        digest = _sha256(path)
        files.append({"path": relative, "sha256": digest})
        combined.update(relative.encode("utf-8"))
        combined.update(digest.encode("ascii"))
    return {"files": files, "combined_sha256": combined.hexdigest()}


def _load_experiments(config_path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("The paper-query config must contain an experiments list.")
    ids = tuple(str(item.get("experiment_id", "")).upper() for item in experiments)
    if ids != EXPECTED_IDS:
        raise ValueError(
            f"Expected paper experiment order {EXPECTED_IDS}, found {ids}."
        )
    for item in experiments:
        if not str(item.get("question", "")).strip():
            raise ValueError(
                f"Experiment {item.get('experiment_id')} has an empty question."
            )
    return payload, experiments


def _validate_paper_trace(experiment_id: str, public_trace: dict) -> None:
    selected_tools = set(public_trace.get("selected_tools", []))
    if experiment_id != "SPAT":
        return
    unsafe = sorted(selected_tools & UNSAFE_PUBLIC_SPATIAL_FIGURE_TOOLS)
    if unsafe:
        raise RuntimeError(
            "SPAT selected exact-location public figure tools despite the "
            f"privacy-safe request: {unsafe}."
        )
    occupancy = sorted(selected_tools & OCCUPANCY_TOOLS)
    if occupancy:
        raise RuntimeError(
            "SPAT selected occupancy tools even though the prespecified question "
            f"asks only for detection hotspots: {occupancy}."
        )
    if "plot_species_distribution_static" not in selected_tools:
        raise RuntimeError(
            "SPAT did not select the privacy-safe single-species distribution figure."
        )
    if "build_hotspot_rank" not in selected_tools:
        raise RuntimeError(
            "SPAT did not select a focal-species detection-hotspot ranking tool."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit the six prespecified paper questions unchanged to CamAgent "
            "and archive privacy-sanitised public traces."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "paper_query_experiments.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "paper_query_experiments"),
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=EXPECTED_IDS,
        help="Run only the selected experiment ID; repeat to select multiple IDs.",
    )
    parser.add_argument("--max-replan-rounds", type=int, default=2)
    parser.add_argument("--max-tool-calls", type=int, default=32)
    parser.add_argument("--max-tool-retries", type=int, default=1)
    parser.add_argument(
        "--confirm-external-api-data-use",
        action="store_true",
        help=(
            "Confirm that the configured model endpoint may receive the natural-"
            "language questions and redacted aggregate tool results."
        ),
    )
    parser.add_argument(
        "--approved-endpoint-host",
        help="Exact approved hostname for the configured model endpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace traces from an earlier paper-query run.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Validate and reuse existing public traces instead of repeating their "
            "external model calls."
        ),
    )
    parser.add_argument(
        "--rerun",
        action="append",
        choices=EXPECTED_IDS,
        help=(
            "Repeat the selected experiment while reusing the other existing "
            "traces. Repeat this option to rerun multiple experiment IDs. "
            "Requires --resume-existing."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "paper_query_experiment_manifest.json"
    pending_manifest_path = (
        output_dir / "paper_query_experiment_manifest.pending.json"
    )
    summary_path = output_dir / "paper_query_experiment_summary.json"
    prior_manifest = {}
    if manifest_path.exists():
        try:
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prior_manifest = {}
    prior_experiments = {
        str(item.get("experiment_id") or "").upper(): item
        for item in prior_manifest.get("experiments", [])
        if item.get("experiment_id")
    }

    config_payload, experiments = _load_experiments(config_path)
    selected_ids = set(args.only or EXPECTED_IDS)
    rerun_ids = set(args.rerun or ())
    if rerun_ids and not args.resume_existing:
        raise SystemExit("--rerun requires --resume-existing.")
    if rerun_ids - selected_ids:
        raise SystemExit(
            "--rerun IDs must also be included in the selected experiment set."
        )
    experiments = [
        item
        for item in experiments
        if str(item["experiment_id"]).upper() in selected_ids
    ]

    reused_existing_ids = {
        str(item["experiment_id"]).upper()
        for item in experiments
        if args.resume_existing
        and str(item["experiment_id"]).upper() not in rerun_ids
        and (output_dir / f"{str(item['experiment_id']).upper()}_public_trace.json").exists()
    }
    requires_external_model = any(
        str(item["experiment_id"]).upper() not in reused_existing_ids
        for item in experiments
    )
    if requires_external_model:
        if not args.confirm_external_api_data_use:
            raise SystemExit(
                "Refusing external model calls without "
                "--confirm-external-api-data-use."
            )
        configure_runtime()
        endpoint_host = _configured_endpoint_host()
        approved_host = str(args.approved_endpoint_host or "").strip().casefold()
        if not approved_host:
            raise SystemExit("--approved-endpoint-host is required.")
        if str(endpoint_host or "").casefold() != approved_host:
            raise SystemExit(
                f"Configured endpoint host {endpoint_host!r} does not match the "
                f"approved host {approved_host!r}."
            )
    else:
        endpoint_host = prior_manifest.get("endpoint_host")

    policy = AgentRunPolicy(
        max_replan_rounds=args.max_replan_rounds,
        max_tool_calls=args.max_tool_calls,
        max_tool_retries=args.max_tool_retries,
    )
    llm = build_llm() if requires_external_model else None
    started_at = datetime.now(timezone.utc)
    manifest = {
        "schema_version": "1.0",
        "suite_id": config_payload.get("suite_id"),
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": None,
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "code": _code_manifest(),
        "endpoint_host": endpoint_host,
        "external_data_scope": "questions_and_redacted_aggregate_tool_results",
        "trace_detail": "public",
        "manifest_assembly_mode": (
            "assembled_from_completed_agent_traces"
            if reused_existing_ids
            else "live_agent_execution"
        ),
        "resumed_existing": bool(reused_existing_ids),
        "agent_policy": {
            "max_replan_rounds": policy.max_replan_rounds,
            "max_tool_calls": policy.max_tool_calls,
            "max_tool_retries": policy.max_tool_retries,
            "stop_on_repeated_plan": policy.stop_on_repeated_plan,
        },
        "experiments": [],
    }
    for field in (
        "review_artifact_postprocessing",
        "review_artifact_code_layout",
    ):
        if field in prior_manifest:
            manifest[field] = prior_manifest[field]
    # Keep the last completed paper manifest intact until the entire selected
    # rerun succeeds. A failed external model call must not replace previously
    # valid trace provenance with a top-level failed manifest.
    _write_json(pending_manifest_path, manifest)

    summaries = []
    try:
        for experiment in experiments:
            experiment_id = str(experiment["experiment_id"]).upper()
            question = str(experiment["question"])
            trace_path = output_dir / f"{experiment_id}_public_trace.json"
            force_rerun = experiment_id in rerun_ids
            reused_existing = False
            if trace_path.exists() and args.resume_existing and not force_rerun:
                public_trace = json.loads(trace_path.read_text(encoding="utf-8"))
                reused_existing = True
                if public_trace.get("question") != question:
                    raise RuntimeError(
                        f"{experiment_id} existing trace does not retain the "
                        "prespecified question unchanged."
                    )
            elif trace_path.exists() and not (args.overwrite or force_rerun):
                raise FileExistsError(
                    f"{trace_path} already exists; pass --resume-existing to reuse "
                    "it, --rerun to repeat a selected experiment, or --overwrite "
                    "to replace it."
                )
            else:
                agent = WildlifeAgent(
                    llm,
                    default_species="Siberian tiger",
                    run_policy=policy,
                )
                record = agent.run(question, experiment_id=experiment_id)
                if record.get("question") != question:
                    raise RuntimeError(
                        f"{experiment_id} did not retain the prespecified question unchanged."
                    )
                public_trace = agent.exportable_trace(record, detail="public")
                _write_json(trace_path, public_trace)
            _validate_paper_trace(experiment_id, public_trace)
            synthesis_mode = (
                public_trace.get("trace", {})
                .get("synthesis", {})
                .get("synthesis_mode")
            )
            goal_status = public_trace.get("autonomy", {}).get("goal_status")
            public_trace_sha256 = _sha256(trace_path)
            prior_experiment = prior_experiments.get(experiment_id, {})
            source_trace_sha256 = (
                prior_experiment.get("source_trace_sha256")
                if reused_existing
                else None
            ) or public_trace_sha256
            summary = {
                "experiment_id": experiment_id,
                "question": question,
                "trace_path": str(trace_path.relative_to(ROOT)),
                "trace_sha256": public_trace_sha256,
                "source_trace_sha256": source_trace_sha256,
                "trace_postprocessing_applied": (
                    source_trace_sha256 != public_trace_sha256
                ),
                "trace_source": (
                    "reused_existing" if reused_existing else "executed_in_this_suite"
                ),
                "trace_started_at_utc": public_trace.get("started_at_utc"),
                "trace_completed_at_utc": public_trace.get("completed_at_utc"),
                "trace_duration_seconds": public_trace.get("duration_seconds"),
                "selected_tools": public_trace.get("selected_tools", []),
                "goal_status": goal_status,
                "termination_reason": public_trace.get("autonomy", {}).get(
                    "termination_reason"
                ),
                "synthesis_mode": synthesis_mode,
                "runtime": public_trace.get("runtime"),
                "figure_count": len(public_trace.get("visualizations", [])),
            }
            manifest["experiments"].append(summary)
            summaries.append(summary)
            _write_json(pending_manifest_path, manifest)

            if goal_status != "completed":
                raise RuntimeError(
                    f"{experiment_id} finished with goal_status={goal_status!r}."
                )
            if synthesis_mode != "llm_synthesis":
                raise RuntimeError(
                    f"{experiment_id} used synthesis_mode={synthesis_mode!r}; "
                    "paper experiments must return successful tool evidence to the LLM."
                )
    except Exception:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        failed_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        failed_manifest_path = (
            output_dir
            / f"paper_query_experiment_manifest.failed_{failed_stamp}.json"
        )
        _write_json(failed_manifest_path, manifest)
        pending_manifest_path.unlink(missing_ok=True)
        raise

    completed_at = datetime.now(timezone.utc)
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = completed_at.isoformat()
    manifest["experiment_count"] = len(summaries)
    manifest["manifest_assembly_duration_seconds"] = round(
        (completed_at - started_at).total_seconds(),
        6,
    )
    _write_json(pending_manifest_path, manifest)
    pending_manifest_path.replace(manifest_path)
    _write_json(
        summary_path,
        {
            "schema_version": "1.0",
            "suite_id": manifest["suite_id"],
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "experiments": summaries,
        },
    )
    questions_path = output_dir / "QUESTIONS.md"
    questions_lines = [
        "# Prespecified natural-language questions",
        "",
        "These questions are submitted unchanged to the common CamAgent entry point.",
        "",
    ]
    for experiment in config_payload["experiments"]:
        questions_lines.extend([
            f"## {experiment['experiment_id']}",
            "",
            str(experiment["question"]),
            "",
        ])
    questions_path.write_text("\n".join(questions_lines), encoding="utf-8")
    print(f"Completed {len(summaries)} paper query experiments.")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
