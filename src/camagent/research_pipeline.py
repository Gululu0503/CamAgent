"""Deterministic, manuscript-oriented camera-trap research pipeline.

The LLM agent is an interface layer. Scientific results are generated here by
explicitly configured Python functions so a reviewer can reproduce a run without
calling an LLM service.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from .agent_router import EcologicalToolset
from .paths import PROJECT_ROOT


PIPELINE_SCHEMA_VERSION = "1.0"


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _sha256(path: Path, hash_large_files: bool, large_file_bytes: int = 100_000_000) -> dict[str, Any]:
    stat = path.stat()
    result = {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size > large_file_bytes and not hash_large_files:
        result["sha256"] = None
        result["hash_status"] = "skipped_large_file"
        return result
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result["sha256"] = digest.hexdigest()
    result["hash_status"] = "complete"
    return result


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]
    config_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        config_path = Path(path).expanduser().resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("project_name", "input", "analysis", "output_dir"):
            if key not in raw:
                raise ValueError(f"Pipeline config is missing required key: {key}")
        if raw["input"].get("type") != "camtrap_dp":
            raise ValueError("The deterministic analysis runner currently requires input.type='camtrap_dp'.")
        if not raw["analysis"].get("species"):
            raise ValueError("analysis.species must contain at least one species.")
        return cls(raw=raw, config_path=config_path)

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        candidates = [self.config_path.parent / path, PROJECT_ROOT / path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return (PROJECT_ROOT / path).resolve()


class ResearchPipeline:
    """Execute traceable analyses in a fixed, dependency-ordered sequence."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.raw = config.raw
        self.analysis = self.raw["analysis"]
        self.random_seed = int(self.analysis.get("random_seed", 42))
        self.output_dir = config.resolve(self.raw["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tools = EcologicalToolset()
        self.warnings: list[dict[str, str]] = []

    def _clock_axis_label(self) -> str:
        """Return a provenance-aware axis label without inferring a timezone."""
        if (
            self.analysis.get("clock_time_verified", False)
            and self.analysis.get("clock_timezone")
            and self.analysis.get("clock_utc_offset")
        ):
            return (
                "Local civil time "
                f"(hour; {self.analysis['clock_timezone']}, "
                f"UTC{self.analysis['clock_utc_offset']})"
            )
        return "Recorded clock time (hour; UTC offset unverified)"

    def _package_dir(self) -> Path:
        value = self.config.resolve(self.raw["input"]["datapackage"])
        return value if value.is_dir() else value.parent

    def _environment(self) -> dict[str, Any]:
        packages = {}
        for name in ("numpy", "pandas", "scipy", "matplotlib", "chromadb", "langchain-core"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        return {
            "python": sys.version,
            "python_executable": Path(sys.executable).name,
            "platform": platform.platform(),
            "packages": packages,
        }

    def _input_manifest(self, package_dir: Path) -> list[dict[str, Any]]:
        hash_large = bool(self.raw.get("reproducibility", {}).get("hash_large_inputs", False))
        paths = [package_dir / name for name in (
            "datapackage.json", "deployments.csv", "media.csv", "observations.csv",
            "agent_independent_events.csv", "conversion_report.json",
            "coordinate_resolution_audit.csv",
            "coordinate_resolution_audit_manifest.json",
        )]
        study_area_map = self.raw.get("manuscript_assets", {}).get("study_area_map")
        if study_area_map:
            paths.append(self.config.resolve(study_area_map))
        manifest = []
        for path in paths:
            if not path.exists():
                continue
            item = _sha256(path, hash_large_files=hash_large)
            item["path"] = path.name
            item["role"] = "study_area_map" if study_area_map and path == self.config.resolve(study_area_map) else "analysis_input"
            manifest.append(item)
        return manifest

    def _code_manifest(self) -> dict[str, Any]:
        candidates = [
            *(PROJECT_ROOT / "src" / "camagent").glob("*.py"),
            *(PROJECT_ROOT / "scripts").glob("*.py"),
            PROJECT_ROOT / "run_agent.py",
            PROJECT_ROOT / "pyproject.toml",
        ]
        relative_paths = sorted({str(path.relative_to(PROJECT_ROOT)) for path in candidates if path.exists()})
        files = []
        for relative_path in relative_paths:
            path = PROJECT_ROOT / relative_path
            if path.exists():
                item = _sha256(path, hash_large_files=True)
                item["path"] = relative_path
                files.append(item)
        return {
            "schema_version": "1.0",
            "file_count": len(files),
            "files": files,
            "note": "These hashes identify the packaged analysis, agent, evaluation, and command-line code available to this run.",
        }

    def _dataset_snapshot(self, package_dir: Path) -> dict[str, Any]:
        report = self.tools._load_camtrap_conversion_report()
        inventory = self.tools.summarize_species_inventory(top_k=1000)
        deployments = self.tools._load_camtrap_deployments_df()
        species_counts = {
            row["species"]: int(row["record_count"])
            for row in inventory.get("species_inventory", [])
        }
        package = json.loads((package_dir / "datapackage.json").read_text(encoding="utf-8"))
        licenses = package.get("licenses", []) if isinstance(package, dict) else []
        return {
            "package_id": package_dir.name,
            "validation": self.tools._validate_camtrap_package(package_dir),
            "media_record_count": inventory.get("total_record_count"),
            "independent_event_count": report.get("independent_events"),
            "species_count": inventory.get("species_count"),
            "species_record_counts": species_counts,
            "date_start": inventory.get("date_start"),
            "date_end": inventory.get("date_end"),
            "deployment_count": int(len(deployments)),
            "camera_count": int(deployments["cameraID"].nunique()) if "cameraID" in deployments else 0,
            "camera_coordinate_conflict_count": report.get("camera_coordinate_conflict_count"),
            "coordinate_resolution_method": report.get(
                "coordinate_resolution_method",
                "median_per_camera_during_conversion",
            ),
            "event_gap_minutes_cached": report.get("filters", {}).get("min_gap_minutes"),
            "conversion_filters": report.get("filters", {}),
            "deployment_effort_source": (
                "detection_span_proxy"
                if "deploymentComments" in deployments
                and deployments["deploymentComments"].astype(str).str.contains("approximated", case=False, na=False).any()
                else "documented_deployment_intervals"
            ),
            "license_status": report.get(
                "license_status",
                "declared" if licenses else "requires_rights_holder_confirmation",
            ),
            "declared_licenses": licenses,
        }

    def _manuscript_consistency(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        expected = self.raw.get("manuscript_expectations", {})
        issues = []
        expected_counts = expected.get("species_record_counts", {})
        actual_counts = snapshot.get("species_record_counts", {})
        for species, expected_count in expected_counts.items():
            actual = actual_counts.get(species)
            if actual != int(expected_count):
                issues.append({
                    "field": f"species_record_counts.{species}",
                    "manuscript": int(expected_count),
                    "data": actual,
                    "severity": "error",
                })
        expected_species = set(expected.get("species", []))
        actual_species = set(actual_counts)
        if expected_species and expected_species != actual_species:
            issues.append({
                "field": "species_set",
                "manuscript": sorted(expected_species),
                "data": sorted(actual_species),
                "missing_from_data": sorted(expected_species - actual_species),
                "not_declared_in_manuscript": sorted(actual_species - expected_species),
                "severity": "error",
            })
        for field in ("date_start", "date_end", "camera_count", "deployment_count"):
            if field in expected and str(snapshot.get(field))[:10] != str(expected[field])[:10]:
                issues.append({
                    "field": field,
                    "manuscript": expected[field],
                    "data": snapshot.get(field),
                    "severity": "error",
                })
        return {"status": "failed" if issues else "passed", "issue_count": len(issues), "issues": issues}

    def _quality_audit(self, species: list[str]) -> dict[str, Any]:
        records = self.tools._load_camtrap_records_df()
        duplicate_mask = records.duplicated(["cameraID", "timestamp", "species"], keep=False)
        duplicate_groups = records.loc[duplicate_mask].groupby(
            ["cameraID", "timestamp", "species"],
            dropna=False,
            sort=False,
        ).agg(
            record_count=("record_id", "size"),
            unique_media_count=("mediaID", "nunique"),
            unique_filename_count=("filename", "nunique"),
            independent_event_count=("eventID", "nunique"),
        ).reset_index()
        if duplicate_groups.empty:
            duplicate_resolution = []
            review_group_count = 0
        else:
            duplicate_groups["duplicate_class"] = duplicate_groups["unique_media_count"].map(
                lambda value: (
                    "single_media_multiple_annotations"
                    if int(value) <= 1 else
                    "multiple_media_same_camera_clock_species"
                )
            )
            duplicate_groups["analysis_resolution"] = duplicate_groups["independent_event_count"].map(
                lambda value: (
                    "collapsed_to_one_independent_event"
                    if int(value) == 1 else
                    "review_required_multiple_event_ids"
                )
            )
            review_group_count = int(
                (duplicate_groups["analysis_resolution"] == "review_required_multiple_event_ids").sum()
            )
            duplicate_resolution = []
            for (duplicate_class, resolution), group in duplicate_groups.groupby(
                ["duplicate_class", "analysis_resolution"],
                sort=True,
            ):
                duplicate_resolution.append({
                    "duplicate_class": duplicate_class,
                    "analysis_resolution": resolution,
                    "group_count": int(len(group)),
                    "record_count": int(group["record_count"].sum()),
                    "excess_record_count": int((group["record_count"] - 1).sum()),
                })
        report = self.tools._load_camtrap_conversion_report()
        package_dir = Path(self.tools._camtrap_package_dir)
        coordinate_audit_path = package_dir / "coordinate_resolution_audit.csv"
        coordinate_manifest_path = package_dir / "coordinate_resolution_audit_manifest.json"
        coordinate_manifest = {}
        coordinate_audit_valid = False
        coordinate_audit_validation_note = "coordinate audit or manifest is missing"
        coordinate_span_summary = {
            "conflict_span_median_km": None,
            "conflict_span_max_km": None,
            "conflict_count_over_0_1_km": None,
            "conflict_count_over_1_km": None,
            "conflict_count_over_10_km": None,
        }
        if coordinate_audit_path.exists() and coordinate_manifest_path.exists():
            try:
                coordinate_manifest = json.loads(
                    coordinate_manifest_path.read_text(encoding="utf-8")
                )
                current_audit_sha = _sha256(
                    coordinate_audit_path,
                    hash_large_files=True,
                )["sha256"]
                coordinate_audit_valid = bool(
                    coordinate_manifest.get("status") == "completed"
                    and coordinate_manifest.get("audit_sha256") == current_audit_sha
                    and int(coordinate_manifest.get("coordinate_conflict_camera_count", -1))
                    == int(report.get("camera_coordinate_conflict_count", -2))
                )
                coordinate_audit_validation_note = (
                    "manifest status, audit hash, and conflict count agree"
                    if coordinate_audit_valid
                    else "manifest status, audit hash, or conflict count does not agree"
                )
                coordinate_rows = pd.read_csv(coordinate_audit_path)
                conflict_rows = coordinate_rows[
                    coordinate_rows["coordinate_conflict"].astype(bool)
                ].copy()
                spans = pd.to_numeric(
                    conflict_rows["maximum_coordinate_span_km"],
                    errors="coerce",
                ).dropna()
                if not spans.empty:
                    coordinate_span_summary = {
                        "conflict_span_median_km": round(float(spans.median()), 6),
                        "conflict_span_max_km": round(float(spans.max()), 6),
                        "conflict_count_over_0_1_km": int((spans > 0.1).sum()),
                        "conflict_count_over_1_km": int((spans > 1).sum()),
                        "conflict_count_over_10_km": int((spans > 10).sum()),
                    }
            except Exception as exc:
                coordinate_audit_validation_note = f"coordinate audit validation failed: {type(exc).__name__}"
        return {
            "species_timestamp_quality": [self.tools.check_timestamp_quality(name) for name in species],
            "missing_values": {
                col: int(records[col].isna().sum() + records[col].astype(str).str.strip().eq("").sum())
                for col in ("cameraID", "timestamp", "species", "latitude", "longitude")
                if col in records
            },
            "duplicate_like_record_count": int(duplicate_mask.sum()),
            "duplicate_like_group_count": int(len(duplicate_groups)),
            "duplicate_resolution_summary": duplicate_resolution,
            "duplicate_groups_requiring_review": review_group_count,
            "coordinate_conflict_summary": {
                "source_camera_conflict_count": int(report.get("camera_coordinate_conflict_count") or 0),
                "package_resolution_method": report.get(
                    "coordinate_resolution_method",
                    "median_per_camera_during_conversion",
                ),
                "raw_coordinate_audit_available": coordinate_audit_path.exists(),
                "audit_manifest_available": coordinate_manifest_path.exists(),
                "audit_manifest_status": coordinate_manifest.get("status"),
                "audit_validation_note": coordinate_audit_validation_note,
                **coordinate_span_summary,
                "status": (
                    "archived_with_deterministic_median_resolution"
                    if coordinate_audit_valid
                    else "median_resolved_but_source_audit_not_archived_or_invalid"
                ),
                "interpretation": (
                    "The audit establishes provenance and the deterministic per-camera median used by "
                    "the package; it does not claim that a human adjudicated each conflicting source row."
                ),
            },
            "quality_score_policy": "No formal ecological conclusion is based on the heuristic camera quality score.",
        }

    def _release_readiness(
        self,
        snapshot: dict[str, Any],
        consistency: dict[str, Any],
        quality: dict[str, Any],
        input_manifest: list[dict[str, Any]],
    ) -> dict[str, Any]:
        clock_verification = self.analysis.get("clock_time_verification", {})
        clock_evidence_complete = bool(
            self.analysis.get("clock_time_verified", False)
            and self.analysis.get("clock_timezone")
            and self.analysis.get("clock_utc_offset")
            and clock_verification.get("status") in {
                "author_confirmed",
                "data_custodian_confirmed",
                "deployment_metadata_verified",
            }
            and clock_verification.get("confirmed_by_role")
            and clock_verification.get("confirmed_at")
        )
        manuscript_scope = self.raw.get("manuscript_scope", {})
        effort_required = bool(
            manuscript_scope.get("reports_occupancy_inference", False)
            or manuscript_scope.get("reports_effort_corrected_rai", False)
        )
        effort_documented = snapshot.get("deployment_effort_source") == "documented_deployment_intervals"
        checks = [
            {
                "check": "manuscript_data_match",
                "passed": consistency.get("status") == "passed",
                "action_if_failed": "Reconcile every reported count and date with the versioned analytical package.",
            },
            {
                "check": "all_declared_inputs_hashed",
                "passed": bool(input_manifest) and all(
                    item.get("hash_status") == "complete" for item in input_manifest
                ),
                "action_if_failed": "Archive cryptographic hashes for all input tables and map assets.",
            },
            {
                "check": "field_timezone_verified",
                "required": True,
                "passed": clock_evidence_complete,
                "status": "verified" if clock_evidence_complete else "required_evidence_missing",
                "evidence": {
                    "timezone": self.analysis.get("clock_timezone"),
                    "utc_offset": self.analysis.get("clock_utc_offset"),
                    "verification": clock_verification,
                },
                "action_if_failed": "Verify the field UTC offset from deployment metadata or data custodians.",
            },
            {
                "check": "independent_effort_documented",
                "required": effort_required,
                "passed": effort_documented if effort_required else True,
                "status": (
                    "documented"
                    if effort_documented else
                    "not_applicable_to_declared_manuscript_scope"
                    if not effort_required else
                    "required_evidence_missing"
                ),
                "evidence": {
                    "deployment_effort_source": snapshot.get("deployment_effort_source"),
                    "reports_occupancy_inference": bool(
                        manuscript_scope.get("reports_occupancy_inference", False)
                    ),
                    "reports_effort_corrected_rai": bool(
                        manuscript_scope.get("reports_effort_corrected_rai", False)
                    ),
                },
                "action_if_failed": (
                    "Add deployment/service logs before reporting occupancy inference or effort-corrected RAI. "
                    "No logs are required when those claims are explicitly outside the manuscript scope."
                ),
            },
            {
                "check": "coordinate_conflict_audit_archived",
                "required": True,
                "passed": (
                    quality.get("coordinate_conflict_summary", {}).get("status")
                    == "archived_with_deterministic_median_resolution"
                ),
                "status": quality.get("coordinate_conflict_summary", {}).get("status"),
                "action_if_failed": "Regenerate the CamtrapDP package with the coordinate-resolution audit.",
            },
            {
                "check": "duplicate_event_resolution_complete",
                "passed": int(quality.get("duplicate_groups_requiring_review", 0)) == 0,
                "action_if_failed": "Adjudicate groups mapped to more than one independent-event ID.",
            },
            {
                "check": "data_and_media_licenses_declared",
                "passed": snapshot.get("license_status") == "declared",
                "action_if_failed": "Confirm data and media licences with the rights holder.",
            },
        ]
        failed = [item["check"] for item in checks if not item["passed"]]
        return {
            "schema_version": "1.0",
            "status": "ready" if not failed else "blocked_pending_human_or_source_metadata",
            "checks": checks,
            "failed_checks": failed,
            "manuscript_scope": manuscript_scope,
            "note": (
                "A failed required check is a visible release blocker. Evidence that is not required for "
                "the declared manuscript scope is recorded as not applicable rather than silently inferred."
            ),
        }

    def _event_sensitivity(self, species: list[str]) -> list[dict[str, Any]]:
        gaps = self.analysis.get("event_gap_sensitivity_minutes") or [self.analysis.get("event_gap_minutes", 30)]
        rows = []
        for gap in sorted({float(value) for value in gaps}):
            for name in species:
                result = self.tools.build_independent_events(name, min_gap_minutes=gap)
                rows.append({
                    "species": name,
                    "min_gap_minutes": gap,
                    "source_record_count": result.get("source_record_count"),
                    "independent_event_count": result.get("event_count"),
                    "event_source": result.get("event_source"),
                })
        pd.DataFrame(rows).to_csv(self.tables_dir / "event_gap_sensitivity.csv", index=False)
        return rows

    def _temporal_bandwidth_sensitivity(
        self,
        pairs: list[list[str]],
        min_gap_minutes: float,
        temporal_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        bandwidths = temporal_cfg.get("sensitivity_smooth_sigma") or [temporal_cfg.get("smooth_sigma", 1.25)]
        rows = []
        for smooth_sigma in sorted({float(value) for value in bandwidths}):
            for species_a, species_b in pairs:
                result = self.tools.estimate_activity_overlap(
                    species_a=species_a,
                    species_b=species_b,
                    min_gap_minutes=min_gap_minutes,
                    smooth_sigma=smooth_sigma,
                    bootstrap_iters=0,
                    bootstrap_interval_method=temporal_cfg.get(
                        "bootstrap_interval_method",
                        "percentile",
                    ),
                    bootstrap_cluster_strategy=temporal_cfg.get(
                        "bootstrap_cluster_strategy",
                        "joint_camera_network",
                    ),
                    permutation_iters=0,
                    random_seed=self.random_seed,
                )
                rows.append({
                    "species_a": result.get("species_a", species_a),
                    "species_b": result.get("species_b", species_b),
                    "min_gap_minutes": min_gap_minutes,
                    "smooth_sigma_hours": smooth_sigma,
                    "delta_hat": result.get("delta_hat"),
                    "event_count_a": result.get("independent_event_count_a"),
                    "event_count_b": result.get("independent_event_count_b"),
                })
        pd.DataFrame(rows).to_csv(self.tables_dir / "temporal_bandwidth_sensitivity.csv", index=False)
        return rows

    def _write_table(self, filename: str, rows: list[dict[str, Any]], columns: list[str] | None = None) -> dict[str, Any]:
        frame = pd.DataFrame(rows)
        if frame.empty and columns:
            frame = pd.DataFrame(columns=columns)
        path = self.tables_dir / filename
        frame.to_csv(path, index=False)
        item = {
            "path": str(path.relative_to(self.output_dir)),
            "row_count": int(len(frame)),
            "columns": list(frame.columns),
        }
        item["sha256"] = _sha256(path, hash_large_files=True)["sha256"]
        return item

    def _export_source_tables(
        self,
        results: dict[str, Any],
        input_manifest: list[dict[str, Any]],
    ) -> dict[str, Any]:
        tables = []
        dataset = results["dataset"]
        tables.append(self._write_table("dataset_summary.csv", [{
            "package_id": dataset.get("package_id"),
            "media_record_count": dataset.get("media_record_count"),
            "independent_event_count": dataset.get("independent_event_count"),
            "species_count": dataset.get("species_count"),
            "date_start": dataset.get("date_start"),
            "date_end": dataset.get("date_end"),
            "deployment_count": dataset.get("deployment_count"),
            "camera_count": dataset.get("camera_count"),
            "deployment_effort_source": dataset.get("deployment_effort_source"),
            "license_status": dataset.get("license_status"),
            "event_gap_minutes_cached": dataset.get("event_gap_minutes_cached"),
        }]))
        tables.append(self._write_table(
            "manuscript_consistency_issues.csv",
            results["manuscript_consistency"].get("issues", []),
            columns=["field", "manuscript", "data", "severity"],
        ))
        tables.append(self._write_table(
            "timestamp_quality.csv",
            results["quality"].get("species_timestamp_quality", []),
        ))
        tables.append(self._write_table("quality_summary.csv", [{
            "duplicate_like_record_count": results["quality"].get("duplicate_like_record_count"),
            "duplicate_like_group_count": results["quality"].get("duplicate_like_group_count"),
            "duplicate_groups_requiring_review": results["quality"].get("duplicate_groups_requiring_review"),
            "coordinate_conflict_count": results["quality"].get("coordinate_conflict_summary", {}).get("source_camera_conflict_count"),
            "coordinate_conflict_status": results["quality"].get("coordinate_conflict_summary", {}).get("status"),
            **{f"missing_{key}": value for key, value in results["quality"].get("missing_values", {}).items()},
        }]))
        tables.append(self._write_table(
            "duplicate_resolution_summary.csv",
            results["quality"].get("duplicate_resolution_summary", []),
            columns=[
                "duplicate_class",
                "analysis_resolution",
                "group_count",
                "record_count",
                "excess_record_count",
            ],
        ))
        tables.append(self._write_table(
            "coordinate_conflict_summary.csv",
            [results["quality"].get("coordinate_conflict_summary", {})],
        ))
        tables.append(self._write_table(
            "release_readiness.csv",
            results.get("release_readiness", {}).get("checks", []),
        ))
        tables.append(self._write_table(
            "event_gap_sensitivity.csv",
            results.get("event_gap_sensitivity", []),
        ))

        community = results.get("community", {})
        community_rows = []
        for row in community.get("top_species", []):
            community_rows.append({
                **row,
                "effort_basis": community.get("effort_basis"),
                "rai_inference_status": community.get("rai_inference_status"),
            })
        tables.append(self._write_table("community_species_summary.csv", community_rows))

        cooccurrence = results.get("spatial_cooccurrence", {})
        cooccurrence_rows = []
        for row in cooccurrence.get("matrix", []):
            for value in row.get("values", []):
                cooccurrence_rows.append({
                    "species_a": row.get("species"),
                    "species_b": value.get("species"),
                    "camera_count_a": row.get("camera_count"),
                    "shared_camera_count": value.get("shared_camera_count"),
                    "jaccard": value.get("jaccard"),
                    "min_gap_minutes": cooccurrence.get("min_gap_minutes"),
                })
        tables.append(self._write_table("camera_cooccurrence_matrix.csv", cooccurrence_rows))
        tables.append(self._write_table(
            "camera_cooccurrence_ranked_pairs.csv",
            cooccurrence.get("top_cooccurring_pairs", []),
        ))

        temporal_summary = []
        temporal_density = []
        for result in results.get("temporal_overlap", []):
            temporal_summary.append({
                "species_a": result.get("species_a"),
                "species_b": result.get("species_b"),
                "min_gap_minutes": result.get("min_gap_minutes"),
                "smooth_sigma_hours": result.get("smooth_sigma"),
                "delta_hat": result.get("delta_hat"),
                "delta_ci_low": result.get("delta_ci_low"),
                "delta_ci_high": result.get("delta_ci_high"),
                "event_count_a": result.get("independent_event_count_a"),
                "event_count_b": result.get("independent_event_count_b"),
                "bootstrap_iters": result.get("bootstrap_iters"),
                "bootstrap_unit": result.get("bootstrap_unit"),
                "bootstrap_interval_method": result.get("bootstrap_interval_method"),
                "bootstrap_cluster_strategy": result.get(
                    "bootstrap_cluster_strategy"
                ),
                "bootstrap_camera_count_common": result.get(
                    "bootstrap_camera_count_common"
                ),
                "bootstrap_camera_count_union": result.get(
                    "bootstrap_camera_count_union"
                ),
                "interval_calibration_status": result.get("interval_calibration_status"),
                "permutation_iters": result.get("permutation_iters"),
                "clock_time_ratio_a": result.get("clock_time_ratio_a"),
                "clock_time_ratio_b": result.get("clock_time_ratio_b"),
                "clock_time_basis": self.analysis.get("clock_time_basis"),
                "clock_time_verified": bool(self.analysis.get("clock_time_verified", False)),
                "clock_timezone": self.analysis.get("clock_timezone"),
                "clock_utc_offset": self.analysis.get("clock_utc_offset"),
                "clock_time_verification": self.analysis.get("clock_time_verification", {}),
            })
            for hour, density_a, density_b, overlap in zip(
                result.get("hour_grid", []),
                result.get("density_a", []),
                result.get("density_b", []),
                result.get("overlap_density", []),
            ):
                temporal_density.append({
                    "species_a": result.get("species_a"),
                    "species_b": result.get("species_b"),
                    "hour": hour,
                    "density_a": density_a,
                    "density_b": density_b,
                    "overlap_density": overlap,
                })
        tables.append(self._write_table("temporal_overlap_summary.csv", temporal_summary))
        tables.append(self._write_table("temporal_overlap_density.csv", temporal_density))
        tables.append(self._write_table(
            "temporal_bandwidth_sensitivity.csv",
            results.get("temporal_bandwidth_sensitivity", []),
        ))
        tables.append(self._write_table(
            "occupancy_guardrail.csv",
            results.get("occupancy", []),
        ))
        tables.append(self._write_table(
            "camera_adjacency_hypotheses.csv",
            results.get("camera_transition_hypotheses", {}).get("rows", []),
            columns=["species", "from_camera", "to_camera", "distance_km", "time_gap_hours", "bearing_degrees"],
        ))
        tables.append(self._write_table("input_manifest.csv", input_manifest))

        manifest = {
            "schema_version": "1.0",
            "table_count": len(tables),
            "tables": tables,
            "note": (
                "These machine-readable tables are generated by the same versioned analysis run as the figures. "
                "No table contains raw media, credentials, or exact threatened-species camera coordinates."
            ),
        }
        _write_json(self.output_dir / "source_data_manifest.json", manifest)
        return manifest

    def _save_figure(self, fig, stem: str) -> dict[str, str]:
        pdf_path = self.figures_dir / f"{stem}.pdf"
        png_path = self.figures_dir / f"{stem}.png"
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        return {
            "pdf": str(pdf_path.relative_to(self.output_dir)),
            "png": str(png_path.relative_to(self.output_dir)),
            "pdf_sha256": _sha256(pdf_path, hash_large_files=True)["sha256"],
            "png_sha256": _sha256(png_path, hash_large_files=True)["sha256"],
        }

    def _register_study_area_map(self, input_manifest: list[dict[str, Any]]) -> dict[str, Any] | None:
        configured = self.raw.get("manuscript_assets", {}).get("study_area_map")
        if not configured:
            return None
        source = self.config.resolve(configured)
        if not source.exists():
            self.warnings.append({
                "code": "study_area_map_missing",
                "message": f"Configured study-area map was not found: {source.name}",
            })
            return None
        destination = self.figures_dir / f"fig_study_area_camera_traps{source.suffix.lower()}"
        shutil.copy2(source, destination)
        source_item = next(
            (item for item in input_manifest if item.get("role") == "study_area_map"),
            None,
        )
        return {
            "figure_id": "study_area_camera_traps",
            "asset_type": "versioned_external_map",
            "path": str(destination.relative_to(self.output_dir)),
            "sha256": _sha256(destination, hash_large_files=True)["sha256"],
            "input_sha256": source_item.get("sha256") if source_item else None,
            "source_data": ["input:study_area_map"],
            "interpretation_boundary": (
                "Prepared study-area map supplied as a hashed manuscript input; the versioned analysis runner "
                "does not reconstruct its DEM or administrative boundary layers."
            ),
        }

    def _write_manuscript_assets(
        self,
        results: dict[str, Any],
        figure_manifest: dict[str, Any],
        duration_seconds: float,
    ) -> None:
        dataset = results["dataset"]
        values = {
            "CamMediaRecords": f"{int(dataset.get('media_record_count') or 0):,}",
            "CamIndependentEvents": f"{int(dataset.get('independent_event_count') or 0):,}",
            "CamCameraCount": f"{int(dataset.get('camera_count') or 0):,}",
            "CamDeploymentCount": f"{int(dataset.get('deployment_count') or 0):,}",
            "CamEventGapMinutes": f"{float(self.analysis.get('event_gap_minutes', 30)):g}",
            "CamPipelineSeconds": f"{duration_seconds:.1f}",
            "CamDuplicateLikeRecords": f"{int(results['quality'].get('duplicate_like_record_count') or 0):,}",
            "CamCoordinateConflictCameras": f"{int(dataset.get('camera_coordinate_conflict_count') or 0):,}",
        }
        pair_aliases = {
            ("Siberian tiger", "Sika deer"): "TigerSika",
            ("Siberian tiger", "Roe deer"): "TigerRoe",
            ("Siberian tiger", "Black bear"): "TigerBear",
            ("Siberian tiger", "Amur leopard"): "TigerLeopard",
        }
        for result in results.get("temporal_overlap", []):
            alias = pair_aliases.get((result.get("species_a"), result.get("species_b")))
            if not alias or result.get("delta_hat") is None:
                continue
            values[f"CamDelta{alias}"] = f"{float(result['delta_hat']):.4f}"
            values[f"CamDelta{alias}Low"] = f"{float(result['delta_ci_low']):.4f}"
            values[f"CamDelta{alias}High"] = f"{float(result['delta_ci_high']):.4f}"
        lines = [
            "% Auto-generated by camagent-pipeline; do not edit numerical values by hand.",
            *[
                rf"\providecommand{{\{command}}}{{{value}}}"
                for command, value in values.items()
            ],
            "",
        ]
        (self.output_dir / "manuscript_values.tex").write_text("\n".join(lines), encoding="utf-8")
        artifact_map = {
            "schema_version": "1.0",
            "latex_values_file": "manuscript_values.tex",
            "values": values,
            "figures": {
                item["figure_id"]: {
                    key: value
                    for key, value in item.items()
                    if key in {"path", "pdf", "png", "sha256", "pdf_sha256", "png_sha256"}
                }
                for item in figure_manifest.get("figures", [])
            },
            "note": "Use these paths and macros to prevent hand-copied manuscript values from drifting from the versioned analysis run.",
        }
        _write_json(self.output_dir / "manuscript_artifact_map.json", artifact_map)

    def _render_manuscript_figures(
        self,
        results: dict[str, Any],
        input_manifest: list[dict[str, Any]],
    ) -> dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_items = []
        study_area_item = self._register_study_area_map(input_manifest)
        if study_area_item:
            figure_items.append(study_area_item)

        community_frame = pd.DataFrame(results.get("community", {}).get("top_species", []))
        cooccurrence_rows = []
        for row in results.get("spatial_cooccurrence", {}).get("matrix", []):
            for value in row.get("values", []):
                cooccurrence_rows.append({
                    "species_a": row.get("species"),
                    "species_b": value.get("species"),
                    "jaccard": value.get("jaccard"),
                })
        cooccurrence_frame = pd.DataFrame(cooccurrence_rows)
        if not community_frame.empty or not cooccurrence_frame.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
            if not community_frame.empty:
                plotted = community_frame.sort_values("independent_event_count", ascending=True)
                axes[0].barh(
                    plotted["species"],
                    plotted["independent_event_count"],
                    color="#2F5D8C",
                )
                axes[0].set_xscale("log")
                axes[0].set_xlabel("Independent events (log scale)")
                axes[0].set_title("A  Community event composition")
                axes[0].grid(axis="x", alpha=0.2)
            else:
                axes[0].axis("off")
            if not cooccurrence_frame.empty:
                species_order = list(self.analysis.get("species", []))
                matrix = cooccurrence_frame.pivot(
                    index="species_a",
                    columns="species_b",
                    values="jaccard",
                ).reindex(index=species_order, columns=species_order)
                image = axes[1].imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="YlGnBu")
                axes[1].set_xticks(
                    range(len(species_order)),
                    species_order,
                    rotation=45,
                    ha="right",
                    fontsize=8,
                )
                axes[1].set_yticks(range(len(species_order)), species_order, fontsize=8)
                axes[1].set_title("B  Camera-level co-detection")
                colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
                colorbar.set_label("Jaccard index")
            else:
                axes[1].axis("off")
            fig.tight_layout()
            paths = self._save_figure(fig, "fig_empirical_community_overview")
            plt.close(fig)
            figure_items.append({
                "figure_id": "empirical_community_overview",
                **paths,
                "source_data": [
                    "tables/community_species_summary.csv",
                    "tables/camera_cooccurrence_matrix.csv",
                ],
                "interpretation_boundary": (
                    "Event counts are descriptive; camera-level Jaccard values do not establish "
                    "species interaction, abundance, or population density."
                ),
            })

        temporal = results.get("temporal_overlap", [])
        renderable_temporal = [
            result
            for result in temporal
            if result.get("delta_hat") is not None
            and result.get("hour_grid")
            and result.get("density_a")
            and result.get("density_b")
            and result.get("overlap_density")
        ]
        if renderable_temporal:
            clock_axis_label = self._clock_axis_label()
            ncols = 2
            nrows = (len(renderable_temporal) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.0 * nrows), sharex=True)
            axes = list(getattr(axes, "flat", [axes]))
            for ax, result in zip(axes, renderable_temporal):
                hours = result.get("hour_grid", [])
                density_a = result.get("density_a", [])
                density_b = result.get("density_b", [])
                overlap = result.get("overlap_density", [])
                ax.plot(hours, density_a, color="#2F5D8C", linewidth=2.0, label=result.get("species_a"))
                ax.plot(hours, density_b, color="#2E7D62", linewidth=1.8, linestyle="--", label=result.get("species_b"))
                ax.fill_between(hours, overlap, color="#9AA5B1", alpha=0.35, label="Overlap")
                ax.set_title(
                    f"{result.get('species_a')} vs {result.get('species_b')}\n"
                    f"Delta = {result.get('delta_hat'):.3f} "
                    f"({result.get('delta_ci_low'):.3f}-{result.get('delta_ci_high'):.3f})",
                    fontsize=10,
                )
                ax.set_xlim(0, 24)
                ax.set_xticks([0, 6, 12, 18, 24])
                ax.set_xlabel(clock_axis_label)
                ax.set_ylabel("Density")
                ax.grid(alpha=0.2)
                ax.legend(frameon=False, fontsize=8)
            for ax in axes[len(renderable_temporal):]:
                ax.axis("off")
            fig.suptitle("Pairwise diel activity overlap from independent events", fontsize=13)
            fig.tight_layout()
            paths = self._save_figure(fig, "fig_temporal_overlap_worked_example")
            plt.close(fig)
            figure_items.append({
                "figure_id": "temporal_overlap_worked_example",
                **paths,
                "source_data": [
                    "tables/temporal_overlap_summary.csv",
                    "tables/temporal_overlap_density.csv",
                ],
                "interpretation_boundary": "Diel similarity is not evidence of direct species interaction.",
            })

        event_frame = pd.DataFrame(results.get("event_gap_sensitivity", []))
        bandwidth_frame = pd.DataFrame(results.get("temporal_bandwidth_sensitivity", []))
        if not event_frame.empty or not bandwidth_frame.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
            if not event_frame.empty:
                for species_name, group in event_frame.groupby("species"):
                    group = group.sort_values("min_gap_minutes")
                    axes[0].plot(
                        group["min_gap_minutes"],
                        group["independent_event_count"],
                        marker="o",
                        linewidth=1.7,
                        label=species_name,
                    )
                axes[0].set_yscale("log")
                axes[0].set_xlabel("Independent-event gap (min)")
                axes[0].set_ylabel("Independent-event count (log scale)")
                axes[0].set_title("Event-definition sensitivity")
                axes[0].grid(alpha=0.2)
                axes[0].legend(frameon=False, fontsize=7)
            else:
                axes[0].axis("off")
            if not bandwidth_frame.empty:
                for (species_a, species_b), group in bandwidth_frame.groupby(["species_a", "species_b"]):
                    group = group.sort_values("smooth_sigma_hours")
                    axes[1].plot(
                        group["smooth_sigma_hours"],
                        group["delta_hat"],
                        marker="o",
                        linewidth=1.7,
                        label=f"{species_a} vs {species_b}",
                    )
                axes[1].set_xlabel("von Mises kernel scale (h)")
                axes[1].set_ylabel("Overlap coefficient (Delta)")
                axes[1].set_ylim(0, 1)
                axes[1].set_title("Temporal-bandwidth sensitivity")
                axes[1].grid(alpha=0.2)
                axes[1].legend(frameon=False, fontsize=7)
            else:
                axes[1].axis("off")
            fig.tight_layout()
            paths = self._save_figure(fig, "fig_sensitivity_analysis")
            plt.close(fig)
            figure_items.append({
                "figure_id": "sensitivity_analysis",
                **paths,
                "source_data": [
                    "tables/event_gap_sensitivity.csv",
                    "tables/temporal_bandwidth_sensitivity.csv",
                ],
                "interpretation_boundary": "Scenario sensitivity does not estimate the true event gap or optimal bandwidth.",
            })

        checks = {
            "Manuscript-data match": results.get("manuscript_consistency", {}).get("status") == "passed",
            "Observed clock values": all(
                item.get("reliability_status") == "passed"
                for item in results.get("quality", {}).get("species_timestamp_quality", [])
            ),
            "Field timezone verified": bool(self.analysis.get("clock_time_verified", False)),
            "Occupancy guard applied": all(
                item.get("model_status") != "fitted"
                for item in results.get("occupancy", [])
            ),
            "All input files hashed": all(item.get("hash_status") == "complete" for item in input_manifest),
        }
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        labels = list(checks)
        values = [1 if checks[label] else 0 for label in labels]
        colors = ["#2E7D62" if value else "#B44C43" for value in values]
        bars = ax.barh(labels, [1] * len(labels), color=colors)
        for bar, value in zip(bars, values):
            ax.text(0.5, bar.get_y() + bar.get_height() / 2, "PASS" if value else "REVIEW", ha="center", va="center", color="white", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title("Versioned-run reproducibility and guardrail audit")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        paths = self._save_figure(fig, "fig_reproducibility_guardrails")
        plt.close(fig)
        figure_items.append({
            "figure_id": "reproducibility_guardrails",
            **paths,
            "source_data": [
                "tables/manuscript_consistency_issues.csv",
                "tables/timestamp_quality.csv",
                "tables/occupancy_guardrail.csv",
                "tables/input_manifest.csv",
            ],
            "checks": checks,
        })

        manifest = {
            "schema_version": "2.0",
            "figure_count": len(figure_items),
            "figures": figure_items,
            "note": (
                "Analytical figures are generated from source-data tables written by the same versioned "
                "analysis run. Any externally prepared cartographic asset is copied from a hashed versioned input "
                "and explicitly labelled as such."
            ),
        }
        _write_json(self.output_dir / "figure_manifest.json", manifest)
        return manifest

    def run(self) -> dict[str, Any]:
        started = datetime.now(timezone.utc)
        package_dir = self._package_dir()
        backend = self.tools.set_camtrap_data_dir(
            package_dir,
            activate=True,
            persist_active=False,
            write_validation=False,
        )
        if backend["validation"].get("error_count", 0):
            raise ValueError("CamtrapDP validation failed; see conversion_validation.json in the package.")

        species = [self.tools._canonical_species_name(value) for value in self.analysis["species"]]
        gap = float(self.analysis.get("event_gap_minutes", 30))
        input_manifest = self._input_manifest(package_dir)
        snapshot = self._dataset_snapshot(package_dir)
        consistency = self._manuscript_consistency(snapshot)
        quality = self._quality_audit(species)
        release_readiness = self._release_readiness(
            snapshot,
            consistency,
            quality,
            input_manifest,
        )
        event_sensitivity = self._event_sensitivity(species)

        community = self.tools.summarize_community_composition(
            top_k=max(len(species), 1),
            min_gap_minutes=gap,
            abundance_metric="rai",
        )
        cooccurrence = self.tools.build_species_cooccurrence_matrix(
            species_list=species,
            top_k=len(species),
            min_gap_minutes=gap,
        )

        temporal_cfg = self.analysis.get("temporal_overlap", {})
        pairwise = []
        for pair in temporal_cfg.get("pairs", []):
            pairwise.append(self.tools.estimate_activity_overlap(
                species_a=pair[0],
                species_b=pair[1],
                min_gap_minutes=gap,
                smooth_sigma=float(temporal_cfg.get("smooth_sigma", 1.25)),
                bootstrap_iters=int(temporal_cfg.get("bootstrap_iters", 1000)),
                bootstrap_interval_method=temporal_cfg.get(
                    "bootstrap_interval_method",
                    "percentile",
                ),
                bootstrap_cluster_strategy=temporal_cfg.get(
                    "bootstrap_cluster_strategy",
                    "joint_camera_network",
                ),
                permutation_iters=int(temporal_cfg.get("permutation_iters", 0)),
                random_seed=self.random_seed,
            ))
        bandwidth_sensitivity = self._temporal_bandwidth_sensitivity(
            temporal_cfg.get("pairs", []),
            gap,
            temporal_cfg,
        )

        occupancy_cfg = self.analysis.get("occupancy", {})
        occupancy = []
        if occupancy_cfg.get("enabled", True):
            for name in occupancy_cfg.get("species", species):
                occupancy.append(self.tools.fit_single_season_occupancy_model(
                    species=name,
                    occasion_days=int(occupancy_cfg.get("occasion_days", 14)),
                    min_gap_minutes=gap,
                    min_active_occasions=int(occupancy_cfg.get("min_active_occasions", 2)),
                    bootstrap_iters=int(occupancy_cfg.get("bootstrap_iters", 1000)),
                    random_seed=self.random_seed,
                    allow_inferred_effort=bool(occupancy_cfg.get("allow_inferred_effort", False)),
                ))

        movement_cfg = self.analysis.get("camera_transition_hypotheses", {})
        transitions = []
        if movement_cfg.get("enabled", False):
            transitions = self.tools.analyze_movement_steps(
                species=movement_cfg.get("species", species[0]),
                max_time_gap_hours=float(movement_cfg.get("max_time_gap_hours", 72)),
                max_distance_km=float(movement_cfg.get("max_distance_km", 100)),
                min_gap_minutes=gap,
            )

        results = {
            "dataset": snapshot,
            "manuscript_consistency": consistency,
            "quality": quality,
            "release_readiness": release_readiness,
            "event_gap_sensitivity": event_sensitivity,
            "community": community,
            "spatial_cooccurrence": cooccurrence,
            "temporal_overlap": pairwise,
            "temporal_bandwidth_sensitivity": bandwidth_sensitivity,
            "occupancy": occupancy,
            "camera_transition_hypotheses": {
                "count": len(transitions),
                "rows": transitions,
                "interpretation_limit": "Camera-detection adjacency only; no individual identity or continuous trajectory is inferred.",
            },
        }
        _write_json(self.output_dir / "analysis_results.json", results)
        _write_json(self.output_dir / "dataset_audit.json", {"dataset": snapshot, "manuscript_consistency": consistency, "quality": quality})
        _write_json(self.output_dir / "release_readiness.json", release_readiness)
        _write_json(self.output_dir / "environment.json", self._environment())

        config_snapshot = self.output_dir / "pipeline_config.snapshot.json"
        _write_json(config_snapshot, self.raw)
        code_manifest = self._code_manifest()
        _write_json(self.output_dir / "code_manifest.json", code_manifest)
        source_data_manifest = self._export_source_tables(results, input_manifest)
        figure_manifest = self._render_manuscript_figures(results, input_manifest)
        completed = datetime.now(timezone.utc)
        duration_seconds = round((completed - started).total_seconds(), 3)
        self._write_manuscript_assets(results, figure_manifest, duration_seconds)
        output_paths = sorted(path for path in self.output_dir.rglob("*") if path.is_file() and path.name != "run_manifest.json")
        output_checksums = []
        for path in output_paths:
            item = _sha256(path, hash_large_files=True)
            item["path"] = str(path.relative_to(self.output_dir))
            output_checksums.append(item)
        manifest = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "experiment_id": "PIPELINE_SMOKE_TEST",
            "experiment_name": "Versioned Pipeline Smoke Test",
            "project_name": self.raw["project_name"],
            "status": "completed_with_manuscript_mismatches" if consistency["issue_count"] else "completed",
            "started_at_utc": started.isoformat(),
            "completed_at_utc": completed.isoformat(),
            "duration_seconds": duration_seconds,
            "random_seed": self.random_seed,
            "event_gap_minutes": gap,
            "config_sha256": _sha256(config_snapshot, True)["sha256"],
            "inputs": input_manifest,
            "source_data_table_count": source_data_manifest["table_count"],
            "figure_count": figure_manifest["figure_count"],
            "code_file_count": code_manifest["file_count"],
            "clock_time_basis": self.analysis.get("clock_time_basis"),
            "clock_time_verified": bool(self.analysis.get("clock_time_verified", False)),
            "clock_timezone": self.analysis.get("clock_timezone"),
            "clock_utc_offset": self.analysis.get("clock_utc_offset"),
            "clock_time_verification": self.analysis.get("clock_time_verification", {}),
            "temporal_overlap_uncertainty": {
                "bootstrap_interval_method": temporal_cfg.get(
                    "bootstrap_interval_method", "percentile"
                ),
                "bootstrap_cluster_strategy": temporal_cfg.get(
                    "bootstrap_cluster_strategy", "joint_camera_network"
                ),
                "bootstrap_iters": int(
                    temporal_cfg.get("bootstrap_iters", 1000)
                ),
                "calibration_status": (
                    "requires_data_generating_process_specific_simulation"
                ),
            },
            "manuscript_scope": self.raw.get("manuscript_scope", {}),
            "release_readiness_status": release_readiness["status"],
            "release_readiness_failed_checks": release_readiness["failed_checks"],
            "warnings": self.warnings,
            "outputs": sorted(
                {str(path.relative_to(self.output_dir)) for path in self.output_dir.rglob("*") if path.is_file()}
                | {"run_manifest.json"}
            ),
            "output_checksums": output_checksums,
            "scientific_guardrails": [
                "Hourly analyses exclude timestamps without an observed clock time.",
                "Independent events are recomputed when the requested gap differs from the package cache.",
                "Occupancy is not fitted by default when effort is inferred from detections.",
                "Transition links are camera-detection adjacency hypotheses, not individual movement tracks.",
                "RAI is independent events per 100 camera-days and is not population density.",
                "Pairwise temporal-overlap uncertainty resamples the shared camera network jointly across species.",
                "Externally prepared cartographic assets are hashed inputs, not silently regenerated figures.",
            ],
        }
        _write_json(self.output_dir / "run_manifest.json", manifest)
        return manifest


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    return ResearchPipeline(PipelineConfig.load(config_path)).run()
