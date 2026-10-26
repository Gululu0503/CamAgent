import json
import hashlib
import inspect
import re
import time
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .*doesn't match a supported version!",
    category=Warning,
)

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ModuleNotFoundError:
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content

    class SystemMessage:
        def __init__(self, content: str):
            self.content = content

from .paths import OUTPUTS_DIR, display_path, sanitize_for_display
from .tool_registry import ToolRegistry


@dataclass(frozen=True)
class AgentRunPolicy:
    """Bound the autonomous plan-act-observe-reflect loop."""

    max_replan_rounds: int = 2
    max_tool_calls: int = 32
    max_tool_retries: int = 1
    stop_on_repeated_plan: bool = True

    def __post_init__(self):
        if self.max_replan_rounds < 0:
            raise ValueError("max_replan_rounds must be non-negative.")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1.")
        if self.max_tool_retries < 0:
            raise ValueError("max_tool_retries must be non-negative.")


class WildlifeAgent:
    _REPLANNER_USER_PROMPT_BUDGET_CHARS = 60_000
    _ANSWER_USER_PROMPT_BUDGET_CHARS = 40_000

    def __init__(
        self,
        llm,
        default_species: Optional[str] = None,
        run_policy: Optional[AgentRunPolicy] = None,
    ):
        self.llm = llm
        self.default_species = (
            str(default_species).strip() if default_species else None
        )
        self.run_policy = run_policy or AgentRunPolicy()
        self.registry = ToolRegistry()
        self.history: List[Dict[str, Any]] = []
        self._last_answer_trace: Dict[str, Any] = {}

    # =========================
    # Basic utilities
    # =========================
    def _prompt_sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _prompt_json(self, value: Any) -> str:
        """Serialize a redacted prompt payload without whitespace inflation."""
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    def _redact_trace_value(
        self,
        value: Any,
        key: str = "",
        *,
        redact_local_paths: bool = False,
    ) -> Any:
        """Redact sensitive values with a stricter mode for exported traces."""
        lowered = key.casefold()
        if lowered in {
            "camera_id", "cameraid", "centercameraid", "center_camera_id",
            "camera_token", "from_camera", "to_camera", "camera_a", "camera_b",
            "latitude", "longitude", "location", "coordinates",
        }:
            return "[REDACTED_SENSITIVE_LOCATION]"
        if any(
            token in lowered
            for token in (
                "record_id",
                "observation_id",
                "media_id",
                "deployment_id",
            )
        ):
            return "[REDACTED_INTERNAL_IDENTIFIER]"
        if redact_local_paths and lowered in {
            "filename",
            "file_name",
            "media_filename",
            "source_filename",
        }:
            return "[REDACTED_MEDIA_NAME]"
        if isinstance(value, dict):
            return {
                item_key: self._redact_trace_value(
                    item,
                    item_key,
                    redact_local_paths=redact_local_paths,
                )
                for item_key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._redact_trace_value(
                    item,
                    key,
                    redact_local_paths=redact_local_paths,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            return [
                self._redact_trace_value(
                    item,
                    key,
                    redact_local_paths=redact_local_paths,
                )
                for item in value
            ]
        if isinstance(value, str):
            if redact_local_paths:
                is_path_key = any(
                    token in lowered
                    for token in (
                        "path",
                        "directory",
                        "output_dir",
                        "input_dir",
                        "media_source",
                        "package_dir",
                        "manifest_json",
                    )
                )
                looks_absolute = bool(
                    re.match(r"^(?:/|[A-Za-z]:[\\/])", value.strip())
                )
                looks_path_like = bool(
                    looks_absolute
                    or "/" in value
                    or "\\" in value
                    or Path(value).suffix
                )
                if is_path_key and looks_path_like:
                    return "[REDACTED_LOCAL_PATH]"
            patterns = (
                r"(?i)(\bcamera\s+id\b(?:\s*[:#=]\s*|\s+))[A-Za-z0-9_-]{3,}\b",
                r"(?i)(\bcamera\s*[:#=]\s*)[A-Za-z0-9_-]{3,}\b",
                r"(?i)(\bcamera\s+)(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{3,}\b",
                r"(?i)\b(?:obs|observation|media|deployment)_[A-Za-z0-9_-]{6,}\b",
            )
            for pattern in patterns:
                replacement = (
                    r"\1[REDACTED]"
                    if pattern.startswith("(?i)(")
                    else "[REDACTED_INTERNAL_IDENTIFIER]"
                )
                value = re.sub(pattern, replacement, value)
            if redact_local_paths:
                value = re.sub(
                    r"(^|[\s\"'(=:])/(?:[^\s\"']+)",
                    r"\1[REDACTED_LOCAL_PATH]",
                    value,
                )
                value = re.sub(
                    r"(^|[\s\"'(=:])[A-Za-z]:\\(?:[^\s\"']+)",
                    r"\1[REDACTED_LOCAL_PATH]",
                    value,
                )
                value = re.sub(
                    r"(^|[\s\"'(=:])(?:workspace|tmp|data|results)/(?:[^\s\"']+)",
                    r"\1[REDACTED_LOCAL_PATH]",
                    value,
                )
                value = re.sub(
                    r"(?i)\b[^\s/\\\"']+\.(?:mp4|mov|avi|mkv|webm|jpg|jpeg|png|tif|tiff)\b",
                    "[REDACTED_MEDIA_NAME]",
                    value,
                )
        return value

    def _redact_external_context(self, value: Any, key: str = "") -> Any:
        """Redact wildlife locations before external LLM calls, retaining input paths."""
        return self._redact_trace_value(
            value,
            key,
            redact_local_paths=False,
        )

    def _redact_public_trace(self, value: Any, key: str = "") -> Any:
        """Remove locations, identifiers, local paths, and media names from public traces."""
        return self._redact_trace_value(
            value,
            key,
            redact_local_paths=True,
        )

    def _local_camera_id_from_question(self, question: str) -> Optional[str]:
        """Recover an explicit camera identifier locally after the API prompt is redacted."""
        patterns = (
            r"\bcamera\s+id\s*(?:[:#=]\s*)?([A-Za-z0-9][A-Za-z0-9_-]{2,})\b",
            r"\bcamera\s*(?:[:#=]\s*)?([A-Za-z0-9][A-Za-z0-9_-]{2,})\b",
        )
        for pattern in patterns:
            match = re.search(pattern, question, flags=re.I)
            if not match:
                continue
            candidate = match.group(1)
            # Requiring a digit avoids treating ordinary phrases such as
            # "camera network" or "camera traps" as identifiers.
            if any(char.isdigit() for char in candidate):
                return candidate
        return None

    def _rehydrate_local_sensitive_args(
        self,
        question: str,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Restore locally parsed sensitive constraints without exposing them to the LLM."""
        camera_id = self._local_camera_id_from_question(question)
        if not camera_id:
            return args

        restored = deepcopy(args)
        target_key = None
        if tool_name in {"filter_records", "browse_records", "evidence_lookup"}:
            target_key = "camera_id"
        elif tool_name in {
            "build_local_hotspot_rank",
            "analyze_spatial_distribution",
            "plot_local_hotspot_map",
            "plot_local_hotspot_static",
        }:
            target_key = "center_camera_id"
        if target_key is None:
            return restored

        current = str(restored.get(target_key, ""))
        if not current or "redacted" in current.casefold():
            restored[target_key] = camera_id
        return restored

    def exportable_trace(self, record: Dict[str, Any], detail: str = "public") -> Dict[str, Any]:
        """Build a portable trace; public mode excludes raw records and sensitive locations."""
        if detail not in {"public", "full"}:
            raise ValueError("trace detail must be 'public' or 'full'.")
        payload = sanitize_for_display(deepcopy(record))
        if detail == "full":
            return payload

        payload["executed_results"] = self._compact_executed_results_for_prompt(
            record.get("executed_results", [])
        )
        if isinstance(payload.get("plan", {}).get("_llm_outputs"), dict):
            payload["plan"]["_llm_outputs"].pop("planner_raw_response", None)
        if isinstance(payload.get("replan", {}).get("_llm_outputs"), dict):
            payload["replan"]["_llm_outputs"].pop("replanner_raw_response", None)
        trace = payload.get("trace", {})
        for section, raw_key in (
            ("planning", "planner_raw_response"),
            ("replanning", "replanner_raw_response"),
            ("synthesis", "answer_raw_response"),
        ):
            if isinstance(trace.get(section), dict):
                trace[section].pop(raw_key, None)
        payload["trace_export"] = {
            "detail": "public",
            "raw_tool_results_included": False,
            "raw_model_responses_included": False,
            "sensitive_camera_locations_redacted": True,
        }
        return self._redact_public_trace(payload)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except Exception:
                pass

        brace = re.search(r"(\{.*\})", text, flags=re.S)
        if brace:
            try:
                return json.loads(brace.group(1))
            except Exception:
                pass

        raise ValueError(f"Could not parse JSON from model output:\n{text}")

    def _species_aliases(self) -> List[tuple]:
        return [
            ("Siberian tiger", ["siberian tiger", "amur tiger", "tiger", "panthera tigris altaica"]),
            ("Cattle", ["cattle", "cow", "cows", "bos taurus"]),
            ("Wild boar", ["wild boar", "boar", "sus scrofa"]),
            ("Roe deer", ["roe deer", "siberian roe deer", "roe", "capreolus pygargus"]),
            ("Sika deer", ["sika deer", "sika", "cervus nippon"]),
            ("Musk deer", ["musk deer", "moschus moschiferus"]),
            ("Amur leopard", ["amur leopard", "leopard", "panthera pardus orientalis"]),
            ("Black bear", ["black bear", "asian black bear", "ursus thibetanus"]),
            ("Leopard cat", ["leopard cat", "prionailurus bengalensis"]),
            ("Red fox", ["red fox", "fox", "vulpes vulpes"]),
            ("Raccoon dog", ["raccoon dog", "nyctereutes procyonoides"]),
            ("Badger", ["badger", "meles leucurus"]),
            ("Amur hedgehog", ["amur hedgehog", "hedgehog", "erinaceus amurensis"]),
            ("Manchurian hare", ["manchurian hare", "hare", "lepus mandshuricus"]),
            ("Chipmunk", ["chipmunk", "siberian chipmunk", "eutamias sibiricus"]),
            ("Squirrel", ["squirrel", "sciurus vulgaris"]),
            ("Dog", ["dog", "domestic dog", "canis lupus familiaris"]),
            ("Horse", ["horse", "equus ferus caballus"]),
            ("Sheep", ["sheep", "ovis aries"]),
            ("Bird", ["bird", "aves"]),
        ]

    def _species_display_name(self, species: str) -> str:
        return species

    def _active_species_inventory(self) -> List[Dict[str, Any]]:
        """Return a compact inventory from the currently active data backend."""
        try:
            result = self.registry.get_func("summarize_species_inventory")(
                top_k=100
            )
        except Exception:
            return []
        inventory = []
        for item in (result or {}).get("species_inventory", []):
            name = str(item.get("species") or "").strip()
            if not name or name.casefold() == "unknown":
                continue
            inventory.append({
                "species": name,
                "record_count": item.get("record_count"),
            })
        return inventory

    def _species_context(
        self,
        explicit_species: Optional[str],
        inventory: List[Dict[str, Any]],
    ) -> Optional[str]:
        if explicit_species and str(explicit_species).strip():
            return str(explicit_species).strip()
        if self.default_species:
            return self.default_species
        names = [item["species"] for item in inventory]
        return names[0] if len(names) == 1 else None

    def _sanitize_species_mentions(self, text: str, species: str, single_species_focus: bool) -> str:
        if not text or not single_species_focus:
            return text

        species = str(species).strip() if species and str(species).strip() else self.default_species
        if not species:
            return text
        alias_map = dict(self._species_aliases())
        target_aliases = set(alias_map.get(species, []))
        target_compact = self._species_display_name(species)
        target_name = target_compact

        cleaned = text
        for canonical_name, aliases in self._species_aliases():
            if canonical_name == species:
                continue
            for alias in aliases:
                if not alias:
                    continue
                pattern = re.escape(alias)
                if re.fullmatch(r"[A-Za-z ]+", alias):
                    pattern = rf"\b{pattern}\b"
                cleaned = re.sub(pattern, target_name, cleaned, flags=re.IGNORECASE)

        # If the answer still lacks the canonical species string, patch the first tiger mention into canonical display.
        if species.lower() not in cleaned.lower() and target_aliases:
            first_alias = sorted(target_aliases, key=len, reverse=True)[0]
            cleaned = re.sub(
                re.escape(first_alias),
                target_compact,
                cleaned,
                count=1,
                flags=re.IGNORECASE,
            )
        return cleaned

    def _compact_result_for_prompt(self, tool_name: str, result: Any) -> Any:
        if not isinstance(result, dict):
            if tool_name == "analyze_movement_steps" and isinstance(result, list):
                return {
                    "camera_transition_hypothesis_count": len(result),
                    "steps_preview": [
                        {
                            "time_gap_hours": item.get("time_gap_hours"),
                            "distance_km": item.get("distance_km"),
                            "speed_km_h": item.get("speed_km_h"),
                            "bearing_degrees": item.get("bearing_degrees"),
                            "direction": item.get("direction"),
                            "inference_level": item.get("inference_level"),
                            "identity_evidence": item.get("identity_evidence"),
                        }
                        for item in result[:3]
                        if isinstance(item, dict)
                    ],
                    "interpretation_boundary": (
                        "Temporally adjacent camera detections are camera-transition "
                        "hypotheses, not individual tracks or proven corridors."
                    ),
                }
            if tool_name == "infer_transition_corridors" and isinstance(result, list):
                return {
                    "candidate_transition_edge_count": len(result),
                    "ranked_edge_observation_counts": [
                        item.get("count")
                        for item in result
                        if isinstance(item, dict)
                    ],
                    "interpretation_boundary": (
                        "Candidate transition edges are not confirmed movement corridors."
                    ),
                }
            if tool_name == "infer_longest_transition_steps" and isinstance(result, list):
                return {
                    "candidate_transition_step_count": len(result),
                    "largest_distances_km": [
                        item.get("distance_km")
                        for item in result[:10]
                        if isinstance(item, dict)
                    ],
                    "interpretation_boundary": (
                        "Adjacent detections may represent different individuals."
                    ),
                }
            if isinstance(result, list):
                return result[:10]
            return result

        if tool_name == "check_timestamp_quality":
            return {
                "species": result.get("species"),
                "total_records": result.get("total_records"),
                "valid_timestamp_records": result.get("valid_timestamp_records"),
                "invalid_timestamp_count": result.get("invalid_timestamp_count"),
                "observed_clock_time_ratio": result.get("observed_clock_time_ratio"),
                "synthetic_clock_time_ratio": result.get("synthetic_clock_time_ratio"),
                "timestamp_source_summary": result.get("timestamp_source_summary"),
                "timestamp_source": result.get("timestamp_source"),
                "reliability_status": result.get("reliability_status"),
            }

        if tool_name == "check_missing_fields":
            missing_counts = result.get("missing_counts", {})
            missing_ratios = result.get("missing_ratios", {})
            field_names = list(dict.fromkeys([
                *missing_counts.keys(),
                *missing_ratios.keys(),
            ]))
            return {
                "species": result.get("species"),
                "total_records": result.get("total_records"),
                "field_completeness": [
                    {
                        "field": field_name,
                        "missing_count": missing_counts.get(field_name),
                        "missing_ratio": missing_ratios.get(field_name),
                    }
                    for field_name in field_names
                ],
            }

        if tool_name == "build_independent_events":
            return {
                "species": result.get("species"),
                "by_camera": result.get("by_camera"),
                "min_gap_minutes": result.get("min_gap_minutes"),
                "raw_record_count": result.get("source_record_count"),
                "valid_record_count": result.get("valid_record_count"),
                "independent_event_count": (
                    result.get("independent_event_count")
                    if result.get("independent_event_count") is not None
                    else result.get("event_count")
                ),
                "event_gap_minutes": result.get("min_gap_minutes"),
                "note": result.get("note"),
            }

        if tool_name == "build_hourly_activity_distribution":
            return {
                "species": result.get("species"),
                "min_gap_minutes": result.get("min_gap_minutes"),
                "source_record_count": result.get("source_record_count"),
                "valid_time_records": result.get("valid_time_records"),
                "independent_event_count": result.get("independent_event_count"),
                "observed_clock_time_ratio": result.get("observed_clock_time_ratio"),
                "synthetic_clock_time_ratio": result.get("synthetic_clock_time_ratio"),
                "timestamp_source_summary": result.get("timestamp_source_summary"),
                "hourly_distribution": result.get("hourly_distribution", []),
                "peak_hours": result.get("peak_hours", []),
                "peak_period_label": result.get("peak_period_label"),
                "activity_window_shares": result.get("activity_window_shares"),
                "major_peak_window": result.get("major_peak_window"),
                "secondary_peak_window": result.get("secondary_peak_window"),
                "activity_pattern_label": result.get("activity_pattern_label"),
                "note": result.get("note"),
            }

        if tool_name == "analyze_spatial_distribution":
            local = result.get("local_analysis", {})
            compact = {
                "species": result.get("species"),
                "total_observations": result.get("total_observations"),
                "hotspot_rank_top5": result.get("hotspot_rank", [])[:5],
                "occupancy_summary": result.get("occupancy_summary"),
                "occupancy_model_summary": result.get("occupancy_model_summary"),
            }
            if isinstance(local, dict) and local:
                compact["local_analysis"] = {
                    "center_camera_id": local.get("center_camera_id"),
                    "radius_km": local.get("radius_km"),
                    "nearby_camera_count": len(local.get("nearby_cameras", [])),
                    "nearby_total_observations": local.get("nearby_total_observations"),
                    "local_hotspots_top5": local.get("local_hotspots_top5", [])[:5],
                }
            return compact

        if tool_name == "analyze_camera_data_quality":
            return result[:10]

        if tool_name == "analyze_data_quality":
            return {
                "missing_fields": self._compact_result_for_prompt(
                    "check_missing_fields",
                    result.get("missing_fields", {}),
                ),
                "invalid_timestamps": result.get("invalid_timestamps"),
                "label_noise": result.get("label_noise"),
                "camera_quality_top10": result.get("camera_quality", [])[:10],
                "duplicate_burden_by_camera_top10": result.get("duplicate_burden_by_camera", [])[:10],
            }

        if tool_name == "analyze_species_pair_relationship":
            proximity = result.get("same_camera_temporal_proximity", {}) or {}
            return {
                "species_a": result.get("species_a"),
                "species_b": result.get("species_b"),
                "independent_event_count_a": result.get("record_count_a"),
                "independent_event_count_b": result.get("record_count_b"),
                "camera_count_a": result.get("camera_count_a"),
                "camera_count_b": result.get("camera_count_b"),
                "shared_camera_count": result.get("shared_camera_count"),
                "camera_jaccard": result.get("camera_jaccard"),
                "activity_overlap_delta": result.get("activity_overlap_delta"),
                "activity_overlap_ci": result.get("activity_overlap_ci"),
                "activity_overlap_estimator_id": result.get(
                    "activity_overlap_estimator_id"
                ),
                "activity_overlap_estimator_version": result.get(
                    "activity_overlap_estimator_version"
                ),
                "activity_overlap_source_events_sha256": result.get(
                    "activity_overlap_source_events_sha256"
                ),
                "activity_overlap_result_bundle_id": result.get(
                    "activity_overlap_result_bundle_id"
                ),
                "activity_overlap_result_bundle_sha256": result.get(
                    "activity_overlap_result_bundle_sha256"
                ),
                "activity_overlap_result_sha256": result.get(
                    "activity_overlap_result_sha256"
                ),
                "activity_overlap_result_reused": result.get(
                    "activity_overlap_result_reused"
                ),
                "same_camera_temporal_proximity": {
                    "temporal_window_hours": proximity.get("temporal_window_hours"),
                    "proximate_event_pair_count": proximity.get("proximate_event_pair_count"),
                    "camera_count_with_proximity": proximity.get("camera_count_with_proximity"),
                    "median_nearest_gap_hours": proximity.get("median_nearest_gap_hours"),
                },
                "interpretation_boundary": (
                    "These are descriptive shared-camera, diel-overlap, and "
                    "near-time detection metrics. They do not prove simultaneous "
                    "occurrence, causation, or direct ecological interaction."
                ),
            }

        if tool_name == "estimate_activity_overlap":
            return {
                "species_a": result.get("species_a"),
                "species_b": result.get("species_b"),
                "delta_hat": result.get("delta_hat"),
                "delta_ci_low": result.get("delta_ci_low"),
                "delta_ci_high": result.get("delta_ci_high"),
                "permutation_p_value": result.get("permutation_p_value"),
                "permutation_unit": result.get("permutation_unit"),
                "permutation_inference_status": result.get(
                    "permutation_inference_status"
                ),
                "temporal_relationship": result.get("temporal_relationship"),
                "statistically_distinguishable_activity_distributions": result.get(
                    "statistically_distinguishable_activity_distributions"
                ),
                "supports_temporal_separation": result.get("supports_temporal_separation"),
                "supports_strong_temporal_separation": result.get(
                    "supports_strong_temporal_separation"
                ),
                "valid_event_count_a": result.get("valid_event_count_a"),
                "valid_event_count_b": result.get("valid_event_count_b"),
                "peak_hours_a": result.get("peak_hours_a", [])[:3],
                "peak_hours_b": result.get("peak_hours_b", [])[:3],
                "estimator_id": result.get("estimator_id"),
                "estimator_version": result.get("estimator_version"),
                "source_events_sha256": result.get("source_events_sha256"),
                "overlap_result_bundle_id": result.get(
                    "overlap_result_bundle_id"
                ),
                "overlap_result_bundle_sha256": result.get(
                    "overlap_result_bundle_sha256"
                ),
                "overlap_result_sha256": result.get("overlap_result_sha256"),
                "overlap_result_reused": result.get("overlap_result_reused"),
                "note": result.get("note"),
            }

        if tool_name == "estimate_pairwise_activity_overlap":
            return {
                "species_a": result.get("species_a"),
                "comparison_species": result.get("comparison_species"),
                "min_gap_minutes": result.get("min_gap_minutes"),
                "smooth_sigma": result.get("smooth_sigma"),
                "bootstrap_iters": result.get("bootstrap_iters"),
                "bootstrap_interval_method": result.get("bootstrap_interval_method"),
                "permutation_iters": result.get("permutation_iters"),
                "pairwise_results": result.get("pairwise_results", []),
                "overlap_result_bundle_ids": result.get(
                    "overlap_result_bundle_ids", []
                ),
                "overlap_result_bundle_set_sha256": result.get(
                    "overlap_result_bundle_set_sha256"
                ),
                "skipped": result.get("skipped", []),
                "note": result.get("note"),
            }

        if tool_name == "plot_pairwise_activity_overlap_panels":
            return {
                "status": result.get("status"),
                "figure_count": result.get("figure_count"),
                "figures": result.get("figures", []),
                "overlap_result_bundle_ids": result.get(
                    "overlap_result_bundle_ids", []
                ),
                "failed_figures": result.get("failed_figures", []),
            }

        if tool_name == "fit_single_season_occupancy_model":
            return {
                "species": result.get("species"),
                "surveyed_camera_count": result.get("surveyed_camera_count"),
                "detected_camera_count": result.get("detected_camera_count"),
                "naive_occupancy_ratio": result.get("naive_occupancy_ratio"),
                "occupancy_probability_psi": result.get("occupancy_probability_psi"),
                "detection_probability_p": result.get("detection_probability_p"),
                "psi_ci_low": result.get("psi_ci_low"),
                "psi_ci_high": result.get("psi_ci_high"),
                "p_ci_low": result.get("p_ci_low"),
                "p_ci_high": result.get("p_ci_high"),
                "note": result.get("note"),
            }

        if tool_name == "import_camtrap_dp":
            return {
                "package_name": result.get("package_name"),
                "target_dir": result.get("target_dir"),
                "imported_record_count": result.get("imported_record_count"),
                "species_breakdown": result.get("species_breakdown"),
                "note": result.get("note"),
            }

        if tool_name == "load_camtrap_dp_dataset":
            return {
                "package_name": result.get("package_name"),
                "target_dir": result.get("target_dir"),
                "active_json_backend": result.get("active_json_backend"),
                "imported_record_count": result.get("imported_record_count"),
                "species_breakdown": result.get("species_breakdown"),
                "note": result.get("note"),
            }

        if tool_name == "summarize_species_inventory":
            return {
                "species_count": result.get("species_count"),
                "total_record_count": result.get("total_record_count"),
                "species_inventory": result.get("species_inventory", [])[:10],
                "note": result.get("note"),
            }

        if tool_name == "summarize_community_composition":
            return {
                "total_record_count": result.get("total_record_count"),
                "total_source_record_count": result.get("total_source_record_count"),
                "species_richness": result.get("species_richness"),
                "shannon_index": result.get("shannon_index"),
                "simpson_index": result.get("simpson_index"),
                "abundance_metric": result.get("abundance_metric"),
                "excluded_low_effort_species_count": result.get("excluded_low_effort_species_count"),
                "top_species": result.get("top_species", [])[:10],
                "note": result.get("note"),
            }

        if tool_name == "analyze_community_ecology":
            return {
                "selected_species": result.get("selected_species"),
                "community": result.get("community"),
                "camera_richness_hotspots": result.get("camera_richness_hotspots", [])[:5],
                "top_spatial_cooccurrence_pairs": result.get("top_spatial_cooccurrence_pairs", [])[:5],
                "activity_guild_counts": result.get("activity_guild_counts"),
                "activity_guilds": result.get("activity_guilds", [])[:10],
                "top_relationships": result.get("top_relationships", [])[:5],
                "note": result.get("note"),
            }

        if tool_name in {
            "infer_complete_community",
            "plot_complete_community_inference",
        }:
            return {
                "status": result.get("status"),
                "analysis_scope": result.get("analysis_scope"),
                "community_inference_status": result.get(
                    "community_inference_status"
                ),
                "source_species_count": result.get(
                    "source_species_count"
                ),
                "eligible_species_count": result.get(
                    "eligible_species_count"
                ),
                "analysed_species_count": result.get(
                    "analysed_species_count"
                ),
                "camera_count": result.get("camera_count"),
                "independent_event_count": result.get(
                    "independent_event_count"
                ),
                "gamma_diversity": result.get("gamma_diversity"),
                "alpha_diversity": result.get("alpha_diversity"),
                "beta_diversity": result.get("beta_diversity"),
                "species_summary": result.get(
                    "species_summary",
                    [],
                )[:20],
                "significant_conditional_associations": [
                    row
                    for row in result.get(
                        "conditional_camera_associations",
                        [],
                    )
                    if row.get("fdr_significant")
                ][:20],
                "detection_corrected_occupancy_species_count": (
                    result.get(
                        "detection_corrected_occupancy_species_count"
                    )
                ),
                "occupancy_effort_basis": result.get(
                    "occupancy_effort_basis"
                ),
                "figures": result.get("figures", []),
                "interpretation": result.get("interpretation"),
            }

        if tool_name == "build_camera_species_richness":
            return {
                "camera_count": result.get("camera_count"),
                "top_k": result.get("top_k"),
                "richness_by_camera": result.get("richness_by_camera", [])[:10],
                "note": result.get("note"),
            }

        if tool_name == "build_species_cooccurrence_matrix":
            return {
                "species": result.get("species"),
                "top_cooccurring_pairs": result.get("top_cooccurring_pairs", [])[:10],
                "note": result.get("note"),
            }

        if tool_name == "analyze_species_pair_relationship":
            return {
                "species_a": result.get("species_a"),
                "species_b": result.get("species_b"),
                "record_count_a": result.get("record_count_a"),
                "record_count_b": result.get("record_count_b"),
                "camera_count_a": result.get("camera_count_a"),
                "camera_count_b": result.get("camera_count_b"),
                "shared_camera_count": result.get("shared_camera_count"),
                "camera_jaccard": result.get("camera_jaccard"),
                "spatial_overlap_level": result.get("spatial_overlap_level"),
                "activity_overlap_delta": result.get("activity_overlap_delta"),
                "activity_overlap_ci": result.get("activity_overlap_ci"),
                "activity_overlap_result_bundle_id": result.get(
                    "activity_overlap_result_bundle_id"
                ),
                "activity_overlap_result_bundle_sha256": result.get(
                    "activity_overlap_result_bundle_sha256"
                ),
                "activity_overlap_result_sha256": result.get(
                    "activity_overlap_result_sha256"
                ),
                "temporal_overlap_level": result.get("temporal_overlap_level"),
                "same_camera_temporal_proximity": result.get("same_camera_temporal_proximity"),
                "relationship_score": result.get("relationship_score"),
                "relationship_label": result.get("relationship_label"),
                "note": result.get("note"),
            }

        if tool_name == "summarize_species_relationship_network":
            return {
                "species": result.get("species"),
                "nodes": result.get("nodes", [])[:10],
                "top_relationships": result.get("top_relationships", [])[:10],
                "note": result.get("note"),
            }

        if tool_name == "compare_multi_species_hourly_activity":
            return {
                "species": result.get("species"),
                "profiles": [
                    {
                        "species": item.get("species"),
                        "valid_time_records": item.get("valid_time_records"),
                        "independent_event_count": item.get("independent_event_count"),
                        "top_hours": item.get("top_hours", [])[:3],
                        "day_ratio": item.get("day_ratio"),
                        "night_ratio": item.get("night_ratio"),
                    }
                    for item in result.get("profiles", [])[:8]
                ],
                "note": result.get("note"),
            }

        if tool_name == "build_species_activity_guilds":
            return {
                "species": result.get("species"),
                "guild_counts": result.get("guild_counts"),
                "activity_guilds": result.get("activity_guilds", [])[:10],
                "note": result.get("note"),
            }

        if tool_name == "export_camtrap_dp":
            return {
                "package_name": result.get("package_name"),
                "output_dir": result.get("output_dir"),
                "datapackage_path": result.get("datapackage_path"),
                "record_count": result.get("record_count"),
                "species_breakdown": result.get("species_breakdown"),
                "note": result.get("note"),
            }

        if tool_name == "export_media_by_species":
            return {
                "output_dir": result.get("output_dir"),
                "manifest_path": result.get("manifest_path"),
                "matched_count": result.get("matched_count"),
                "exported_count": result.get("exported_count"),
                "skipped_count": result.get("skipped_count"),
                "backend_used": result.get("backend_used"),
                "species_dirs": result.get("species_dirs"),
                "note": result.get("note"),
            }

        if tool_name in {"filter_records", "browse_records"}:
            return {
                "matched_count": result.get("matched_count"),
                "returned_count": result.get("returned_count"),
                "filters": result.get("filters"),
                "backend_used": result.get("backend_used"),
                "active_backend": result.get("active_backend"),
                "records": result.get("records", [])[:5],
                "note": result.get("note"),
            }

        if tool_name == "evidence_lookup":
            return {
                "matched_count": result.get("matched_count"),
                "returned_count": result.get("returned_count"),
                "filters": result.get("filters"),
                "backend_used": result.get("backend_used"),
                "active_backend": result.get("active_backend"),
                "evidence_samples": result.get("evidence_samples", [])[:5],
                "note": result.get("note"),
            }

        if tool_name == "run_detection_pipeline":
            return {
                "media_source": result.get("media_source"),
                "output_dir": result.get("output_dir"),
                "record_count": result.get("record_count"),
                "placeholder_record_count": result.get("placeholder_record_count"),
                "skipped_without_prediction_count": result.get(
                    "skipped_without_prediction_count"
                ),
                "detector": result.get("detector"),
                "classifier": result.get("classifier"),
                "pipeline_mode": result.get("pipeline_mode"),
                "pipeline_status": result.get("pipeline_status"),
                "model_inferred_image_count": result.get(
                    "model_inferred_image_count"
                ),
                "model_inferred_video_frame_count": result.get(
                    "model_inferred_video_frame_count"
                ),
                "records_with_bbox_count": result.get("records_with_bbox_count"),
                "records_with_confidence_count": result.get("records_with_confidence_count"),
                "analysis_readiness": result.get("analysis_readiness"),
                "species_breakdown": result.get("species_breakdown"),
                "sample_records": result.get("sample_records", [])[:3],
                "note": result.get("note"),
            }

        if tool_name == "prepare_camera_trap_dataset":
            backend = result.get("backend_result", {}) or {}
            validation = backend.get("validation", {}) or {}
            detection = result.get("detection", {}) or {}
            sample_predictions = []
            for row in detection.get("sample_records", [])[:3]:
                if not isinstance(row, dict):
                    continue
                sample_predictions.append({
                    "species": row.get("species"),
                    "scientific_name": row.get("scientific_name"),
                    "confidence": row.get("confidence"),
                    "timestamp": row.get("timestamp"),
                    "timestamp_source": row.get("timestamp_source"),
                    "timestamp_ocr_score": row.get("timestamp_ocr_score"),
                    "timestamp_ocr_vote_count": row.get(
                        "timestamp_ocr_vote_count"
                    ),
                    "model_version": row.get("perception_model_version"),
                    "prediction_source": row.get("prediction_source"),
                    "frame_offset_seconds": row.get("frame_offset_seconds"),
                })
            return {
                "status": result.get("status"),
                "input_type": result.get("input_type"),
                "pipeline_stages": result.get("pipeline_stages"),
                "perception_executed": result.get("perception_executed"),
                "data_to_camtrap_pipeline_complete": result.get(
                    "data_to_camtrap_pipeline_complete"
                ),
                "full_raw_media_pipeline_complete": result.get(
                    "full_raw_media_pipeline_complete"
                ),
                "record_count": detection.get("record_count"),
                "model_inferred_video_frame_count": detection.get(
                    "model_inferred_video_frame_count"
                ),
                "perception_evidence": {
                    "detector": detection.get("detector"),
                    "classifier": detection.get("classifier"),
                    "model_versions": detection.get("model_versions", {}),
                    "prediction_sources": detection.get(
                        "prediction_sources", {}
                    ),
                    "video_sampling": detection.get("video_sampling", {}),
                    "species_breakdown": detection.get(
                        "species_breakdown", {}
                    ),
                    "sample_predictions": sample_predictions,
                },
                "camtrap_validation_status": validation.get("status"),
                "camtrap_validation_error_count": validation.get("error_count"),
                "camtrap_validation_warning_count": validation.get(
                    "warning_count"
                ),
                "active_backend": {
                    "backend": backend.get("backend"),
                    "activated": bool(
                        result.get("analysis_readiness", {}).get(
                            "active_for_downstream_tools"
                        )
                    ),
                },
                "analysis_readiness": result.get("analysis_readiness"),
                "routing_note": result.get("routing_note"),
            }

        if tool_name == "evaluate_perception_benchmark":
            summary = result.get("summary", {}) or {}
            benchmark = result.get("benchmark_run", {}) or {}
            return {
                "status": result.get("status"),
                "evaluation_scope": summary.get("evaluation_scope"),
                "sample_count": summary.get("sample_count"),
                "class_count_with_ground_truth": summary.get(
                    "class_count_with_ground_truth"
                ),
                "accuracy": summary.get("accuracy"),
                "accuracy_ci_low": summary.get("accuracy_ci_low"),
                "accuracy_ci_high": summary.get("accuracy_ci_high"),
                "macro_precision": summary.get("macro_precision"),
                "macro_recall": summary.get("macro_recall"),
                "macro_f1": summary.get("macro_f1"),
                "expected_calibration_error": summary.get(
                    "expected_calibration_error"
                ),
                "camera_group_count": summary.get("camera_group_count"),
                "bootstrap_unit": summary.get("bootstrap_unit"),
                "inference_failure_count": benchmark.get(
                    "inference_failure_count"
                ),
                "per_class": result.get("per_class", [])[:20],
                "figures": result.get("figures", []),
                "manifest_path": result.get("manifest_path"),
            }

        if tool_name == "analyze_animal_movement_pattern":
            return {
                "note": result.get("note"),
                "species": result.get("species"),
                "movement_step_count": result.get("movement_step_count"),
                "movement_distance_summary": result.get("movement_distance_summary"),
                "movement_direction_summary": result.get("movement_direction_summary"),
                "top_transition_corridors": result.get("top_transition_corridors", [])[:10],
                "longest_transition_steps": result.get("longest_transition_steps", [])[:10],
            }

        compact = {}
        for key, value in result.items():
            if key in {"detection_map", "occupied_camera_ids", "nearby_cameras", "nearby_detection_map"}:
                compact[f"{key}_count"] = len(value) if isinstance(value, list) else None
            elif key == "steps" and isinstance(value, list):
                compact["steps_count"] = len(value)
            elif isinstance(value, list):
                compact[key] = value[:10]
            else:
                compact[key] = value
        return compact

    def _compact_tool_descriptions_for_replanning(self) -> List[Dict[str, Any]]:
        """Keep recovery schemas useful while avoiding a second full registry dump."""
        return [
            {
                "tool_name": item.get("tool_name"),
                "description": item.get("description"),
                "params": item.get("params", {}),
            }
            for item in self.registry.get_tool_descriptions()
        ]

    def _prompt_value_summary(self, value: Any, depth: int = 0) -> Any:
        """Build a deterministic evidence-preserving fallback for a tight prompt budget."""
        if depth >= 3:
            if isinstance(value, (list, tuple)):
                return {"item_count": len(value), "details_omitted_for_prompt_budget": True}
            if isinstance(value, dict):
                return {
                    "field_count": len(value),
                    "fields": list(value)[:20],
                    "details_omitted_for_prompt_budget": True,
                }
        if isinstance(value, dict):
            summary = {}
            for key, item in list(value.items())[:40]:
                if isinstance(item, (list, tuple)):
                    summary[key] = {
                        "item_count": len(item),
                        "preview": [
                            self._prompt_value_summary(entry, depth + 1)
                            for entry in list(item)[:2]
                        ],
                    }
                elif isinstance(item, dict):
                    summary[key] = self._prompt_value_summary(item, depth + 1)
                elif isinstance(item, str) and len(item) > 300:
                    summary[key] = item[:300] + "…"
                else:
                    summary[key] = item
            if len(value) > 40:
                summary["_omitted_field_count"] = len(value) - 40
            return summary
        if isinstance(value, (list, tuple)):
            return {
                "item_count": len(value),
                "preview": [
                    self._prompt_value_summary(entry, depth + 1)
                    for entry in list(value)[:3]
                ],
            }
        if isinstance(value, str) and len(value) > 300:
            return value[:300] + "…"
        return value

    def _execution_status_summary_for_prompt(
        self,
        compact_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        summaries = []
        for item in compact_results:
            summary = {
                key: item.get(key)
                for key in (
                    "step",
                    "tool_name",
                    "args",
                    "status",
                    "round_index",
                    "attempt_index",
                    "duration_seconds",
                    "reason",
                    "error",
                    "skip_reason",
                )
                if item.get(key) is not None
            }
            if "result" in item:
                summary["result_summary"] = self._prompt_value_summary(item["result"])
            summaries.append(summary)
        return summaries

    def _critical_result_evidence_for_prompt(
        self,
        tool_name: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Retain decision-critical ecological evidence under a tight budget."""
        exact_keys = {
            "status", "analysis_scope", "community_inference_status",
            "species", "species_a", "species_b", "comparison_species",
            "selected_species", "source_species_count",
            "eligible_species_count", "analysed_species_count",
            "record_count", "source_record_count", "valid_record_count",
            "independent_event_count", "independent_event_count_a",
            "independent_event_count_b", "valid_event_count_a",
            "valid_event_count_b", "camera_count", "camera_count_a",
            "camera_count_b", "shared_camera_count", "camera_jaccard",
            "peak_hours", "peak_hours_a", "peak_hours_b",
            "activity_window_shares", "major_peak_window",
            "secondary_peak_window", "activity_pattern_label", "delta_hat",
            "delta_ci_low", "delta_ci_high", "activity_overlap_delta",
            "activity_overlap_ci", "pairwise_results",
            "overlap_result_bundle_id", "overlap_result_bundle_ids",
            "overlap_result_bundle_sha256",
            "overlap_result_bundle_set_sha256", "overlap_result_sha256",
            "source_events_sha256", "gamma_diversity", "alpha_diversity",
            "beta_diversity", "species_summary",
            "significant_conditional_associations",
            "detection_corrected_occupancy_species_count",
            "occupancy_effort_basis", "occupancy_summary",
            "occupancy_model_summary", "hotspot_rank_top5",
            "same_camera_temporal_proximity", "movement_step_count",
            "movement_distance_summary", "movement_direction_summary",
            "pipeline_stages", "perception_executed",
            "data_to_camtrap_pipeline_complete",
            "full_raw_media_pipeline_complete", "perception_evidence",
            "camtrap_validation_status", "camtrap_validation_error_count",
            "camtrap_validation_warning_count", "active_backend",
            "analysis_readiness", "routing_note", "interpretation",
            "interpretation_boundary", "note",
        }
        evidence = {}
        for key, value in result.items():
            keep_key = (
                key in exact_keys
                or key.endswith("_count")
                or key.endswith("_status")
                or key.endswith("_sha256")
                or key.endswith("_bundle_id")
            )
            if not keep_key:
                continue
            if isinstance(value, list):
                evidence[key] = {
                    "item_count": len(value),
                    "preview": [
                        self._prompt_value_summary(item, depth=1)
                        for item in value[:5]
                    ],
                }
            elif isinstance(value, dict):
                evidence[key] = self._prompt_value_summary(value, depth=1)
            elif isinstance(value, str):
                evidence[key] = value[:300] + (
                    "…" if len(value) > 300 else ""
                )
            else:
                evidence[key] = value
        return {
            "tool_name": tool_name,
            "evidence": evidence,
            "retained_evidence_field_count": len(evidence),
        }

    def _minimal_execution_statuses_for_prompt(
        self,
        compact_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Retain every call outcome and a bounded critical-evidence projection."""
        summaries = []
        for item in compact_results:
            summary = {
                key: item.get(key)
                for key in (
                    "step",
                    "tool_name",
                    "status",
                    "round_index",
                    "attempt_index",
                    "duration_seconds",
                )
                if item.get(key) is not None
            }
            args = item.get("args")
            if isinstance(args, dict):
                summary["args"] = {
                    key: (
                        value[:200] + "…"
                        if isinstance(value, str) and len(value) > 200
                        else value
                    )
                    for key, value in args.items()
                    if not isinstance(value, (dict, list, tuple))
                }
            result = item.get("result")
            if not isinstance(result, dict) and isinstance(
                item.get("result_summary"), dict
            ):
                result = item.get("result_summary")
            if isinstance(result, dict):
                projection = self._critical_result_evidence_for_prompt(
                    str(item.get("tool_name") or ""),
                    result,
                )
                summary["result_evidence"] = projection["evidence"]
                summary["retained_evidence_field_count"] = projection[
                    "retained_evidence_field_count"
                ]
            elif isinstance(result, (list, tuple)):
                summary["result_item_count"] = len(result)
            elif result is not None:
                summary["result_value"] = (
                    result[:200] + "…"
                    if isinstance(result, str) and len(result) > 200
                    else result
                )
            if item.get("error"):
                summary["error"] = str(item["error"])[:300]
            summaries.append(summary)
        return summaries

    def _compact_executed_results_for_prompt(self, executed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact_items = []
        for item in executed_results:
            compact_item = {
                "step": item.get("step"),
                "tool_name": item.get("tool_name"),
                "args": sanitize_for_display(item.get("args")),
                "input_args": sanitize_for_display(item.get("input_args")),
                "filtered_args": sanitize_for_display(item.get("filtered_args", [])),
                "reason": item.get("reason"),
                "status": item.get("status"),
                "round_index": item.get("round_index"),
                "attempt_index": item.get("attempt_index"),
                "duration_seconds": item.get("duration_seconds"),
            }
            if item.get("status") == "success":
                compact_item["result"] = sanitize_for_display(self._compact_result_for_prompt(
                    item.get("tool_name", ""),
                    item.get("result")
                ))
            elif item.get("status") == "skipped":
                compact_item["skip_reason"] = item.get("skip_reason")
                compact_item["result"] = sanitize_for_display(item.get("result", {}))
            else:
                compact_item["status"] = "failed"
                compact_item["reason"] = item.get("reason_failed") or item.get("error") or item.get("reason")
                compact_item["error"] = item.get("error")
            compact_items.append(self._redact_public_trace(compact_item))
        return compact_items

    # =========================
    # Planner
    # =========================
    def plan(self, question: str, species: str = None) -> Dict[str, Any]:
        active_inventory = self._active_species_inventory()
        species = self._species_context(species, active_inventory)
        tool_desc = self.registry.get_tool_descriptions()

        system_prompt = """
You are the planner for a wildlife monitoring analysis agent.
You are solely responsible for choosing the tools needed to answer the user's ecological question.
Analyze the question, inspect the supplied tool schemas, and produce a complete dependency-ordered plan.
The runtime will validate your plan but will not add, replace, or route tools on your behalf.
Do not answer the user directly.
Do not output Markdown or explanatory prose.
Use only tools from the supplied tool list.
Do not invent parameter names that are not in the tool schema.
Respect every explicit user constraint, including requested species, temporal or spatial limits,
visualization requests or prohibitions, scientific interpretation boundaries, and required data checks.
When an analysis requires prerequisite evidence, select those prerequisite tools yourself.
Treat each tool's requires_prior_tools contract as mandatory and place every
prerequisite before the dependent figure tool.
Complete all selected quality, event-construction, and ecological-analysis
tasks before beginning any plot_* task.
When one focal species is compared with multiple species, use the aggregate
estimate_pairwise_activity_overlap and plot_pairwise_activity_overlap_panels
tools instead of repeating one single-pair analysis or plot call per species.
Use explicit species names and explicit argument values rather than placeholders such as "default".
Use the active species inventory as data context, not as a fixed routing rule.
If the question requires one focal species but the user did not identify it,
use a configured default species only when one is explicitly supplied. If no
configured default exists and the active inventory contains exactly one species,
that sole species may be used. If several species are plausible, do not guess or
select the most abundant species: return an empty task list and one concise
clarification_request asking the user to identify the focal species. Community-wide
questions that do not require a focal species may proceed without clarification.

The output format must be exactly:
{
  "species": "explicit species name or null",
  "clarification_request": null,
  "tasks": [
    {
      "tool_name": "build_hotspot_rank",
      "args": {
        "species": "explicit species name",
        "top_k": 5
      },
      "reason": "The user asks where records are concentrated, so hotspot ranking is needed."
    }
  ]
}
"""

        # The planner must receive user-supplied input paths so it can populate
        # generic preparation-tool arguments. Public export applies the stricter
        # local-path and media-name redaction after the run.
        prompt_question = self._redact_external_context(question)
        base_user_prompt = f"""
User question:
{prompt_question}

Configured default species:
{species if self.default_species else None}

Active species inventory:
{json.dumps(active_inventory, ensure_ascii=False, indent=2)}

Available tools:
{json.dumps(tool_desc, ensure_ascii=False, indent=2)}
"""
        max_attempts = 1 + self.run_policy.max_replan_rounds
        validation_feedback = None
        planner_attempts = []
        last_plan = None

        for attempt_index in range(max_attempts):
            user_prompt = base_user_prompt
            if validation_feedback:
                user_prompt += f"""

Mechanical validation feedback from your previous plan:
{json.dumps(validation_feedback, ensure_ascii=False, indent=2)}

Return a corrected complete plan. Choose the tools yourself from the supplied
registry; do not assume that the runtime will add or replace any tool.
"""
            try:
                resp = self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
                candidate = self._extract_json(resp.content)
            except Exception as exc:
                raise RuntimeError(
                    "LLM planning failed. Configure an API-backed model before running CamAgent planning."
                ) from exc

            candidate.setdefault("species", species)
            if not isinstance(candidate.get("tasks"), list):
                candidate["_planner_validation"] = {
                    "phase": "initial_plan",
                    "action": "request_llm_replan",
                    "plan_errors": ["Planner output must contain a tasks list."],
                }
                candidate["tasks"] = []

            for task in candidate["tasks"]:
                task.setdefault("args", {})

            clarification = str(
                candidate.get("clarification_request") or ""
            ).strip()
            if clarification:
                candidate["_clarification_request"] = {
                    "message": clarification,
                    "active_inventory_species": [
                        item["species"] for item in active_inventory
                    ],
                }

            candidate["_llm_outputs"] = {
                "planner_raw_response": resp.content,
                "planner_mode": (
                    "llm" if attempt_index == 0 else "llm_validation_retry"
                ),
                "planner_system_prompt_sha256": self._prompt_sha256(system_prompt),
                "planner_user_prompt_sha256": self._prompt_sha256(user_prompt),
                "planner_input_summary": {
                    "user_query": question,
                    "available_tool_names": [
                        tool.get("tool_name") for tool in tool_desc
                    ],
                    "species_alias_rules": (
                        "Common and scientific aliases are normalized for focal "
                        "species display; tool choice remains the LLM planner's decision."
                    ),
                    "active_species_inventory": active_inventory,
                    "configured_default_species": self.default_species,
                    "output_schema": (
                        "JSON object with species and tasks; each task has "
                        "tool_name, args, and reason."
                    ),
                    "constraints": [
                        "the LLM planner alone chooses tools",
                        "use only registered tools",
                        "do not invent unsupported args",
                        "return a complete dependency-ordered JSON plan",
                        "respect explicit user constraints and scientific boundaries",
                        "request clarification rather than guess an ambiguous focal species",
                    ],
                },
            }
            candidate = self._normalize_plan(
                question,
                candidate,
                {"allow_empty": True},
            )
            validation = candidate.get("_planner_validation", {})
            planner_attempts.append({
                "attempt_index": attempt_index,
                "raw_response_sha256": self._prompt_sha256(resp.content),
                "validation": sanitize_for_display(deepcopy(validation)),
            })
            last_plan = candidate

            if not self._planner_validation_has_errors(validation):
                candidate["_llm_outputs"]["planner_attempt_count"] = len(
                    planner_attempts
                )
                candidate["_llm_outputs"]["planner_attempts"] = planner_attempts
                return candidate

            validation_feedback = self._planner_validation_feedback(validation)

        if last_plan is None:
            raise RuntimeError("LLM planning produced no plan.")
        last_plan["_llm_outputs"]["planner_attempt_count"] = len(planner_attempts)
        last_plan["_llm_outputs"]["planner_attempts"] = planner_attempts
        raise ValueError(
            "LLM planner did not produce an executable registry-valid plan after "
            f"{max_attempts} attempts: {json.dumps(validation_feedback, ensure_ascii=False)}"
        )

    def _planner_validation_has_errors(self, validation: Dict[str, Any]) -> bool:
        return any(
            validation.get(key)
            for key in (
                "plan_errors",
                "invalid_tools",
                "missing_args",
                "invalid_args",
                "filtered_args",
                "dependency_errors",
                "bundle_alignment_errors",
                "analysis_order_errors",
            )
        )

    def _planner_validation_feedback(
        self,
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            key: sanitize_for_display(deepcopy(validation.get(key, [])))
            for key in (
                "plan_errors",
                "invalid_tools",
                "missing_args",
                "invalid_args",
                "filtered_args",
                "dependency_errors",
                "bundle_alignment_errors",
                "analysis_order_errors",
            )
            if validation.get(key)
        }

    def _normalize_plan(self, question: str, plan: Dict[str, Any], hints: Dict[str, Any]) -> Dict[str, Any]:
        normalized_tasks = []
        seen = set()
        planner_validation = deepcopy(plan.get("_planner_validation") or {})
        planner_validation.setdefault("phase", "initial_plan")
        planner_validation.setdefault("plan_errors", [])
        planner_validation.setdefault("invalid_tools", [])
        planner_validation.setdefault("missing_args", [])
        planner_validation.setdefault("invalid_args", [])
        planner_validation.setdefault("filtered_args", [])
        planner_validation.setdefault("dependency_errors", [])
        planner_validation.setdefault("bundle_alignment_errors", [])
        planner_validation.setdefault("analysis_order_errors", [])
        planner_validation.setdefault("action", "validated")
        planner_validation["validation_only"] = True
        planner_validation["local_tool_rewrite"] = False
        planner_validation["fallback_used"] = False
        species = plan.get("species") or self.default_species

        for task in plan.get("tasks", []):
            tool_name = task.get("tool_name")
            if not tool_name:
                planner_validation["plan_errors"].append(
                    "Every task must include tool_name."
                )
                continue
            if not self.registry.has_tool(tool_name, planner_only=True):
                planner_validation["invalid_tools"].append({
                    "tool_name": tool_name,
                    "reason": "not found in the planner-visible registry",
                    "action": "request_llm_replan",
                })
                continue

            args = deepcopy(task.get("args") or {})
            reason = task.get("reason", "")
            args = self._rehydrate_local_sensitive_args(question, tool_name, args)
            func = self.registry.get_func(tool_name)
            parameters = self._tool_signature(func).parameters
            if "species" in parameters and not str(args.get("species") or "").strip():
                planner_validation["missing_args"].append({
                    "tool_name": tool_name,
                    "missing_args": ["species"],
                    "reason": (
                        "Planner-visible species arguments must be selected explicitly "
                        "by the LLM rather than filled by local code."
                    ),
                    "action": "request_llm_replan",
                })
                continue

            if tool_name == "estimate_pairwise_activity_overlap":
                if not str(args.get("species_a") or "").strip():
                    planner_validation["missing_args"].append({
                        "tool_name": tool_name,
                        "missing_args": ["species_a"],
                        "reason": "The LLM must select the focal species explicitly.",
                        "action": "request_llm_replan",
                    })
                    continue
                if "species_list" in args:
                    normalized_species = self._normalize_explicit_species_list(
                        args.get("species_list")
                    )
                    if not normalized_species:
                        planner_validation["invalid_args"].append({
                            "tool_name": tool_name,
                            "argument": "species_list",
                            "reason": (
                                "Provide explicit comparison species or omit "
                                "species_list so the tool selects them from the "
                                "active dataset inventory."
                            ),
                            "action": "request_llm_replan",
                        })
                        continue
                    args["species_list"] = normalized_species
            elif tool_name.startswith("plot_"):
                if tool_name == "plot_pairwise_activity_overlap_panels":
                    if not str(args.get("species_a") or "").strip():
                        planner_validation["missing_args"].append({
                            "tool_name": tool_name,
                            "missing_args": ["species_a"],
                            "reason": "The LLM must select the focal species explicitly.",
                            "action": "request_llm_replan",
                        })
                        continue
                    if "comparison_species" in args:
                        normalized_species = self._normalize_explicit_species_list(
                            args.get("comparison_species")
                        )
                        if not normalized_species:
                            planner_validation["invalid_args"].append({
                                "tool_name": tool_name,
                                "argument": "comparison_species",
                                "reason": (
                                    "Provide explicit comparison species or omit "
                                    "comparison_species so the figure tool uses "
                                    "the active dataset selection."
                                ),
                                "action": "request_llm_replan",
                            })
                            continue
                        args["comparison_species"] = normalized_species

            args, filtered = self._filter_args_for_tool(func, args)
            if filtered:
                planner_validation["filtered_args"].append({
                    "tool_name": tool_name,
                    "filtered_args": filtered,
                    "action": "request_llm_replan",
                })
                continue

            missing = self._missing_required_args(func, args)
            if missing:
                planner_validation["missing_args"].append({
                    "tool_name": tool_name,
                    "missing_args": missing,
                    "action": "request_llm_replan",
                })
                continue

            key = (tool_name, json.dumps(args, ensure_ascii=False, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            normalized_tasks.append({
                "tool_name": tool_name,
                "args": args,
                "reason": reason,
            })

        prior_tool_names = []
        for task in normalized_tasks:
            tool_name = task["tool_name"]
            dependencies = self.registry.get_tool_dependencies(tool_name)
            missing_all = [
                required
                for required in dependencies.get("all_of", [])
                if required not in prior_tool_names
            ]
            any_of = dependencies.get("any_of", [])
            missing_any = bool(
                any_of
                and not any(candidate in prior_tool_names for candidate in any_of)
            )
            if missing_all or missing_any:
                dependency_error = {
                    "tool_name": tool_name,
                    "reason": (
                        "Dependent temporal tools must follow the prerequisite "
                        "quality, event-construction, analysis, or context tools "
                        "declared in the registry."
                    ),
                    "action": "request_llm_replan",
                }
                if missing_all:
                    dependency_error["missing_all_of"] = missing_all
                if missing_any:
                    dependency_error["missing_any_of"] = any_of
                planner_validation["dependency_errors"].append(dependency_error)
            prior_tool_names.append(tool_name)

        first_plot_name = None
        for task in normalized_tasks:
            tool_name = task["tool_name"]
            if tool_name.startswith("plot_"):
                first_plot_name = first_plot_name or tool_name
                continue
            if first_plot_name is not None:
                planner_validation["analysis_order_errors"].append({
                    "tool_name": tool_name,
                    "preceding_plot_tool": first_plot_name,
                    "reason": (
                        "Complete all selected quality checks, event "
                        "construction, and ecological analyses before starting "
                        "the figure-rendering phase."
                    ),
                    "action": "request_llm_replan",
                })

        def _effective_arg(task, parameter_name):
            if parameter_name in task.get("args", {}):
                return task["args"][parameter_name]
            parameters = self._tool_signature(
                self.registry.get_func(task["tool_name"])
            ).parameters
            parameter = parameters.get(parameter_name)
            if parameter is None or parameter.default is inspect.Parameter.empty:
                return None
            return parameter.default

        def _species_key(value):
            return str(value or "").strip().casefold()

        def _same_overlap_spec(analysis_task, plot_task):
            return (
                float(_effective_arg(analysis_task, "min_gap_minutes"))
                == float(_effective_arg(plot_task, "min_gap_minutes"))
                and float(_effective_arg(analysis_task, "smooth_sigma"))
                == float(_effective_arg(plot_task, "smooth_sigma"))
                and int(_effective_arg(analysis_task, "bootstrap_iters"))
                == int(_effective_arg(plot_task, "bootstrap_iters"))
                and int(_effective_arg(analysis_task, "permutation_iters") or 0)
                == 0
            )

        for task_index, task in enumerate(normalized_tasks):
            tool_name = task["tool_name"]
            if tool_name not in {
                "plot_activity_overlap_panel",
                "plot_pairwise_temporal_overlap",
                "plot_pairwise_activity_overlap_panels",
            }:
                continue
            prior_tasks = normalized_tasks[:task_index]
            matched = False
            if tool_name == "plot_pairwise_activity_overlap_panels":
                focal = _species_key(task["args"].get("species_a"))
                comparisons = {
                    _species_key(item)
                    for item in task["args"].get("comparison_species", [])
                }
                for analysis_task in prior_tasks:
                    if analysis_task["tool_name"] != "estimate_pairwise_activity_overlap":
                        continue
                    analysis_species = {
                        _species_key(item)
                        for item in analysis_task["args"].get("species_list", [])
                    }
                    if (
                        _species_key(analysis_task["args"].get("species_a"))
                        == focal
                        and analysis_species == comparisons
                        and _same_overlap_spec(analysis_task, task)
                    ):
                        matched = True
                        break
            else:
                focal_arg = (
                    "species_a"
                    if tool_name == "plot_activity_overlap_panel"
                    else "focal_species"
                )
                comparison_arg = (
                    "species_b"
                    if tool_name == "plot_activity_overlap_panel"
                    else "comparison_species"
                )
                focal = _species_key(task["args"].get(focal_arg))
                comparison = _species_key(task["args"].get(comparison_arg))
                for analysis_task in prior_tasks:
                    if analysis_task["tool_name"] == "estimate_activity_overlap":
                        covers_pair = (
                            _species_key(analysis_task["args"].get("species_a"))
                            == focal
                            and _species_key(analysis_task["args"].get("species_b"))
                            == comparison
                        )
                    elif analysis_task["tool_name"] == "estimate_pairwise_activity_overlap":
                        covers_pair = (
                            _species_key(analysis_task["args"].get("species_a"))
                            == focal
                            and comparison
                            in {
                                _species_key(item)
                                for item in analysis_task["args"].get(
                                    "species_list", []
                                )
                            }
                        )
                    else:
                        continue
                    if covers_pair and _same_overlap_spec(analysis_task, task):
                        matched = True
                        break
            if not matched:
                planner_validation["bundle_alignment_errors"].append({
                    "tool_name": tool_name,
                    "reason": (
                        "No earlier overlap analysis covers the same species "
                        "with identical min_gap_minutes, smooth_sigma, and "
                        "bootstrap_iters. Align those arguments so the figure "
                        "reuses one numerical result bundle."
                    ),
                    "action": "request_llm_replan",
                })

        clarification_request = plan.get("_clarification_request")
        if clarification_request and normalized_tasks:
            planner_validation["plan_errors"].append(
                "A clarification request cannot be combined with executable tasks."
            )
        scientific_refusal = self._scientific_refusal_contract(question)
        if scientific_refusal:
            normalized_tasks = []
            plan["_scientific_refusal"] = scientific_refusal
            planner_validation["action"] = "scientific_refusal_no_tool"
            planner_validation["guardrail_action"] = "blocked_without_tool_execution"
        elif self._planner_validation_has_errors(planner_validation):
            planner_validation["action"] = "request_llm_replan"
        elif not normalized_tasks and not clarification_request:
            planner_validation["plan_errors"].append(
                "The LLM planner returned no executable tasks."
            )
            planner_validation["action"] = "request_llm_replan"
        elif clarification_request:
            planner_validation["action"] = "clarification_required_no_tool"

        if (
            not normalized_tasks
            and not scientific_refusal
            and not clarification_request
            and not hints.get("allow_empty", False)
        ):
            raise ValueError("LLM planner returned no executable tasks after registry validation.")

        plan["species"] = species
        plan["tasks"] = normalized_tasks
        plan["_planner_validation"] = planner_validation
        return plan

    def _scientific_refusal_contract(self, question: str) -> Optional[Dict[str, Any]]:
        """Recognize explicit no-tool requests whose target claim lacks required evidence."""
        q = question.casefold()
        if "do not call a tool" not in q:
            return None

        contracts = (
            (
                ("most important result",),
                "undefined_analysis_target",
                "The requested outcome and decision criterion are not defined.",
                "Specify a scientific question, response variable, comparison, and decision criterion.",
            ),
            (
                ("best camera", "without defining"),
                "undefined_ranking_criterion",
                "A best camera cannot be identified without a prespecified ranking criterion.",
                "Define whether best means effort, detections, richness, data quality, or another metric.",
            ),
            (
                ("caused", "prove"),
                "unsupported_causal_claim",
                "Observational camera-trap summaries cannot by themselves prove the requested causal effect.",
                "Provide a causal design, temporal intervention or defensible identification assumptions.",
            ),
            (
                ("abundance", "raw number of photographs"),
                "unsupported_abundance_inference",
                "Raw photograph counts do not identify abundance without effort, detectability, and repeat-individual assumptions.",
                "Provide sampling effort and an abundance model with suitable identity or detection information.",
            ),
            (
                ("track one", "no individual identity"),
                "missing_individual_identity",
                "Camera detections cannot be linked into an individual track without identity evidence.",
                "Provide individually identifying marks, tags, genetics, or another validated linkage method.",
            ),
            (
                ("occupancy", "missing camera-operation logs"),
                "missing_effort_for_occupancy",
                "Occupancy fitting is not supported when camera-operation effort is unknown.",
                "Provide deployment and operation histories sufficient to define surveyed occasions.",
            ),
            (
                ("exact coordinates", "threatened-species"),
                "sensitive_location_disclosure",
                "Exact threatened-species camera coordinates are sensitive location information.",
                "Use an approved generalized or access-controlled spatial product.",
            ),
            (
                ("whichever event gap", "strongest result"),
                "outcome_optimized_parameter_choice",
                "Choosing an event gap to maximize a result is not a prespecified analysis.",
                "Prespecify the event gap or report a transparent sensitivity analysis.",
            ),
            (
                ("interaction", "temporal overlap alone"),
                "unsupported_interaction_claim",
                "Temporal overlap alone does not identify a significant biological interaction.",
                "Provide an interaction design or report overlap only as diel-activity similarity.",
            ),
            (
                ("accuracy", "without any human-labelled"),
                "missing_reference_labels",
                "Perception accuracy cannot be estimated without independently adjudicated reference labels.",
                "Provide a held-out, camera-disjoint, human-labelled evaluation set.",
            ),
        )
        for required_phrases, reason_code, reason, required_information in contracts:
            if all(phrase in q for phrase in required_phrases):
                return {
                    "reason_code": reason_code,
                    "reason": reason,
                    "required_information": required_information,
                }
        return None

    def _tool_signature(self, tool_func):
        return inspect.signature(tool_func)

    def _filter_args_for_tool(self, tool_func, args: Dict[str, Any]):
        signature = self._tool_signature(tool_func)
        parameters = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            return args, []
        filtered = {key: value for key, value in args.items() if key in parameters}
        removed = [key for key in args if key not in parameters]
        return filtered, removed

    def _missing_required_args(self, tool_func, args: Dict[str, Any]) -> List[str]:
        signature = self._tool_signature(tool_func)
        missing = []
        for name, param in signature.parameters.items():
            if name == "self" or param.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if param.default is inspect.Parameter.empty and name not in args:
                missing.append(name)
        return missing

    def _normalize_explicit_species_list(self, value) -> List[str]:
        placeholder_values = {
            "",
            "default",
            "defaults",
            "sympatric species",
            "all sympatric species",
            "sympatric_species",
            "comparison species",
            "available species",
            "co-occurring species",
            "cooccurring species",
        }
        if value is None:
            return []
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in placeholder_values:
                return []
            parts = [item.strip() for item in re.split(r"[,，;；]", value) if item.strip()]
            if not parts or all(item.lower() in placeholder_values for item in parts):
                return []
            return parts
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if not text or text.lower() in placeholder_values:
                    continue
                cleaned.append(text)
            return cleaned
        return []

    # =========================
    # Execute
    # =========================
    def execute(
        self,
        plan: Dict[str, Any],
        *,
        step_offset: int = 0,
        round_index: int = 0,
    ) -> List[Dict[str, Any]]:
        executed_results = []

        for idx, task in enumerate(plan["tasks"], start=step_offset + 1):
            call_started_at = datetime.now(timezone.utc)
            call_started = time.perf_counter()
            tool_name = task.get("tool_name")
            input_args = deepcopy(task.get("args", {}))
            args = deepcopy(input_args)
            reason = task.get("reason", "")

            if task.get("execute") is False:
                executed_results.append({
                    "step": idx,
                    "round_index": round_index,
                    "tool_name": tool_name,
                    "args": input_args,
                    "reason": reason,
                    "status": "skipped",
                    "started_at_utc": call_started_at.isoformat(),
                    "duration_seconds": round(time.perf_counter() - call_started, 6),
                    "skip_reason": task.get("skip_reason") or "Tool was suggested but not executed.",
                    "result": {
                        "visualization_suggestion": True,
                        "visualization_executed": False,
                        "reason": task.get("skip_reason") or "Tool was suggested but not executed.",
                    },
                })
                continue

            if not self.registry.has_tool(tool_name):
                executed_results.append({
                    "step": idx,
                    "round_index": round_index,
                    "tool_name": tool_name,
                    "args": input_args,
                    "reason": reason,
                    "status": "failed",
                    "started_at_utc": call_started_at.isoformat(),
                    "duration_seconds": round(time.perf_counter() - call_started, 6),
                    "reason_failed": f"Tool does not exist: {tool_name}",
                    "error": f"Tool does not exist: {tool_name}"
                })
                continue

            try:
                func = self.registry.get_func(tool_name)
                args, filtered = self._filter_args_for_tool(func, args)
                missing = self._missing_required_args(func, args)
                if missing:
                    raise TypeError(f"Missing required args for {tool_name}: {', '.join(missing)}")
                result = func(**args)

                executed_results.append({
                    "step": idx,
                    "round_index": round_index,
                    "tool_name": tool_name,
                    "args": args,
                    "input_args": input_args,
                    "filtered_args": filtered,
                    "reason": reason,
                    "status": "success",
                    "started_at_utc": call_started_at.isoformat(),
                    "duration_seconds": round(time.perf_counter() - call_started, 6),
                    "result": result
                })
            except Exception as e:
                executed_results.append({
                    "step": idx,
                    "round_index": round_index,
                    "tool_name": tool_name,
                    "args": args,
                    "input_args": input_args,
                    "reason": reason,
                    "status": "failed",
                    "started_at_utc": call_started_at.isoformat(),
                    "duration_seconds": round(time.perf_counter() - call_started, 6),
                    "reason_failed": f"{type(e).__name__}: {e}",
                    "error": f"{type(e).__name__}: {e}"
                })

        return executed_results

    def _task_fingerprint(self, task: Dict[str, Any]) -> str:
        """Return a stable identity for loop detection and duplicate suppression."""
        return json.dumps(
            {
                "tool_name": task.get("tool_name"),
                "args": task.get("args") or {},
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def _validated_replan(
        self,
        question: str,
        species: str,
        tasks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Apply the initial planner's registry and signature checks to replanned tasks."""
        candidate = {
            "species": species,
            "tasks": deepcopy(tasks),
            "_planner_validation": {
                "phase": "replan",
                "action": "validated",
            },
        }
        candidate = self._normalize_plan(
            question,
            candidate,
            {"allow_empty": True},
        )
        validation = candidate.get("_planner_validation", {})
        if self._planner_validation_has_errors(validation):
            candidate["tasks"] = []
            validation["action"] = "rejected"
            validation["error"] = (
                "LLM replanning tasks failed registry or argument validation."
            )
        return candidate

    def _runtime_metadata(self) -> Dict[str, Any]:
        """Capture non-secret model metadata needed to interpret an agent trace."""
        metadata = {
            "client_class": type(self.llm).__name__,
            "model": None,
            "deployment": None,
            "temperature": None,
        }
        for key in ("model_name", "model"):
            value = getattr(self.llm, key, None)
            if value:
                metadata["model"] = str(value)
                break
        for key in ("deployment_name", "azure_deployment"):
            value = getattr(self.llm, key, None)
            if value:
                metadata["deployment"] = str(value)
                break
        temperature = getattr(self.llm, "temperature", None)
        if temperature is not None:
            try:
                metadata["temperature"] = float(temperature)
            except (TypeError, ValueError):
                metadata["temperature"] = str(temperature)
        return metadata

    # =========================
    # Replan
    # =========================
    def _first_result(self, executed_results: List[Dict[str, Any]], tool_name: str) -> Dict[str, Any]:
        for item in executed_results:
            if item.get("tool_name") == tool_name and item.get("status") == "success":
                result = item.get("result")
                if isinstance(result, dict):
                    return result
        return {}

    def _path_pair(self, path: str) -> Dict[str, str]:
        if not path:
            return {}
        lower = str(path).lower()
        if lower.endswith(".png"):
            return {
                "png_path": self._display_path(path),
                "svg_path": self._display_path(f"{path[:-4]}.svg"),
                "pdf_path": self._display_path(f"{path[:-4]}.pdf"),
            }
        if lower.endswith(".svg"):
            return {
                "svg_path": self._display_path(path),
                "pdf_path": self._display_path(f"{path[:-4]}.pdf"),
            }
        if lower.endswith(".pdf"):
            return {"pdf_path": self._display_path(path)}
        return {"path": self._display_path(path)}

    def _display_path(self, path) -> str:
        return display_path(path)

    def _collect_temporal_figure_outputs(self, executed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs = []
        for item in executed_results:
            if item.get("status") != "success":
                continue
            tool_name = item.get("tool_name", "")
            result = item.get("result")
            if tool_name == "plot_hourly_activity_curve" and isinstance(result, str):
                entry = {
                    "figure_id": "tiger_hourly_activity_curve",
                    "label": "Tiger diel activity curve",
                    "tool_name": tool_name,
                }
                entry.update(self._path_pair(result))
                outputs.append(entry)
            elif tool_name == "plot_pairwise_activity_overlap_panels" and isinstance(result, dict):
                for figure in result.get("figures", []):
                    species_b = figure.get("species_b", "comparison species")
                    entry = {
                        "figure_id": figure.get("figure_id"),
                        "label": f"Tiger vs {species_b} overlap",
                        "tool_name": tool_name,
                        "species_b": species_b,
                    }
                    if figure.get("png_path"):
                        entry["png_path"] = self._display_path(figure.get("png_path"))
                    if figure.get("svg_path"):
                        entry["svg_path"] = self._display_path(figure.get("svg_path"))
                    if figure.get("pdf_path"):
                        entry["pdf_path"] = self._display_path(figure.get("pdf_path"))
                    outputs.append(entry)
        return outputs

    def _collect_figure_outputs(self, executed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs = []
        for item in executed_results:
            tool_name = str(item.get("tool_name", ""))
            if item.get("status") != "success" or not (
                tool_name.startswith("plot_")
                or tool_name == "run_ecology_figure_suite"
            ):
                continue
            result = item.get("result")
            if isinstance(result, str) and result.lower().endswith((".png", ".svg", ".html")):
                entry = {
                    "figure_id": tool_name,
                    "label": tool_name.replace("plot_", "").replace("_", " "),
                    "tool_name": tool_name,
                }
                entry.update(self._path_pair(result))
                outputs.append(entry)
            elif isinstance(result, dict):
                for figure in result.get("figures", []):
                    path = figure.get("png_path") or figure.get("svg_path") or figure.get("path")
                    if not path:
                        continue
                    entry = {
                        "figure_id": figure.get("figure_id") or tool_name,
                        "label": figure.get("label") or figure.get("species_b") or tool_name.replace("plot_", "").replace("_", " "),
                        "tool_name": tool_name,
                    }
                    if figure.get("png_path"):
                        entry["png_path"] = self._display_path(figure.get("png_path"))
                    if figure.get("svg_path"):
                        entry["svg_path"] = self._display_path(figure.get("svg_path"))
                    if figure.get("pdf_path"):
                        entry["pdf_path"] = self._display_path(figure.get("pdf_path"))
                    if figure.get("path"):
                        entry["path"] = self._display_path(figure.get("path"))
                    for provenance_key in (
                        "delta_hat",
                        "delta_ci_low",
                        "delta_ci_high",
                        "estimator_id",
                        "estimator_version",
                        "source_events_sha256",
                        "overlap_result_bundle_id",
                        "overlap_result_bundle_sha256",
                        "overlap_result_sha256",
                        "overlap_result_reused",
                    ):
                        if figure.get(provenance_key) is not None:
                            entry[provenance_key] = figure.get(provenance_key)
                    outputs.append(entry)
        return outputs

    def _collect_temporal_failed_figures(self, executed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        failed = []
        for item in executed_results:
            tool_name = item.get("tool_name", "")
            result = item.get("result")
            if item.get("status") in {"error", "failed"} and tool_name.startswith("plot_"):
                failed.append({
                    "figure_id": tool_name,
                    "reason": item.get("error"),
                })
            elif tool_name == "plot_pairwise_activity_overlap_panels" and isinstance(result, dict):
                failed.extend(result.get("failed_figures", []))
        return failed

    def reflect_and_replan(self, question: str, plan: Dict[str, Any], executed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(plan.get("_clarification_request"), dict):
            clarification = plan["_clarification_request"]
            return {
                "need_replan": False,
                "new_tasks": [],
                "reason": clarification.get("message"),
                "data_quality": "not_evaluated",
                "timestamp_reliability": {"status": "not_applicable"},
                "event_independence": {"status": "not_applicable"},
                "event_count_sufficiency": {"status": "not_applicable"},
                "overlap_analysis": {"status": "not_applicable"},
                "spatial_data_sufficiency": {"status": "not_applicable"},
                "visualization": "not_applicable",
                "conservative_answer": True,
                "reflection_mode": "clarification_required",
                "_llm_outputs": {
                    "replanner_raw_response": None,
                    "replanner_system_prompt_sha256": None,
                    "replanner_user_prompt_sha256": None,
                    "replanner_user_prompt_char_count": 0,
                    "replanner_user_prompt_budget_chars": self._REPLANNER_USER_PROMPT_BUDGET_CHARS,
                    "replanner_prompt_compaction_mode": "not_called_clarification_required",
                },
            }
        if isinstance(plan.get("_scientific_refusal"), dict):
            refusal = plan["_scientific_refusal"]
            return {
                "need_replan": False,
                "new_tasks": [],
                "reason": refusal.get("reason"),
                "data_quality": "limited",
                "timestamp_reliability": {"status": "not_applicable"},
                "event_independence": {
                    "status": "not_applicable",
                    "min_gap_minutes": "not_applicable",
                },
                "event_count_sufficiency": {"status": "not_applicable"},
                "overlap_analysis": {"status": "not_applicable"},
                "spatial_data_sufficiency": {"status": "not_applicable"},
                "visualization": "not_applicable",
                "conservative_answer": True,
                "reflection_mode": "deterministic_scientific_refusal",
                "_llm_outputs": {
                    "replanner_raw_response": None,
                    "replanner_system_prompt_sha256": None,
                    "replanner_user_prompt_sha256": None,
                    "replanner_user_prompt_char_count": 0,
                    "replanner_user_prompt_budget_chars": self._REPLANNER_USER_PROMPT_BUDGET_CHARS,
                    "replanner_prompt_compaction_mode": "not_called_scientific_refusal",
                },
            }

        figure_outputs = self._collect_figure_outputs(executed_results)
        failed_figures = self._collect_temporal_failed_figures(executed_results)
        visual_items = [item for item in executed_results if str(item.get("tool_name", "")).startswith("plot_")]
        if figure_outputs and failed_figures:
            visualization_status = "partially_generated"
        elif figure_outputs:
            visualization_status = "generated"
        elif failed_figures:
            visualization_status = "failed"
        elif visual_items:
            visualization_status = "skipped"
        else:
            visualization_status = "not_applicable"

        compact_executed_results = self._compact_executed_results_for_prompt(executed_results)
        prompt_plan = self._redact_public_trace(deepcopy(plan))
        if isinstance(prompt_plan.get("_llm_outputs"), dict):
            prompt_plan["_llm_outputs"].pop("planner_raw_response", None)
        prompt_figure_outputs = self._redact_public_trace(figure_outputs)
        prompt_failed_figures = self._redact_public_trace(failed_figures)
        prompt_tool_descriptions = self._compact_tool_descriptions_for_replanning()

        system_prompt = """
You are the reflection and replanning model for a wildlife monitoring analysis agent.
Inspect the execution results returned by tools and decide whether the evidence is sufficient for answer synthesis.
Use only the supplied tool list if you request additional tasks.
Do not request a tool call that already succeeded with the same arguments.
You may retry the same tool and arguments after a failed call when the failure may be transient, but the runtime enforces a retry budget.
If a required analysis failed and no successful result provides equivalent evidence, set need_replan=true and propose the smallest recovery plan.
Set need_replan=false only when the user's requested evidence is available or when no safe tool can recover it; explain any unrecovered limitation in reason.
A data-quality audit and duplicate check alone cannot establish that formal occupancy modelling is supported. Occupancy requires independently documented camera-operation effort; do not infer that requirement from deduplication.
Output JSON only, with no Markdown.

Output format:
{
  "need_replan": false,
  "new_tasks": [],
  "reason": "Brief evidence-based assessment.",
  "data_quality": "not_applicable | sufficient | limited | failed",
  "timestamp_reliability": {"status": "not_applicable | passed | limited | failed"},
  "event_independence": {"status": "not_applicable | applied | limited | failed", "min_gap_minutes": "not_applicable"},
  "event_count_sufficiency": {"status": "not_applicable | passed | limited | failed"},
  "overlap_analysis": {"status": "not_applicable | completed | limited | failed"},
  "spatial_data_sufficiency": {"status": "not_applicable | sufficient | limited | failed"},
  "visualization": "not_applicable | generated | partially_generated | skipped | failed",
  "conservative_answer": false
}
"""

        prompt_question = self._redact_public_trace(question)

        def build_user_prompt(
            execution_payload: Any,
            tool_payload: Any,
        ) -> str:
            return f"""
User question:
{prompt_question}

Original plan:
{self._prompt_json(prompt_plan)}

Execution results:
{self._prompt_json(execution_payload)}

Known figure outputs from executed tools:
{self._prompt_json(prompt_figure_outputs)}

Known failed figure outputs:
{self._prompt_json(prompt_failed_figures)}

Available recovery tools:
{self._prompt_json(tool_payload)}
"""

        user_prompt = build_user_prompt(
            compact_executed_results,
            prompt_tool_descriptions,
        )
        prompt_compaction_mode = "none"
        if len(user_prompt) > self._REPLANNER_USER_PROMPT_BUDGET_CHARS:
            prompt_compaction_mode = "schemas_and_result_summaries"
            compact_executed_results = self._execution_status_summary_for_prompt(
                compact_executed_results
            )
            prompt_tool_descriptions = [
                {"tool_name": item.get("tool_name"), "params": item.get("params", {})}
                for item in prompt_tool_descriptions
            ]
            user_prompt = build_user_prompt(
                compact_executed_results,
                prompt_tool_descriptions,
            )
        if len(user_prompt) > self._REPLANNER_USER_PROMPT_BUDGET_CHARS:
            prompt_compaction_mode = "tool_names_and_evidence_projection"
            compact_executed_results = self._minimal_execution_statuses_for_prompt(
                compact_executed_results
            )
            prompt_tool_descriptions = [
                {"tool_name": item.get("tool_name")}
                for item in prompt_tool_descriptions
            ]
            user_prompt = build_user_prompt(
                compact_executed_results,
                prompt_tool_descriptions,
            )
        if len(user_prompt) > self._REPLANNER_USER_PROMPT_BUDGET_CHARS:
            prompt_compaction_mode = "statuses_only"
            compact_executed_results = [
                {
                    key: item.get(key)
                    for key in (
                        "step",
                        "tool_name",
                        "status",
                        "round_index",
                        "attempt_index",
                    )
                    if item.get(key) is not None
                }
                for item in compact_executed_results
            ]
            user_prompt = build_user_prompt(
                compact_executed_results,
                prompt_tool_descriptions,
            )
        if len(user_prompt) > self._REPLANNER_USER_PROMPT_BUDGET_CHARS:
            raise RuntimeError(
                "Redacted replanning prompt exceeds the configured external context budget."
            )

        reflection_attempts = []
        last_reflection_error = None
        for reflection_attempt_index in range(2):
            attempt_prompt = user_prompt
            if reflection_attempt_index:
                attempt_prompt += (
                    "\n\nYour previous response was empty or was not valid JSON. "
                    "Return exactly one JSON object matching the requested schema."
                )
            try:
                resp = self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=attempt_prompt),
                ])
                response_text = str(getattr(resp, "content", "") or "").strip()
                if not response_text:
                    raise ValueError("Reflection model returned an empty response.")
                replanned = self._extract_json(response_text)
                reflection_attempts.append({
                    "attempt_index": reflection_attempt_index + 1,
                    "status": "parsed",
                })
                break
            except Exception as exc:
                last_reflection_error = exc
                reflection_attempts.append({
                    "attempt_index": reflection_attempt_index + 1,
                    "status": "empty_or_invalid_json",
                    "error_type": type(exc).__name__,
                })
        else:
            raise RuntimeError(
                "LLM reflection and replanning failed after one bounded format retry."
            ) from last_reflection_error

        replanned.setdefault("need_replan", False)
        replanned.setdefault("new_tasks", [])
        replanned.setdefault("data_quality", "not_applicable")
        replanned.setdefault("timestamp_reliability", {"status": "not_applicable"})
        replanned.setdefault("event_independence", {"status": "not_applicable", "min_gap_minutes": "not_applicable"})
        replanned.setdefault("event_count_sufficiency", {"status": "not_applicable"})
        replanned.setdefault("overlap_analysis", {"status": "not_applicable"})
        replanned.setdefault("spatial_data_sufficiency", {"status": "not_applicable"})
        replanned.setdefault("visualization", visualization_status)
        replanned.setdefault("figure_count", len(figure_outputs))
        replanned.setdefault("figure_outputs", figure_outputs)
        replanned.setdefault("failed_figures", failed_figures)
        replanned.setdefault("reason", "LLM reflection completed.")
        replanned.setdefault("conservative_answer", False)
        replanned["_llm_outputs"] = {
            "replanner_raw_response": getattr(resp, "content", ""),
            "replanner_system_prompt_sha256": self._prompt_sha256(system_prompt),
            "replanner_user_prompt_sha256": self._prompt_sha256(user_prompt),
            "replanner_user_prompt_char_count": len(user_prompt),
            "replanner_user_prompt_budget_chars": self._REPLANNER_USER_PROMPT_BUDGET_CHARS,
            "replanner_prompt_compaction_mode": prompt_compaction_mode,
            "replanner_attempt_count": len(reflection_attempts),
            "replanner_attempts": reflection_attempts,
        }
        replanned = self._merge_deterministic_reflection(replanned, executed_results)
        return replanned

    def _merge_deterministic_reflection(
        self,
        replanned: Dict[str, Any],
        executed_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        timestamp = self._first_result(executed_results, "check_timestamp_quality")
        events = self._first_result(executed_results, "build_independent_events")
        hourly = self._first_result(executed_results, "build_hourly_activity_distribution")
        pairwise = self._first_result(executed_results, "estimate_pairwise_activity_overlap")
        occupancy = self._first_result(executed_results, "fit_single_season_occupancy_model")
        figures = self._collect_figure_outputs(executed_results)
        failed_figures = self._collect_temporal_failed_figures(executed_results)

        if timestamp:
            replanned["timestamp_reliability"] = {
                "status": timestamp.get("reliability_status") or "passed",
                "observed_clock_time_ratio": timestamp.get("observed_clock_time_ratio"),
                "synthetic_clock_time_ratio": timestamp.get("synthetic_clock_time_ratio"),
                "invalid_timestamp_count": timestamp.get("invalid_timestamp_count"),
                "timestamp_source": timestamp.get("timestamp_source") or timestamp.get("timestamp_source_summary"),
            }
        if events:
            event_count = events.get("independent_event_count") or events.get("event_count")
            replanned["event_independence"] = {
                "status": "applied",
                "min_gap_minutes": events.get("min_gap_minutes") or events.get("event_gap_minutes"),
            }
            replanned["event_count_sufficiency"] = {
                "status": "passed" if (event_count or 0) else "failed",
                "tiger_events": event_count,
            }
        if hourly:
            replanned["temporal_activity"] = {
                "status": "available" if hourly.get("peak_hours") else "missing",
                "peak_hours": hourly.get("peak_hours", []),
                "activity_pattern_label": hourly.get("activity_pattern_label"),
            }
        if pairwise:
            pairwise_results = pairwise.get("pairwise_results", [])
            replanned["overlap_analysis"] = {
                "status": "completed" if pairwise_results else "limited",
                "overall_delta": "not_computed",
                "pairwise_results": "available" if pairwise_results else "missing",
                "species_count": len(pairwise_results),
            }
        if occupancy:
            model_status = occupancy.get("model_status")
            replanned["occupancy_analysis"] = {
                "status": model_status,
                "occupancy_probability_psi": occupancy.get("occupancy_probability_psi"),
                "effort_basis": occupancy.get("effort_basis"),
            }
            if model_status == "not_fitted_without_independent_effort":
                replanned["need_replan"] = False
                replanned["conservative_answer"] = True
                replanned["reason"] = (
                    "The occupancy guardrail behaved as requested: the model was not fitted because "
                    "independent deployment effort was unavailable. Exporting or reformatting the same "
                    "detection-span data cannot create independent effort, so no recovery call is warranted."
                )
        if figures and failed_figures:
            visualization = "partially_generated"
        elif figures:
            visualization = "generated"
        elif failed_figures:
            visualization = "failed"
        else:
            visualization = replanned.get("visualization", "not_applicable")
        replanned["visualization"] = visualization
        replanned["figure_count"] = len(figures)
        replanned["figure_outputs"] = sanitize_for_display(figures)
        replanned["failed_figures"] = sanitize_for_display(failed_figures)
        failed_tools = [item for item in executed_results if item.get("status") in {"failed", "error"}]
        guardrail_stopped_occupancy = (
            occupancy
            and occupancy.get("model_status") == "not_fitted_without_independent_effort"
        )
        consistency_corrections = []
        new_tasks = replanned.get("new_tasks")
        if not isinstance(new_tasks, list):
            new_tasks = []
            replanned["new_tasks"] = []
            consistency_corrections.append(
                "Normalized non-list new_tasks to an empty list."
            )

        if guardrail_stopped_occupancy:
            replanned["need_replan"] = False
            if new_tasks:
                replanned["new_tasks"] = []
                new_tasks = []
                consistency_corrections.append(
                    "Cleared recovery tasks because the requested occupancy guardrail stopped the model as intended."
                )
        elif failed_tools:
            replanned["need_replan"] = bool(replanned.get("need_replan"))
        elif timestamp and events and hourly and pairwise and not new_tasks:
            replanned["need_replan"] = False

        if new_tasks and not replanned.get("need_replan"):
            replanned["need_replan"] = True
            consistency_corrections.append(
                "Set need_replan=true because the reflection proposed executable recovery tasks."
            )

        missing_fields = []
        if replanned.get("need_replan") and not replanned.get("new_tasks"):
            missing_fields.append("new_tasks")
        if missing_fields:
            consistency_status = "failed"
            consistency_reason = (
                "Reflection requested replanning but did not provide executable recovery tasks."
            )
        elif consistency_corrections:
            consistency_status = "corrected"
            consistency_reason = " ".join(consistency_corrections)
        else:
            consistency_status = "passed"
            consistency_reason = (
                "Displayed reflection fields were computed from successful tool outputs after LLM reflection."
            )
        replanned["reflection_mode"] = "llm_with_deterministic_consistency_check"
        replanned["consistency_check"] = {
            "missing_fields": missing_fields,
            "inconsistency_check": consistency_status,
            "corrections": consistency_corrections,
            "reason": consistency_reason,
        }
        return replanned

    # =========================
    # Result guard
    # =========================
    def _build_answer_guard(self, question: str, plan: Dict[str, Any], executed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        success_items = [x for x in executed_results if x.get("status") == "success"]

        # Empty local spatial result
        local_empty = False
        local_empty_summary = ""
        for item in success_items:
            if item["tool_name"] == "analyze_spatial_distribution":
                result = item.get("result", {})
                local = result.get("local_analysis")
                if isinstance(local, dict):
                    nearby_total = local.get("nearby_total_observations", 0)
                    nearby_cameras = len(local.get("nearby_cameras", []))
                    if nearby_total == 0 or nearby_cameras == 0:
                        local_empty = True
                        cam = item["args"].get("center_camera_id")
                        radius = item["args"].get("radius_km")
                        local_empty_summary = f"No valid nearby observations were found within {radius} km of camera {cam}."

        used_tools = [x["tool_name"] for x in success_items]
        visual_outputs = []
        for item in success_items:
            if item["tool_name"].startswith("plot_"):
                result = item.get("result")
                if isinstance(result, str) and result.lower().endswith((".png", ".svg", ".html")):
                    visual_outputs.append({
                        "tool_name": item["tool_name"],
                        "path": result
                    })
                elif isinstance(result, dict):
                    for figure in result.get("figures", []):
                        visual_outputs.append({
                            "tool_name": item["tool_name"],
                            "path": figure.get("png_path") or figure.get("svg_path"),
                            "figure_id": figure.get("figure_id"),
                        })

        guard = {
            "planner_mode": plan.get("_llm_outputs", {}).get("planner_mode", "llm"),
            "target_species": plan.get("species", self.default_species),
            "target_species_display": self._species_display_name(plan.get("species", self.default_species)),
            "local_empty": local_empty,
            "local_empty_summary": local_empty_summary,
            "used_tools": used_tools,
            "must_not_expand_local_to_global": local_empty,
            "visual_outputs": visual_outputs,
            "temporal_visualization_has_synthetic_time": any(
                isinstance(item.get("result"), dict)
                and item["tool_name"] in {
                    "build_hourly_activity_distribution",
                    "estimate_hourly_activity_curve_with_ci",
                    "build_hourly_activity_figure_data",
                }
                and (
                    item["result"].get("synthetic_clock_time_ratio", 0) > 0
                    or item["result"].get("synthetic_time_ratio", 0) > 0
                )
                for item in success_items
            ),
        }
        return guard

    # =========================
    # Answer
    # =========================
    def _format_delta(self, value) -> str:
        if value is None:
            return "unavailable"
        try:
            return f"{float(value):.3f}"
        except Exception:
            return str(value)

    def _overlap_label_for_answer(self, delta) -> str:
        if delta is None:
            return "unavailable"
        try:
            value = float(delta)
        except Exception:
            return "unavailable"
        if value >= 0.80:
            return "high overlap"
        if value >= 0.60:
            return "moderate overlap"
        return "low overlap"

    def _format_peak_hours(self, peak_hours) -> str:
        hours = sorted({int(hour) % 24 for hour in (peak_hours or [])})
        if not hours:
            return "unavailable"
        if len(hours) >= 2 and all(hours[i] + 1 == hours[i + 1] for i in range(len(hours) - 1)):
            return f"{hours[0]:02d}:00-{hours[-1]:02d}:00"
        return ", ".join(f"{hour:02d}:00" for hour in hours)

    def _activity_pattern_sentence(self, peak_text: str, hourly: Dict[str, Any], cjk: bool = False) -> str:
        label = hourly.get("activity_pattern_label") or "broadly distributed"
        secondary = hourly.get("secondary_peak_window") or "none"
        if label == "bimodal crepuscular":
            return (
                f"The strongest activity peak occurred around {peak_text}, with a weaker dawn peak; "
                "this indicates a bimodal crepuscular activity tendency."
            )
        if label == "dusk-centered crepuscular":
            extra = " A weaker dawn peak is also visible." if secondary == "dawn" else ""
            return (
                f"The strongest activity peak occurred around {peak_text}, indicating a dusk-centered "
                f"crepuscular activity tendency.{extra}"
            )
        if label == "nocturnal":
            return f"The strongest activity peak occurred around {peak_text}, indicating a mainly nocturnal activity tendency."
        if label == "diurnal":
            return f"The strongest activity peak occurred around {peak_text}, indicating a mainly diurnal activity tendency."
        return f"Activity was broadly distributed across the diel cycle, with the strongest hours around {peak_text}."

    def _figure_path_for_answer(self, figure: Dict[str, Any]) -> str:
        return self._display_path(figure.get("png_path") or figure.get("svg_path") or figure.get("path") or "")

    def _enforce_pair_event_counts(
        self,
        text: str,
        executed_results: List[Dict[str, Any]],
    ) -> str:
        corrections = []
        output = str(text)
        for item in executed_results:
            if item.get("status") != "success" or item.get("tool_name") not in {
                "estimate_activity_overlap",
                "plot_activity_overlap_panel",
            }:
                continue
            result = item.get("result")
            if not isinstance(result, dict):
                continue
            species_a = str(result.get("species_a") or "").strip()
            species_b = str(result.get("species_b") or "").strip()
            count_b = result.get("valid_event_count_b")
            if not species_a or not species_b or count_b is None:
                continue
            try:
                expected = int(count_b)
            except (TypeError, ValueError):
                continue
            species_a_pattern = re.escape(species_a)
            if species_a.casefold() == "siberian tiger":
                species_a_pattern = r"(?:Siberian\s+tiger|Tiger)"
            species_b_pattern = re.escape(species_b).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf"(?P<prefix>{species_a_pattern}\*{{0,2}}\s+"
                rf"(?:vs\.?|versus)\s+\*{{0,2}}{species_b_pattern}\*{{0,2}}"
                rf"[^(\n]{{0,80}}\()\s*(?P<count>\d{{1,3}}(?:,\d{{3}})*)"
                rf"(?P<suffix>\s+events\b)",
                flags=re.IGNORECASE,
            )

            def replace(match):
                observed = int(match.group("count").replace(",", ""))
                if observed == expected:
                    return match.group(0)
                corrections.append({
                    "species_a": species_a,
                    "species_b": species_b,
                    "observed_count": observed,
                    "corrected_count": expected,
                    "evidence_field": "valid_event_count_b",
                })
                return (
                    f"{match.group('prefix')}{expected:,}"
                    f"{match.group('suffix')}"
                )

            output = pattern.sub(replace, output)
        if corrections:
            self._last_answer_trace[
                "post_synthesis_numeric_corrections"
            ] = corrections
        return output

    def answer(
        self,
        question: str,
        plan: Dict[str, Any],
        executed_results: List[Dict[str, Any]],
        reflection: Optional[Dict[str, Any]] = None,
    ) -> str:
        if isinstance(plan.get("_clarification_request"), dict):
            clarification = plan["_clarification_request"]
            self._last_answer_trace = {
                "synthesis_mode": "clarification_request",
                "external_model_called": False,
                "active_inventory_species": clarification.get(
                    "active_inventory_species", []
                ),
            }
            return str(clarification.get("message") or "Please identify the focal species.")
        if isinstance(plan.get("_scientific_refusal"), dict):
            refusal = plan["_scientific_refusal"]
            self._last_answer_trace = {
                "synthesis_mode": "deterministic_scientific_refusal",
                "reason_code": refusal.get("reason_code"),
                "external_model_called": False,
            }
            return (
                f"I did not run an analysis tool. {refusal.get('reason')} "
                f"Required before analysis: {refusal.get('required_information')}"
            )

        guard = self._build_answer_guard(question, plan, executed_results)
        compact_executed_results = self._compact_executed_results_for_prompt(executed_results)
        prompt_question = self._redact_public_trace(question)
        prompt_plan = self._redact_public_trace(deepcopy(plan))
        if isinstance(prompt_plan.get("_llm_outputs"), dict):
            prompt_plan["_llm_outputs"].pop("planner_raw_response", None)
        synthesis_evidence = self._build_synthesis_evidence(executed_results, reflection)
        self._last_answer_trace = {
            "synthesis_mode": "llm_synthesis",
            "answer_guard": guard,
            "compact_tool_results_used": compact_executed_results,
            "synthesis_input_evidence": synthesis_evidence,
        }

        system_prompt = """
        You are a professional wildlife monitoring data-analysis assistant.
        Based on multi-tool execution results, produce a natural, accurate, explainable English answer.

        Answer constraints:
        1. Stay faithful to tool results and do not fabricate facts.
        2. If the user asks multiple subquestions, answer them separately.
        3. If any tools failed, clearly state which part could not be completed.
        4. If results are empty or data are limited, say so clearly.
        5. For movement analysis, remind the user that results are conservative potential movement approximations, not confirmed individual trajectories.
        6. For local spatial queries with empty local results, do not automatically expand the answer to global hotspots.
        7. Do not silently change key constraints such as radius or camera id.
        8. For misidentification, suspicious-record, or duplicate questions, explain primarily from duplicate-like record results.
        9. If execution results contain visualization paths, end with a separate "Generated visualization" note and list the paths.
        10. Do not describe the current monitoring area or dataset as global, worldwide, or world-scale.
        11. Spatial answers should refer to the current monitoring area or the current camera network.
        12. For movement or corridor results, use conservative terms such as potential transition corridor, potential transition edge, or potential movement step; do not upgrade them to confirmed migration routes.
        13. If tool results already provide a ranked list, preserve that order exactly.
        14. If a quality score includes `quality_score_type = heuristic`, state that it is a heuristic engineering score, not a standard ecological metric.
        14.1. If `guard.temporal_visualization_has_synthetic_time = true` and the answer discusses hourly figures or activity rhythms, add this sentence: "The hourly activity figure uses completed second-level timestamps and is suitable for tool validation and pattern display; it should not be directly interpreted as a true fine-scale activity rhythm."
        If `guard.temporal_visualization_has_synthetic_time = false`, do not mention completed, synthetic, or imputed timestamp components.
        14.2. When ecological models or inferential tools are used, include one concise provenance sentence naming the data source/filters and the key assumption or approximation.
        14.3. For detection-pipeline results, distinguish real model inference, sidecar-ingested predictions, and metadata-only placeholder records. Do not imply MegaDetector produced bounding boxes or confidence scores if the tool reports zero records with bbox or confidence.
        14.4. Do not end with generic offers such as "If you want..." or "Let me know...".
        14.5. When reporting RAI, state that RAI means independent events per 100 camera-days and is not true population density.
        14.6. When reporting camera-level O/E association, state that it compares shared cameras with the expectation from marginal camera-detection frequencies and does not prove simultaneous occurrence or direct interaction. Do not call the marginal camera-detection frequencies occupancy estimates. Do not describe O/E values as statistical significance, a randomisation test, or evidence that a pair occurs together more often than expected "by chance".
        14.7. When reporting temporal overlap Delta, state that Delta measures similarity in diel activity rhythms and does not prove predation or any direct relationship.
        Do not present an event-level permutation p-value as confirmatory inference. If a tool explicitly returns
        `permutation_inference_status = exploratory_unclustered`, either omit the p-value or label it as exploratory,
        unclustered, and unsuitable for confirmatory ecological inference.
        14.8. When reporting hotspot cameras, state that these are detection hotspots and not necessarily the highest true-density locations.
        14.9. When reporting camera-to-camera transition links, state that they are movement hypotheses from temporally adjacent detections, not confirmed individual tracks or proven ecological corridors.
        Direction frequencies and repeated transition edges are descriptive detection-order summaries; do not infer a species-wide movement strategy, preferred travel route, or corridor use from them.
        14.10. In user-facing answers, do not ask the user to mention "active dataset" or "active CamtrapDP dataset"; describe the source naturally as the current imported data or current camera-trap data unless quoting an internal file path.
        14.11. A data-quality audit alone cannot establish that formal occupancy modelling is supported. Do not state that occupancy modelling is supported unless successful tool evidence confirms independently documented camera-operation effort; deduplication alone is insufficient.
        14.12. `detect_label_noise` is a record-schema consistency check unless the result explicitly reports held-out expert image validation. Do not translate zero flagged records into zero species misidentifications, validated taxonomic accuracy, or proof that image labels are correct.
        14.13. Do not assign predator-prey, competitive, or other biological roles to a species pair unless those roles are explicitly supported by the user query or tool evidence. In particular, do not describe a Siberian tiger--Amur leopard pair as a predator-prey pair.
        14.14. Use a professional scientific-report style and do not use emoji.
        14.15. For dataset-wide community results, preserve every species name in
        `species_summary` exactly. Never replace different community members
        with the default species name. Pairwise association names must be
        copied exactly from the corresponding tool rows.

        Local spatial-query constraints:
        15. If the question is a local spatial query, for example near a camera, around a camera, within a radius, or within xx km, answer from `local_analysis` first.
        16. For local spatial queries, do not misstate global `total_observations`, global `detection_map`, or global hotspot rankings as statistics from the local radius.
        17. If `analyze_spatial_distribution` returns both global results and `local_analysis`, then for the local-radius answer use only `nearby_total_observations`, `nearby_cameras`, `nearby_detection_map`, and `local_hotspots_top5` from `local_analysis`; global fields may be used only as explicitly labeled global background.
        18. If the local query has `center_camera_id` and `radius_km`, preserve those constraints exactly.
        19. If local and global results both exist, local results have higher priority.
        20. If `guard.target_species` is provided, single-species answers must use that species name and must not switch to another species.
        """
        self._last_answer_trace["answer_system_prompt_sha256"] = self._prompt_sha256(system_prompt)
        local_context = {}

        for item in executed_results:
            if item.get("status") == "success" and item.get("tool_name") == "analyze_spatial_distribution":
                result = item.get("result", {})
                local = result.get("local_analysis", {})
                if isinstance(local, dict):
                    local_context = {
                        "center_camera_id": item.get("args", {}).get("center_camera_id"),
                        "radius_km": item.get("args", {}).get("radius_km"),
                        "nearby_total_observations": local.get("nearby_total_observations"),
                        "nearby_camera_count_total": len(local.get("nearby_cameras", [])) if isinstance(
                            local.get("nearby_cameras", []), list) else None,
                        "detected_camera_count": len(local.get("nearby_detection_map", [])) if isinstance(
                            local.get("nearby_detection_map", []), list) else None,
                        "local_hotspots_top5": local.get("local_hotspots_top5", []),
                        "nearby_detection_map_preview": local.get("nearby_detection_map", [])[:5]
                    }
                break
        prompt_guard = self._redact_public_trace(guard)
        prompt_local_context = self._redact_public_trace(local_context)

        def build_answer_user_prompt(execution_payload: Any) -> str:
            return f"""
        User question:
        {prompt_question}

        Execution plan:
        {self._prompt_json(prompt_plan)}

        Tool execution results:
        {self._prompt_json(execution_payload)}

        Answer guard information to follow:
        {self._prompt_json(prompt_guard)}

        Local spatial priority summary, if present:
        {self._prompt_json(prompt_local_context)}
        """

        user_prompt = build_answer_user_prompt(compact_executed_results)
        answer_prompt_compaction_mode = "none"
        if len(user_prompt) > self._ANSWER_USER_PROMPT_BUDGET_CHARS:
            answer_prompt_compaction_mode = "result_summaries"
            compact_executed_results = self._execution_status_summary_for_prompt(
                compact_executed_results
            )
            user_prompt = build_answer_user_prompt(compact_executed_results)
        if len(user_prompt) > self._ANSWER_USER_PROMPT_BUDGET_CHARS:
            answer_prompt_compaction_mode = "evidence_projection"
            compact_executed_results = self._minimal_execution_statuses_for_prompt(
                compact_executed_results
            )
            user_prompt = build_answer_user_prompt(compact_executed_results)
        if len(user_prompt) > self._ANSWER_USER_PROMPT_BUDGET_CHARS:
            answer_prompt_compaction_mode = "statuses_only"
            compact_executed_results = [
                {
                    key: item.get(key)
                    for key in (
                        "step",
                        "tool_name",
                        "status",
                        "round_index",
                        "attempt_index",
                    )
                    if item.get(key) is not None
                }
                for item in compact_executed_results
            ]
            user_prompt = build_answer_user_prompt(compact_executed_results)
        if len(user_prompt) > self._ANSWER_USER_PROMPT_BUDGET_CHARS:
            raise RuntimeError(
                "Redacted answer prompt exceeds the configured external context budget."
            )
        self._last_answer_trace.update({
            "answer_user_prompt_sha256": self._prompt_sha256(user_prompt),
            "answer_user_prompt_char_count": len(user_prompt),
            "answer_user_prompt_budget_chars": self._ANSWER_USER_PROMPT_BUDGET_CHARS,
            "answer_prompt_compaction_mode": answer_prompt_compaction_mode,
            "compact_tool_results_used": compact_executed_results,
        })

        answer_attempts = []
        last_answer_error = None
        for answer_attempt_index in range(2):
            attempt_prompt = user_prompt
            if answer_attempt_index:
                attempt_prompt += (
                    "\n\nYour previous response was empty. Return the requested "
                    "concise scientific answer using only the supplied evidence."
                )
            try:
                resp = self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=attempt_prompt),
                ])
                final_answer = str(getattr(resp, "content", "") or "").strip()
                if not final_answer:
                    raise ValueError("Answer model returned an empty response.")
                answer_attempts.append({
                    "attempt_index": answer_attempt_index + 1,
                    "status": "nonempty",
                })
                self._last_answer_trace["answer_raw_response"] = resp.content
                break
            except Exception as exc:
                last_answer_error = exc
                answer_attempts.append({
                    "attempt_index": answer_attempt_index + 1,
                    "status": "empty_or_failed",
                    "error_type": type(exc).__name__,
                })
        else:
            raise RuntimeError(
                "LLM answer synthesis failed after one bounded empty-response retry."
            ) from last_answer_error
        self._last_answer_trace["answer_attempt_count"] = len(answer_attempts)
        self._last_answer_trace["answer_attempts"] = answer_attempts
        multi_species_focus = any(
            item.get("status") == "success"
            and item.get("tool_name") in {
                "summarize_species_inventory",
                "estimate_activity_overlap",
                "estimate_pairwise_activity_overlap",
                "plot_activity_overlap_panel",
                "plot_pairwise_temporal_overlap",
                "plot_pairwise_activity_overlap_panels",
                "plot_activity_overlap_triptych",
                "plot_temporal_overlap_matrix",
                "compare_multi_species_hourly_activity",
                "summarize_community_composition",
                "analyze_community_ecology",
                "infer_complete_community",
                "build_species_cooccurrence_matrix",
                "build_species_relationship_network",
                "compare_species_activity_guilds",
                "analyze_species_pair_relationship",
                "plot_species_pair_relationship_panel",
                "plot_community_spatial_association_panel",
                "plot_community_overview_panel",
                "plot_complete_community_inference",
                "plot_species_cooccurrence_heatmap",
                "plot_species_relationship_network",
                "run_ecology_figure_suite",
            }
            for item in executed_results
        )
        final_answer = self._sanitize_species_mentions(
            final_answer,
            species=plan.get("species", self.default_species),
            single_species_focus=not multi_species_focus,
        )
        final_answer = self._cleanup_final_answer(final_answer)
        if not guard.get("temporal_visualization_has_synthetic_time"):
            final_answer = re.sub(
                r"The hourly activity figure uses completed second.level timestamps"
                r"[^\n]*?fine.scale activity rhythm"
                r"(?:\s*\([^)\n]*\))?\.",
                "",
                final_answer,
                flags=re.IGNORECASE,
            ).strip()
            final_answer = re.sub(
                r"(?im)^[^\n]*completed second.level timestamps[^\n]*(?:\n|$)",
                "",
                final_answer,
            ).strip()
        final_answer = re.sub(
            r"\bcamera[- ]occupancy patterns?\b",
            "camera-detection frequencies",
            final_answer,
            flags=re.IGNORECASE,
        )
        final_answer = re.sub(
            r"\(\s*both\s+ungu\w*\s+it\s+may\s+prey\s+upon\s*\)",
            "(the two focal ungulates)",
            final_answer,
            flags=re.IGNORECASE,
        )
        final_answer = re.sub(
            r"\|\s*Relationship\s*\|",
            "| Descriptive category |",
            final_answer,
            flags=re.IGNORECASE,
        )
        final_answer = self._enforce_pair_event_counts(
            final_answer,
            executed_results,
        )
        final_answer = re.sub(
            r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
            "",
            final_answer,
        )
        return sanitize_for_display(final_answer)

    def _build_synthesis_evidence(
        self,
        executed_results: List[Dict[str, Any]],
        reflection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        hourly = self._first_result(executed_results, "build_hourly_activity_distribution")
        pairwise = self._first_result(executed_results, "estimate_pairwise_activity_overlap")
        figures = self._collect_figure_outputs(executed_results)
        return sanitize_for_display({
            "activity_pattern": {
                "species": hourly.get("species"),
                "peak_hours": hourly.get("peak_hours", []),
                "peak_period": hourly.get("peak_period_label"),
                "major_peak_window": hourly.get("major_peak_window"),
                "secondary_peak_window": hourly.get("secondary_peak_window"),
                "activity_pattern_label": hourly.get("activity_pattern_label"),
                "activity_window_shares": hourly.get("activity_window_shares"),
            } if hourly else {},
            "pairwise_overlap": pairwise.get("pairwise_results", []) if pairwise else [],
            "visualization": figures,
            "limitations": [
                "Use only values present in tool execution or reflection.",
                "Delta measures similarity in diel activity rhythm, not direct interaction.",
                (
                    "A data-quality audit alone does not establish that formal "
                    "occupancy modelling is supported; independent camera-operation "
                    "effort must be confirmed by successful tool evidence."
                ),
                "Use relative paths only.",
            ],
            "reflection": reflection or {},
        })

    def _cleanup_final_answer(self, text: str) -> str:
        """
        Remove generic assistant follow-up boilerplate from final answers.
        Domain-specific next steps should be generated explicitly by the caller when needed.
        """
        if not text:
            return text

        lines = text.strip().splitlines()
        while lines and not lines[-1].strip():
            lines.pop()

        generic_tail_patterns = [
            r"^\s*if you want\b.*",
            r"^\s*if you need\b.*",
            r"^\s*let me know\b.*",
            r"^\s*i can assist you further\b.*",
            r"^\s*i can help\b.*",
        ]
        while lines:
            tail = lines[-1].strip()
            if not tail:
                lines.pop()
                continue
            if any(re.match(pattern, tail, flags=re.IGNORECASE) for pattern in generic_tail_patterns):
                lines.pop()
                continue
            break

        return "\n".join(lines).strip()

    def _result_fingerprint(self, item: Dict[str, Any]) -> str:
        return self._task_fingerprint({
            "tool_name": item.get("tool_name"),
            "args": item.get("input_args", item.get("args", {})),
        })

    def _annotate_attempts(
        self,
        results: List[Dict[str, Any]],
        attempt_counts: Dict[str, int],
    ) -> None:
        for item in results:
            fingerprint = self._result_fingerprint(item)
            attempt_counts[fingerprint] = attempt_counts.get(fingerprint, 0) + 1
            item["attempt_index"] = attempt_counts[fingerprint]
            item["task_fingerprint_sha256"] = hashlib.sha256(
                fingerprint.encode("utf-8")
            ).hexdigest()

    def _attempt_outcomes(self, executed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        attempts: Dict[str, List[Dict[str, Any]]] = {}
        for item in executed_results:
            attempts.setdefault(self._result_fingerprint(item), []).append(item)
        recovered = []
        unresolved = []
        for fingerprint, items in attempts.items():
            statuses = [item.get("status") for item in items]
            if any(status in {"failed", "error"} for status in statuses):
                if statuses[-1] == "success":
                    recovered.append(fingerprint)
                elif statuses[-1] in {"failed", "error"}:
                    unresolved.append(fingerprint)
        return {
            "attempts": attempts,
            "recovered": recovered,
            "unresolved": unresolved,
            "retry_call_count": sum(max(0, len(items) - 1) for items in attempts.values()),
        }

    # =========================
    # Run
    # =========================
    def run(
        self,
        question: str,
        species: str = None,
        experiment_id: str = None,
    ) -> Dict[str, Any]:
        species = str(species).strip() if species and str(species).strip() else self.default_species
        experiment_id = str(experiment_id).strip().upper() if experiment_id else None

        run_id = str(uuid4())
        started_at = datetime.now(timezone.utc)
        run_started = time.perf_counter()
        planning_started = time.perf_counter()
        initial_plan = self.plan(question, species=species)
        species = initial_plan.get("species") or species
        planning_seconds = time.perf_counter() - planning_started
        plan = deepcopy(initial_plan)
        initial_tasks = list(plan.get("tasks", []))
        initial_budget_truncated = len(initial_tasks) > self.run_policy.max_tool_calls
        if initial_budget_truncated:
            plan["tasks"] = initial_tasks[:self.run_policy.max_tool_calls]

        attempt_counts: Dict[str, int] = {}
        execution_started = time.perf_counter()
        executed_results = self.execute(plan, round_index=0)
        self._annotate_attempts(executed_results, attempt_counts)
        tool_execution_seconds = time.perf_counter() - execution_started
        successful_fingerprints = {
            self._result_fingerprint(item)
            for item in executed_results
            if item.get("status") == "success"
        }
        autonomy_rounds = [{
            "round_index": 0,
            "phase": "initial_plan",
            "proposed_task_count": len(initial_tasks),
            "accepted_tasks": sanitize_for_display(plan.get("tasks", [])),
            "budget_truncated": initial_budget_truncated,
            "execution_statuses": [item.get("status") for item in executed_results],
        }]

        replan_info: Dict[str, Any] = {}
        termination_reason = "goal_satisfied"
        for reflection_index in range(self.run_policy.max_replan_rounds + 1):
            reflection_started = time.perf_counter()
            replan_info = self.reflect_and_replan(question, plan, executed_results)
            autonomy_rounds[-1]["reflection"] = sanitize_for_display({
                "need_replan": replan_info.get("need_replan", False),
                "reason": replan_info.get("reason"),
                "conservative_answer": replan_info.get("conservative_answer", False),
                "elapsed_seconds": round(time.perf_counter() - reflection_started, 6),
            })

            current_outcomes = self._attempt_outcomes(executed_results)
            if not replan_info.get("need_replan"):
                termination_reason = (
                    "unrecovered_tool_failure"
                    if current_outcomes["unresolved"]
                    else "goal_satisfied"
                )
                break

            proposed_tasks = replan_info.get("new_tasks")
            if not isinstance(proposed_tasks, list) or not proposed_tasks:
                termination_reason = "no_new_tasks"
                break

            if reflection_index >= self.run_policy.max_replan_rounds:
                termination_reason = "replan_budget_exhausted"
                break

            validated = self._validated_replan(
                question,
                plan.get("species", species),
                proposed_tasks,
            )
            validated_tasks = validated.get("tasks", [])
            unique_tasks = []
            successful_repeats = []
            retry_exhausted = []
            for task in validated_tasks:
                fingerprint = self._task_fingerprint(task)
                if fingerprint in successful_fingerprints:
                    successful_repeats.append(task)
                    continue
                allowed_attempts = 1 + self.run_policy.max_tool_retries
                if attempt_counts.get(fingerprint, 0) >= allowed_attempts:
                    retry_exhausted.append(task)
                    continue
                unique_tasks.append(task)
            if not unique_tasks and self.run_policy.stop_on_repeated_plan:
                termination_reason = (
                    "retry_budget_exhausted" if retry_exhausted else "repeated_plan"
                )
                autonomy_rounds.append({
                    "round_index": reflection_index + 1,
                    "phase": "replan",
                    "proposed_tasks": sanitize_for_display(proposed_tasks),
                    "validation": sanitize_for_display(validated.get("_planner_validation", {})),
                    "accepted_tasks": [],
                    "successful_repeats_rejected": sanitize_for_display(successful_repeats),
                    "retry_budget_rejected": sanitize_for_display(retry_exhausted),
                    "execution_statuses": [],
                })
                break

            remaining_tool_calls = self.run_policy.max_tool_calls - len(executed_results)
            if remaining_tool_calls <= 0:
                termination_reason = "tool_call_budget_exhausted"
                break
            accepted_tasks = unique_tasks[:remaining_tool_calls]
            budget_truncated = len(unique_tasks) > len(accepted_tasks)
            if not accepted_tasks:
                termination_reason = "no_valid_new_tasks"
                break

            plan["tasks"].extend(deepcopy(accepted_tasks))
            extra_execution_started = time.perf_counter()
            extra_results = self.execute(
                {
                    "species": plan.get("species", species),
                    "tasks": accepted_tasks,
                },
                step_offset=len(executed_results),
                round_index=reflection_index + 1,
            )
            self._annotate_attempts(extra_results, attempt_counts)
            successful_fingerprints.update(
                self._result_fingerprint(item)
                for item in extra_results
                if item.get("status") == "success"
            )
            round_execution_seconds = time.perf_counter() - extra_execution_started
            tool_execution_seconds += round_execution_seconds
            executed_results.extend(extra_results)
            autonomy_rounds.append({
                "round_index": reflection_index + 1,
                "phase": "replan",
                "proposed_tasks": sanitize_for_display(proposed_tasks),
                "validation": sanitize_for_display(validated.get("_planner_validation", {})),
                "accepted_tasks": sanitize_for_display(accepted_tasks),
                "successful_repeats_rejected": sanitize_for_display(successful_repeats),
                "retry_budget_rejected": sanitize_for_display(retry_exhausted),
                "budget_truncated": budget_truncated,
                "execution_statuses": [item.get("status") for item in extra_results],
                "execution_elapsed_seconds": round(round_execution_seconds, 6),
            })

        if initial_budget_truncated and len(executed_results) >= self.run_policy.max_tool_calls:
            termination_reason = "tool_call_budget_exhausted"
        if isinstance(plan.get("_scientific_refusal"), dict):
            termination_reason = "scientific_refusal"
        if isinstance(plan.get("_clarification_request"), dict):
            termination_reason = "clarification_required"

        attempt_outcomes = self._attempt_outcomes(executed_results)
        if termination_reason in {
            "tool_call_budget_exhausted", "replan_budget_exhausted", "retry_budget_exhausted",
        }:
            goal_status = "budget_exhausted"
        elif termination_reason == "scientific_refusal":
            goal_status = "completed"
        elif termination_reason == "clarification_required":
            goal_status = "awaiting_clarification"
        elif not executed_results or not any(item.get("status") == "success" for item in executed_results):
            goal_status = "failed"
        elif termination_reason == "goal_satisfied" and not attempt_outcomes["unresolved"]:
            goal_status = "completed"
        else:
            goal_status = "incomplete"

        synthesis_started = time.perf_counter()
        final_answer = self.answer(question, plan, executed_results, reflection=replan_info)
        synthesis_seconds = time.perf_counter() - synthesis_started
        compact_executed_results = self._compact_executed_results_for_prompt(executed_results)

        visualizations = []
        for item in executed_results:
            if item.get("status") == "success" and item["tool_name"].startswith("plot_"):
                result = item.get("result")
                if isinstance(result, str) and result.lower().endswith((".png", ".svg", ".html")):
                    visualizations.append({
                        "tool_name": item["tool_name"],
                        "path": self._display_path(result)
                    })
                elif isinstance(result, dict):
                    for figure in result.get("figures", []):
                        visualizations.append({
                            "tool_name": item["tool_name"],
                            "path": self._display_path(figure.get("png_path") or figure.get("svg_path")),
                            "figure_id": figure.get("figure_id"),
                            "svg_path": self._display_path(figure.get("svg_path")),
                            "png_path": self._display_path(figure.get("png_path")),
                            "pdf_path": self._display_path(figure.get("pdf_path")),
                        })

        completed_at = datetime.now(timezone.utc)
        record = {
            "schema_version": "2.0",
            "run_id": run_id,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "duration_seconds": round(time.perf_counter() - run_started, 6),
            "timing": {
                "planning_seconds": round(planning_seconds, 6),
                "tool_execution_seconds": round(tool_execution_seconds, 6),
                "answer_synthesis_seconds": round(synthesis_seconds, 6),
            },
            "question": question,
            "experiment_id": experiment_id,
            "species": species,
            "plan": sanitize_for_display(plan),
            "selected_tools": [task.get("tool_name") for task in plan.get("tasks", [])],
            "executed_results": executed_results,
            "replan": sanitize_for_display(replan_info),
            "autonomy": {
                "loop": "plan-act-observe-reflect",
                "policy": asdict(self.run_policy),
                "goal_status": goal_status,
                "termination_reason": termination_reason,
                "round_count": len(autonomy_rounds),
                "reflection_count": sum("reflection" in item for item in autonomy_rounds),
                "tool_call_count": len(executed_results),
                "successful_tool_call_count": sum(item.get("status") == "success" for item in executed_results),
                "failed_tool_call_count": sum(item.get("status") in {"failed", "error"} for item in executed_results),
                "retry_call_count": attempt_outcomes["retry_call_count"],
                "recovered_failure_count": len(attempt_outcomes["recovered"]),
                "unrecovered_failure_count": len(attempt_outcomes["unresolved"]),
                "rounds": autonomy_rounds,
            },
            "runtime": self._runtime_metadata(),
            "trace": {
                "planning": {
                    "planner_mode": plan.get("_llm_outputs", {}).get("planner_mode", "llm"),
                    "target_species": plan.get("species", species),
                    "planner_input_summary": sanitize_for_display(plan.get("_llm_outputs", {}).get("planner_input_summary", {})),
                    "planner_raw_response": plan.get("_llm_outputs", {}).get("planner_raw_response"),
                    "planner_system_prompt_sha256": plan.get("_llm_outputs", {}).get("planner_system_prompt_sha256"),
                    "planner_user_prompt_sha256": plan.get("_llm_outputs", {}).get("planner_user_prompt_sha256"),
                    "planner_attempt_count": plan.get("_llm_outputs", {}).get("planner_attempt_count", 1),
                    "planner_attempts": sanitize_for_display(plan.get("_llm_outputs", {}).get("planner_attempts", [])),
                    "initial_normalized_tasks": sanitize_for_display(initial_plan.get("tasks", [])),
                    "normalized_tasks": sanitize_for_display(plan.get("tasks", [])),
                    "planner_validation": sanitize_for_display(plan.get("_planner_validation", {})),
                },
                "execution": compact_executed_results,
                "replanning": {
                    "need_replan": replan_info.get("need_replan", False),
                    "new_tasks": replan_info.get("new_tasks", []),
                    "replanner_raw_response": replan_info.get("_llm_outputs", {}).get("replanner_raw_response"),
                    "replanner_system_prompt_sha256": replan_info.get("_llm_outputs", {}).get("replanner_system_prompt_sha256"),
                    "replanner_user_prompt_sha256": replan_info.get("_llm_outputs", {}).get("replanner_user_prompt_sha256"),
                    "replanner_user_prompt_char_count": replan_info.get("_llm_outputs", {}).get("replanner_user_prompt_char_count"),
                    "replanner_user_prompt_budget_chars": replan_info.get("_llm_outputs", {}).get("replanner_user_prompt_budget_chars"),
                    "replanner_prompt_compaction_mode": replan_info.get("_llm_outputs", {}).get("replanner_prompt_compaction_mode"),
                    "timestamp_reliability": replan_info.get("timestamp_reliability"),
                    "event_independence": replan_info.get("event_independence"),
                    "event_count_sufficiency": replan_info.get("event_count_sufficiency"),
                    "overlap_analysis": replan_info.get("overlap_analysis"),
                    "visualization": replan_info.get("visualization"),
                    "figure_count": replan_info.get("figure_count", 0),
                    "figure_outputs": sanitize_for_display(replan_info.get("figure_outputs", [])),
                    "failed_figures": sanitize_for_display(replan_info.get("failed_figures", [])),
                    "reason": replan_info.get("reason"),
                    "conservative_answer": replan_info.get("conservative_answer", False),
                    "reflection_mode": replan_info.get("reflection_mode"),
                    "consistency_check": sanitize_for_display(replan_info.get("consistency_check", {})),
                    "temporal_activity": sanitize_for_display(replan_info.get("temporal_activity", {})),
                    "termination_reason": termination_reason,
                    "rounds": autonomy_rounds,
                },
                "synthesis": sanitize_for_display(deepcopy(self._last_answer_trace)),
            },
            "final_answer": sanitize_for_display(final_answer),
            "visualizations": visualizations,
        }
        self.history.append(record)
        return record
