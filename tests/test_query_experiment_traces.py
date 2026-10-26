from __future__ import annotations

import json
import importlib.util
import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camagent.llm_agent import WildlifeAgent
from camagent.agent_router import AgentRouter, EcologicalToolset
from camagent.paths import PARK_SHAPEFILE_PATH
from camagent.visualization_tools import VisualizationTools
from run_agent import (
    _intent_flags_for_display,
    _result_check_summary,
    _visible_reasoning_trace_en,
    build_parser,
)


class _NoNetworkLLM:
    def invoke(self, _messages):
        raise AssertionError("This regression test must not call an external model.")


class _MinimalRunAgent(WildlifeAgent):
    def plan(self, _question, species=None):
        return {
            "species": species or "Siberian tiger",
            "tasks": [
                {
                    "tool_name": "summarize_species_inventory",
                    "args": {"top_k": 5},
                    "reason": "Return a deterministic inventory summary.",
                }
            ],
            "_llm_outputs": {"planner_mode": "test"},
            "_planner_validation": {"action": "validated"},
        }

    def execute(self, plan, *, step_offset=0, round_index=0):
        return [
            {
                "step": step_offset + 1,
                "round_index": round_index,
                "tool_name": plan["tasks"][0]["tool_name"],
                "args": plan["tasks"][0]["args"],
                "status": "success",
                "result": {"species_richness": 5},
            }
        ]

    def reflect_and_replan(self, _question, _plan, _executed_results):
        return {
            "need_replan": False,
            "new_tasks": [],
            "reason": "The requested evidence is available.",
            "consistency_check": {
                "missing_fields": [],
                "inconsistency_check": "passed",
            },
        }

    def answer(self, _question, _plan, _executed_results, reflection=None):
        self._last_answer_trace = {"synthesis_mode": "test"}
        return "Five focal species were summarized."


def _trace_text(tasks):
    planning = {"target_species": "Siberian tiger", "planner_mode": "test"}
    return "\n".join(
        _visible_reasoning_trace_en("Representative ecological question", planning, tasks)
    )


def test_quality_trace_is_not_hard_coded_as_temporal():
    text = _trace_text(
        [{"tool_name": "analyze_data_quality", "args": {"species": "Siberian tiger"}}]
    )
    assert "data-quality audit" in text
    assert "diel activity pattern" not in text
    assert "temporal niche overlap" not in text


def test_each_query_domain_produces_its_own_visible_trace_language():
    spatial = _trace_text(
        [{"tool_name": "build_hotspot_rank", "args": {"species": "Siberian tiger"}}]
    )
    community = _trace_text(
        [{"tool_name": "build_species_cooccurrence_matrix", "args": {}}]
    )
    movement = _trace_text(
        [
            {
                "tool_name": "analyze_movement_steps",
                "args": {
                    "species": "Amur leopard",
                    "max_time_gap_hours": 72,
                    "max_distance_km": 100,
                },
            }
        ]
    )
    assert "detection hotspots from occupancy or density" in spatial
    assert "co-detection from direct interaction" in community
    assert "hypotheses, not tracks or confirmed corridors" in movement


def test_integrated_trace_reports_all_selected_domains():
    tasks = [
        {"tool_name": "analyze_data_quality", "args": {}},
        {"tool_name": "estimate_activity_overlap", "args": {}},
        {"tool_name": "build_hotspot_rank", "args": {}},
        {"tool_name": "build_species_cooccurrence_matrix", "args": {}},
    ]
    text = _trace_text(tasks)
    flags = _intent_flags_for_display({}, tasks)
    assert "integrated_multi_domain" in flags
    assert "quality_analysis" in flags
    assert "temporal_analysis" in flags
    assert "spatial_analysis" in flags
    assert "community_analysis" in flags
    assert "dependency-ordered multi-domain workflow" in text


def test_result_check_is_domain_agnostic():
    summary = _result_check_summary(
        [
            {
                "tool_name": "analyze_data_quality",
                "status": "success",
                "result": {"status": "completed"},
            },
            {
                "tool_name": "plot_camera_quality_problem_barh",
                "status": "failed",
                "error": "Injected failure",
            },
        ],
        {
            "visualization": "failed",
            "figure_count": 0,
            "need_replan": True,
            "consistency_check": {"inconsistency_check": "passed"},
        },
    )
    assert summary["successful_tools"] == ["analyze_data_quality"]
    assert summary["failed_tools"] == ["plot_camera_quality_problem_barh"]
    assert summary["returned_result_count"] == 1
    assert summary["need_replan"] is True
    assert "timestamp reliability" not in summary


def test_reflection_replanning_state_is_normalized():
    agent = WildlifeAgent(_NoNetworkLLM())
    reflection = agent._merge_deterministic_reflection(
        {
            "need_replan": False,
            "new_tasks": [
                {
                    "tool_name": "check_timestamp_quality",
                    "args": {"species": "Sika deer"},
                }
            ],
            "reason": "Additional timestamp evidence is required.",
        },
        [],
    )
    assert reflection["need_replan"] is True
    assert reflection["new_tasks"]
    assert reflection["consistency_check"]["inconsistency_check"] == "corrected"


def test_public_redaction_preserves_ordinary_camera_language():
    agent = WildlifeAgent(_NoNetworkLLM())
    ordinary = "Same-species, same-camera records were grouped into events."
    assert agent._redact_public_trace(ordinary) == ordinary
    ordinary_plural = (
        "Camera-clustered bootstrap intervals are used when camera identifiers "
        "are available."
    )
    assert agent._redact_public_trace(ordinary_plural) == ordinary_plural
    assert "[REDACTED]" in agent._redact_public_trace(
        "Records from camera 213215 were retained."
    )
    assert "[REDACTED]" in agent._redact_public_trace(
        "Records from camera ID CAM123 were retained."
    )
    assert "obs_123456abcdef" not in agent._redact_public_trace(
        "Example record obs_123456abcdef was retained."
    )
    assert agent._redact_public_trace(
        ["obs_123456abcdef"],
        "example_record_ids",
    ) == "[REDACTED_INTERNAL_IDENTIFIER]"
    movement = agent._redact_public_trace({
        "from_camera": "140669",
        "to_camera": "140668",
        "distance_km": 4.2,
    })
    assert movement == {
        "from_camera": "[REDACTED_SENSITIVE_LOCATION]",
        "to_camera": "[REDACTED_SENSITIVE_LOCATION]",
        "distance_km": 4.2,
    }


def test_missing_field_summary_preserves_aggregate_counts_without_locations():
    agent = WildlifeAgent(_NoNetworkLLM())
    compact = agent._compact_result_for_prompt(
        "check_missing_fields",
        {
            "species": "Siberian tiger",
            "total_records": 10,
            "missing_counts": {"camera_id": 0, "latitude": 2},
            "missing_ratios": {"camera_id": 0.0, "latitude": 0.2},
        },
    )
    public = agent._redact_public_trace(compact)
    assert public["field_completeness"] == [
        {"field": "camera_id", "missing_count": 0, "missing_ratio": 0.0},
        {"field": "latitude", "missing_count": 2, "missing_ratio": 0.2},
    ]


def test_experiment_id_is_stored_in_run_and_cli():
    record = _MinimalRunAgent(_NoNetworkLLM()).run(
        "List the focal species.",
        experiment_id="comm",
    )
    assert record["experiment_id"] == "COMM"

    args = build_parser().parse_args(
        [
            "chat",
            "--experiment-id",
            "TEMP",
            "--question",
            "Describe tiger activity.",
        ]
    )
    assert args.experiment_id == "TEMP"


def test_paper_query_experiment_config_defines_all_six_ids():
    project_root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (project_root / "configs" / "paper_query_experiments.json").read_text(
            encoding="utf-8"
        )
    )
    experiments = payload["experiments"]
    assert [item["experiment_id"] for item in experiments] == [
        "QUAL",
        "TEMP",
        "SPAT",
        "COMM",
        "MOVE",
        "INTEG",
    ]
    assert all(item["question"] for item in experiments)
    assert all(item["trace_output"].endswith("_public_trace.json") for item in experiments)
    temp = next(item for item in experiments if item["experiment_id"] == "TEMP")
    assert temp["question"] == (
        "What is the activity pattern of the Siberian tiger, and how much "
        "temporal niche overlap does it have with other species?"
    )


def test_pairwise_overlap_defaults_are_resolved_from_active_inventory():
    from camagent.temporal_tools import TemporalTools

    tools = TemporalTools.__new__(TemporalTools)
    tools.summarize_species_inventory = lambda top_k=50: {
        "species_inventory": [
            {"species": "Roe deer", "record_count": 100},
            {"species": "Siberian tiger", "record_count": 90},
            {"species": "Sika deer", "record_count": 80},
            {"species": "Black bear", "record_count": 70},
            {"species": "Amur leopard", "record_count": 60},
        ]
    }
    assert tools._resolve_comparison_species("Siberian tiger", top_k=4) == [
        "Roe deer",
        "Sika deer",
        "Black bear",
        "Amur leopard",
    ]
    source = inspect.getsource(TemporalTools.estimate_pairwise_activity_overlap)
    assert '["Sika deer", "Roe deer", "Black bear", "Amur leopard"]' not in source


def test_agent_router_is_only_a_backward_compatible_toolset_alias():
    assert AgentRouter is EcologicalToolset


def test_public_spatial_figure_defaults_to_masked_coordinates_without_occupancy():
    signature = inspect.signature(VisualizationTools.plot_species_distribution_static)
    assert signature.parameters["privacy_grid_degrees"].default == 0.1
    source = inspect.getsource(VisualizationTools.plot_species_distribution_static)
    assert "build_occupancy_summary" not in source
    assert "fit_single_season_occupancy_model" not in source
    assert "camera identifiers are omitted" in source.lower()
    assert '"aggregation_unit": (' in source
    assert '"masked_grid_cell"' in source
    assert "annotated cells are not" in source


class _InventoryAwareLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return type("Response", (), {"content": json.dumps(self.payload)})()


def test_agent_has_no_implicit_tiger_default_and_plans_for_active_species():
    llm = _InventoryAwareLLM({
        "species": "Red fox",
        "clarification_request": None,
        "tasks": [{
            "tool_name": "check_timestamp_quality",
            "args": {"species": "Red fox"},
            "reason": "Validate the requested species before temporal analysis.",
        }],
    })
    agent = WildlifeAgent(llm)
    agent._active_species_inventory = lambda: [
        {"species": "Red fox", "record_count": 80},
        {"species": "Wild boar", "record_count": 60},
    ]
    plan = agent.plan("Describe the diel activity of Red fox.")
    assert agent.default_species is None
    assert plan["species"] == "Red fox"
    assert plan["tasks"][0]["args"]["species"] == "Red fox"
    cli = build_parser().parse_args(["chat", "--question", "Describe activity."])
    assert cli.default_species is None


def test_ambiguous_species_question_returns_clarification_without_tools():
    llm = _InventoryAwareLLM({
        "species": None,
        "clarification_request": "Which species should be the focal species?",
        "tasks": [],
    })
    agent = WildlifeAgent(llm)
    agent._active_species_inventory = lambda: [
        {"species": "Red fox", "record_count": 80},
        {"species": "Wild boar", "record_count": 60},
    ]
    record = agent.run("Describe its activity pattern.")
    assert record["selected_tools"] == []
    assert record["autonomy"]["goal_status"] == "awaiting_clarification"
    assert record["autonomy"]["termination_reason"] == "clarification_required"
    assert "Which species" in record["final_answer"]
    assert llm.calls == 1


def test_dynamic_other_species_selection_does_not_require_tiger():
    from camagent.temporal_tools import TemporalTools

    tools = TemporalTools.__new__(TemporalTools)
    tools.summarize_species_inventory = lambda top_k=50: {
        "species_inventory": [
            {"species": "Red fox", "record_count": 100},
            {"species": "Wild boar", "record_count": 90},
            {"species": "Raccoon dog", "record_count": 70},
        ]
    }
    assert tools._resolve_comparison_species("Red fox", top_k=2) == [
        "Wild boar",
        "Raccoon dog",
    ]


def test_generic_visual_entries_have_no_fixed_focal_species_or_role_list():
    from camagent.ecology_config import normalize_ecological_role_mapping

    for method_name, parameter_name in (
        ("plot_hourly_activity_curve", "species"),
        ("plot_species_distribution_static", "species"),
        ("plot_pairwise_activity_overlap_panels", "species_a"),
        ("plot_activity_overlap_triptych", "reference_species"),
    ):
        method = getattr(VisualizationTools, method_name)
        assert inspect.signature(method).parameters[parameter_name].default is None
        assert "Siberian tiger" not in inspect.getsource(method)

    assert normalize_ecological_role_mapping(None, ["Red fox", "Wild boar"]) == {
        "predators": [],
        "prey": [],
        "source": "user_or_dataset_configuration",
    }
    configured = normalize_ecological_role_mapping(
        {"predators": ["Red fox"], "prey": ["Wild boar"], "source": "study metadata"},
        ["Red fox", "Wild boar"],
    )
    assert configured == {
        "predators": ["Red fox"],
        "prey": ["Wild boar"],
        "source": "study metadata",
    }


def test_published_temp_and_integ_trace_contracts_and_bundle_alignment():
    trace_dir = PROJECT_ROOT / "results" / "paper_query_experiments"
    temp = json.loads((trace_dir / "TEMP_public_trace.json").read_text(encoding="utf-8"))
    integ = json.loads((trace_dir / "INTEG_public_trace.json").read_text(encoding="utf-8"))
    assert temp["autonomy"]["tool_call_count"] == 8
    assert integ["autonomy"]["tool_call_count"] == 12
    assert integ["autonomy"]["round_count"] == 1
    assert integ["replan"]["need_replan"] is False

    def _result(trace, tool_name):
        return next(
            item["result"]
            for item in trace["executed_results"]
            if item["tool_name"] == tool_name
        )

    temp_pairs = _result(temp, "estimate_pairwise_activity_overlap")["pairwise_results"]
    temp_sika = next(item for item in temp_pairs if item["species_b"] == "Sika deer")
    integ_pairs = _result(integ, "estimate_pairwise_activity_overlap")["pairwise_results"]
    integ_sika = next(item for item in integ_pairs if item["species_b"] == "Sika deer")
    integ_figures = _result(integ, "plot_pairwise_activity_overlap_panels")["figures"]
    integ_sika_figure = next(item for item in integ_figures if item["species_b"] == "Sika deer")
    assert temp_sika["delta_hat"] == integ_sika["delta_hat"] == 0.8843
    assert temp_sika["overlap_result_bundle_sha256"] == integ_sika["overlap_result_bundle_sha256"]
    assert integ_sika_figure["overlap_result_bundle_sha256"] == integ_sika["overlap_result_bundle_sha256"]
    assert (
        integ_sika_figure["delta_hat"],
        integ_sika_figure["delta_ci_low"],
        integ_sika_figure["delta_ci_high"],
    ) == (
        integ_sika["delta_hat"],
        integ_sika["delta_ci_low"],
        integ_sika["delta_ci_high"],
    )


def test_community_panel_with_proxy_effort_reports_event_counts_not_rai():
    source = inspect.getsource(
        VisualizationTools.plot_community_spatial_association_panel
    )
    assert '"not_reported_without_independent_effort"' in source
    assert '"independent_event_count"' in source
    assert 'str(row["camera_id"])' in source


def test_camtrap_community_overview_delegates_to_effort_safe_panel():
    source = inspect.getsource(VisualizationTools.plot_community_overview_panel)
    assert "if self._using_camtrap_backend()" in source
    assert "self.plot_community_spatial_association_panel(" in source


def test_registered_pairwise_overlap_figures_share_500_camera_bootstraps():
    triptych = inspect.signature(
        VisualizationTools.plot_activity_overlap_triptych
    )
    pairwise = inspect.signature(
        VisualizationTools.plot_pairwise_activity_overlap_panels
    )
    assert triptych.parameters["bootstrap_iters"].default == 500
    assert pairwise.parameters["bootstrap_iters"].default == 500


def test_complete_community_figure_supports_optional_privacy_safe_park_map():
    signature = inspect.signature(
        VisualizationTools.plot_complete_community_inference
    )
    assert signature.parameters["privacy_grid_degrees"].default == 0.1
    assert PARK_SHAPEFILE_PATH.name == "东北虎豹国家公园.shp"
    source = inspect.getsource(
        VisualizationTools.plot_complete_community_inference
    )
    assert "Privacy-safe community richness" in source
    assert "_load_park_gdf()" in source
    assert "if park_gdf is not None and not park_gdf.empty:" in source
    assert "elif not grid_summary.empty:" in source
    assert '"_agenttool_skip_tight_layout"' in source


def test_answer_numeric_guard_corrects_pair_event_count_from_tool_evidence():
    agent = WildlifeAgent.__new__(WildlifeAgent)
    agent._last_answer_trace = {}
    answer = agent._enforce_pair_event_counts(
        "- **Tiger vs Sika deer** (34,530 events) – high overlap.",
        [{
            "status": "success",
            "tool_name": "plot_activity_overlap_panel",
            "result": {
                "species_a": "Siberian tiger",
                "species_b": "Sika deer",
                "valid_event_count_b": 345300,
            },
        }],
    )
    assert "(345,300 events)" in answer
    assert agent._last_answer_trace["post_synthesis_numeric_corrections"] == [{
        "species_a": "Siberian tiger",
        "species_b": "Sika deer",
        "observed_count": 34530,
        "corrected_count": 345300,
        "evidence_field": "valid_event_count_b",
    }]


def test_paper_query_runner_loads_the_same_six_prespecified_questions():
    runner_path = PROJECT_ROOT / "scripts" / "run_paper_query_experiments.py"
    spec = importlib.util.spec_from_file_location("paper_query_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _, experiments = module._load_experiments(
        PROJECT_ROOT / "configs" / "paper_query_experiments.json"
    )
    assert tuple(item["experiment_id"] for item in experiments) == module.EXPECTED_IDS


def test_paper_query_runner_accepts_explicit_single_experiment_rerun():
    runner_path = PROJECT_ROOT / "scripts" / "run_paper_query_experiments.py"
    spec = importlib.util.spec_from_file_location("paper_query_runner_rerun", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    args = module._build_parser().parse_args(
        ["--resume-existing", "--rerun", "COMM"]
    )
    assert args.resume_existing is True
    assert args.rerun == ["COMM"]
    assert args.only is None


def test_paper_runner_rejects_spat_occupancy_or_exact_location_figures():
    runner_path = PROJECT_ROOT / "scripts" / "run_paper_query_experiments.py"
    spec = importlib.util.spec_from_file_location("paper_query_runner_validation", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    valid = {
        "selected_tools": [
            "build_detection_map",
            "build_hotspot_rank",
            "plot_species_distribution_static",
        ]
    }
    module._validate_paper_trace("SPAT", valid)

    invalid = {
        "selected_tools": [
            "fit_single_season_occupancy_model",
            "plot_species_distribution_map",
        ]
    }
    try:
        module._validate_paper_trace("SPAT", invalid)
    except RuntimeError as exc:
        assert "exact-location" in str(exc) or "occupancy" in str(exc)
    else:
        raise AssertionError("Unsafe SPAT tools should be rejected after LLM planning.")
