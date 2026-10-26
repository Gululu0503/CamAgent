import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from camagent.llm_agent import AgentRunPolicy, WildlifeAgent


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeAPILLM:
    def invoke(self, _messages):
        return FakeResponse("""
{
  "species": "Siberian tiger",
  "tasks": [
    {"tool_name": "check_timestamp_quality", "args": {"species": "Siberian tiger"}, "reason": "The LLM selects timestamp validation before temporal analysis."},
    {"tool_name": "build_independent_events", "args": {"species": "Siberian tiger", "by_camera": true, "min_gap_minutes": 30}, "reason": "The LLM selects independent-event construction before activity estimation."},
    {"tool_name": "build_hourly_activity_distribution", "args": {"species": "Siberian tiger", "min_gap_minutes": 30}, "reason": "The LLM selects hourly activity estimation for the diel pattern."},
    {"tool_name": "summarize_community_composition", "args": {"top_k": 10}, "reason": "The LLM documents the community context used to select comparison species."},
    {"tool_name": "estimate_pairwise_activity_overlap", "args": {"species_a": "Siberian tiger", "species_list": ["Sika deer", "Roe deer", "Black bear", "Amur leopard"], "min_gap_minutes": 30}, "reason": "The LLM selects pairwise temporal-overlap estimation."},
    {"tool_name": "plot_hourly_activity_curve", "args": {"species": "Siberian tiger", "min_gap_minutes": 30}, "reason": "The LLM selects a diel activity figure."},
    {"tool_name": "plot_pairwise_activity_overlap_panels", "args": {"species_a": "Siberian tiger", "comparison_species": ["Sika deer", "Roe deer", "Black bear", "Amur leopard"], "min_gap_minutes": 30}, "reason": "The LLM selects temporal-overlap panels."}
  ]
}
""")


class FakeDefaultComparisonLLM:
    def __init__(self):
        self.calls = 0
        self.messages = []

    def invoke(self, messages):
        self.calls += 1
        self.messages.append(messages)
        if self.calls == 1:
            return FakeResponse("""
{
  "species": "Siberian tiger",
  "tasks": [
    {"tool_name": "estimate_pairwise_activity_overlap", "args": {"species_a": "Siberian tiger", "species_list": "default"}, "reason": "The LLM asks for default sympatric species."},
    {"tool_name": "plot_pairwise_activity_overlap_panels", "args": {"species_a": "Siberian tiger", "comparison_species": "default"}, "reason": "The LLM asks for default overlap panels."}
  ]
}
""")
        return FakeResponse("""
{
  "species": "Siberian tiger",
  "tasks": [
    {"tool_name": "check_timestamp_quality", "args": {"species": "Siberian tiger"}, "reason": "Validate focal-species timestamps."},
    {"tool_name": "build_independent_events", "args": {"species": "Siberian tiger"}, "reason": "Build focal-species independent events."},
    {"tool_name": "build_hourly_activity_distribution", "args": {"species": "Siberian tiger"}, "reason": "Build the focal hourly distribution."},
    {"tool_name": "summarize_community_composition", "args": {"top_k": 10}, "reason": "Document comparison-species context."},
    {"tool_name": "estimate_pairwise_activity_overlap", "args": {"species_a": "Siberian tiger", "species_list": ["Sika deer", "Roe deer", "Black bear", "Amur leopard"]}, "reason": "The LLM corrects the invalid placeholder by selecting explicit comparison species."},
    {"tool_name": "plot_pairwise_activity_overlap_panels", "args": {"species_a": "Siberian tiger", "comparison_species": ["Sika deer", "Roe deer", "Black bear", "Amur leopard"]}, "reason": "The LLM corrects the invalid placeholder for the requested panels."}
  ]
}
""")


class FakeInvalidToolLLM:
    def __init__(self):
        self.calls = 0
        self.messages = []

    def invoke(self, messages):
        self.calls += 1
        self.messages.append(messages)
        if self.calls == 1:
            return FakeResponse("""
{
  "species": "Siberian tiger",
  "tasks": [
    {"tool_name": "analyze_nonexistent_activity_pattern", "args": {"species": "Siberian tiger"}, "reason": "Invalid planner output for validation test."}
  ]
}
""")
        return FakeResponse("""
{
  "species": "Siberian tiger",
  "tasks": [
    {"tool_name": "summarize_species_inventory", "args": {"top_k": 5}, "reason": "The LLM selects a registry-visible tool after receiving validation feedback."}
  ]
}
""")


class FakeHiddenToolLLM:
    def __init__(self):
        self.calls = 0
        self.messages = []

    def invoke(self, messages):
        self.calls += 1
        self.messages.append(messages)
        if self.calls == 1:
            return FakeResponse("""
{
  "species": "Siberian tiger",
  "tasks": [
    {"tool_name": "get_all_records", "args": {}, "reason": "Attempt to invoke an internal-only data primitive."}
  ]
}
""")
        return FakeResponse("""
{
  "species": "Siberian tiger",
  "tasks": [
    {"tool_name": "summarize_species_inventory", "args": {"top_k": 5}, "reason": "The LLM selects a planner-visible inventory tool after validation rejects the internal primitive."}
  ]
}
""")


class FakeAlwaysInvalidToolLLM:
    def invoke(self, _messages):
        return FakeResponse("""
{
  "species": "Siberian tiger",
  "tasks": [
    {"tool_name": "analyze_nonexistent_activity_pattern", "args": {"species": "Siberian tiger"}, "reason": "Repeat an invalid tool."}
  ]
}
""")


class FakeUnsafeFallbackLLM:
    def invoke(self, _messages):
        return FakeResponse("""
{
  "species": "Siberian tiger",
  "tasks": [
    {"tool_name": "summarize_species_inventory", "args": {}, "reason": "Run a generic fallback."}
  ]
}
""")


class FakeInventoryAnswerLLM:
    def invoke(self, messages):
        system_text = str(messages[0].content)
        if "produce a natural, accurate, explainable English answer" in system_text:
            return FakeResponse(
                "Roe deer: 727,090; Sika deer: 625,249; "
                "Amur leopard: 11,704; Siberian tiger: 7,745; Black bear: 4,229."
            )
        return FakeResponse('{"need_replan": false, "new_tasks": [], "reason": "Complete."}')


class FakeTemporalAnswerLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return FakeResponse(
            "Siberian tiger versus Sika deer: Delta = 0.8861. "
            "No bootstrap interval was computed because bootstrap_iters was 0. "
            "This is similarity in recorded-clock activity, not evidence of direct interaction."
        )


class FakeCompleteCommunityAnswerLLM:
    def invoke(self, _messages):
        return FakeResponse(
            "The eligible community comprised Roe deer, Sika deer, "
            "Amur leopard, Siberian tiger, and Black bear. "
            "The strongest conditional association was Amur leopard--"
            "Siberian tiger."
        )


class CapturingReflectionLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return FakeResponse(
            '{"need_replan": false, "new_tasks": [], "reason": "Evidence is sufficient."}'
        )


class CapturingAnswerLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return FakeResponse("The requested analysis completed with the stated limitations.")


class FakeAutonomousAgent(WildlifeAgent):
    def __init__(self, reflections, policy=None):
        super().__init__(FakeAPILLM(), run_policy=policy or AgentRunPolicy())
        self.reflections = list(reflections)

    def plan(self, _question, species=None):
        return {
            "species": species or "Siberian tiger",
            "tasks": [{
                "tool_name": "summarize_species_inventory",
                "args": {"top_k": 5},
                "reason": "Initial inventory step.",
            }],
            "_llm_outputs": {"planner_mode": "test"},
            "_planner_validation": {"action": "validated"},
        }

    def execute(self, plan, *, step_offset=0, round_index=0):
        return [{
            "step": step_offset + index,
            "round_index": round_index,
            "tool_name": task["tool_name"],
            "args": task.get("args", {}),
            "status": "success",
            "result": {},
        } for index, task in enumerate(plan["tasks"], start=1)]

    def reflect_and_replan(self, _question, _plan, _executed_results):
        if self.reflections:
            return self.reflections.pop(0)
        return {"need_replan": False, "new_tasks": [], "reason": "Complete."}

    def answer(self, _question, _plan, _executed_results, reflection=None):
        return "Grounded test answer."


class FakeRecoveringAgent(FakeAutonomousAgent):
    def __init__(self):
        super().__init__([
            {
                "need_replan": True,
                "reason": "Retry the transiently failed inventory call.",
                "new_tasks": [{"tool_name": "summarize_species_inventory", "args": {"top_k": 5}}],
            },
            {"need_replan": False, "new_tasks": [], "reason": "Recovered evidence is sufficient."},
        ])
        self.execution_count = 0

    def execute(self, plan, *, step_offset=0, round_index=0):
        self.execution_count += 1
        status = "failed" if self.execution_count == 1 else "success"
        return [{
            "step": step_offset + index,
            "round_index": round_index,
            "tool_name": task["tool_name"],
            "args": task.get("args", {}),
            "status": status,
            "result": {} if status == "success" else None,
            "error": "Injected transient failure" if status == "failed" else None,
        } for index, task in enumerate(plan["tasks"], start=1)]


class TemporalOverlapPlanningTest(unittest.TestCase):
    def setUp(self):
        self.agent = WildlifeAgent(FakeAPILLM())

    def assert_temporal_overlap_plan(self, question):
        plan = self.agent.plan(question, species="Siberian tiger")
        self.assertEqual(
            [task["tool_name"] for task in plan["tasks"]],
            [
                "check_timestamp_quality",
                "build_independent_events",
                "build_hourly_activity_distribution",
                "summarize_community_composition",
                "estimate_pairwise_activity_overlap",
                "plot_hourly_activity_curve",
                "plot_pairwise_activity_overlap_panels",
            ],
        )

    def test_english_sympatric_temporal_overlap_query(self):
        question = (
            "What is the diel activity pattern of the Siberian tiger, "
            "and how does its temporal niche overlap with sympatric species?"
        )
        self.assert_temporal_overlap_plan(question)

    def test_invalid_species_placeholder_is_returned_to_llm_for_correction(self):
        llm = FakeDefaultComparisonLLM()
        agent = WildlifeAgent(llm)
        plan = agent.plan(
            "What is the diel activity pattern of the Siberian tiger, "
            "and how does its temporal niche overlap with sympatric species?",
            species="Siberian tiger",
        )
        pairwise = next(task for task in plan["tasks"] if task["tool_name"] == "estimate_pairwise_activity_overlap")
        panels = next(task for task in plan["tasks"] if task["tool_name"] == "plot_pairwise_activity_overlap_panels")
        self.assertEqual(pairwise["args"]["species_list"], ["Sika deer", "Roe deer", "Black bear", "Amur leopard"])
        self.assertEqual(panels["args"]["comparison_species"], ["Sika deer", "Roe deer", "Black bear", "Amur leopard"])
        self.assertNotIn("min_gap_minutes", pairwise["args"])
        self.assertNotIn("bootstrap_iters", pairwise["args"])
        self.assertNotIn("min_gap_minutes", panels["args"])
        self.assertEqual(llm.calls, 2)
        self.assertEqual(plan["_llm_outputs"]["planner_attempt_count"], 2)
        self.assertEqual(plan["_llm_outputs"]["planner_mode"], "llm_validation_retry")
        self.assertIn("invalid_args", llm.messages[1][1].content)
        first_validation = plan["_llm_outputs"]["planner_attempts"][0]["validation"]
        self.assertTrue(first_validation["invalid_args"])
        self.assertFalse(first_validation["local_tool_rewrite"])

    def test_invalid_tool_feedback_goes_to_llm_and_llm_selects_replacement(self):
        llm = FakeInvalidToolLLM()
        agent = WildlifeAgent(llm)
        plan = agent.plan(
            "What is the diel activity pattern of the Siberian tiger, "
            "and how does its temporal niche overlap with sympatric species?",
            species="Siberian tiger",
        )
        self.assertEqual(
            [task["tool_name"] for task in plan["tasks"]],
            ["summarize_species_inventory"],
        )
        first_validation = plan["_llm_outputs"]["planner_attempts"][0]["validation"]
        self.assertEqual(
            first_validation["invalid_tools"][0]["tool_name"],
            "analyze_nonexistent_activity_pattern",
        )
        self.assertEqual(llm.calls, 2)
        self.assertIn("invalid_tools", llm.messages[1][1].content)

    def test_internal_tool_is_rejected_and_llm_selects_visible_replacement(self):
        llm = FakeHiddenToolLLM()
        agent = WildlifeAgent(llm)
        plan = agent.plan("List the available species.", species="Siberian tiger")
        first_validation = plan["_llm_outputs"]["planner_attempts"][0]["validation"]
        self.assertEqual(first_validation["invalid_tools"][0]["tool_name"], "get_all_records")
        self.assertEqual(
            [task["tool_name"] for task in plan["tasks"]],
            ["summarize_species_inventory"],
        )
        self.assertEqual(llm.calls, 2)
        self.assertFalse(agent.registry.is_planner_exposed("get_all_records"))
        self.assertTrue(agent.registry.is_planner_exposed("summarize_species_inventory"))

    def test_planner_fails_closed_when_llm_repeats_invalid_tool(self):
        agent = WildlifeAgent(
            FakeAlwaysInvalidToolLLM(),
            run_policy=AgentRunPolicy(max_replan_rounds=1),
        )
        with self.assertRaisesRegex(
            ValueError,
            "did not produce an executable registry-valid plan",
        ):
            agent.plan("List the available species.", species="Siberian tiger")

    def test_validator_does_not_fill_missing_species_for_the_llm(self):
        plan = self.agent._normalize_plan(
            "Rank local tiger hotspots.",
            {
                "species": "Siberian tiger",
                "tasks": [{
                    "tool_name": "build_hotspot_rank",
                    "args": {"top_k": 5},
                    "reason": "The LLM omitted the focal species.",
                }],
            },
            {"allow_empty": True},
        )
        self.assertEqual(plan["tasks"], [])
        self.assertEqual(
            plan["_planner_validation"]["missing_args"][0]["missing_args"],
            ["species"],
        )
        self.assertEqual(
            plan["_planner_validation"]["action"],
            "request_llm_replan",
        )

    def test_explicit_scientific_refusal_runs_no_tool(self):
        agent = WildlifeAgent(FakeUnsafeFallbackLLM())
        record = agent.run(
            "Prove that the predators caused prey avoidance. If the request is not "
            "scientifically identifiable, do not call a tool; explain the missing "
            "evidence or choice.",
            species="Siberian tiger",
        )
        self.assertEqual(record["selected_tools"], [])
        self.assertEqual(record["executed_results"], [])
        self.assertEqual(record["autonomy"]["goal_status"], "completed")
        self.assertEqual(
            record["autonomy"]["termination_reason"],
            "scientific_refusal",
        )
        self.assertIn("did not run an analysis tool", record["final_answer"])
        self.assertEqual(
            record["trace"]["synthesis"]["synthesis_mode"],
            "deterministic_scientific_refusal",
        )

    def test_validator_does_not_remove_llm_selected_plot_tools_by_keyword(self):
        plan = self.agent._normalize_plan(
            "Compare diel activity overlap between Siberian tiger and Sika deer "
            "using a 15-minute independent-event gap. Do not plot.",
            {
                "species": "Siberian tiger",
                "tasks": [
                    {"tool_name": "check_timestamp_quality", "args": {"species": "Siberian tiger"}, "reason": "test"},
                    {"tool_name": "build_independent_events", "args": {"species": "Siberian tiger", "min_gap_minutes": 15}, "reason": "test"},
                    {"tool_name": "build_hourly_activity_distribution", "args": {"species": "Siberian tiger", "min_gap_minutes": 15}, "reason": "test"},
                    {"tool_name": "estimate_pairwise_activity_overlap", "args": {"species_a": "Siberian tiger", "species_list": ["Sika deer"], "min_gap_minutes": 15}, "reason": "test"},
                    {"tool_name": "plot_hourly_activity_curve", "args": {"species": "Siberian tiger", "min_gap_minutes": 15}, "reason": "test"},
                    {"tool_name": "plot_pairwise_activity_overlap_panels", "args": {"species_a": "Siberian tiger", "comparison_species": ["Sika deer"], "min_gap_minutes": 15}, "reason": "test"},
                ],
            },
            {},
        )
        assert [task["tool_name"] for task in plan["tasks"]] == [
            "check_timestamp_quality",
            "build_independent_events",
            "build_hourly_activity_distribution",
            "estimate_pairwise_activity_overlap",
            "plot_hourly_activity_curve",
            "plot_pairwise_activity_overlap_panels",
        ]
        assert plan["_planner_validation"]["local_tool_rewrite"] is False
        assert plan["_planner_validation"]["fallback_used"] is False

    def test_validator_preserves_llm_selected_analysis_tools(self):
        movement = self.agent._normalize_plan(
            "List temporally adjacent camera detections for Roe deer using maximum "
            "gaps of 72 hours and 80 km. Treat them only as camera-transition "
            "hypotheses, never as individual tracks or corridors.",
            {
                "species": "Roe deer",
                "tasks": [{
                    "tool_name": "infer_transition_corridors",
                    "args": {
                        "species": "Roe deer",
                        "max_time_gap_hours": 72,
                        "max_distance_km": 80,
                    },
                    "reason": "test",
                }],
            },
            {},
        )
        assert movement["tasks"] == [{
            "tool_name": "infer_transition_corridors",
            "args": {
                "species": "Roe deer",
                "max_time_gap_hours": 72,
                "max_distance_km": 80,
            },
            "reason": "test",
        }]

        spatial = self.agent._normalize_plan(
            "Summarize Amur leopard detections within 10 km of camera 213215 "
            "and rank local hotspots; do not infer movement corridors.",
            {
                "species": "Amur leopard",
                "tasks": [
                    {
                        "tool_name": "filter_records",
                        "args": {"species": "Amur leopard", "camera_id": "213215"},
                        "reason": "test",
                    },
                    {
                        "tool_name": "analyze_spatial_distribution",
                        "args": {
                            "species": "Amur leopard",
                            "center_camera_id": "213215",
                            "radius_km": 10,
                        },
                        "reason": "test",
                    },
                ],
            },
            {},
        )
        assert [task["tool_name"] for task in spatial["tasks"]] == [
            "filter_records",
            "analyze_spatial_distribution"
        ]
        assert spatial["tasks"][1]["args"]["center_camera_id"] == "213215"
        assert spatial["tasks"][1]["args"]["radius_km"] == 10

        occupancy = self.agent._normalize_plan(
            "Attempt an intercept-only single-season occupancy analysis for "
            "Siberian tiger with 7-day occasions, but refuse to fit occupancy "
            "when effort comes only from detection spans.",
            {
                "species": "Siberian tiger",
                "tasks": [
                    {
                        "tool_name": "build_detection_history_matrix",
                        "args": {"species": "Siberian tiger", "occasion_days": 7},
                        "reason": "test",
                    },
                    {
                        "tool_name": "fit_single_season_occupancy_model",
                        "args": {"species": "Siberian tiger", "occasion_days": 7},
                        "reason": "test",
                    },
                ],
            },
            {},
        )
        assert [task["tool_name"] for task in occupancy["tasks"]] == [
            "build_detection_history_matrix",
            "fit_single_season_occupancy_model"
        ]
        assert occupancy["tasks"][1]["args"]["occasion_days"] == 7

    def test_autonomous_loop_validates_replan_and_suppresses_duplicate_tasks(self):
        agent = FakeAutonomousAgent([
            {
                "need_replan": True,
                "reason": "A spatial summary is still needed.",
                "new_tasks": [
                    {"tool_name": "summarize_species_inventory", "args": {"top_k": 5}},
                    {"tool_name": "build_hotspot_rank", "args": {"species": "Siberian tiger", "top_k": 3}},
                ],
            },
            {"need_replan": False, "new_tasks": [], "reason": "Evidence is sufficient."},
        ])
        record = agent.run("Show tiger detections and hotspots.", species="Siberian tiger")
        self.assertEqual(record["autonomy"]["termination_reason"], "goal_satisfied")
        self.assertEqual(record["autonomy"]["round_count"], 2)
        self.assertEqual(record["selected_tools"], ["summarize_species_inventory", "build_hotspot_rank"])
        self.assertEqual([item["step"] for item in record["executed_results"]], [1, 2])

    def test_autonomous_loop_stops_when_replanner_repeats_executed_task(self):
        agent = FakeAutonomousAgent([{
            "need_replan": True,
            "reason": "Repeat the same step.",
            "new_tasks": [{"tool_name": "summarize_species_inventory", "args": {"top_k": 5}}],
        }])
        record = agent.run("List species.", species="Siberian tiger")
        self.assertEqual(record["autonomy"]["termination_reason"], "repeated_plan")
        self.assertEqual(record["autonomy"]["tool_call_count"], 1)

    def test_autonomous_loop_retries_failed_task_and_records_recovery(self):
        record = FakeRecoveringAgent().run("List species.", species="Siberian tiger")
        self.assertEqual(record["autonomy"]["termination_reason"], "goal_satisfied")
        self.assertEqual(record["autonomy"]["goal_status"], "completed")
        self.assertEqual(record["autonomy"]["retry_call_count"], 1)
        self.assertEqual(record["autonomy"]["recovered_failure_count"], 1)
        self.assertEqual([item["attempt_index"] for item in record["executed_results"]], [1, 2])

    def test_public_trace_removes_raw_results_and_sensitive_locations(self):
        agent = WildlifeAgent(FakeAPILLM())
        record = {
            "question": "Show records near camera 213215.",
            "plan": {"tasks": [], "_llm_outputs": {"planner_raw_response": "raw"}},
            "replan": {"_llm_outputs": {"replanner_raw_response": "raw"}},
            "executed_results": [{
                "step": 1,
                "tool_name": "filter_records",
                "args": {"camera_id": "213215"},
                "status": "success",
                "result": {"camera_id": "213215", "latitude": 43.1, "records": [{"cameraID": "213215"}]},
            }],
            "trace": {
                "planning": {"planner_raw_response": "raw"},
                "replanning": {"replanner_raw_response": "raw"},
                "synthesis": {"answer_raw_response": "raw"},
            },
        }
        exported = agent.exportable_trace(record)
        self.assertNotIn("planner_raw_response", exported["trace"]["planning"])
        self.assertNotIn("planner_raw_response", exported["plan"]["_llm_outputs"])
        self.assertNotIn("replanner_raw_response", exported["replan"]["_llm_outputs"])
        self.assertEqual(exported["executed_results"][0]["args"]["camera_id"], "[REDACTED_SENSITIVE_LOCATION]")
        self.assertNotIn("latitude", exported["executed_results"][0]["result"])
        self.assertEqual(
            exported["executed_results"][0]["result"]["records"][0]["cameraID"],
            "[REDACTED_SENSITIVE_LOCATION]",
        )

    def test_external_prompt_compaction_redacts_sensitive_locations(self):
        agent = WildlifeAgent(FakeAPILLM())
        compact = agent._compact_executed_results_for_prompt([{
            "step": 1,
            "tool_name": "filter_records",
            "args": {"camera_id": "213215"},
            "status": "success",
            "result": {
                "latitude": 43.1,
                "longitude": 130.2,
                "records": [{"cameraID": "213215"}],
            },
        }])
        self.assertEqual(compact[0]["args"]["camera_id"], "[REDACTED_SENSITIVE_LOCATION]")
        self.assertNotIn("latitude", compact[0]["result"])
        self.assertNotIn("longitude", compact[0]["result"])
        self.assertEqual(
            compact[0]["result"]["records"][0]["cameraID"],
            "[REDACTED_SENSITIVE_LOCATION]",
        )

    def test_replanner_prompt_has_hard_budget_and_compacts_movement_hypotheses(self):
        llm = CapturingReflectionLLM()
        agent = WildlifeAgent(llm)
        movement_steps = [
            {
                "from_camera": f"camera-{index}",
                "to_camera": f"camera-{index + 1}",
                "time_gap_hours": 1.5,
                "distance_km": 2.5,
                "speed_km_h": 1.667,
                "bearing_degrees": 45.0,
                "direction": "northeast",
                "inference_level": "camera_detection_adjacency_hypothesis",
                "identity_evidence": "none",
                "warning": "This is not an individual track.",
            }
            for index in range(5000)
        ]
        result = agent.reflect_and_replan(
            "List adjacent Roe deer detections as hypotheses only.",
            {
                "species": "Roe deer",
                "tasks": [{
                    "tool_name": "analyze_movement_steps",
                    "args": {
                        "species": "Roe deer",
                        "max_time_gap_hours": 72,
                        "max_distance_km": 80,
                    },
                }],
                "_llm_outputs": {"planner_raw_response": "x" * 100_000},
            },
            [{
                "step": 1,
                "tool_name": "analyze_movement_steps",
                "args": {
                    "species": "Roe deer",
                    "max_time_gap_hours": 72,
                    "max_distance_km": 80,
                },
                "status": "success",
                "result": movement_steps,
            }],
        )
        user_prompt = llm.messages[1].content
        self.assertLessEqual(
            len(user_prompt),
            agent._REPLANNER_USER_PROMPT_BUDGET_CHARS,
        )
        self.assertNotIn("camera-4999", user_prompt)
        self.assertNotIn("x" * 1000, user_prompt)
        self.assertIn("camera_transition_hypothesis_count", user_prompt)
        self.assertEqual(
            result["_llm_outputs"]["replanner_user_prompt_char_count"],
            len(user_prompt),
        )

    def test_replanner_evidence_compaction_guarantees_prompt_budget(self):
        llm = CapturingReflectionLLM()
        agent = WildlifeAgent(llm)
        large_results = [
            {
                "step": index,
                "tool_name": "build_hotspot_rank",
                "args": {"species": "Roe deer", "top_k": 10},
                "status": "success",
                "result": {
                    f"large_scalar_{field_index}": "x" * 1000
                    for field_index in range(50)
                },
            }
            for index in range(1, 33)
        ]
        large_results[0]["tool_name"] = "analyze_species_pair_relationship"
        large_results[0]["result"].update({
            "species_a": "Siberian tiger",
            "species_b": "Sika deer",
            "activity_overlap_delta": 0.8843,
            "activity_overlap_ci": {
                "lower": 0.8728,
                "upper": 0.8950,
            },
            "activity_overlap_result_bundle_sha256": "bundle-sha256-8843",
            "interpretation_boundary": (
                "Temporal overlap does not demonstrate direct interaction."
            ),
        })
        result = agent.reflect_and_replan(
            "Summarize the completed tool evidence.",
            {"species": "Roe deer", "tasks": []},
            large_results,
        )
        user_prompt = llm.messages[1].content
        self.assertLessEqual(
            len(user_prompt),
            agent._REPLANNER_USER_PROMPT_BUDGET_CHARS,
        )
        self.assertEqual(
            result["_llm_outputs"]["replanner_prompt_compaction_mode"],
            "tool_names_and_evidence_projection",
        )
        self.assertIn('"tool_name":"build_hotspot_rank"', user_prompt)
        self.assertIn('"activity_overlap_delta":0.8843', user_prompt)
        self.assertIn('"lower":0.8728', user_prompt)
        self.assertIn('"upper":0.895', user_prompt)
        self.assertIn("bundle-sha256-8843", user_prompt)

    def test_answer_prompt_has_independent_hard_budget(self):
        llm = CapturingAnswerLLM()
        agent = WildlifeAgent(llm)
        large_results = [
            {
                "step": index,
                "tool_name": "build_hotspot_rank",
                "args": {"species": "Roe deer", "top_k": 10},
                "status": "success",
                "result": {
                    f"large_scalar_{field_index}": "x" * 1000
                    for field_index in range(50)
                },
            }
            for index in range(1, 33)
        ]
        agent.answer(
            "Summarize the completed tool evidence.",
            {
                "species": "Roe deer",
                "tasks": [],
                "_llm_outputs": {"planner_raw_response": "x" * 100_000},
            },
            large_results,
        )
        user_prompt = llm.messages[1].content
        self.assertLessEqual(
            len(user_prompt),
            agent._ANSWER_USER_PROMPT_BUDGET_CHARS,
        )
        self.assertNotIn("x" * 1000, user_prompt)
        self.assertIn(
            agent._last_answer_trace["answer_prompt_compaction_mode"],
            {"evidence_projection", "statuses_only"},
        )
        self.assertEqual(
            agent._last_answer_trace["answer_user_prompt_char_count"],
            len(user_prompt),
        )

    def test_community_evidence_projection_retains_inferential_results(self):
        agent = WildlifeAgent(CapturingReflectionLLM())
        projection = agent._critical_result_evidence_for_prompt(
            "infer_complete_community",
            {
                "status": "ok",
                "eligible_species_count": 5,
                "gamma_diversity": {"observed_richness": 5},
                "beta_diversity": {"mean_jaccard_dissimilarity": 0.61},
                "significant_conditional_associations": [{
                    "species_a": "Siberian tiger",
                    "species_b": "Sika deer",
                    "standardized_effect_size": 3.8,
                    "fdr_significant": True,
                }],
                "detection_corrected_occupancy_species_count": 0,
                "occupancy_effort_basis": "not eligible without effort histories",
                "interpretation_boundary": (
                    "Incidence associations do not prove direct interaction."
                ),
            },
        )["evidence"]
        self.assertEqual(projection["eligible_species_count"], 5)
        self.assertEqual(
            projection["gamma_diversity"]["observed_richness"], 5
        )
        self.assertEqual(
            projection["significant_conditional_associations"]["item_count"], 1
        )
        self.assertIn("occupancy_effort_basis", projection)

    def test_inventory_answer_is_synthesized_by_llm_and_preserves_species_names(self):
        agent = WildlifeAgent(FakeInventoryAnswerLLM())
        answer = agent.answer(
            "List species.",
            {"species": "Siberian tiger", "tasks": []},
            [{
                "tool_name": "summarize_species_inventory",
                "status": "success",
                "result": {
                    "species_inventory": [
                        {"species": "Roe deer", "record_count": 727090},
                        {"species": "Sika deer", "record_count": 625249},
                    ],
                },
            }],
        )
        self.assertIn("Roe deer", answer)
        self.assertIn("Sika deer", answer)
        self.assertEqual(
            agent._last_answer_trace["synthesis_mode"],
            "llm_synthesis",
        )
        self.assertEqual(
            agent._last_answer_trace["compact_tool_results_used"][0]["tool_name"],
            "summarize_species_inventory",
        )

    def test_complete_community_answer_preserves_all_species_names(self):
        agent = WildlifeAgent(FakeCompleteCommunityAnswerLLM())
        answer = agent.answer(
            "Infer the complete community.",
            {"species": "Siberian tiger", "tasks": []},
            [{
                "tool_name": "infer_complete_community",
                "status": "success",
                "result": {
                    "status": "ok",
                    "analysis_scope": "all eligible species",
                    "species_summary": [
                        {"species": "Roe deer"},
                        {"species": "Sika deer"},
                        {"species": "Amur leopard"},
                        {"species": "Siberian tiger"},
                        {"species": "Black bear"},
                    ],
                    "conditional_camera_associations": [{
                        "species_a": "Amur leopard",
                        "species_b": "Siberian tiger",
                        "fdr_significant": True,
                    }],
                },
            }],
        )
        for species in (
            "Roe deer",
            "Sika deer",
            "Amur leopard",
            "Siberian tiger",
            "Black bear",
        ):
            self.assertIn(species, answer)

    def test_temporal_answer_is_synthesized_by_llm_from_exact_tool_evidence(self):
        llm = FakeTemporalAnswerLLM()
        agent = WildlifeAgent(llm)
        answer = agent.answer(
            (
                "Compare diel activity overlap between Siberian tiger and Sika deer "
                "using a 60-minute gap without a plot."
            ),
            {"species": "Siberian tiger", "tasks": []},
            [
                {
                    "tool_name": "check_timestamp_quality",
                    "status": "success",
                    "result": {
                        "total_records": 100,
                        "valid_timestamp_records": 100,
                        "observed_clock_time_ratio": 1.0,
                        "synthetic_clock_time_ratio": 0.0,
                    },
                },
                {
                    "tool_name": "build_independent_events",
                    "status": "success",
                    "result": {
                        "min_gap_minutes": 60,
                        "independent_event_count": 80,
                    },
                },
                {
                    "tool_name": "build_hourly_activity_distribution",
                    "status": "success",
                    "result": {"peak_hours": [17, 18, 19]},
                },
                {
                    "tool_name": "estimate_pairwise_activity_overlap",
                    "status": "success",
                    "result": {
                        "species_a": "Siberian tiger",
                        "min_gap_minutes": 60,
                        "bootstrap_iters": 0,
                        "permutation_iters": 0,
                        "pairwise_results": [{
                            "species_a": "Siberian tiger",
                            "species_b": "Sika deer",
                            "delta_hat": 0.8861,
                            "delta_ci_low": None,
                            "delta_ci_high": None,
                            "valid_event_count_a": 80,
                            "valid_event_count_b": 120,
                        }],
                    },
                },
            ],
        )
        self.assertIn("Delta = 0.8861", answer)
        self.assertIn("No bootstrap interval was computed", answer)
        self.assertNotIn("0.8861–0.8861", answer)
        self.assertEqual(
            agent._last_answer_trace["synthesis_mode"],
            "llm_synthesis",
        )
        prompt = llm.messages[1].content
        self.assertIn('"delta_hat":0.8861', prompt)
        self.assertIn('"bootstrap_iters":0', prompt)
        self.assertIn('"independent_event_count":80', prompt)

    def test_validator_never_canonicalizes_llm_tool_choices(self):
        quality = self.agent._normalize_plan(
            "Audit missing fields, invalid timestamps, label noise, and duplicate burden for Siberian tiger records.",
            {
                "species": "Siberian tiger",
                "tasks": [
                    {"tool_name": "check_missing_fields", "args": {"species": "Siberian tiger"}},
                    {"tool_name": "detect_invalid_timestamps", "args": {"species": "Siberian tiger"}},
                    {"tool_name": "detect_duplicate_like_records", "args": {"species": "Siberian tiger"}},
                ],
            },
            {},
        )
        self.assertEqual(
            [task["tool_name"] for task in quality["tasks"]],
            [
                "check_missing_fields",
                "detect_invalid_timestamps",
                "detect_duplicate_like_records",
            ],
        )

        records = self.agent._normalize_plan(
            "Return at most 20 Amur leopard records from camera 213215, sorted chronologically.",
            {
                "species": "Amur leopard",
                "tasks": [{
                    "tool_name": "browse_records",
                    "args": {
                        "species": "Amur leopard",
                        "camera_id": "213215",
                        "limit": 20,
                        "sort_by": "timestamp_asc",
                    },
                }],
            },
            {},
        )
        self.assertEqual(records["tasks"][0]["tool_name"], "browse_records")

        local = self.agent._normalize_plan(
            "Summarize Siberian tiger detections within 20 km of camera 213215 and rank local hotspots.",
            {
                "species": "Siberian tiger",
                "tasks": [{
                    "tool_name": "build_local_hotspot_rank",
                    "args": {
                        "species": "Siberian tiger",
                        "center_camera_id": "213215",
                        "radius_km": 20,
                        "top_k": 5,
                    },
                }],
            },
            {},
        )
        self.assertEqual(local["tasks"][0]["tool_name"], "build_local_hotspot_rank")
        for candidate in (quality, records, local):
            self.assertFalse(candidate["_planner_validation"]["local_tool_rewrite"])

    def test_sensitive_camera_constraint_is_rehydrated_locally_after_prompt_redaction(self):
        records = self.agent._normalize_plan(
            "Return at most 20 Amur leopard records from camera 213215, sorted chronologically.",
            {
                "species": "Amur leopard",
                "tasks": [{
                    "tool_name": "browse_records",
                    "args": {
                        "species": "Amur leopard",
                        "camera_id": "[REDACTED]",
                        "limit": 20,
                        "sort_by": "timestamp_asc",
                    },
                }],
            },
            {},
        )
        self.assertEqual(records["tasks"][0]["tool_name"], "browse_records")
        self.assertEqual(records["tasks"][0]["args"]["camera_id"], "213215")

        local = self.agent._normalize_plan(
            "Summarize Siberian tiger detections within 20 km of camera 213215.",
            {
                "species": "Siberian tiger",
                "tasks": [{
                    "tool_name": "build_local_hotspot_rank",
                    "args": {
                        "species": "Siberian tiger",
                        "center_camera_id": "[REDACTED]",
                        "radius_km": 20,
                    },
                }],
            },
            {},
        )
        self.assertEqual(local["tasks"][0]["args"]["center_camera_id"], "213215")
        self.assertIsNone(self.agent._local_camera_id_from_question("Summarize the camera network."))

    def test_validator_does_not_auto_complete_temporal_tool_chain(self):
        plan = self.agent._normalize_plan(
            (
                "Compare diel activity overlap between Siberian tiger and Sika deer using a "
                "30-minute gap; no plot is needed."
            ),
            {
                "species": "Siberian tiger",
                "tasks": [{
                    "tool_name": "estimate_pairwise_activity_overlap",
                    "args": {
                        "species_a": "Siberian tiger",
                        "species_list": ["Sika deer"],
                        "min_gap_minutes": 30,
                    },
                }],
            },
            {},
        )
        self.assertEqual(
            [task["tool_name"] for task in plan["tasks"]],
            ["estimate_pairwise_activity_overlap"],
        )
        self.assertEqual(plan["tasks"][0]["args"]["species_list"], ["Sika deer"])
        self.assertFalse(plan["_planner_validation"]["fallback_used"])
        self.assertFalse(plan["_planner_validation"]["local_tool_rewrite"])

    def test_validator_accepts_dataset_derived_pairwise_comparisons(self):
        plan = self.agent._normalize_plan(
            "Describe tiger activity and overlap with other species.",
            {
                "species": "Siberian tiger",
                "tasks": [
                    {
                        "tool_name": "check_timestamp_quality",
                        "args": {"species": "Siberian tiger"},
                    },
                    {
                        "tool_name": "build_independent_events",
                        "args": {"species": "Siberian tiger"},
                    },
                    {
                        "tool_name": "build_hourly_activity_distribution",
                        "args": {"species": "Siberian tiger"},
                    },
                    {
                        "tool_name": "summarize_community_composition",
                        "args": {"top_k": 20},
                    },
                    {
                        "tool_name": "estimate_pairwise_activity_overlap",
                        "args": {"species_a": "Siberian tiger", "top_k": 4},
                    },
                    {
                        "tool_name": "plot_pairwise_activity_overlap_panels",
                        "args": {"species_a": "Siberian tiger", "top_k": 4},
                    },
                ],
            },
            {},
        )
        validation = plan["_planner_validation"]
        self.assertEqual(validation["action"], "validated")
        self.assertFalse(validation["invalid_args"])
        pairwise = next(
            task
            for task in plan["tasks"]
            if task["tool_name"] == "estimate_pairwise_activity_overlap"
        )
        panels = next(
            task
            for task in plan["tasks"]
            if task["tool_name"] == "plot_pairwise_activity_overlap_panels"
        )
        self.assertNotIn("species_list", pairwise["args"])
        self.assertNotIn("comparison_species", panels["args"])

    def test_validator_rejects_overlap_figure_without_prior_analysis(self):
        plan = self.agent._normalize_plan(
            "Plot tiger and Sika deer temporal overlap.",
            {
                "species": "Siberian tiger",
                "tasks": [{
                    "tool_name": "plot_activity_overlap_panel",
                    "args": {
                        "species_a": "Siberian tiger",
                        "species_b": "Sika deer",
                    },
                    "reason": "Render the requested overlap figure.",
                }],
            },
            {"allow_empty": True},
        )
        validation = plan["_planner_validation"]
        self.assertEqual(validation["action"], "request_llm_replan")
        self.assertEqual(
            validation["dependency_errors"][0]["tool_name"],
            "plot_activity_overlap_panel",
        )
        self.assertTrue(validation["bundle_alignment_errors"])

    def test_validator_rejects_mismatched_overlap_analysis_and_plot_specs(self):
        plan = self.agent._normalize_plan(
            "Estimate and plot tiger and Sika deer temporal overlap.",
            {
                "species": "Siberian tiger",
                "tasks": [
                    {
                        "tool_name": "check_timestamp_quality",
                        "args": {"species": "Siberian tiger"},
                        "reason": "Validate timestamps.",
                    },
                    {
                        "tool_name": "build_independent_events",
                        "args": {
                            "species": "Siberian tiger",
                            "min_gap_minutes": 30,
                        },
                        "reason": "Build independent events.",
                    },
                    {
                        "tool_name": "build_hourly_activity_distribution",
                        "args": {
                            "species": "Siberian tiger",
                            "min_gap_minutes": 30,
                        },
                        "reason": "Build hourly activity.",
                    },
                    {
                        "tool_name": "estimate_activity_overlap",
                        "args": {
                            "species_a": "Siberian tiger",
                            "species_b": "Sika deer",
                            "min_gap_minutes": 30,
                            "smooth_sigma": 1.25,
                            "bootstrap_iters": 500,
                        },
                        "reason": "Estimate overlap.",
                    },
                    {
                        "tool_name": "plot_activity_overlap_panel",
                        "args": {
                            "species_a": "Siberian tiger",
                            "species_b": "Sika deer",
                            "min_gap_minutes": 30,
                            "smooth_sigma": 1.5,
                            "bootstrap_iters": 500,
                        },
                        "reason": "Render overlap.",
                    },
                ],
            },
            {"allow_empty": True},
        )
        validation = plan["_planner_validation"]
        self.assertFalse(validation["dependency_errors"])
        self.assertEqual(validation["action"], "request_llm_replan")
        self.assertEqual(
            validation["bundle_alignment_errors"][0]["tool_name"],
            "plot_activity_overlap_panel",
        )

    def test_validator_requires_all_analyses_before_plotting_phase(self):
        plan = self.agent._normalize_plan(
            "Analyze and plot tiger activity and overlap with Sika deer.",
            {
                "species": "Siberian tiger",
                "tasks": [
                    {
                        "tool_name": "check_timestamp_quality",
                        "args": {"species": "Siberian tiger"},
                    },
                    {
                        "tool_name": "build_independent_events",
                        "args": {"species": "Siberian tiger"},
                    },
                    {
                        "tool_name": "build_hourly_activity_distribution",
                        "args": {"species": "Siberian tiger"},
                    },
                    {
                        "tool_name": "plot_hourly_activity_curve",
                        "args": {"species": "Siberian tiger"},
                    },
                    {
                        "tool_name": "estimate_activity_overlap",
                        "args": {
                            "species_a": "Siberian tiger",
                            "species_b": "Sika deer",
                        },
                    },
                ],
            },
            {"allow_empty": True},
        )
        validation = plan["_planner_validation"]
        self.assertEqual(validation["action"], "request_llm_replan")
        self.assertEqual(
            validation["analysis_order_errors"][0]["tool_name"],
            "estimate_activity_overlap",
        )

    def test_occupancy_guardrail_is_a_successful_terminal_state(self):
        reflection = self.agent._merge_deterministic_reflection(
            {
                "need_replan": True,
                "new_tasks": [{"tool_name": "export_camtrap_dp", "args": {}}],
                "reason": "Try to recover.",
            },
            [{
                "tool_name": "fit_single_season_occupancy_model",
                "status": "success",
                "result": {
                    "model_status": "not_fitted_without_independent_effort",
                    "occupancy_probability_psi": None,
                    "effort_basis": "detection_span_proxy",
                },
            }],
        )
        self.assertFalse(reflection["need_replan"])
        self.assertTrue(reflection["conservative_answer"])
        self.assertIn("no recovery call is warranted", reflection["reason"])


if __name__ == "__main__":
    unittest.main()
