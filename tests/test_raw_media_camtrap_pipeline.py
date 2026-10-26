import json
from pathlib import Path

import pandas as pd

import camagent.base_tools as base_tools_module
from camagent.tool_registry import ToolRegistry
from camagent.visualization_tools import VisualizationTools
from camagent.llm_agent import WildlifeAgent
from camagent.perception_evaluation import run_perception_benchmark


def _write_video_sidecar(media_path: Path):
    sidecar = {
        "species": "Siberian tiger",
        "scientific_name": "Panthera tigris altaica",
        "confidence": 0.94,
        "bbox": [0.1, 0.2, 0.5, 0.7],
        "camera_id": "CAM_001",
        "timestamp": "2024-01-15 12:30:00",
        "latitude": 43.25,
        "longitude": 130.75,
        "count": 1,
    }
    media_path.with_suffix(media_path.suffix + ".json").write_text(
        json.dumps(sidecar),
        encoding="utf-8",
    )


def test_registry_uses_one_stateful_tool_instance():
    registry = ToolRegistry()
    assert registry.tools is registry.viz
    planner_names = {
        item["tool_name"]
        for item in registry.get_tool_descriptions()
    }
    assert "prepare_camera_trap_dataset" in planner_names
    assert "plot_tiger_spatial_transition_panel" not in planner_names
    assert "plot_predator_prey_temporal_panel" not in planner_names


def test_raw_video_sidecar_to_active_camtrap_dp(tmp_path):
    media_dir = tmp_path / "raw_media"
    media_dir.mkdir()
    video_path = media_dir / "CAM_001_20240115_123000.mp4"
    video_path.write_bytes(b"test-video-placeholder")
    _write_video_sidecar(video_path)

    tools = VisualizationTools(output_dir=tmp_path / "figures")
    result = tools.prepare_camera_trap_dataset(
        input_path=media_dir,
        input_type="raw_media",
        output_dir=tmp_path / "prepared",
        run_inference=False,
        source_timezone="Asia/Shanghai",
        activate_backend=True,
        persist_active=False,
    )

    assert result["status"] == "ok"
    assert result["data_to_camtrap_pipeline_complete"] is True
    assert result["full_raw_media_pipeline_complete"] is False
    assert result["perception_executed"] is False
    assert result["analysis_readiness"]["camtrap_dp_ready"] is True
    assert tools._using_camtrap_backend() is True
    assert len(tools.get_all_records()) == 1

    package_dir = Path(result["datapackage_path"]).parent
    media = pd.read_csv(package_dir / "media.csv", dtype=str)
    observations = pd.read_csv(package_dir / "observations.csv", dtype=str)
    assert media.loc[0, "timestamp"].startswith("2024-01-15T04:30:00")
    media_observation = observations.loc[
        observations["observationLevel"] == "media"
    ].iloc[0]
    assert float(media_observation["bboxX"]) == 0.1
    assert float(media_observation["bboxWidth"]) == 0.4

    figure = tools.plot_species_distribution_static(
        species="Siberian tiger",
        annotate_top_k=1,
        privacy_grid_degrees=0.1,
        save_path=tmp_path / "figures" / "distribution.svg",
    )
    svg_path = Path(figure["figures"][0]["svg_path"])
    assert svg_path.exists()
    assert svg_path.with_suffix(".pdf").exists()


def test_video_inference_branch_creates_frame_level_records(
    tmp_path,
    monkeypatch,
):
    media_dir = tmp_path / "raw_media"
    media_dir.mkdir()
    video_path = media_dir / "CAM_002_20240201_010203.mp4"
    video_path.write_bytes(b"test-video-placeholder")
    video_path.with_suffix(".mp4.json").write_text(
        json.dumps({
            "camera_id": "CAM_002",
            "timestamp": "2024-02-01 01:02:03",
            "latitude": 43.5,
            "longitude": 130.5,
        }),
        encoding="utf-8",
    )

    def fake_video_inference(*args, **kwargs):
        return {
            "manifest_json": str(tmp_path / "video_predictions.json"),
            "failures": [],
            "predictions": [
                {
                    "species": "Roe deer",
                    "scientific_name": "Capreolus pygargus",
                    "confidence": 0.91,
                    "model_version": "4.0.2a",
                    "prediction_source": "classifier",
                    "bbox": [0.1, 0.1, 0.3, 0.4],
                    "frame_path": str(tmp_path / "frame_1.jpg"),
                    "frame_offset_seconds": 0.0,
                    "predictions_json": str(tmp_path / "frame_1.json"),
                },
                {
                    "species": "Roe deer",
                    "scientific_name": "Capreolus pygargus",
                    "confidence": 0.87,
                    "model_version": "4.0.2a",
                    "prediction_source": "classifier",
                    "bbox": [0.2, 0.2, 0.2, 0.3],
                    "frame_path": str(tmp_path / "frame_2.jpg"),
                    "frame_offset_seconds": 2.0,
                    "predictions_json": str(tmp_path / "frame_2.json"),
                },
            ],
        }

    monkeypatch.setattr(
        base_tools_module,
        "run_speciesnet_on_video",
        fake_video_inference,
    )
    tools = VisualizationTools(output_dir=tmp_path / "figures")
    result = tools.prepare_camera_trap_dataset(
        input_path=media_dir,
        input_type="raw_media",
        output_dir=tmp_path / "prepared",
        run_inference=True,
        persist_active=False,
    )

    detection = result["detection"]
    assert result["full_raw_media_pipeline_complete"] is True
    assert detection["pipeline_mode"] == "raw_media_model_inference"
    assert detection["model_inferred_video_frame_count"] == 2
    assert detection["model_versions"] == {"4.0.2a": 2}
    assert detection["prediction_sources"] == {"classifier": 2}
    assert detection["video_sampling"] == {
        "frame_interval_seconds": 1.0,
        "max_frames": 12,
    }
    assert detection["record_count"] == 2
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in Path(result["structured_records_dir"]).glob("*.json")
    ]
    assert {
        record["timestamp"]
        for record in records
    } == {
        "2024-02-01 01:02:03",
        "2024-02-01 01:02:05",
    }
    assert all(
        record["camtrap_file_path"] == str(video_path)
        for record in records
    )


def test_public_raw_media_trace_retains_evidence_and_redacts_locations(tmp_path):
    agent = WildlifeAgent(llm=object())
    media_path = tmp_path / "CAM_190790_20240922.mp4"
    output_dir = tmp_path / "prepared"
    private_record = {
        "schema_version": "2.0",
        "question": f"Prepare {media_path} and write it to {output_dir}.",
        "experiment_id": "RAW_MEDIA_PIPELINE",
        "plan": {"tasks": [{
            "tool_name": "prepare_camera_trap_dataset",
            "args": {
                "input_path": str(media_path),
                "output_dir": str(output_dir),
                "input_type": "raw_media",
                "video_frame_interval_seconds": 1.0,
                "max_video_frames": 1,
            },
        }]},
        "selected_tools": ["prepare_camera_trap_dataset"],
        "executed_results": [{
            "step": 1,
            "tool_name": "prepare_camera_trap_dataset",
            "args": {
                "input_path": str(media_path),
                "output_dir": str(output_dir),
                "input_type": "raw_media",
            },
            "status": "success",
            "result": {
                "status": "ok",
                "input_type": "raw_media",
                "pipeline_stages": [
                    "sampled-video-frame perception",
                    "CamtrapDP validation",
                    "backend activation",
                ],
                "perception_executed": True,
                "data_to_camtrap_pipeline_complete": True,
                "full_raw_media_pipeline_complete": True,
                "detection": {
                    "record_count": 1,
                    "model_inferred_video_frame_count": 1,
                    "detector": "megadetector",
                    "classifier": "speciesnet",
                    "model_versions": {"4.0.2a": 1},
                    "prediction_sources": {"classifier": 1},
                    "video_sampling": {
                        "frame_interval_seconds": 1.0,
                        "max_frames": 1,
                    },
                    "species_breakdown": {"tiger": 1},
                    "sample_records": [{
                        "id": "CAM_190790_private",
                        "camera_id": "190790",
                        "latitude": 43.123456,
                        "longitude": 130.654321,
                        "filename": media_path.name,
                        "species": "tiger",
                        "confidence": 0.9989,
                        "timestamp": "2024-09-22 16:56:28",
                        "timestamp_source": "paddleocr_video_timestamp",
                        "perception_model_version": "4.0.2a",
                        "prediction_source": "classifier",
                    }],
                },
                "backend_result": {
                    "backend": "camtrap_dp",
                    "validation": {
                        "status": "ok",
                        "error_count": 0,
                        "warning_count": 0,
                    },
                },
                "analysis_readiness": {
                    "active_for_downstream_tools": True,
                },
            },
        }],
        "replan": {"reason": "Preparation evidence is complete."},
        "trace": {"replanning": {"reason": "Preparation evidence is complete."}},
        "final_answer": (
            "SpeciesNet 4.0.2a returned tiger at 0.9989; PaddleOCR recovered "
            "2024-09-22 16:56:28; validation returned 0 errors and 0 warnings."
        ),
    }
    public_text = json.dumps(
        agent.exportable_trace(private_record, detail="public"),
        ensure_ascii=False,
    )
    assert str(tmp_path) not in public_text
    assert "190790" not in public_text
    assert "43.123456" not in public_text
    assert "130.654321" not in public_text
    assert "prepare_camera_trap_dataset" in public_text
    assert "4.0.2a" in public_text
    assert "0.9989" in public_text
    assert "2024-09-22 16:56:28" in public_text
    assert '"camtrap_validation_error_count": 0' in public_text
    assert '"camtrap_validation_warning_count": 0' in public_text
    assert '"activated": true' in public_text
    assert agent._redact_public_trace("Asia/Shanghai") == "Asia/Shanghai"
    assert agent._redact_public_trace("detector/classifier") == "detector/classifier"
    assert agent._redact_public_trace("use workspace/private/video.mp4") == (
        "use [REDACTED_LOCAL_PATH]"
    )


def test_one_agent_plan_prepares_then_analyses_and_plots_new_backend(tmp_path):
    media_dir = tmp_path / "raw_media"
    media_dir.mkdir()
    video_path = media_dir / "CAM_003_20240304_050607.mp4"
    video_path.write_bytes(b"test-video-placeholder")
    _write_video_sidecar(video_path)

    agent = WildlifeAgent(llm=object())
    plan = {
        "species": "Siberian tiger",
        "tasks": [
            {
                "tool_name": "prepare_camera_trap_dataset",
                "args": {
                    "input_path": str(media_dir),
                    "input_type": "raw_media",
                    "output_dir": str(tmp_path / "prepared"),
                    "run_inference": False,
                    "activate_backend": True,
                    "persist_active": False,
                },
                "reason": "Prepare the supplied raw media as CamtrapDP.",
            },
            {
                "tool_name": "summarize_species_inventory",
                "args": {"top_k": 5},
                "reason": "Summarize the newly activated dataset.",
            },
            {
                "tool_name": "plot_species_distribution_static",
                "args": {
                    "species": "Siberian tiger",
                    "annotate_top_k": 1,
                    "privacy_grid_degrees": 0.1,
                    "save_path": str(tmp_path / "figures" / "same_run.svg"),
                },
                "reason": "Plot the newly activated dataset.",
            },
        ],
    }
    executed = agent.execute(plan)

    assert [item["status"] for item in executed] == [
        "success",
        "success",
        "success",
    ], executed
    assert executed[1]["result"]["total_record_count"] == 1
    figure_path = Path(
        executed[2]["result"]["figures"][0]["svg_path"]
    )
    assert figure_path.exists()
    assert figure_path.with_suffix(".pdf").exists()


def test_mock_llm_selects_preparation_then_uses_new_backend(tmp_path):
    media_dir = tmp_path / "raw_media"
    media_dir.mkdir()
    video_path = media_dir / "CAM_004_20240405_060708.mp4"
    video_path.write_bytes(b"test-video-placeholder")
    _write_video_sidecar(video_path)
    prepared_dir = tmp_path / "prepared"
    figure_path = tmp_path / "figures" / "same_agent_run.svg"

    class _Response:
        def __init__(self, content):
            self.content = content

    class _LLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return _Response(json.dumps({
                    "species": "Siberian tiger",
                    "tasks": [
                        {
                            "tool_name": "prepare_camera_trap_dataset",
                            "args": {
                                "input_path": str(media_dir),
                                "input_type": "raw_media",
                                "output_dir": str(prepared_dir),
                                "run_inference": False,
                                "activate_backend": True,
                                "persist_active": False,
                            },
                            "reason": "Prepare and activate the raw media.",
                        },
                        {
                            "tool_name": "summarize_species_inventory",
                            "args": {"top_k": 5},
                            "reason": "Inspect the activated package.",
                        },
                        {
                            "tool_name": "plot_species_distribution_static",
                            "args": {
                                "species": "Siberian tiger",
                                "annotate_top_k": 1,
                                "privacy_grid_degrees": 0.1,
                                "save_path": str(figure_path),
                            },
                            "reason": "Verify downstream plotting.",
                        },
                    ],
                }))
            if self.calls == 2:
                return _Response(json.dumps({
                    "need_replan": False,
                    "new_tasks": [],
                    "reason": "All requested preparation checks succeeded.",
                    "data_quality": "sufficient",
                    "timestamp_reliability": {"status": "passed"},
                    "event_independence": {"status": "not_applicable"},
                    "event_count_sufficiency": {"status": "passed"},
                    "overlap_analysis": {"status": "not_applicable"},
                    "spatial_data_sufficiency": {"status": "sufficient"},
                    "visualization": "generated",
                }))
            return _Response("The LLM-selected preparation workflow succeeded.")

    llm = _LLM()
    agent = WildlifeAgent(llm)
    agent._active_species_inventory = lambda: []
    record = agent.run(
        f"Prepare and analyse the raw videos at {media_dir}.",
        experiment_id="RAW_MEDIA_PIPELINE",
    )
    assert record["selected_tools"] == [
        "prepare_camera_trap_dataset",
        "summarize_species_inventory",
        "plot_species_distribution_static",
    ]
    assert record["autonomy"]["successful_tool_call_count"] == 3
    assert figure_path.exists()
    assert llm.calls == 3
    public_text = json.dumps(
        agent.exportable_trace(record, detail="public"),
        ensure_ascii=False,
    )
    assert str(tmp_path) not in public_text
    assert "CAM_004" not in public_text


def test_registered_perception_benchmark_scores_batch_predictions(tmp_path):
    annotations = pd.DataFrame([
        {
            "sample_id": "s1",
            "media_path": "one.mp4",
            "camera_id": "A",
            "true_label": "Tiger",
            "predicted_label": "Tiger",
            "confidence": 0.95,
        },
        {
            "sample_id": "s2",
            "media_path": "two.mp4",
            "camera_id": "B",
            "true_label": "Tiger",
            "predicted_label": "Leopard",
            "confidence": 0.70,
        },
        {
            "sample_id": "s3",
            "media_path": "three.mp4",
            "camera_id": "B",
            "true_label": "Leopard",
            "predicted_label": "Leopard",
            "confidence": 0.90,
        },
    ])
    annotation_path = tmp_path / "annotations.csv"
    annotations.to_csv(annotation_path, index=False)

    result = run_perception_benchmark(
        annotation_path=annotation_path,
        output_dir=tmp_path / "perception",
        run_inference=False,
        bootstrap_iters=20,
        calibration_bins=3,
        minimum_class_support=1,
    )

    assert result["status"] == "success"
    assert result["summary"]["sample_count"] == 3
    assert result["summary"]["accuracy"] == 0.666667
    assert result["summary"]["bootstrap_unit"] == "camera_id"
    assert Path(result["figures"][0]["pdf_path"]).exists()
    assert "evaluate_perception_benchmark" in {
        item["tool_name"]
        for item in ToolRegistry().get_tool_descriptions()
    }


def test_complete_community_inference_uses_all_eligible_species(tmp_path):
    table = pd.DataFrame([
        {
            "species": species,
            "camera_id": camera,
            "timestamp": f"2024-01-{day:02d} 12:00:00",
            "latitude": 43.0 + int(camera) * 0.01,
            "longitude": 130.0 + int(camera) * 0.01,
            "media": f"{species}_{camera}_{day}.jpg",
        }
        for species, camera, day in [
            ("Tiger", "1", 1),
            ("Tiger", "2", 2),
            ("Leopard", "1", 3),
            ("Leopard", "3", 4),
            ("Sika deer", "2", 5),
            ("Sika deer", "3", 6),
        ]
    ])
    table_path = tmp_path / "community.csv"
    table.to_csv(table_path, index=False)
    tools = VisualizationTools(output_dir=tmp_path / "figures")
    tools.convert_table_to_camtrap_dp(
        table_path=table_path,
        output_dir=tmp_path / "community_dp",
        source_timezone="Asia/Shanghai",
        activate_backend=True,
        persist_active=False,
    )

    result = tools.infer_complete_community(
        top_k=2,
        min_independent_events=1,
        min_detected_cameras=1,
        bootstrap_iters=20,
        accumulation_permutations=20,
        fit_detection_corrected_occupancy=False,
    )
    # A ranking/display limit must never truncate complete-community inference.
    assert result["analysed_species_count"] == 3
    assert result["gamma_diversity"]["observed_richness"] == 3
    assert result["conditional_camera_associations"]
    assert result["association_model"]["multiple_testing"] == (
        "Benjamini-Hochberg FDR"
    )
    assert result["beta_diversity"]["mean_turnover_component"] is not None
    assert result["beta_diversity"]["mean_nestedness_component"] is not None

    figure = tools.plot_complete_community_inference(
        min_independent_events=1,
        min_detected_cameras=1,
        bootstrap_iters=10,
        accumulation_permutations=10,
        fit_detection_corrected_occupancy=False,
        display_top_species=3,
        save_path=tmp_path / "figures" / "community.svg",
    )
    assert Path(figure["figures"][0]["svg_path"]).exists()
    assert Path(figure["figures"][0]["pdf_path"]).exists()
