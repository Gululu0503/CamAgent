from __future__ import annotations

import json

import pandas as pd

from camagent.base_tools import BaseTools
from camagent.agent_router import AgentRouter
from camagent.camtrap_conversion import (
    _coerce_best_timestamp,
    add_independent_events,
    write_coordinate_resolution_audit,
)
from camagent.ecology_config import DEFAULT_0525_SPECIES_CONFIG
from camagent.research_pipeline import run_pipeline
from camagent.spatial_tools import SpatialTools


def test_filename_timestamp_is_used_when_recordtime_has_date_only():
    recordtime = pd.Series(["2024-01-02", "2024-01-03 08:15"])
    filename = pd.Series(["100001_20240102112233.mp4", "100001_20240103123456.mp4"])
    best, source, _, observed, synthetic = _coerce_best_timestamp(recordtime, filename)

    assert best.dt.tz_convert("Asia/Shanghai").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    ).tolist() == [
        "2024-01-02 11:22:33",
        "2024-01-03 08:15:00",
    ]
    assert source.tolist() == ["videoname_timestamp", "recordtime"]
    assert observed.tolist() == [True, True]
    assert synthetic.tolist() == [False, False]


def test_default_species_config_recognizes_current_chinese_csv_stems():
    assert {
        "东北虎",
        "东北豹",
        "黑熊",
        "梅花鹿",
        "袍",
    }.issubset(DEFAULT_0525_SPECIES_CONFIG)


def test_independent_events_use_consecutive_same_camera_gaps():
    data = pd.DataFrame({
        "species": ["Sika deer"] * 3,
        "camera_id": ["1"] * 3,
        "timestamp_dt": pd.to_datetime([
            "2024-01-01 00:00:00Z",
            "2024-01-01 00:20:00Z",
            "2024-01-01 00:50:00Z",
        ]),
        "mediaID": ["a", "b", "c"],
        "deploymentID": ["dep_1"] * 3,
        "scientificName": ["Cervus nippon"] * 3,
        "latitude": [43.0] * 3,
        "longitude": [130.0] * 3,
        "timestamp_has_observed_clock": [True] * 3,
        "timestamp_is_synthetic": [False] * 3,
    })

    _, events_30 = add_independent_events(data, min_gap_minutes=30)
    _, events_15 = add_independent_events(data, min_gap_minutes=15)
    assert len(events_30) == 1
    assert len(events_15) == 3


def test_coordinate_resolution_audit_is_privacy_preserving(tmp_path):
    data = pd.DataFrame({
        "camera_id": ["sensitive-1", "sensitive-1", "sensitive-2"],
        "latitude": [43.0, 43.01, 44.0],
        "longitude": [130.0, 130.02, 131.0],
    })
    output_path = tmp_path / "coordinate_resolution_audit.csv"
    audit = write_coordinate_resolution_audit(
        data,
        output_path,
        package_name="test-package",
    )
    public_text = output_path.read_text(encoding="utf-8")
    assert "sensitive-1" not in public_text
    assert "43.0" not in public_text
    assert int(audit["coordinate_conflict"].sum()) == 1
    assert set(pd.read_csv(output_path).columns) == {
        "camera_token",
        "source_record_count",
        "unique_coordinate_pair_count",
        "maximum_coordinate_span_km",
        "coordinate_conflict",
        "resolution_method",
    }


def _write_small_camtrap_package(package_dir):
    package_dir.mkdir()
    pd.DataFrame([{
        "deploymentID": "dep_1", "cameraID": "1", "locationID": "1",
        "latitude": 43.0, "longitude": 130.0,
        "deploymentStart": "2024-01-01T00:00:00Z",
        "deploymentEnd": "2024-01-02T00:00:00Z",
        "deploymentComments": "Deployment approximated from camera ID in source CSV.",
    }]).to_csv(package_dir / "deployments.csv", index=False)
    times = ["2024-01-01T00:00:00Z", "2024-01-01T00:20:00Z", "2024-01-01T00:50:00Z"]
    pd.DataFrame([{
        "mediaID": f"m{i}", "deploymentID": "dep_1", "timestamp": timestamp,
        "filePath": f"{i}.jpg", "fileName": f"{i}.jpg", "fileMediatype": "image/jpeg",
        "captureMethod": "activityDetection",
        "mediaComments": "source_recordtime=2024-01-01 00:00; timestamp_source=recordtime",
    } for i, timestamp in enumerate(times)]).to_csv(package_dir / "media.csv", index=False)
    pd.DataFrame([{
        "observationID": f"o{i}", "deploymentID": "dep_1", "mediaID": f"m{i}",
        "eventID": "cached", "eventStart": timestamp, "eventEnd": timestamp,
        "observationLevel": "media", "observationType": "animal",
        "scientificName": "Cervus nippon", "count": 1,
    } for i, timestamp in enumerate(times)]).to_csv(package_dir / "observations.csv", index=False)
    pd.DataFrame([{
        "event_id": "cached", "deploymentID": "dep_1", "camera_id": "1",
        "species": "Sika deer", "scientificName": "Cervus nippon",
        "event_time": "2024-01-01T00:25:00Z", "start_time": times[0], "end_time": times[-1],
        "event_record_count": 3, "latitude": 43.0, "longitude": 130.0,
    }]).to_csv(package_dir / "agent_independent_events.csv", index=False)
    (package_dir / "datapackage.json").write_text(json.dumps({"resources": [
        {"name": "deployments", "path": "deployments.csv"},
        {"name": "media", "path": "media.csv"},
        {"name": "observations", "path": "observations.csv"},
    ]}))
    (package_dir / "conversion_report.json").write_text(json.dumps({
        "filters": {"min_gap_minutes": 30},
        "species_summary": {"Sika deer": {"media_records": 3, "independent_events": 1}},
    }))


def test_camtrap_backend_recomputes_non_cached_event_gap(tmp_path):
    package_dir = tmp_path / "package"
    _write_small_camtrap_package(package_dir)
    tools = BaseTools(db_path=tmp_path / "chroma")
    tools.set_camtrap_data_dir(package_dir, activate=True, persist_active=False)

    cached = tools.build_independent_events("Sika deer", min_gap_minutes=30)
    recomputed = tools.build_independent_events("Sika deer", min_gap_minutes=15)
    assert cached["event_count"] == 1
    assert cached["event_source"] == "cached"
    assert recomputed["event_count"] == 3
    assert recomputed["event_source"] == "recomputed_from_media"


def test_specific_deer_queries_do_not_expand_to_other_deer_species():
    tools = BaseTools.__new__(BaseTools)
    assert "roe deer" not in tools._expand_species_query_tokens("Sika deer")
    assert "sika deer" not in tools._expand_species_query_tokens("Roe deer")
    assert {"sika deer", "roe deer", "musk deer"}.issubset(
        tools._expand_species_query_tokens("deer")
    )


def test_camtrap_utc_storage_is_restored_to_declared_field_clock(tmp_path):
    package_dir = tmp_path / "package"
    _write_small_camtrap_package(package_dir)
    report_path = package_dir / "conversion_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_timezone"] = "Asia/Shanghai"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    tools = BaseTools(db_path=tmp_path / "chroma")
    tools.set_camtrap_data_dir(package_dir, activate=True, persist_active=False)
    parsed = tools._parse_camtrap_timestamp_series(
        pd.Series(["2024-01-01T09:30:00Z"]),
        local_clock=True,
    )
    assert parsed.dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == [
        "2024-01-01 17:30:00"
    ]
    assert tools.safe_parse_timestamp("2024-01-01T09:30:00Z").hour == 17


def test_overlap_without_bootstrap_does_not_report_degenerate_interval(tmp_path):
    package_dir = tmp_path / "package"
    _write_small_camtrap_package(package_dir)
    tools = AgentRouter(db_path=tmp_path / "chroma")
    tools.set_camtrap_data_dir(package_dir, activate=True, persist_active=False)
    result = tools.estimate_activity_overlap(
        "Sika deer",
        "Sika deer",
        min_gap_minutes=15,
        bootstrap_iters=0,
        permutation_iters=0,
    )
    assert result["delta_hat"] is not None
    assert result["delta_ci_low"] is None
    assert result["delta_ci_high"] is None
    assert result["bootstrap_unit"] is None
    assert result["interval_calibration_status"] == "not_applicable_no_interval_computed"


def test_overlap_result_bundle_is_reused_across_tool_instances(tmp_path):
    package_dir = tmp_path / "package"
    _write_small_camtrap_package(package_dir)
    bundle_dir = tmp_path / "overlap_bundles"

    first_tools = AgentRouter(db_path=tmp_path / "chroma_first")
    first_tools.set_camtrap_data_dir(
        package_dir,
        activate=True,
        persist_active=False,
    )
    first_tools._overlap_result_bundle_dir = bundle_dir
    first_tools._overlap_result_memory_cache = {}
    first = first_tools.estimate_activity_overlap(
        "Sika deer",
        "Sika deer",
        min_gap_minutes=15,
        bootstrap_iters=500,
        permutation_iters=0,
    )

    assert first["overlap_result_reused"] is False
    assert first["estimator_id"] == "camagent.circular_kde_activity_overlap"
    assert first["estimator_version"] == "1.0.0"
    bundle_path = bundle_dir / f"{first['overlap_result_bundle_id']}.json"
    assert bundle_path.exists()
    stored = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert stored["bundle_sha256"] == first["overlap_result_bundle_sha256"]
    assert stored["result_sha256"] == first["overlap_result_sha256"]

    second_tools = AgentRouter(db_path=tmp_path / "chroma_second")
    second_tools.set_camtrap_data_dir(
        package_dir,
        activate=True,
        persist_active=False,
    )
    second_tools._overlap_result_bundle_dir = bundle_dir
    second_tools._overlap_result_memory_cache = {}
    second = second_tools.estimate_activity_overlap(
        "Sika deer",
        "Sika deer",
        min_gap_minutes=15,
        bootstrap_iters=500,
        permutation_iters=0,
    )

    assert second["overlap_result_reused"] is True
    assert second["overlap_result_bundle_id"] == first["overlap_result_bundle_id"]
    assert second["overlap_result_bundle_sha256"] == first["overlap_result_bundle_sha256"]
    assert second["delta_hat"] == first["delta_hat"]

    relationship = second_tools.analyze_species_pair_relationship(
        "Sika deer",
        "Sika deer",
        min_gap_minutes=15,
    )
    assert relationship["activity_overlap_delta"] == first["delta_hat"]
    assert relationship["activity_overlap_ci"] == {
        "low": first["delta_ci_low"],
        "high": first["delta_ci_high"],
    }
    assert (
        relationship["activity_overlap_result_bundle_id"]
        == first["overlap_result_bundle_id"]
    )
    assert relationship["activity_overlap_result_reused"] is True


def test_occupancy_refuses_detection_derived_effort_by_default(monkeypatch):
    tools = SpatialTools.__new__(SpatialTools)
    monkeypatch.setattr(tools, "build_detection_history_matrix", lambda **_: {
        "camera_detection_histories": [{"detection_history": [1, 0], "detected": True}],
        "effort_basis": "detection_span_proxy",
        "effort_supports_occupancy_inference": False,
    })
    result = tools.fit_single_season_occupancy_model("Siberian tiger")
    assert result["model_status"] == "not_fitted_without_independent_effort"
    assert result["occupancy_probability_psi"] is None


def test_agent_router_is_only_a_tool_aggregation_layer():
    router = AgentRouter.__new__(AgentRouter)
    assert not hasattr(router, "answer_spatial_question")
    assert not hasattr(router, "answer_temporal_question")
    assert not hasattr(router, "answer_data_quality_question")
    assert not hasattr(router, "answer_movement_question")


def test_research_pipeline_writes_reproducibility_artifacts(tmp_path):
    package_dir = tmp_path / "package"
    _write_small_camtrap_package(package_dir)
    output_dir = tmp_path / "output"
    config = {
        "project_name": "synthetic integration test",
        "input": {"type": "camtrap_dp", "datapackage": str(package_dir / "datapackage.json")},
        "output_dir": str(output_dir),
        "analysis": {
            "species": ["Sika deer"],
            "event_gap_minutes": 30,
            "event_gap_sensitivity_minutes": [15, 30],
            "random_seed": 7,
            "temporal_overlap": {"pairs": []},
            "occupancy": {"enabled": False},
            "camera_transition_hypotheses": {"enabled": False},
        },
        "manuscript_expectations": {"species": ["Sika deer"], "species_record_counts": {"Sika deer": 3}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = run_pipeline(config_path)
    assert manifest["status"] == "completed"
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "dataset_audit.json").exists()
    assert (output_dir / "tables" / "event_gap_sensitivity.csv").exists()
    assert (output_dir / "source_data_manifest.json").exists()
    assert (output_dir / "figure_manifest.json").exists()
    assert (output_dir / "code_manifest.json").exists()
    assert (output_dir / "tables" / "dataset_summary.csv").exists()
    assert (output_dir / "figures" / "fig_sensitivity_analysis.pdf").exists()
    assert (output_dir / "figures" / "fig_reproducibility_guardrails.png").exists()
    assert (output_dir / "tables" / "duplicate_resolution_summary.csv").exists()
    assert (output_dir / "release_readiness.json").exists()
    assert (output_dir / "manuscript_values.tex").exists()
    assert (output_dir / "manuscript_artifact_map.json").exists()
