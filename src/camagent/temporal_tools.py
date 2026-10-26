import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from .base_tools import BaseTools


def _integrate_trapezoid(y, x):
    integrator = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return integrator(y, x)


class TemporalTools(BaseTools):
    _ACTIVITY_OVERLAP_ESTIMATOR_ID = "camagent.circular_kde_activity_overlap"
    _ACTIVITY_OVERLAP_ESTIMATOR_VERSION = "1.0.0"
    _ACTIVITY_OVERLAP_BUNDLE_SCHEMA_VERSION = "1.0"

    def _resolve_comparison_species(
        self,
        focal_species,
        species_list=None,
        top_k=4,
    ):
        """Resolve comparison taxa from the active dataset without fixed names."""
        focal_species = self._canonical_species_name(focal_species)
        top_k = max(int(top_k), 1)

        if isinstance(species_list, str):
            requested = [
                item.strip()
                for item in species_list.replace("，", ",").replace("；", ";").replace(";", ",").split(",")
                if item.strip()
            ]
        elif isinstance(species_list, (list, tuple, set)):
            requested = [str(item).strip() for item in species_list if str(item).strip()]
        else:
            requested = []

        if requested:
            candidates = requested
        else:
            inventory = self.summarize_species_inventory(top_k=top_k + 10)
            candidates = [
                item.get("species")
                for item in inventory.get("species_inventory", [])
            ]

        selected = []
        seen = {focal_species.casefold(), "unknown"}
        for value in candidates:
            species = self._canonical_species_name(value)
            key = species.casefold()
            if not species or key in seen:
                continue
            seen.add(key)
            selected.append(species)
            if len(selected) >= top_k:
                break
        return selected

    def _overlap_canonical_json(self, value):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def _overlap_sha256(self, value):
        return hashlib.sha256(
            self._overlap_canonical_json(value).encode("utf-8")
        ).hexdigest()

    def _activity_overlap_source_evidence(self, event_a, event_b):
        def _rows(event):
            hours = list(event.get("event_hours") or [])
            cameras = list(event.get("event_cameras") or [])
            if len(cameras) != len(hours):
                cameras = [""] * len(hours)
            return sorted(
                [
                    {
                        "camera_id": str(camera),
                        "hour": round(float(hour) % 24.0, 9),
                    }
                    for hour, camera in zip(hours, cameras)
                ],
                key=lambda item: (item["camera_id"], item["hour"]),
            )

        evidence = {
            "species_a_events": _rows(event_a),
            "species_b_events": _rows(event_b),
        }
        return {
            "source_event_count_a": len(evidence["species_a_events"]),
            "source_event_count_b": len(evidence["species_b_events"]),
            "source_events_sha256": self._overlap_sha256(evidence),
        }

    def _activity_overlap_bundle_context(
        self,
        *,
        species_a,
        species_b,
        event_a,
        event_b,
        min_gap_minutes,
        smooth_sigma,
        bootstrap_iters,
        bootstrap_interval_method,
        bootstrap_cluster_strategy,
        permutation_iters,
        random_seed,
    ):
        source = self._activity_overlap_source_evidence(event_a, event_b)
        estimator = {
            "estimator_id": self._ACTIVITY_OVERLAP_ESTIMATOR_ID,
            "estimator_version": self._ACTIVITY_OVERLAP_ESTIMATOR_VERSION,
            "density_method": "circular_von_mises_kde",
            "integration_method": "trapezoid_minimum_density",
            "density_resolution_intervals": 288,
        }
        request = {
            "result_type": "two_species_activity_overlap",
            "species_a": str(species_a),
            "species_b": str(species_b),
            "min_gap_minutes": int(min_gap_minutes),
            "smooth_sigma_hours": float(smooth_sigma),
            "bootstrap_iters": int(bootstrap_iters),
            "bootstrap_interval_method": str(bootstrap_interval_method),
            "bootstrap_cluster_strategy": str(bootstrap_cluster_strategy),
            "permutation_iters": int(permutation_iters),
            "random_seed": int(random_seed),
            "source_events_sha256": source["source_events_sha256"],
            "estimator": estimator,
        }
        request_sha256 = self._overlap_sha256(request)
        return {
            "bundle_id": f"activity_overlap_{request_sha256[:20]}",
            "request": request,
            "request_sha256": request_sha256,
            "source": source,
            "estimator": estimator,
        }

    def _activity_overlap_bundle_path(self, bundle_id):
        bundle_dir = getattr(self, "_overlap_result_bundle_dir", None)
        if bundle_dir is None:
            return None
        return Path(bundle_dir) / f"{bundle_id}.json"

    def _validate_activity_overlap_bundle(self, payload, context):
        if not isinstance(payload, dict):
            return False
        if payload.get("schema_version") != self._ACTIVITY_OVERLAP_BUNDLE_SCHEMA_VERSION:
            return False
        if payload.get("request_sha256") != context["request_sha256"]:
            return False
        if payload.get("request") != context["request"]:
            return False
        result = payload.get("result")
        if not isinstance(result, dict):
            return False
        if payload.get("result_sha256") != self._overlap_sha256(result):
            return False
        unsigned = {
            key: value
            for key, value in payload.items()
            if key != "bundle_sha256"
        }
        return payload.get("bundle_sha256") == self._overlap_sha256(unsigned)

    def _attach_activity_overlap_bundle_metadata(
        self,
        result,
        payload,
        *,
        reused,
    ):
        attached = deepcopy(result)
        bundle_path = self._activity_overlap_bundle_path(payload["bundle_id"])
        attached.update({
            "estimator_id": payload["estimator"]["estimator_id"],
            "estimator_version": payload["estimator"]["estimator_version"],
            "source_events_sha256": payload["source"]["source_events_sha256"],
            "overlap_result_bundle_id": payload["bundle_id"],
            "overlap_result_bundle_sha256": payload["bundle_sha256"],
            "overlap_result_sha256": payload["result_sha256"],
            "overlap_result_bundle_path": (
                str(bundle_path) if bundle_path is not None else None
            ),
            "overlap_result_reused": bool(reused),
        })
        return attached

    def _load_activity_overlap_bundle(self, context):
        cache = getattr(self, "_overlap_result_memory_cache", None)
        if cache is None:
            cache = {}
            self._overlap_result_memory_cache = cache
        payload = cache.get(context["bundle_id"])
        if self._validate_activity_overlap_bundle(payload, context):
            return self._attach_activity_overlap_bundle_metadata(
                payload["result"],
                payload,
                reused=True,
            )

        bundle_path = self._activity_overlap_bundle_path(context["bundle_id"])
        if bundle_path is None or not bundle_path.exists():
            return None
        try:
            payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not self._validate_activity_overlap_bundle(payload, context):
            return None
        cache[context["bundle_id"]] = payload
        return self._attach_activity_overlap_bundle_metadata(
            payload["result"],
            payload,
            reused=True,
        )

    def _store_activity_overlap_bundle(self, result, context):
        core_result = deepcopy(result)
        for key in list(core_result):
            if key.startswith("overlap_result_") or key in {
                "estimator_id",
                "estimator_version",
                "source_events_sha256",
            }:
                core_result.pop(key, None)

        payload = {
            "schema_version": self._ACTIVITY_OVERLAP_BUNDLE_SCHEMA_VERSION,
            "bundle_id": context["bundle_id"],
            "result_type": "two_species_activity_overlap",
            "estimator": context["estimator"],
            "source": context["source"],
            "request": context["request"],
            "request_sha256": context["request_sha256"],
            "result": core_result,
            "result_sha256": self._overlap_sha256(core_result),
        }
        payload["bundle_sha256"] = self._overlap_sha256(payload)

        cache = getattr(self, "_overlap_result_memory_cache", None)
        if cache is None:
            cache = {}
            self._overlap_result_memory_cache = cache
        cache[context["bundle_id"]] = payload

        bundle_path = self._activity_overlap_bundle_path(context["bundle_id"])
        if bundle_path is not None:
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = bundle_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(bundle_path)

        return self._attach_activity_overlap_bundle_metadata(
            core_result,
            payload,
            reused=False,
        )

    def _activity_overlap_label(self, delta):
        if delta is None:
            return "unavailable"
        delta = float(delta)
        if delta >= 0.80:
            return "high_overlap"
        if delta >= 0.60:
            return "moderate_overlap"
        return "low_overlap"

    def _peak_period_label(self, peak_hours):
        hours = {int(hour) % 24 for hour in peak_hours or []}
        if not hours:
            return "insufficient_data"
        if hours & {5, 6, 7} and hours & {17, 18, 19}:
            return "crepuscular"
        if hours <= {5, 6, 7, 8} or hours & {5, 6, 7}:
            return "morning / dawn"
        if hours <= {16, 17, 18, 19, 20} or hours & {17, 18, 19}:
            return "evening / dusk"
        if all(6 <= hour <= 17 for hour in hours):
            return "daytime"
        if all(hour >= 20 or hour <= 5 for hour in hours):
            return "night"
        return "mixed"

    def _classify_activity_windows(self, hourly_distribution):
        windows = {
            "dawn": set(range(4, 8)),
            "daytime": set(range(8, 16)),
            "dusk": set(range(16, 21)),
            "night": {21, 22, 23, 0, 1, 2, 3},
        }
        shares = {name: 0.0 for name in windows}
        for item in hourly_distribution or []:
            hour = int(item.get("hour", 0)) % 24
            ratio = float(item.get("ratio", 0.0) or 0.0)
            for name, hour_set in windows.items():
                if hour in hour_set:
                    shares[name] += ratio
                    break

        ranked = sorted(shares.items(), key=lambda pair: pair[1], reverse=True)
        major_window, major_share = ranked[0] if ranked else ("none", 0.0)
        second_window, second_share = ranked[1] if len(ranked) > 1 else ("none", 0.0)
        secondary_window = "none"
        if major_share > 0 and second_share >= major_share * 0.60:
            secondary_window = second_window

        if major_share > 0 and second_share >= major_share * 0.85 and major_share < 0.35:
            label = "broadly distributed"
        elif major_window == "dusk" and secondary_window == "dawn":
            label = "bimodal crepuscular"
        elif major_window == "dusk":
            label = "dusk-centered crepuscular"
        elif major_window == "dawn" and secondary_window == "dusk":
            label = "bimodal crepuscular"
        elif major_window == "night":
            label = "nocturnal"
        elif major_window == "daytime":
            label = "diurnal"
        else:
            label = "broadly distributed"

        return {
            "activity_window_shares": {key: round(value, 4) for key, value in shares.items()},
            "major_peak_window": major_window if major_share > 0 else "none",
            "secondary_peak_window": secondary_window,
            "activity_pattern_label": label,
        }

    def _check_time_resolution(self, species="Siberian tiger"):
        """
        Check whether the current species records contain real hour-minute information.
        """
        if self._using_camtrap_backend():
            species = self._canonical_species_name(species)
            df = self._load_camtrap_event_df()
            if not df.empty and "species" not in df.columns:
                df = df.copy()
                df["species"] = df.get("scientificName", "").map(self._canonical_species_name)
            df = df[df["species"].map(self._canonical_species_name).eq(species)].copy() if not df.empty else df
            event_count = int(len(df))
            observed = int(pd.to_numeric(df.get("observed_clock_time_records", 0), errors="coerce").fillna(0).gt(0).sum()) if not df.empty else 0
            synthetic = int(pd.to_numeric(df.get("synthetic_time_records", 0), errors="coerce").fillna(0).gt(0).sum()) if not df.empty else 0
            ratio = observed / event_count if event_count else 0.0
            return {
                "species": species,
                "valid_time_records": event_count,
                "records_with_clock_time": observed,
                "clock_time_ratio": round(ratio, 4),
                "synthetic_time_records": synthetic,
                "observed_time_records": observed,
                "synthetic_clock_time_records": synthetic,
                "observed_clock_time_records": observed,
                "synthetic_time_ratio": round(synthetic / event_count, 4) if event_count else 0.0,
                "observed_time_ratio": round(ratio, 4),
                "synthetic_clock_time_ratio": round(synthetic / event_count, 4) if event_count else 0.0,
                "observed_clock_time_ratio": round(ratio, 4),
                "timestamp_source_summary": {"camtrap_dp_observed_clock": observed, "camtrap_dp_without_observed_clock": event_count - observed},
                "is_hourly_reliable": ratio >= 0.8,
            }

        records = self.get_species_records_with_parsed_time(species)
        summary = self.summarize_timestamp_sources(records)
        ratio = summary["observed_clock_time_ratio"]

        return {
            "species": species,
            "valid_time_records": summary["valid_time_records"],
            "records_with_clock_time": summary["observed_clock_time_records"],
            "clock_time_ratio": round(ratio, 4),
            "synthetic_time_records": summary["synthetic_time_records"],
            "observed_time_records": summary["observed_time_records"],
            "synthetic_clock_time_records": summary["synthetic_clock_time_records"],
            "observed_clock_time_records": summary["observed_clock_time_records"],
            "synthetic_time_ratio": summary["synthetic_time_ratio"],
            "observed_time_ratio": summary["observed_time_ratio"],
            "synthetic_clock_time_ratio": summary["synthetic_clock_time_ratio"],
            "observed_clock_time_ratio": summary["observed_clock_time_ratio"],
            "timestamp_source_summary": summary["timestamp_source_breakdown"],
                "is_hourly_reliable": ratio >= 0.8
        }

    def check_timestamp_quality(self, species="Siberian tiger"):
        """
        Lightweight timestamp reliability audit for traceable temporal analysis.
        """
        species = self._canonical_species_name(species)
        if self._using_camtrap_backend():
            records = self._load_camtrap_records_df()
            if records.empty:
                total_records = 0
                valid_records = 0
                source_summary = {"camtrap_dp_unspecified": 0}
            else:
                df = records.copy()
                if "species" not in df.columns:
                    df["species"] = df.get("scientificName", "").map(self._canonical_species_name)
                df["species"] = df["species"].map(self._canonical_species_name)
                df = df[df["species"] == species].copy()
                total_records = int(len(df))
                timestamp_col = "timestamp" if "timestamp" in df.columns else "eventStart"
                parsed = pd.to_datetime(df.get(timestamp_col), errors="coerce")
                valid_records = int(parsed.notna().sum())
                source_summary = df.loc[parsed.notna(), "timestamp_source"].fillna("camtrap_dp_unspecified").value_counts().to_dict()

            invalid_count = max(total_records - valid_records, 0)
            observed_clock_count = int(df["timestamp_has_observed_clock"].sum()) if total_records else 0
            synthetic_count = int(df["timestamp_is_synthetic"].sum()) if total_records else 0
            observed_ratio = round(observed_clock_count / total_records, 4) if total_records else 0.0
            return {
                "species": species,
                "total_records": total_records,
                "valid_timestamp_records": valid_records,
                "invalid_timestamp_count": invalid_count,
                "records_with_clock_time": observed_clock_count,
                "observed_clock_time_ratio": observed_ratio,
                "synthetic_clock_time_ratio": round(synthetic_count / total_records, 4) if total_records else 0.0,
                "timestamp_source_summary": source_summary,
                "timestamp_source": "CamtrapDP tabular",
                "reliability_status": "passed" if observed_ratio >= 0.8 else "failed",
                "note": "Timestamp quality is checked before hourly activity and overlap analysis.",
            }

        records = self.get_species_records_with_parsed_time(species)
        summary = self.summarize_timestamp_sources(records)
        total_records = len(records)
        valid_records = int(summary.get("valid_time_records", 0))
        observed_ratio = float(summary.get("observed_clock_time_ratio", 0.0))
        return {
            "species": species,
            "total_records": total_records,
            "valid_timestamp_records": valid_records,
            "invalid_timestamp_count": max(total_records - valid_records, 0),
            "records_with_clock_time": int(summary.get("records_with_clock_time", 0)),
            "observed_clock_time_ratio": observed_ratio,
            "synthetic_clock_time_ratio": float(summary.get("synthetic_clock_time_ratio", 0.0)),
            "timestamp_source_summary": summary.get("timestamp_source_breakdown", {}),
            "timestamp_source": "JSON / Chroma metadata",
            "reliability_status": "passed" if observed_ratio >= 0.8 else "failed",
            "note": "Timestamp quality is checked before hourly activity and overlap analysis.",
        }

    def _build_timestamp_quality_note(self, resolution, context="hourly analysis"):
        if not resolution["is_hourly_reliable"]:
            return (
                f"{context.capitalize()} is not reliable because most timestamps do not contain explicit clock time."
            )

        if resolution.get("synthetic_clock_time_records", 0) > 0:
            return (
                f"{context.capitalize()} uses observed clock-time records only; synthetic clock times are skipped."
            )

        return f"{context.capitalize()} is available because sufficient records contain explicit clock time."

    def _detect_curve_extrema(
        self,
        curve_x,
        curve_y,
        kind="peak",
        top_k=3,
        min_separation_hours=2.0,
    ):
        curve_x = np.asarray(curve_x, dtype=float)
        curve_y = np.asarray(curve_y, dtype=float)
        if curve_x.size < 3 or curve_y.size < 3:
            return []

        candidates = []
        for idx in range(1, len(curve_y) - 1):
            prev_y = curve_y[idx - 1]
            curr_y = curve_y[idx]
            next_y = curve_y[idx + 1]

            if kind == "peak":
                is_extreme = curr_y >= prev_y and curr_y >= next_y
                score = curr_y
            else:
                is_extreme = curr_y <= prev_y and curr_y <= next_y
                score = -curr_y

            if is_extreme:
                hour = float(curve_x[idx] % 24.0)
                candidates.append({
                    "hour": round(hour, 3),
                    "value": round(float(curr_y), 6),
                    "_score": float(score),
                })

        candidates.sort(key=lambda item: item["_score"], reverse=True)
        selected = []
        for item in candidates:
            if all(abs(item["hour"] - chosen["hour"]) >= min_separation_hours for chosen in selected):
                selected.append(item)
            if len(selected) >= int(top_k):
                break

        for item in selected:
            item.pop("_score", None)
        return selected

    def _extract_event_hours(self, species="Siberian tiger", min_gap_minutes=30):
        if self._using_camtrap_backend():
            species = self._canonical_species_name(species)
            df = self._load_camtrap_event_df(min_gap_minutes=min_gap_minutes)
            if df.empty:
                return {
                    "species": species,
                    "min_gap_minutes": min_gap_minutes,
                    "clock_time_ratio": 0.0,
                    "synthetic_clock_time_ratio": 0.0,
                    "timestamp_source_summary": {"camtrap_dp_unspecified": 0},
                    "is_hourly_reliable": False,
                    "valid_event_count": 0,
                    "independent_event_count": 0,
                    "event_hours": [],
                    "event_cameras": [],
                }
            df = df.copy()
            if "species" not in df.columns:
                df["species"] = df.get("scientificName", "").map(self._canonical_species_name)
            df["species"] = df["species"].map(self._canonical_species_name)
            df = df[df["species"] == species].copy()
            total_event_count = int(len(df))
            if "has_observed_clock_time" in df.columns:
                df = df[df["has_observed_clock_time"].astype(bool)].copy()
            if "event_time_dt" in df.columns:
                df["_event_dt"] = df["event_time_dt"]
            elif "event_time" in df.columns:
                df["_event_dt"] = pd.to_datetime(df["event_time"], errors="coerce")
            else:
                df["_event_dt"] = pd.to_datetime(df.get("eventStart"), errors="coerce")
            df = df.dropna(subset=["_event_dt"])
            dt = df["_event_dt"]
            hours = (dt.dt.hour + dt.dt.minute / 60.0 + dt.dt.second / 3600.0).astype(float).tolist()
            camera_col = "camera_id" if "camera_id" in df.columns else "cameraID"
            cameras = df[camera_col].astype(str).tolist()
            return {
                "species": species,
                "min_gap_minutes": min_gap_minutes,
                "clock_time_ratio": round(len(hours) / total_event_count, 4) if total_event_count else 0.0,
                "synthetic_clock_time_ratio": 0.0,
                "timestamp_source_summary": {"camtrap_dp_observed_clock": len(hours)},
                "is_hourly_reliable": bool(hours),
                "valid_event_count": len(hours),
                "independent_event_count": total_event_count,
                "event_hours": hours,
                "event_cameras": cameras,
            }

        resolution = self._check_time_resolution(species)
        event_result = self.build_independent_events(
            species=species,
            by_camera=True,
            min_gap_minutes=min_gap_minutes,
        )

        event_hours = []
        event_cameras = []
        for event in event_result["events"]:
            if not event.get("has_observed_clock_time"):
                continue
            dt = self.safe_parse_timestamp(event.get("observed_event_time") or event.get("event_time"))
            if dt is None:
                continue
            event_hours.append(dt.hour + dt.minute / 60.0 + dt.second / 3600.0)
            event_cameras.append(str(event.get("camera_id", "")))

        return {
            "species": species,
            "min_gap_minutes": min_gap_minutes,
            "clock_time_ratio": resolution["clock_time_ratio"],
            "synthetic_clock_time_ratio": resolution["synthetic_clock_time_ratio"],
            "timestamp_source_summary": resolution["timestamp_source_summary"],
            "is_hourly_reliable": resolution["is_hourly_reliable"],
            "valid_event_count": len(event_hours),
            "independent_event_count": event_result["event_count"],
            "event_hours": event_hours,
            "event_cameras": event_cameras,
        }

    def _extract_other_species_event_hours(self, focal_species="Siberian tiger", min_gap_minutes=30):
        focal_species = self._canonical_species_name(focal_species)

        if self._using_camtrap_backend():
            df = self._load_camtrap_event_df(min_gap_minutes=min_gap_minutes)
            if df.empty:
                return {
                    "species": "all_other_species",
                    "focal_species": focal_species,
                    "min_gap_minutes": min_gap_minutes,
                    "clock_time_ratio": 0.0,
                    "synthetic_clock_time_ratio": 0.0,
                    "timestamp_source_summary": {"camtrap_dp_unspecified": 0},
                    "is_hourly_reliable": False,
                    "valid_event_count": 0,
                    "independent_event_count": 0,
                    "event_hours": [],
                    "event_cameras": [],
                }

            df = df.copy()
            if "species" not in df.columns:
                df["species"] = df.get("scientificName", "").map(self._canonical_species_name)
            df["species"] = df["species"].map(self._canonical_species_name)
            df = df[(df["species"] != focal_species) & (df["species"].astype(str).str.strip() != "")].copy()
            total_event_count = int(len(df))
            if "has_observed_clock_time" in df.columns:
                df = df[df["has_observed_clock_time"].astype(bool)].copy()

            if "event_time_dt" in df.columns:
                df["_event_dt"] = df["event_time_dt"]
            elif "event_time" in df.columns:
                df["_event_dt"] = pd.to_datetime(df["event_time"], errors="coerce")
            else:
                df["_event_dt"] = pd.to_datetime(df.get("eventStart"), errors="coerce")
            df = df.dropna(subset=["_event_dt"])
            dt = df["_event_dt"]
            hours = (dt.dt.hour + dt.dt.minute / 60.0 + dt.dt.second / 3600.0).astype(float).tolist()
            camera_col = "camera_id" if "camera_id" in df.columns else "cameraID"
            cameras = df[camera_col].astype(str).tolist()
            return {
                "species": "all_other_species",
                "focal_species": focal_species,
                "min_gap_minutes": min_gap_minutes,
                "clock_time_ratio": round(len(hours) / total_event_count, 4) if total_event_count else 0.0,
                "synthetic_clock_time_ratio": 0.0,
                "timestamp_source_summary": {"camtrap_dp_observed_clock": len(hours)},
                "is_hourly_reliable": bool(hours),
                "valid_event_count": len(hours),
                "independent_event_count": total_event_count,
                "event_hours": hours,
                "event_cameras": cameras,
            }

        event_hours = []
        event_cameras = []
        source_counts = {}
        for record in self.get_all_records():
            metadata = record.get("metadata", {}) or {}
            species = self._canonical_species_name(metadata.get("species") or metadata.get("scientific_name") or "")
            if not species or species == focal_species:
                continue
            parsed_dt = self.safe_parse_timestamp(metadata.get("timestamp"))
            if parsed_dt is None or self.is_synthetic_timestamp_metadata(metadata):
                continue
            source = str(metadata.get("timestamp_source", "")).strip() or "unspecified"
            source_counts[source] = source_counts.get(source, 0) + 1
            event_hours.append(parsed_dt.hour + parsed_dt.minute / 60.0 + parsed_dt.second / 3600.0)
            event_cameras.append(str(metadata.get("camera_id", "")))

        return {
            "species": "all_other_species",
            "focal_species": focal_species,
            "min_gap_minutes": min_gap_minutes,
            "clock_time_ratio": 1.0 if event_hours else 0.0,
            "synthetic_clock_time_ratio": 0.0,
            "timestamp_source_summary": source_counts or {"observed_records": 0},
            "is_hourly_reliable": bool(event_hours),
            "valid_event_count": len(event_hours),
            "independent_event_count": len(event_hours),
            "event_hours": event_hours,
            "event_cameras": event_cameras,
        }

    def _circular_density_from_histogram(self, counts, smooth_sigma=1.25):
        counts = np.asarray(counts, dtype=float)
        resolution = len(counts)
        resolution = max(int(resolution), 48)
        grid = np.linspace(0.0, 24.0, resolution, endpoint=False)
        weights = counts.astype(float) / max(float(counts.sum()), 1.0)
        sigma_radians = max(float(smooth_sigma), 0.25) / 24.0 * (2.0 * np.pi)
        kappa = min(1.0 / (sigma_radians ** 2), 500.0)
        normalizer = 2.0 * np.pi * np.i0(kappa)
        offsets = np.arange(resolution, dtype=float) / resolution * (2.0 * np.pi)
        kernel = np.exp(kappa * np.cos(offsets)) / normalizer * (2.0 * np.pi / 24.0)
        density = np.fft.ifft(np.fft.fft(weights) * np.fft.fft(kernel)).real
        density = np.maximum(density, 0.0)
        dense_x = np.append(grid, 24.0)
        dense_y = np.append(density, density[0])

        area = _integrate_trapezoid(dense_y, dense_x)
        if area > 0:
            dense_y = dense_y / area

        return dense_x, dense_y

    def _smooth_circular_density_from_event_hours(self, event_hours, smooth_sigma=1.25, resolution=288):
        resolution = max(int(resolution), 48)
        if event_hours is None or len(event_hours) == 0:
            dense_x = np.linspace(0.0, 24.0, resolution + 1)
            return dense_x, np.zeros_like(dense_x)
        hours = np.mod(np.asarray(event_hours, dtype=float), 24.0)
        counts, _ = np.histogram(hours, bins=np.linspace(0.0, 24.0, resolution + 1))
        return self._circular_density_from_histogram(counts, smooth_sigma=smooth_sigma)

    def _compute_overlap_from_event_hours(self, hours_a, hours_b, smooth_sigma=1.25, resolution=288):
        hour_grid, density_a = self._smooth_circular_density_from_event_hours(
            hours_a,
            smooth_sigma=smooth_sigma,
            resolution=resolution,
        )
        _, density_b = self._smooth_circular_density_from_event_hours(
            hours_b,
            smooth_sigma=smooth_sigma,
            resolution=resolution,
        )
        overlap_density = np.minimum(density_a, density_b)
        delta_hat = float(_integrate_trapezoid(overlap_density, hour_grid))
        return hour_grid, density_a, density_b, overlap_density, delta_hat

    def estimate_activity_overlap(
        self,
        species_a,
        species_b,
        min_gap_minutes=30,
        smooth_sigma=1.25,
        bootstrap_iters=500,
        bootstrap_interval_method="percentile",
        bootstrap_cluster_strategy="joint_camera_network",
        permutation_iters=0,
        random_seed=42,
    ):
        """
        Estimate two-species 24-hour activity overlap from independent events.
        Uses smoothed 24-hour density curves and bootstrap intervals for the overlap coefficient.
        """
        bootstrap_interval_method = str(bootstrap_interval_method).strip().casefold()
        if bootstrap_interval_method not in {"percentile", "basic_boundary_clipped"}:
            raise ValueError(
                "bootstrap_interval_method must be 'percentile' or 'basic_boundary_clipped'."
            )
        bootstrap_cluster_strategy = str(
            bootstrap_cluster_strategy
        ).strip().casefold()
        if bootstrap_cluster_strategy not in {
            "joint_camera_network",
            "independent_species_camera_sets",
        }:
            raise ValueError(
                "bootstrap_cluster_strategy must be 'joint_camera_network' "
                "or 'independent_species_camera_sets'."
            )
        species_a = self._canonical_species_name(species_a)
        species_b_text = str(species_b).strip()
        if species_b_text.casefold() in {
            "all_other_species",
            "all other species",
            "other_species",
            "other species",
        }:
            species_b = "all_other_species"
        else:
            species_b = self._canonical_species_name(species_b_text)

        event_a = self._extract_event_hours(species_a, min_gap_minutes=min_gap_minutes)
        if species_b == "all_other_species":
            event_b = self._extract_other_species_event_hours(species_a, min_gap_minutes=min_gap_minutes)
        else:
            event_b = self._extract_event_hours(species_b, min_gap_minutes=min_gap_minutes)

        bundle_context = self._activity_overlap_bundle_context(
            species_a=species_a,
            species_b=species_b,
            event_a=event_a,
            event_b=event_b,
            min_gap_minutes=min_gap_minutes,
            smooth_sigma=smooth_sigma,
            bootstrap_iters=bootstrap_iters,
            bootstrap_interval_method=bootstrap_interval_method,
            bootstrap_cluster_strategy=bootstrap_cluster_strategy,
            permutation_iters=permutation_iters,
            random_seed=random_seed,
        )
        reused_result = self._load_activity_overlap_bundle(bundle_context)
        if reused_result is not None:
            return reused_result

        hours_a = event_a["event_hours"]
        hours_b = event_b["event_hours"]
        cameras_a = event_a.get("event_cameras", [])
        cameras_b = event_b.get("event_cameras", [])

        if not hours_a or not hours_b:
            unavailable_result = {
                "species_a": species_a,
                "species_b": species_b,
                "min_gap_minutes": min_gap_minutes,
                "smooth_sigma": smooth_sigma,
                "bootstrap_iters": bootstrap_iters,
                "bootstrap_interval_method": bootstrap_interval_method,
                "bootstrap_cluster_strategy": bootstrap_cluster_strategy,
                "permutation_iters": permutation_iters,
                "permutation_unit": None,
                "permutation_inference_status": "not_computed",
                "clock_time_ratio_a": event_a["clock_time_ratio"],
                "clock_time_ratio_b": event_b["clock_time_ratio"],
                "synthetic_clock_time_ratio_a": event_a["synthetic_clock_time_ratio"],
                "synthetic_clock_time_ratio_b": event_b["synthetic_clock_time_ratio"],
                "timestamp_source_summary_a": event_a["timestamp_source_summary"],
                "timestamp_source_summary_b": event_b["timestamp_source_summary"],
                "valid_event_count_a": event_a["valid_event_count"],
                "valid_event_count_b": event_b["valid_event_count"],
                "delta_hat": None,
                "delta_ci_low": None,
                "delta_ci_high": None,
                "permutation_p_value": None,
                "temporal_relationship": None,
                "supports_temporal_separation": None,
                "hour_grid": [],
                "density_a": [],
                "density_b": [],
                "overlap_density": [],
                "peak_hours_a": [],
                "peak_hours_b": [],
                "note": "Activity overlap is unavailable because one or both species have no independent events with explicit clock time.",
            }
            return self._store_activity_overlap_bundle(
                unavailable_result,
                bundle_context,
            )

        hour_grid, density_a, density_b, overlap_density, delta_hat = self._compute_overlap_from_event_hours(
            hours_a,
            hours_b,
            smooth_sigma=smooth_sigma,
            resolution=288,
        )

        if bootstrap_iters and bootstrap_iters > 1 and len(hours_a) > 1 and len(hours_b) > 1:
            rng = np.random.default_rng(int(random_seed))
            boot_values = []
            arr_a = np.asarray(hours_a, dtype=float)
            arr_b = np.asarray(hours_b, dtype=float)

            resolution = len(hour_grid) - 1

            def _camera_histograms(values, cameras):
                grouped = {}
                if len(values) != len(cameras):
                    return grouped
                for value, camera in zip(values, cameras):
                    grouped.setdefault(str(camera), []).append(float(value))
                bins = np.linspace(0.0, 24.0, resolution + 1)
                return {
                    key: np.histogram(np.mod(np.asarray(items, dtype=float), 24.0), bins=bins)[0]
                    for key, items in grouped.items()
                    if key
                }

            groups_a = _camera_histograms(arr_a, cameras_a)
            groups_b = _camera_histograms(arr_b, cameras_b)
            camera_bootstrap = len(groups_a) > 1 and len(groups_b) > 1
            common_camera_count = len(set(groups_a) & set(groups_b))
            union_camera_count = len(set(groups_a) | set(groups_b))
            actual_cluster_strategy = (
                bootstrap_cluster_strategy if camera_bootstrap else "record"
            )

            if camera_bootstrap:
                def _weighted_camera_histograms(matrix, weights):
                    try:
                        from scipy.sparse import csr_matrix

                        return np.asarray(weights @ csr_matrix(matrix))
                    except ImportError:
                        return weights @ matrix

                iterations = int(bootstrap_iters)
                if bootstrap_cluster_strategy == "joint_camera_network":
                    keys = np.asarray(sorted(set(groups_a) | set(groups_b)), dtype=object)
                    zero_histogram = np.zeros(resolution, dtype=int)
                    matrix_a = np.vstack([
                        groups_a.get(key, zero_histogram) for key in keys
                    ])
                    matrix_b = np.vstack([
                        groups_b.get(key, zero_histogram) for key in keys
                    ])
                    probabilities = np.full(len(keys), 1.0 / len(keys))
                    weights = rng.multinomial(len(keys), probabilities, size=iterations)
                    bootstrap_hist_a = _weighted_camera_histograms(matrix_a, weights)
                    bootstrap_hist_b = _weighted_camera_histograms(matrix_b, weights)
                else:
                    keys_a = np.asarray(sorted(groups_a), dtype=object)
                    keys_b = np.asarray(sorted(groups_b), dtype=object)
                    matrix_a = np.vstack([groups_a[key] for key in keys_a])
                    matrix_b = np.vstack([groups_b[key] for key in keys_b])
                    probabilities_a = np.full(len(keys_a), 1.0 / len(keys_a))
                    probabilities_b = np.full(len(keys_b), 1.0 / len(keys_b))
                    weights_a = rng.multinomial(
                        len(keys_a), probabilities_a, size=iterations
                    )
                    weights_b = rng.multinomial(
                        len(keys_b), probabilities_b, size=iterations
                    )
                    bootstrap_hist_a = _weighted_camera_histograms(
                        matrix_a, weights_a
                    )
                    bootstrap_hist_b = _weighted_camera_histograms(
                        matrix_b, weights_b
                    )
                for counts_a, counts_b in zip(bootstrap_hist_a, bootstrap_hist_b):
                    if counts_a.sum() == 0 or counts_b.sum() == 0:
                        continue
                    boot_x, boot_density_a = self._circular_density_from_histogram(
                        counts_a, smooth_sigma=smooth_sigma
                    )
                    _, boot_density_b = self._circular_density_from_histogram(
                        counts_b, smooth_sigma=smooth_sigma
                    )
                    boot_values.append(float(_integrate_trapezoid(np.minimum(boot_density_a, boot_density_b), boot_x)))
            else:
                common_camera_count = 0
                union_camera_count = 0
                actual_cluster_strategy = "record"
                for _ in range(int(bootstrap_iters)):
                    sample_a = rng.choice(arr_a, size=len(arr_a), replace=True)
                    sample_b = rng.choice(arr_b, size=len(arr_b), replace=True)
                    boot_x, boot_density_a = self._smooth_circular_density_from_event_hours(
                        sample_a,
                        smooth_sigma=smooth_sigma,
                        resolution=resolution,
                    )
                    _, boot_density_b = self._smooth_circular_density_from_event_hours(
                        sample_b,
                        smooth_sigma=smooth_sigma,
                        resolution=resolution,
                    )
                    boot_values.append(float(_integrate_trapezoid(np.minimum(boot_density_a, boot_density_b), boot_x)))

            if len(boot_values) > 1:
                percentile_low, percentile_high = np.percentile(
                    boot_values, [2.5, 97.5]
                ).tolist()
                if bootstrap_interval_method == "basic_boundary_clipped":
                    delta_ci_low = max(
                        0.0, 2.0 * delta_hat - float(percentile_high)
                    )
                    delta_ci_high = min(
                        1.0, 2.0 * delta_hat - float(percentile_low)
                    )
                else:
                    delta_ci_low, delta_ci_high = (
                        percentile_low,
                        percentile_high,
                    )
            else:
                delta_ci_low = None
                delta_ci_high = None
        else:
            delta_ci_low = None
            delta_ci_high = None
            camera_bootstrap = False
            common_camera_count = 0
            union_camera_count = 0
            actual_cluster_strategy = "not_applicable"

        permutation_p_value = None
        if permutation_iters and permutation_iters > 1 and len(hours_a) > 1 and len(hours_b) > 1:
            rng = np.random.default_rng(int(random_seed) + 1)
            pooled = np.asarray(list(hours_a) + list(hours_b), dtype=float)
            n_a = len(hours_a)
            perm_deltas = []

            for _ in range(int(permutation_iters)):
                permuted = rng.permutation(pooled)
                perm_a = permuted[:n_a]
                perm_b = permuted[n_a:]
                _, _, _, _, perm_delta = self._compute_overlap_from_event_hours(
                    perm_a,
                    perm_b,
                    smooth_sigma=smooth_sigma,
                    resolution=len(hour_grid) - 1,
                )
                perm_deltas.append(perm_delta)

            if perm_deltas:
                smaller_or_equal = sum(1 for value in perm_deltas if value <= delta_hat)
                permutation_p_value = (smaller_or_equal + 1.0) / (len(perm_deltas) + 1.0)

        temporal_relationship = self._activity_overlap_label(delta_hat)

        statistically_distinguishable_activity_distributions = (
            permutation_p_value <= 0.05
            if permutation_p_value is not None
            else None
        )
        supports_strong_temporal_separation = (
            statistically_distinguishable_activity_distributions
            and delta_hat < 0.5
            if statistically_distinguishable_activity_distributions is not None
            else None
        )

        peak_hours_a = [int(item["hour"]) for item in self.detect_activity_peaks(
            species_a,
            top_k=3,
            min_gap_minutes=min_gap_minutes,
        ) if item.get("count", 0) > 0]
        peak_hours_b = [int(item["hour"]) for item in self.detect_activity_peaks(
            species_b,
            top_k=3,
            min_gap_minutes=min_gap_minutes,
        ) if item.get("count", 0) > 0]

        overlap_result = {
            "species_a": species_a,
            "species_b": species_b,
            "min_gap_minutes": min_gap_minutes,
            "smooth_sigma": smooth_sigma,
            "bootstrap_iters": int(bootstrap_iters),
            "bootstrap_interval_method": bootstrap_interval_method,
            "bootstrap_cluster_strategy": actual_cluster_strategy,
            "bootstrap_camera_count_common": common_camera_count,
            "bootstrap_camera_count_union": union_camera_count,
            "interval_calibration_status": (
                "requires_data_generating_process_specific_simulation"
                if delta_ci_low is not None
                else "not_applicable_no_interval_computed"
            ),
            "permutation_iters": int(permutation_iters),
            "permutation_unit": "event" if permutation_p_value is not None else None,
            "permutation_inference_status": (
                "exploratory_unclustered"
                if permutation_p_value is not None
                else "not_computed"
            ),
            "random_seed": int(random_seed),
            "clock_time_ratio_a": event_a["clock_time_ratio"],
            "clock_time_ratio_b": event_b["clock_time_ratio"],
            "synthetic_clock_time_ratio_a": event_a["synthetic_clock_time_ratio"],
            "synthetic_clock_time_ratio_b": event_b["synthetic_clock_time_ratio"],
            "timestamp_source_summary_a": event_a["timestamp_source_summary"],
            "timestamp_source_summary_b": event_b["timestamp_source_summary"],
            "valid_event_count_a": event_a["valid_event_count"],
            "valid_event_count_b": event_b["valid_event_count"],
            "independent_event_count_a": event_a["independent_event_count"],
            "independent_event_count_b": event_b["independent_event_count"],
            "delta_hat": round(delta_hat, 4),
            "delta_ci_low": round(float(delta_ci_low), 4) if delta_ci_low is not None else None,
            "delta_ci_high": round(float(delta_ci_high), 4) if delta_ci_high is not None else None,
            "bootstrap_unit": "camera" if camera_bootstrap else None,
            "permutation_p_value": round(float(permutation_p_value), 4) if permutation_p_value is not None else None,
            "separation_permutation_p_value": round(float(permutation_p_value), 4) if permutation_p_value is not None else None,
            "temporal_relationship": temporal_relationship,
            "statistically_distinguishable_activity_distributions": (
                statistically_distinguishable_activity_distributions
            ),
            "supports_temporal_separation": (
                statistically_distinguishable_activity_distributions
            ),
            "supports_strong_temporal_separation": supports_strong_temporal_separation,
            "hour_grid": [round(float(x), 4) for x in hour_grid.tolist()],
            "density_a": [round(float(x), 6) for x in density_a.tolist()],
            "density_b": [round(float(x), 6) for x in density_b.tolist()],
            "overlap_density": [round(float(x), 6) for x in overlap_density.tolist()],
            "peak_hours_a": sorted(set(peak_hours_a)),
            "peak_hours_b": sorted(set(peak_hours_b)),
            "note": (
                "Activity overlap is estimated from independent events with explicit clock time. "
                "The overlap coefficient is the area under the minimum of two smoothed 24 h density curves, "
                "with camera-clustered bootstrap intervals when camera identifiers are available. "
                "No permutation p-value is computed by default. If explicitly requested, the current event-label "
                "permutation is exploratory and unclustered; it must not be treated as confirmatory ecological inference. "
                "The descriptive bands are high (Delta >= 0.80), moderate (0.60 <= Delta < 0.80), and low (Delta < 0.60)."
            ),
        }
        return self._store_activity_overlap_bundle(
            overlap_result,
            bundle_context,
        )

    def estimate_hourly_activity_curve_with_ci(
        self,
        species="Siberian tiger",
        min_gap_minutes=30,
        smooth_sigma=1.15,
        bootstrap_iters=500,
        resolution=288,
    ):
        event_result = self._extract_event_hours(
            species=species,
            min_gap_minutes=min_gap_minutes,
        )

        event_hours = event_result["event_hours"]
        if not event_hours:
            return {
                "species": species,
                "min_gap_minutes": min_gap_minutes,
                "smooth_sigma": smooth_sigma,
                "bootstrap_iters": int(bootstrap_iters),
                "hour_grid": [],
                "smooth_mean": [],
                "ci_low": [],
                "ci_high": [],
                "valid_event_count": 0,
                "clock_time_ratio": event_result["clock_time_ratio"],
                "synthetic_clock_time_ratio": event_result["synthetic_clock_time_ratio"],
                "timestamp_source_summary": event_result["timestamp_source_summary"],
                "note": self._build_timestamp_quality_note(
                    {
                        "is_hourly_reliable": event_result["is_hourly_reliable"],
                        "synthetic_clock_time_records": 1 if event_result["synthetic_clock_time_ratio"] > 0 else 0,
                        "synthetic_clock_time_ratio": event_result["synthetic_clock_time_ratio"],
                    },
                    context="hourly curve estimation",
                ),
            }

        hour_grid, smooth_mean = self._smooth_circular_density_from_event_hours(
            event_hours,
            smooth_sigma=smooth_sigma,
            resolution=resolution,
        )
        ci_low = np.array(smooth_mean, copy=True)
        ci_high = np.array(smooth_mean, copy=True)

        if bootstrap_iters and bootstrap_iters > 1 and len(event_hours) > 1:
            rng = np.random.default_rng(42)
            arr = np.asarray(event_hours, dtype=float)
            boot_curves = []
            for _ in range(int(bootstrap_iters)):
                sample = rng.choice(arr, size=len(arr), replace=True)
                _, boot_curve = self._smooth_circular_density_from_event_hours(
                    sample,
                    smooth_sigma=smooth_sigma,
                    resolution=resolution,
                )
                boot_curves.append(boot_curve)

            boot_matrix = np.vstack(boot_curves)
            ci_low = np.percentile(boot_matrix, 2.5, axis=0)
            ci_high = np.percentile(boot_matrix, 97.5, axis=0)

        return {
            "species": species,
            "min_gap_minutes": min_gap_minutes,
            "smooth_sigma": float(smooth_sigma),
            "bootstrap_iters": int(bootstrap_iters),
            "hour_grid": [round(float(x), 4) for x in hour_grid.tolist()],
            "smooth_mean": [round(float(x), 6) for x in smooth_mean.tolist()],
            "ci_low": [round(float(x), 6) for x in ci_low.tolist()],
            "ci_high": [round(float(x), 6) for x in ci_high.tolist()],
            "valid_event_count": int(event_result["valid_event_count"]),
            "clock_time_ratio": event_result["clock_time_ratio"],
            "synthetic_clock_time_ratio": event_result["synthetic_clock_time_ratio"],
            "timestamp_source_summary": event_result["timestamp_source_summary"],
            "note": self._build_timestamp_quality_note(
                {
                    "is_hourly_reliable": event_result["is_hourly_reliable"],
                    "synthetic_clock_time_records": 1 if event_result["synthetic_clock_time_ratio"] > 0 else 0,
                    "synthetic_clock_time_ratio": event_result["synthetic_clock_time_ratio"],
                },
                context="hourly curve estimation",
            ),
        }

    def build_hourly_activity_figure_data(
        self,
        species="Siberian tiger",
        min_gap_minutes=30,
        smooth_sigma=1.15,
        bootstrap_iters=500,
    ):
        resolution = self._check_time_resolution(species)
        hourly_result = self.build_hourly_activity_distribution(
            species=species,
            min_gap_minutes=min_gap_minutes,
        )
        curve_result = self.estimate_hourly_activity_curve_with_ci(
            species=species,
            min_gap_minutes=min_gap_minutes,
            smooth_sigma=smooth_sigma,
            bootstrap_iters=bootstrap_iters,
        )
        circular_summary = self.circular_time_summary(
            species=species,
            min_gap_minutes=min_gap_minutes,
        )
        day_night_summary = self.compare_day_vs_night_activity(
            species=species,
            min_gap_minutes=min_gap_minutes,
        )

        major_peaks = []
        secondary_peaks = []
        troughs = []
        if curve_result.get("hour_grid"):
            major_peaks = self._detect_curve_extrema(
                curve_result["hour_grid"],
                curve_result["smooth_mean"],
                kind="peak",
                top_k=1,
            )
            secondary_peaks = self._detect_curve_extrema(
                curve_result["hour_grid"],
                curve_result["smooth_mean"],
                kind="peak",
                top_k=3,
            )[1:2]
            troughs = self._detect_curve_extrema(
                curve_result["hour_grid"],
                curve_result["smooth_mean"],
                kind="trough",
                top_k=2,
            )[:1]

        return {
            "species": species,
            "min_gap_minutes": min_gap_minutes,
            "smooth_sigma": float(smooth_sigma),
            "bootstrap_iters": int(bootstrap_iters),
            "event_count": hourly_result.get("independent_event_count", 0),
            "valid_time_records": hourly_result.get("valid_time_records", 0),
            "clock_time_ratio": hourly_result.get("clock_time_ratio", 0.0),
            "synthetic_time_ratio": resolution.get("synthetic_time_ratio", 0.0),
            "observed_time_ratio": resolution.get("observed_time_ratio", 0.0),
            "synthetic_clock_time_ratio": hourly_result.get("synthetic_clock_time_ratio", 0.0),
            "observed_clock_time_ratio": hourly_result.get("observed_clock_time_ratio", 0.0),
            "timestamp_source_summary": hourly_result.get("timestamp_source_summary", {}),
            "timestamp_quality_note": hourly_result.get("note"),
            "hourly_distribution": hourly_result.get("hourly_distribution", []),
            "smoothed_curve": {
                "hour": curve_result.get("hour_grid", []),
                "smooth_mean": curve_result.get("smooth_mean", []),
                "ci_low": curve_result.get("ci_low", []),
                "ci_high": curve_result.get("ci_high", []),
            },
            "peak_hours": self.detect_activity_peaks(
                species=species,
                top_k=5,
                min_gap_minutes=min_gap_minutes,
            ),
            "major_peak": major_peaks[0] if major_peaks else None,
            "secondary_peak": secondary_peaks[0] if secondary_peaks else None,
            "trough": troughs[0] if troughs else None,
            "mean_hour": circular_summary.get("mean_hour"),
            "activity_concentration": circular_summary.get("resultant_length"),
            "day_night_summary": day_night_summary,
            "note": hourly_result.get("note"),
        }

    def build_hourly_activity_distribution(self, species="Siberian tiger", min_gap_minutes=30):
        """
        Summarize 24-hour activity distribution.
        Raw records are first merged into independent events by default, then counted by event time.
        Results are reliable only when most raw timestamps include explicit hour-minute values.
        """
        if self._using_camtrap_backend():
            event_hours = self._extract_event_hours(species=species, min_gap_minutes=min_gap_minutes)
            hour_counter = {h: 0 for h in range(24)}
            for hour_value in event_hours.get("event_hours", []):
                hour_counter[int(hour_value) % 24] += 1
            valid_event_count = int(sum(hour_counter.values()))
            distribution = [
                {
                    "hour": h,
                    "count": int(hour_counter[h]),
                    "ratio": round(hour_counter[h] / valid_event_count, 4) if valid_event_count else 0.0,
                }
                for h in range(24)
            ]
            ranked_hours = sorted(distribution, key=lambda item: item["count"], reverse=True)
            peak_hours = [int(item["hour"]) for item in ranked_hours[:3] if item.get("count", 0) > 0]
            window_summary = self._classify_activity_windows(distribution)
            return {
                "species": self._canonical_species_name(species),
                "min_gap_minutes": min_gap_minutes,
                "source_record_count": valid_event_count,
                "valid_time_records": valid_event_count,
                "independent_event_count": valid_event_count,
                "records_with_clock_time": valid_event_count,
                "clock_time_ratio": 1.0 if valid_event_count else 0.0,
                "synthetic_clock_time_ratio": 0.0,
                "observed_clock_time_ratio": 1.0 if valid_event_count else 0.0,
                "timestamp_source_summary": {"camtrap_dp_tabular": valid_event_count},
                "hourly_distribution": distribution,
                "peak_hours": sorted(peak_hours),
                "peak_period_label": self._peak_period_label(peak_hours),
                **window_summary,
                "note": "Hourly analysis uses CamtrapDP independent events with explicit timestamps.",
            }

        resolution = self._check_time_resolution(species)
        if not resolution["is_hourly_reliable"]:
            return {
                "species": species,
                "valid_time_records": resolution["valid_time_records"],
                "records_with_clock_time": resolution["records_with_clock_time"],
                "clock_time_ratio": resolution["clock_time_ratio"],
                "synthetic_clock_time_ratio": resolution["synthetic_clock_time_ratio"],
                "observed_clock_time_ratio": resolution["observed_clock_time_ratio"],
                "timestamp_source_summary": resolution["timestamp_source_summary"],
                "hourly_distribution": [],
                "peak_hours": [],
                "peak_period_label": "insufficient_data",
                "activity_window_shares": {},
                "major_peak_window": "none",
                "secondary_peak_window": "none",
                "activity_pattern_label": "insufficient_data",
                "note": self._build_timestamp_quality_note(resolution, context="hourly analysis")
            }

        event_result = self.build_independent_events(
            species=species,
            by_camera=True,
            min_gap_minutes=min_gap_minutes,
        )
        hour_counter = {h: 0 for h in range(24)}

        valid_event_count = 0
        for event in event_result["events"]:
            if not event.get("has_observed_clock_time"):
                continue
            dt = self.safe_parse_timestamp(event.get("observed_event_time") or event.get("event_time"))
            if dt is None:
                continue
            hour_counter[dt.hour] += 1
            valid_event_count += 1

        distribution = []
        for h in range(24):
            count = hour_counter[h]
            ratio = count / valid_event_count if valid_event_count > 0 else 0.0
            distribution.append({
                "hour": h,
                "count": count,
                "ratio": round(ratio, 4)
            })
        ranked_hours = sorted(distribution, key=lambda item: item["count"], reverse=True)
        peak_hours = [int(item["hour"]) for item in ranked_hours[:3] if item.get("count", 0) > 0]
        window_summary = self._classify_activity_windows(distribution)

        return {
            "species": species,
            "min_gap_minutes": min_gap_minutes,
            "source_record_count": event_result["source_record_count"],
            "valid_time_records": valid_event_count,
            "independent_event_count": event_result["event_count"],
            "records_with_clock_time": resolution["records_with_clock_time"],
            "clock_time_ratio": resolution["clock_time_ratio"],
            "synthetic_clock_time_ratio": resolution["synthetic_clock_time_ratio"],
            "observed_clock_time_ratio": resolution["observed_clock_time_ratio"],
            "timestamp_source_summary": resolution["timestamp_source_summary"],
            "hourly_distribution": distribution,
            "peak_hours": sorted(peak_hours),
            "peak_period_label": self._peak_period_label(peak_hours),
            **window_summary,
            "note": self._build_timestamp_quality_note(resolution, context="hourly analysis")
        }

    def detect_activity_peaks(self, species="Siberian tiger", top_k=3, min_gap_minutes=30):
        """
        Activity peak periods.
        """
        hourly = self.build_hourly_activity_distribution(
            species,
            min_gap_minutes=min_gap_minutes,
        )["hourly_distribution"]
        ranked = sorted(hourly, key=lambda x: x["count"], reverse=True)
        return ranked[:top_k]

    def compute_hourly_relative_frequency(self, species="Siberian tiger", min_gap_minutes=30):
        """
        Simplified hourly relative-frequency summary.
        This is not a strict ecological RAI metric.
        """
        result = self.build_hourly_activity_distribution(
            species,
            min_gap_minutes=min_gap_minutes,
        )
        result["metric_name"] = "hourly_relative_frequency"
        result["is_formal_rai"] = False
        result["note"] = "This output is hourly relative frequency, not a formal RAI estimate."
        return result

    def compute_rai_by_hour(self, species="Siberian tiger", min_gap_minutes=30):
        """
        Backward-compatible alias.
        New code should use compute_hourly_relative_frequency.
        """
        return self.compute_hourly_relative_frequency(
            species,
            min_gap_minutes=min_gap_minutes,
        )

    def compare_multi_species_hourly_activity(self, species_list=None, top_k=6, min_gap_minutes=30):
        """
        Compare 24-hour activity distributions across multiple species.
        Defaults to the top_k non-Unknown species by record count.
        """
        selected_species = self._resolve_species_list(
            species_list=species_list,
            top_k=top_k,
            include_unknown=False,
        )

        rows = []
        for species in selected_species:
            hourly = self.build_hourly_activity_distribution(
                species=species,
                min_gap_minutes=min_gap_minutes,
            )
            distribution = hourly.get("hourly_distribution", [])
            ranked_hours = sorted(distribution, key=lambda item: item.get("count", 0), reverse=True)
            day_night = self.compare_day_vs_night_activity(
                species=species,
                min_gap_minutes=min_gap_minutes,
            )
            rows.append({
                "species": species,
                "valid_time_records": hourly.get("valid_time_records", 0),
                "independent_event_count": hourly.get("independent_event_count", 0),
                "clock_time_ratio": hourly.get("clock_time_ratio", 0.0),
                "top_hours": ranked_hours[:3],
                "day_ratio": day_night.get("day_ratio"),
                "night_ratio": day_night.get("night_ratio"),
                "hourly_distribution": distribution,
                "note": hourly.get("note"),
            })

        return {
            "species": selected_species,
            "top_k": int(top_k),
            "min_gap_minutes": min_gap_minutes,
            "profiles": rows,
            "note": (
                "Multi-species hourly profiles are based on independent events with real clock time. "
                "They are descriptive comparisons, not formal tests of temporal niche separation."
            ),
        }

    def estimate_pairwise_activity_overlap(
        self,
        species_a="Siberian tiger",
        species_list=None,
        top_k=4,
        min_gap_minutes=30,
        smooth_sigma=1.25,
        bootstrap_iters=500,
        bootstrap_interval_method="percentile",
        bootstrap_cluster_strategy="joint_camera_network",
        permutation_iters=0,
        min_events=10,
        random_seed=42,
    ):
        """
        Estimate focal-vs-selected-species activity overlap, skipping pairs with too few events.
        """
        species_a = self._canonical_species_name(species_a)
        selected = self._resolve_comparison_species(
            focal_species=species_a,
            species_list=species_list,
            top_k=top_k,
        )

        pairwise_results = []
        skipped = []
        for species_b in selected:
            species_b = self._canonical_species_name(species_b)
            if species_b == species_a:
                continue
            overlap = self.estimate_activity_overlap(
                species_a=species_a,
                species_b=species_b,
                min_gap_minutes=min_gap_minutes,
                smooth_sigma=smooth_sigma,
                bootstrap_iters=bootstrap_iters,
                bootstrap_interval_method=bootstrap_interval_method,
                bootstrap_cluster_strategy=bootstrap_cluster_strategy,
                permutation_iters=permutation_iters,
                random_seed=random_seed,
            )
            count_a = int(overlap.get("valid_event_count_a") or 0)
            count_b = int(overlap.get("valid_event_count_b") or 0)
            if count_a < int(min_events) or count_b < int(min_events) or overlap.get("delta_hat") is None:
                skipped.append({
                    "species": species_b,
                    "reason": "skipped_due_to_insufficient_events",
                    "valid_event_count_a": count_a,
                    "valid_event_count_b": count_b,
                    "min_events": int(min_events),
                })
                continue
            pairwise_results.append({
                "species_a": species_a,
                "species_b": species_b,
                "delta_hat": overlap.get("delta_hat"),
                "delta_ci_low": overlap.get("delta_ci_low"),
                "delta_ci_high": overlap.get("delta_ci_high"),
                "relationship": self._activity_overlap_label(overlap.get("delta_hat")),
                "valid_event_count_a": count_a,
                "valid_event_count_b": count_b,
                "peak_hours_a": overlap.get("peak_hours_a", []),
                "peak_hours_b": overlap.get("peak_hours_b", []),
                "estimator_id": overlap.get("estimator_id"),
                "estimator_version": overlap.get("estimator_version"),
                "source_events_sha256": overlap.get("source_events_sha256"),
                "overlap_result_bundle_id": overlap.get(
                    "overlap_result_bundle_id"
                ),
                "overlap_result_bundle_sha256": overlap.get(
                    "overlap_result_bundle_sha256"
                ),
                "overlap_result_sha256": overlap.get("overlap_result_sha256"),
                "overlap_result_reused": overlap.get("overlap_result_reused"),
            })

        result_bundle_ids = [
            item["overlap_result_bundle_id"]
            for item in pairwise_results
            if item.get("overlap_result_bundle_id")
        ]
        return {
            "species_a": species_a,
            "comparison_species": selected,
            "comparison_selection": (
                "explicit_species_list"
                if species_list is not None
                else "top_non_focal_species_from_active_dataset"
            ),
            "comparison_top_k": int(top_k),
            "min_gap_minutes": int(min_gap_minutes),
            "smooth_sigma": float(smooth_sigma),
            "bootstrap_iters": int(bootstrap_iters),
            "bootstrap_interval_method": bootstrap_interval_method,
            "bootstrap_cluster_strategy": bootstrap_cluster_strategy,
            "interval_calibration_status": (
                "requires_data_generating_process_specific_simulation"
                if int(bootstrap_iters) > 1
                else "not_applicable_no_interval_computed"
            ),
            "permutation_iters": int(permutation_iters),
            "min_events": int(min_events),
            "random_seed": int(random_seed),
            "pairwise_results": pairwise_results,
            "overlap_result_bundle_ids": result_bundle_ids,
            "overlap_result_bundle_set_sha256": self._overlap_sha256(
                result_bundle_ids
            ),
            "skipped": skipped,
            "note": (
                "Pairwise Delta values use versioned, tamper-evident result "
                "bundles from the same independent-event activity-overlap "
                "estimator. Plot tools reuse these bundles rather than "
                "re-estimating a separate value."
            ),
        }

    def build_species_activity_guilds(self, species_list=None, top_k=10, min_gap_minutes=30):
        """
        Use independent events with real clock times to roughly group species as diurnal, nocturnal, crepuscular, or mixed.
        This is a descriptive guild classification for ecological interpretation and figure panels.
        """
        selected_species = self._resolve_species_list(
            species_list=species_list,
            top_k=top_k,
            include_unknown=False,
        )

        rows = []
        guild_counts = {}
        for species in selected_species:
            hourly = self.build_hourly_activity_distribution(
                species=species,
                min_gap_minutes=min_gap_minutes,
            )
            distribution = hourly.get("hourly_distribution", [])
            if not distribution:
                continue

            day_ratio = 0.0
            night_ratio = 0.0
            crepuscular_ratio = 0.0
            total_ratio = 0.0
            peak_hours = []
            for item in distribution:
                hour = int(item.get("hour", 0))
                ratio = float(item.get("ratio", 0.0))
                total_ratio += ratio
                if 6 <= hour <= 17:
                    day_ratio += ratio
                else:
                    night_ratio += ratio
                if hour in {5, 6, 7, 17, 18, 19}:
                    crepuscular_ratio += ratio

            ranked = sorted(distribution, key=lambda item: item.get("count", 0), reverse=True)
            peak_hours = [int(item.get("hour", 0)) for item in ranked[:3]]
            circular = self.circular_time_summary(species=species, min_gap_minutes=min_gap_minutes)
            resultant_length = circular.get("resultant_length")

            if hourly.get("independent_event_count", 0) == 0 or total_ratio <= 0:
                guild = "insufficient_time_data"
            elif crepuscular_ratio >= 0.45:
                guild = "crepuscular_or_edge-active"
            elif night_ratio >= 0.62:
                guild = "mostly_nocturnal"
            elif day_ratio >= 0.62:
                guild = "mostly_diurnal"
            else:
                guild = "mixed_or_cathemeral"

            guild_counts[guild] = guild_counts.get(guild, 0) + 1
            rows.append({
                "species": species,
                "activity_guild": guild,
                "independent_event_count": hourly.get("independent_event_count", 0),
                "valid_time_records": hourly.get("valid_time_records", 0),
                "clock_time_ratio": hourly.get("clock_time_ratio", 0.0),
                "day_ratio": round(day_ratio, 4),
                "night_ratio": round(night_ratio, 4),
                "crepuscular_ratio": round(crepuscular_ratio, 4),
                "peak_hours": peak_hours,
                "mean_activity_hour": circular.get("mean_hour"),
                "resultant_length": resultant_length,
            })

        rows.sort(
            key=lambda item: (
                item["activity_guild"],
                -item["independent_event_count"],
                item["species"],
            )
        )
        return {
            "species": selected_species,
            "top_k": int(top_k),
            "min_gap_minutes": min_gap_minutes,
            "guild_counts": guild_counts,
            "activity_guilds": rows,
            "note": (
                "Activity guilds are descriptive classes from real clock-time independent events. "
                "They summarize diel niche patterns but are not a formal behavioral classification."
            ),
        }

    def circular_time_summary(self, species="Siberian tiger", min_gap_minutes=30):
        """
        Simple circular summary: convert hours into angles and compute mean activity direction.
        """
        resolution = self._check_time_resolution(species)
        if not resolution["is_hourly_reliable"]:
            return {
                "species": species,
                "mean_hour": None,
                "resultant_length": None,
                "note": self._build_timestamp_quality_note(resolution, context="circular time summary")
            }
        event_result = self.build_independent_events(
            species=species,
            by_camera=True,
            min_gap_minutes=min_gap_minutes,
        )

        angles = []
        for event in event_result["events"]:
            if not event.get("has_observed_clock_time"):
                continue
            dt = self.safe_parse_timestamp(event.get("observed_event_time") or event.get("event_time"))
            if dt is None:
                continue
            hour_float = dt.hour + dt.minute / 60.0
            angle = 2 * math.pi * hour_float / 24.0
            angles.append(angle)

        if not angles:
            return {
                "species": species,
                "mean_hour": None,
                "resultant_length": None
            }

        x = sum(math.cos(a) for a in angles) / len(angles)
        y = sum(math.sin(a) for a in angles) / len(angles)

        mean_angle = math.atan2(y, x)
        if mean_angle < 0:
            mean_angle += 2 * math.pi

        mean_hour = 24 * mean_angle / (2 * math.pi)
        resultant_length = math.sqrt(x * x + y * y)

        return {
            "species": species,
            "min_gap_minutes": min_gap_minutes,
            "mean_hour": round(mean_hour, 3),
            "resultant_length": round(resultant_length, 4)
        }

    def calculate_time_overlap_between_two_sets(self, records_a, records_b):
        """
        Compute simplified overlap between two 24-hour distributions.
        overlap equals the sum of the per-hour minimum proportions.
        """
        hour_a = [0] * 24
        hour_b = [0] * 24

        valid_a = 0
        valid_b = 0

        for r in records_a:
            dt = r.get("parsed_time")
            if dt is not None:
                hour_a[dt.hour] += 1
                valid_a += 1

        for r in records_b:
            dt = r.get("parsed_time")
            if dt is not None:
                hour_b[dt.hour] += 1
                valid_b += 1

        if valid_a == 0 or valid_b == 0:
            return 0.0

        prop_a = [x / valid_a for x in hour_a]
        prop_b = [x / valid_b for x in hour_b]

        overlap = sum(min(prop_a[h], prop_b[h]) for h in range(24))
        return round(overlap, 4)

    def compare_day_vs_night_activity(self, species="Siberian tiger", min_gap_minutes=30):
        """
        Compare daytime and nighttime activity.
        Daytime: 06-17
        Nighttime: 18-05
        """
        resolution = self._check_time_resolution(species)
        if not resolution["is_hourly_reliable"]:
            return {
                "species": species,
                "min_gap_minutes": min_gap_minutes,
                "day_count": 0,
                "night_count": 0,
                "day_ratio": 0.0,
                "night_ratio": 0.0,
                "note": self._build_timestamp_quality_note(resolution, context="day-vs-night analysis"),
            }

        event_result = self.build_independent_events(
            species=species,
            by_camera=True,
            min_gap_minutes=min_gap_minutes,
        )

        day_count = 0
        night_count = 0

        for event in event_result["events"]:
            if not event.get("has_observed_clock_time"):
                continue
            dt = self.safe_parse_timestamp(event.get("observed_event_time") or event.get("event_time"))
            if dt is None:
                continue
            if 6 <= dt.hour <= 17:
                day_count += 1
            else:
                night_count += 1

        total = day_count + night_count
        return {
            "species": species,
            "min_gap_minutes": min_gap_minutes,
            "day_count": day_count,
            "night_count": night_count,
            "day_ratio": round(day_count / total, 4) if total > 0 else 0.0,
            "night_ratio": round(night_count / total, 4) if total > 0 else 0.0
        }

    def build_monthly_activity_distribution(self, species="Siberian tiger", min_gap_minutes=30):
        """
        Summarize activity distribution by month.
        """
        if self._using_camtrap_backend():
            species = self._canonical_species_name(species)
            df = self._load_camtrap_event_df(min_gap_minutes=min_gap_minutes)
            if not df.empty:
                df = df.copy()
                if "species" not in df.columns:
                    df["species"] = df.get("scientificName", "").map(self._canonical_species_name)
                df["species"] = df["species"].map(self._canonical_species_name)
                dt = pd.to_datetime(df.loc[df["species"] == species, "event_time"], errors="coerce").dropna()
                month_counts = dt.dt.month.value_counts().to_dict()
                valid_count = int(len(dt))
                return {
                    "species": species,
                    "valid_date_records": valid_count,
                    "monthly_distribution": [
                        {"month": m, "count": int(month_counts.get(m, 0)), "ratio": round(int(month_counts.get(m, 0)) / valid_count, 4) if valid_count else 0.0}
                        for m in range(1, 13)
                    ],
                }
            records = []
        else:
            records = self.get_species_records_with_parsed_time(species)
        month_counter = {m: 0 for m in range(1, 13)}
        valid_count = 0

        for r in records:
            dt = r["parsed_time"]
            if dt is not None:
                month_counter[dt.month] += 1
                valid_count += 1

        distribution = []
        for m in range(1, 13):
            count = month_counter[m]
            ratio = count / valid_count if valid_count > 0 else 0.0
            distribution.append({
                "month": m,
                "count": count,
                "ratio": round(ratio, 4)
            })

        return {
            "species": species,
            "valid_date_records": valid_count,
            "monthly_distribution": distribution
        }

    def detect_monthly_activity_peaks(self, species="Siberian tiger", top_k=3, min_gap_minutes=30):
        monthly = self.build_monthly_activity_distribution(species, min_gap_minutes=min_gap_minutes)["monthly_distribution"]
        ranked = sorted(monthly, key=lambda x: x["count"], reverse=True)
        return ranked[:top_k]

    def compare_seasonal_activity(self, species="Siberian tiger", min_gap_minutes=30):
        """
        Compare seasonal activity differences at date level.
        """
        if self._using_camtrap_backend():
            species = self._canonical_species_name(species)
            df = self._load_camtrap_event_df(min_gap_minutes=min_gap_minutes)
            if not df.empty:
                df = df.copy()
                if "species" not in df.columns:
                    df["species"] = df.get("scientificName", "").map(self._canonical_species_name)
                df["species"] = df["species"].map(self._canonical_species_name)
                dt = pd.to_datetime(df.loc[df["species"] == species, "event_time"], errors="coerce").dropna()
                records = [{"parsed_time": ts.to_pydatetime()} for ts in dt]
            else:
                records = []
        else:
            records = self.get_species_records_with_parsed_time(species)

        season_map = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"
        }

        season_total = {
            "spring": 0,
            "summer": 0,
            "autumn": 0,
            "winter": 0
        }

        month_in_season_counter = {
            "spring": {3: 0, 4: 0, 5: 0},
            "summer": {6: 0, 7: 0, 8: 0},
            "autumn": {9: 0, 10: 0, 11: 0},
            "winter": {12: 0, 1: 0, 2: 0}
        }

        valid_count = 0
        for r in records:
            dt = r["parsed_time"]
            if dt is None:
                continue

            season = season_map[dt.month]
            season_total[season] += 1
            month_in_season_counter[season][dt.month] += 1
            valid_count += 1

        season_ratio = {
            season: round(count / valid_count, 4) if valid_count > 0 else 0.0
            for season, count in season_total.items()
        }

        return {
            "species": species,
            "season_total_records": season_total,
            "season_ratio": season_ratio,
            "month_in_season_counter": month_in_season_counter
        }

    def build_weekday_activity_distribution(self, species="Siberian tiger", min_gap_minutes=30):
        """
        Summarize activity distribution by weekday.
        0=Monday, 6=Sunday
        """
        if self._using_camtrap_backend():
            species = self._canonical_species_name(species)
            df = self._load_camtrap_event_df(min_gap_minutes=min_gap_minutes)
            if not df.empty:
                df = df.copy()
                if "species" not in df.columns:
                    df["species"] = df.get("scientificName", "").map(self._canonical_species_name)
                df["species"] = df["species"].map(self._canonical_species_name)
                dt = pd.to_datetime(df.loc[df["species"] == species, "event_time"], errors="coerce").dropna()
                weekday_counts = dt.dt.weekday.value_counts().to_dict()
                valid_count = int(len(dt))
                name_map = {
                    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                    4: "Friday", 5: "Saturday", 6: "Sunday"
                }
                return {
                    "species": species,
                    "weekday_distribution": [
                        {"weekday": d, "name": name_map[d], "count": int(weekday_counts.get(d, 0)), "ratio": round(int(weekday_counts.get(d, 0)) / valid_count, 4) if valid_count else 0.0}
                        for d in range(7)
                    ],
                }
            records = []
        else:
            records = self.get_species_records_with_parsed_time(species)
        weekday_counter = {d: 0 for d in range(7)}
        valid_count = 0

        for r in records:
            dt = r["parsed_time"]
            if dt is not None:
                weekday_counter[dt.weekday()] += 1
                valid_count += 1

        name_map = {
            0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
            4: "Friday", 5: "Saturday", 6: "Sunday"
        }

        distribution = []
        for d in range(7):
            count = weekday_counter[d]
            ratio = count / valid_count if valid_count > 0 else 0.0
            distribution.append({
                "weekday": d,
                "name": name_map[d],
                "count": count,
                "ratio": round(ratio, 4)
            })

        return {
            "species": species,
            "weekday_distribution": distribution
        }

    def analyze_temporal_activity_pattern(self, species="Siberian tiger", min_gap_minutes=30):
        """
        Integrated temporal activity-pattern analysis.
        When timestamps are mostly date-level only, return only month, season, and weekday analyses; do not provide reliable hourly-rhythm conclusions by default.
        """
        resolution = self._check_time_resolution(species)
        event_result = self.build_independent_events(
            species=species,
            by_camera=True,
            min_gap_minutes=min_gap_minutes,
        )

        return {
            "species": species,
            "min_gap_minutes": min_gap_minutes,
            "time_resolution_check": resolution,
            "independent_event_summary": {
                "event_count": event_result["event_count"],
                "valid_record_count": event_result["valid_record_count"],
            },
            "note": (
                self._build_timestamp_quality_note(resolution, context="hourly rhythm / day-vs-night analysis")
            ),
            "monthly_activity_distribution": self.build_monthly_activity_distribution(species, min_gap_minutes=min_gap_minutes),
            "monthly_activity_peaks": self.detect_monthly_activity_peaks(species, min_gap_minutes=min_gap_minutes),
            "seasonal_activity": self.compare_seasonal_activity(species, min_gap_minutes=min_gap_minutes),
            "weekday_activity_distribution": self.build_weekday_activity_distribution(species, min_gap_minutes=min_gap_minutes),
            "timestamp_source_summary": resolution["timestamp_source_summary"],
            "synthetic_time_ratio": resolution["synthetic_time_ratio"],
            "observed_time_ratio": resolution["observed_time_ratio"],
            "hourly_activity_distribution": self.build_hourly_activity_distribution(
                species,
                min_gap_minutes=min_gap_minutes,
            ) if resolution["is_hourly_reliable"] else {
                "species": species,
                "hourly_distribution": [],
                "note": "Hourly analysis is not reliable because most timestamps do not contain explicit clock time.",
            }
        }
