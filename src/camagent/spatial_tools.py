import math
from collections import Counter, defaultdict
from statistics import median

import numpy as np
import pandas as pd

from .base_tools import BaseTools


def _conditional_shared_camera_distribution(
    site_count,
    detected_a,
    detected_b,
):
    """Hypergeometric support/probabilities without an optional SciPy import."""
    site_count = int(site_count)
    detected_a = int(detected_a)
    detected_b = int(detected_b)
    lower = max(0, detected_b - (site_count - detected_a))
    upper = min(detected_a, detected_b)

    def _log_combination(n, k):
        if k < 0 or k > n:
            return float("-inf")
        return (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
        )

    denominator = _log_combination(site_count, detected_b)
    support = np.arange(lower, upper + 1, dtype=int)
    log_probabilities = np.asarray([
        (
            _log_combination(detected_a, int(value))
            + _log_combination(
                site_count - detected_a,
                detected_b - int(value),
            )
            - denominator
        )
        for value in support
    ])
    maximum = float(log_probabilities.max())
    probabilities = np.exp(log_probabilities - maximum)
    probabilities /= probabilities.sum()
    return support, probabilities


class SpatialTools(BaseTools):
    def build_detection_history_matrix(
        self,
        species,
        occasion_days=14,
        min_gap_minutes=30,
        min_active_occasions=2,
    ):
        """
        Build a camera-by-occasion detection-history matrix.
        Whether an occasion was surveyed is approximated by any record from that camera in the time window; target-species detections are based on independent events.
        """
        species = species.strip()
        occasion_days = max(int(occasion_days), 1)
        min_active_occasions = max(int(min_active_occasions), 1)

        if self._using_camtrap_backend():
            deployments = self._load_camtrap_deployments_df()
            if deployments.empty:
                return {
                    "species": species,
                    "occasion_days": occasion_days,
                    "min_gap_minutes": min_gap_minutes,
                    "min_active_occasions": min_active_occasions,
                    "survey_start": None,
                    "survey_end": None,
                    "total_occasions": 0,
                    "surveyed_camera_count": 0,
                    "excluded_low_effort_camera_count": 0,
                    "camera_detection_histories": [],
                    "note": "Detection history could not be built because no valid CamtrapDP deployment timestamps were available.",
                }
            dep = deployments[["cameraID", "deploymentStart", "deploymentEnd"]].copy()
            inferred_effort = bool(
                "deploymentComments" in deployments.columns
                and deployments["deploymentComments"].astype(str).str.contains("approximated", case=False, na=False).any()
            )
            effort_basis = "detection_span_proxy" if inferred_effort else "documented_deployment_intervals"
            dep["start_dt"] = pd.to_datetime(dep["deploymentStart"], errors="coerce", utc=True)
            dep["end_dt"] = pd.to_datetime(dep["deploymentEnd"], errors="coerce", utc=True)
            dep = dep.dropna(subset=["cameraID", "start_dt", "end_dt"])
            if dep.empty:
                return {
                    "species": species,
                    "occasion_days": occasion_days,
                    "min_gap_minutes": min_gap_minutes,
                    "min_active_occasions": min_active_occasions,
                    "survey_start": None,
                    "survey_end": None,
                    "total_occasions": 0,
                    "surveyed_camera_count": 0,
                    "excluded_low_effort_camera_count": 0,
                    "camera_detection_histories": [],
                    "note": "Detection history could not be built because deployment dates were unavailable.",
                }
            survey_start = dep["start_dt"].min()
            survey_end = dep["end_dt"].max()
            total_occasions = int(((survey_end - survey_start).days // occasion_days) + 1)
            active_occasions_by_camera = {}
            for row in dep.itertuples(index=False):
                data = row._asdict()
                start_idx = max(int(((data["start_dt"] - survey_start).days // occasion_days)), 0)
                end_idx = min(int(((data["end_dt"] - survey_start).days // occasion_days)), total_occasions - 1)
                active_occasions_by_camera[str(data["cameraID"])] = set(range(start_idx, end_idx + 1))

            detection_occasions_by_camera = defaultdict(set)
            event_result = self.build_independent_events(
                species=species,
                by_camera=True,
                min_gap_minutes=min_gap_minutes,
            )
            for event in event_result.get("events", []):
                camera_id = str(event.get("camera_id", "")).strip()
                event_time = self.safe_parse_timestamp(event.get("event_time"))
                if camera_id and event_time is not None:
                    event_ts = pd.Timestamp(event_time)
                    if event_ts.tzinfo is None:
                        event_ts = event_ts.tz_localize("UTC")
                    else:
                        event_ts = event_ts.tz_convert("UTC")
                    occasion_index = (event_ts - survey_start).days // occasion_days
                    detection_occasions_by_camera[camera_id].add(int(occasion_index))

            camera_locations = self.get_camera_locations()
            camera_ids = sorted(set(camera_locations) | set(active_occasions_by_camera) | set(detection_occasions_by_camera))
            surveyed_rows = []
            excluded_low_effort = 0
            for camera_id in camera_ids:
                active_occasions = sorted(active_occasions_by_camera.get(camera_id, set()))
                if len(active_occasions) < min_active_occasions:
                    if active_occasions:
                        excluded_low_effort += 1
                    continue
                detected_occasions = detection_occasions_by_camera.get(camera_id, set())
                history = [1 if occ in detected_occasions else 0 for occ in active_occasions]
                location = camera_locations.get(camera_id, {})
                surveyed_rows.append({
                    "camera_id": camera_id,
                    "occasion_indices": active_occasions,
                    "detection_history": history,
                    "occasion_count": len(active_occasions),
                    "detection_count": int(sum(history)),
                    "detected": any(history),
                    "latitude": location.get("latitude"),
                    "longitude": location.get("longitude"),
                })

            return {
                "species": species,
                "occasion_days": occasion_days,
                "min_gap_minutes": min_gap_minutes,
                "min_active_occasions": min_active_occasions,
                "survey_start": survey_start.strftime("%Y-%m-%d %H:%M:%S"),
                "survey_end": survey_end.strftime("%Y-%m-%d %H:%M:%S"),
                "total_occasions": total_occasions,
                "surveyed_camera_count": len(surveyed_rows),
                "excluded_low_effort_camera_count": excluded_low_effort,
                "effort_basis": effort_basis,
                "effort_supports_occupancy_inference": not inferred_effort,
                "camera_detection_histories": surveyed_rows,
                "note": (
                    "Detection history was built from CamtrapDP deployment intervals. "
                    + ("Those intervals were inferred from first/last detections, so non-detections are not validated survey occasions." if inferred_effort else "Deployment intervals are documented independently of detections.")
                ),
            }

        valid_all_records = []
        for record in self.get_all_records():
            metadata = record.get("metadata", {}) or {}
            parsed_time = self.safe_parse_timestamp(metadata.get("timestamp"))
            camera_id = str(metadata.get("camera_id", "")).strip()
            if parsed_time is None or not camera_id:
                continue
            valid_all_records.append({
                "camera_id": camera_id,
                "parsed_time": parsed_time,
            })

        if not valid_all_records:
            return {
                "species": species,
                "occasion_days": occasion_days,
                "min_gap_minutes": min_gap_minutes,
                "min_active_occasions": min_active_occasions,
                "survey_start": None,
                "survey_end": None,
                "total_occasions": 0,
                "surveyed_camera_count": 0,
                "excluded_low_effort_camera_count": 0,
                "camera_detection_histories": [],
                "note": "Detection history could not be built because no valid camera timestamps were available.",
            }

        survey_start = min(item["parsed_time"] for item in valid_all_records)
        survey_end = max(item["parsed_time"] for item in valid_all_records)
        total_occasions = ((survey_end - survey_start).days // occasion_days) + 1

        active_occasions_by_camera = defaultdict(set)
        for item in valid_all_records:
            occasion_index = (item["parsed_time"] - survey_start).days // occasion_days
            active_occasions_by_camera[item["camera_id"]].add(int(occasion_index))

        detection_occasions_by_camera = defaultdict(set)
        event_result = self.build_independent_events(
            species=species,
            by_camera=True,
            min_gap_minutes=min_gap_minutes,
        )
        for event in event_result.get("events", []):
            camera_id = str(event.get("camera_id", "")).strip()
            event_time = self.safe_parse_timestamp(event.get("event_time"))
            if not camera_id or event_time is None:
                continue
            occasion_index = (event_time - survey_start).days // occasion_days
            detection_occasions_by_camera[camera_id].add(int(occasion_index))

        camera_locations = self.get_camera_locations()
        camera_ids = sorted(
            set(camera_locations.keys())
            | set(active_occasions_by_camera.keys())
            | set(detection_occasions_by_camera.keys())
        )

        surveyed_rows = []
        excluded_low_effort = 0
        for camera_id in camera_ids:
            active_occasions = sorted(active_occasions_by_camera.get(camera_id, set()))
            if len(active_occasions) < min_active_occasions:
                if active_occasions:
                    excluded_low_effort += 1
                continue

            detected_occasions = detection_occasions_by_camera.get(camera_id, set())
            history = [1 if occ in detected_occasions else 0 for occ in active_occasions]
            location = camera_locations.get(camera_id, {})

            surveyed_rows.append({
                "camera_id": camera_id,
                "occasion_indices": active_occasions,
                "detection_history": history,
                "occasion_count": len(active_occasions),
                "detection_count": int(sum(history)),
                "detected": any(history),
                "latitude": location.get("latitude"),
                "longitude": location.get("longitude"),
            })

        return {
            "species": species,
            "occasion_days": occasion_days,
            "min_gap_minutes": min_gap_minutes,
            "min_active_occasions": min_active_occasions,
            "survey_start": survey_start.strftime("%Y-%m-%d %H:%M:%S"),
            "survey_end": survey_end.strftime("%Y-%m-%d %H:%M:%S"),
            "total_occasions": total_occasions,
            "surveyed_camera_count": len(surveyed_rows),
            "excluded_low_effort_camera_count": excluded_low_effort,
            "effort_basis": "records_present_proxy",
            "effort_supports_occupancy_inference": False,
            "camera_detection_histories": surveyed_rows,
            "note": (
                "Survey occasions are approximated from camera-level activity windows with any record present. "
                "Target-species detections are aggregated as independent events inside those occasions."
            ),
        }

    def _fit_constant_occupancy_em(self, detection_histories, max_iter=200, tol=1e-7):
        if not detection_histories:
            return None

        visit_counts = [len(history) for history in detection_histories]
        detection_counts = [sum(history) for history in detection_histories]
        site_count = len(detection_histories)
        total_detections = sum(detection_counts)
        total_visits = sum(visit_counts)

        if site_count == 0 or total_visits == 0:
            return None

        naive_occupancy = sum(1 for count in detection_counts if count > 0) / site_count
        psi_hat = min(max(naive_occupancy + 0.05, 1e-4), 0.999)
        p_hat = min(max(total_detections / total_visits, 1e-4), 0.999)

        converged = False
        iterations = 0
        for iteration in range(1, max_iter + 1):
            posterior_occupancy = []
            for visits, detections in zip(visit_counts, detection_counts):
                if detections > 0:
                    posterior_occupancy.append(1.0)
                    continue

                absent_component = max(1.0 - psi_hat, 1e-12)
                occupied_zero_component = psi_hat * ((1.0 - p_hat) ** visits)
                denom = absent_component + occupied_zero_component
                posterior_occupancy.append(occupied_zero_component / denom if denom > 0 else 0.0)

            psi_new = min(max(sum(posterior_occupancy) / site_count, 1e-4), 0.999)
            occupied_visits = sum(
                posterior * visits
                for posterior, visits in zip(posterior_occupancy, visit_counts)
            )
            if occupied_visits <= 0:
                p_new = p_hat
            else:
                p_new = min(max(total_detections / occupied_visits, 1e-4), 0.999)

            if abs(psi_new - psi_hat) < tol and abs(p_new - p_hat) < tol:
                converged = True
                psi_hat = psi_new
                p_hat = p_new
                iterations = iteration
                break

            psi_hat = psi_new
            p_hat = p_new
            iterations = iteration

        log_likelihood = 0.0
        for visits, detections in zip(visit_counts, detection_counts):
            if detections > 0:
                likelihood = psi_hat * (p_hat ** detections) * ((1.0 - p_hat) ** (visits - detections))
            else:
                likelihood = (1.0 - psi_hat) + psi_hat * ((1.0 - p_hat) ** visits)
            log_likelihood += math.log(max(likelihood, 1e-12))

        return {
            "psi_hat": psi_hat,
            "p_hat": p_hat,
            "iterations": iterations,
            "converged": converged,
            "log_likelihood": log_likelihood,
            "aic": 4.0 - 2.0 * log_likelihood,
        }

    def fit_single_season_occupancy_model(
        self,
        species,
        occasion_days=14,
        min_gap_minutes=30,
        min_active_occasions=2,
        bootstrap_iters=200,
        random_seed=42,
        allow_inferred_effort=False,
    ):
        """
        Single-season occupancy model, intercept-only:
        - psi: occupancy probability
        - p: detection probability
        Estimate with EM and provide intervals from camera-level bootstrap.
        """
        if getattr(self, "_prefer_camtrap_backend", False) and self._camera_effort_basis() == "detection_span_proxy" and not allow_inferred_effort:
            deployments = self._load_camtrap_deployments_df()
            return {
                "species": species,
                "model_name": "single_season_occupancy_intercept_only",
                "model_status": "not_fitted_without_independent_effort",
                "effort_basis": "detection_span_proxy",
                "surveyed_camera_count": int(deployments["cameraID"].nunique()) if "cameraID" in deployments else 0,
                "naive_occupancy_ratio": None,
                "occupancy_probability_psi": None,
                "detection_probability_p": None,
                "psi_ci_low": None,
                "psi_ci_high": None,
                "p_ci_low": None,
                "p_ci_high": None,
                "note": (
                    "The model was not fitted because camera operation intervals were inferred from detections. "
                    "Supply independently documented deployment/operational effort, or set allow_inferred_effort=True "
                    "only for an explicitly exploratory sensitivity analysis."
                ),
            }
        history_result = self.build_detection_history_matrix(
            species=species,
            occasion_days=occasion_days,
            min_gap_minutes=min_gap_minutes,
            min_active_occasions=min_active_occasions,
        )
        surveyed_rows = history_result.get("camera_detection_histories", [])
        detection_histories = [row["detection_history"] for row in surveyed_rows]

        if not history_result.get("effort_supports_occupancy_inference", False) and not allow_inferred_effort:
            return {
                "species": species,
                "model_name": "single_season_occupancy_intercept_only",
                "model_status": "not_fitted_without_independent_effort",
                "effort_basis": history_result.get("effort_basis", "unknown"),
                "surveyed_camera_count": len(surveyed_rows),
                "naive_occupancy_ratio": None,
                "occupancy_probability_psi": None,
                "detection_probability_p": None,
                "psi_ci_low": None,
                "psi_ci_high": None,
                "p_ci_low": None,
                "p_ci_high": None,
                "note": (
                    "The model was not fitted because camera operation intervals were inferred from detections. "
                    "Supply independently documented deployment/operational effort, or set allow_inferred_effort=True "
                    "only for an explicitly exploratory sensitivity analysis."
                ),
            }

        site_count = len(detection_histories)
        detected_site_count = sum(1 for row in surveyed_rows if row.get("detected"))
        naive_occupancy_ratio = detected_site_count / site_count if site_count > 0 else 0.0

        if site_count == 0:
            return {
                "species": species,
                "model_name": "single_season_occupancy_intercept_only",
                "surveyed_camera_count": 0,
                "naive_occupancy_ratio": 0.0,
                "occupancy_probability_psi": None,
                "detection_probability_p": None,
                "psi_ci_low": None,
                "psi_ci_high": None,
                "p_ci_low": None,
                "p_ci_high": None,
                "note": "Occupancy modeling is unavailable because no surveyed cameras met the minimum active-occasion threshold.",
            }

        if detected_site_count == 0:
            return {
                "species": species,
                "model_name": "single_season_occupancy_intercept_only",
                "surveyed_camera_count": site_count,
                "naive_occupancy_ratio": round(naive_occupancy_ratio, 4),
                "occupancy_probability_psi": 0.0,
                "detection_probability_p": None,
                "psi_ci_low": 0.0,
                "psi_ci_high": 0.0,
                "p_ci_low": None,
                "p_ci_high": None,
                "survey_start": history_result.get("survey_start"),
                "survey_end": history_result.get("survey_end"),
                "occasion_days": occasion_days,
                "min_active_occasions": min_active_occasions,
                "excluded_low_effort_camera_count": history_result.get("excluded_low_effort_camera_count", 0),
                "camera_detection_history_preview": surveyed_rows[:10],
                "note": (
                    "All surveyed cameras were non-detections for the target species, so a detection-corrected occupancy "
                    "estimate is not identifiable beyond zero observed occupancy in this dataset."
                ),
            }

        fit = self._fit_constant_occupancy_em(detection_histories)
        psi_hat = fit["psi_hat"]
        p_hat = fit["p_hat"]

        psi_ci_low = psi_hat
        psi_ci_high = psi_hat
        p_ci_low = p_hat
        p_ci_high = p_hat

        bootstrap_iters = max(int(bootstrap_iters), 0)
        if bootstrap_iters > 1 and site_count > 1:
            import numpy as np

            rng = np.random.default_rng(int(random_seed))
            psi_samples = []
            p_samples = []
            for _ in range(bootstrap_iters):
                sample_idx = rng.choice(site_count, size=site_count, replace=True)
                sample_histories = [detection_histories[int(idx)] for idx in sample_idx]
                sample_fit = self._fit_constant_occupancy_em(sample_histories)
                if sample_fit is None:
                    continue
                psi_samples.append(sample_fit["psi_hat"])
                p_samples.append(sample_fit["p_hat"])

            if psi_samples and p_samples:
                psi_ci_low, psi_ci_high = np.percentile(psi_samples, [2.5, 97.5]).tolist()
                p_ci_low, p_ci_high = np.percentile(p_samples, [2.5, 97.5]).tolist()

        mean_occasions = (
            sum(row["occasion_count"] for row in surveyed_rows) / len(surveyed_rows)
            if surveyed_rows else 0.0
        )

        return {
            "species": species,
            "model_name": "single_season_occupancy_intercept_only",
            "model_status": "fitted" if history_result.get("effort_supports_occupancy_inference", False) else "exploratory_inferred_effort",
            "effort_basis": history_result.get("effort_basis", "unknown"),
            "random_seed": int(random_seed),
            "survey_start": history_result.get("survey_start"),
            "survey_end": history_result.get("survey_end"),
            "occasion_days": occasion_days,
            "min_gap_minutes": min_gap_minutes,
            "min_active_occasions": min_active_occasions,
            "surveyed_camera_count": site_count,
            "detected_camera_count": detected_site_count,
            "excluded_low_effort_camera_count": history_result.get("excluded_low_effort_camera_count", 0),
            "naive_occupancy_ratio": round(naive_occupancy_ratio, 4),
            "occupancy_probability_psi": round(float(psi_hat), 4),
            "detection_probability_p": round(float(p_hat), 4),
            "psi_ci_low": round(float(psi_ci_low), 4),
            "psi_ci_high": round(float(psi_ci_high), 4),
            "p_ci_low": round(float(p_ci_low), 4),
            "p_ci_high": round(float(p_ci_high), 4),
            "expected_occupied_cameras": round(float(psi_hat * site_count), 2),
            "mean_active_occasions_per_camera": round(float(mean_occasions), 2),
            "log_likelihood": round(float(fit["log_likelihood"]), 4),
            "aic": round(float(fit["aic"]), 4),
            "converged": bool(fit["converged"]),
            "iterations": int(fit["iterations"]),
            "camera_detection_history_preview": surveyed_rows[:10],
            "note": (
                "This is a detection-corrected single-season occupancy model with constant psi and p. "
                "Survey effort is approximated from camera occasions that contain any record, so the result is more "
                "ecologically defensible than naive occupancy but still depends on the available monitoring metadata."
            ),
        }

    def build_detection_map(self, species):
        """
        Detection distribution: count independent events at each camera.
        """
        if self._using_camtrap_backend():
            species = self._canonical_species_name(species)
            df = self._load_camtrap_event_df()
            if df.empty:
                return []
            df = df.copy()
            if "species" not in df.columns:
                df["species"] = df.get("scientificName", "").map(self._canonical_species_name)
            df["species"] = df["species"].map(self._canonical_species_name)
            camera_col = "camera_id" if "camera_id" in df.columns else "cameraID"
            counter = df[df["species"] == species].groupby(camera_col).size().to_dict()
            camera_locations = self.get_camera_locations()
            result = []
            for cam_id, count in counter.items():
                loc = camera_locations.get(cam_id, {})
                result.append({
                    "camera_id": cam_id,
                    "count": count,
                    "latitude": loc.get("latitude"),
                    "longitude": loc.get("longitude"),
                    "metric": "independent_events",
                })
            result.sort(key=lambda x: x["count"], reverse=True)
            return result

        records = self.query_by_species(species)
        counter = defaultdict(int)
        camera_locations = self.get_camera_locations()

        for r in records:
            cam = str(r["metadata"].get("camera_id", "")).strip()
            if cam:
                counter[cam] += 1

        result = []
        for cam_id, count in counter.items():
            loc = camera_locations.get(cam_id, {})
            result.append({
                "camera_id": cam_id,
                "count": count,
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude")
            })

        result.sort(key=lambda x: x["count"], reverse=True)
        return result

    def build_hotspot_rank(self, species, top_k=5):
        """
        Observation hotspot ranking: top_k cameras with the highest counts.
        """
        detection_map = self.build_detection_map(species)
        return detection_map[:top_k]

    def build_occupancy_summary(self, species):
        """
        naive occupancy summary：
        Count how many camera sites detected the species.
        This is not detection-corrected and is not a strict occupancy model result.
        """
        all_cameras = set(self.get_camera_locations().keys())
        detection_map = self.build_detection_map(species)
        occupied_cameras = set(item["camera_id"] for item in detection_map)

        total_camera_count = len(all_cameras)
        occupied_count = len(occupied_cameras)
        naive_occupancy_ratio = occupied_count / total_camera_count if total_camera_count > 0 else 0.0

        return {
            "species": species,
            "total_cameras": total_camera_count,
            "occupied_cameras": occupied_count,
            "naive_occupancy_ratio": round(naive_occupancy_ratio, 4),
            "occupied_camera_ids": sorted(list(occupied_cameras)),
            "note": "This is a naive camera-level occupancy proportion, not a formal occupancy estimate with detection correction."
        }

    def build_local_hotspot_rank(self, species, center_camera_id, radius_km=20, top_k=5):
        """
        Rank species hotspot cameras within a radius around a center camera.
        """
        detection_map = self.build_detection_map(species)
        nearby = self.get_cameras_in_radius(center_camera_id, radius_km)
        nearby_ids = set([str(center_camera_id)] + [x["camera_id"] for x in nearby])

        local_map = [item for item in detection_map if item["camera_id"] in nearby_ids]
        local_map.sort(key=lambda x: x["count"], reverse=True)

        return local_map[:top_k]

    def summarize_species_by_camera(self, camera_id, top_k=10):
        """
        List species detected by a camera.
        """
        records = self.query_by_camera(camera_id)
        species_counter = defaultdict(int)

        for r in records:
            species = self._canonical_species_name(r["metadata"].get("species"))
            if species:
                species_counter[species] += 1

        result = []
        for species, count in species_counter.items():
            result.append({
                "species": species,
                "count": count
            })

        result.sort(key=lambda x: x["count"], reverse=True)
        return result[:top_k]

    def _camera_effort_days_from_records(self):
        if self._using_camtrap_backend():
            deployments = self._load_camtrap_deployments_df()
            if not deployments.empty and {"deploymentStart", "deploymentEnd"}.issubset(deployments.columns):
                effort_days = {}
                for row in deployments.itertuples(index=False):
                    data = row._asdict()
                    camera_id = str(data.get("cameraID") or str(data.get("deploymentID", "")).replace("dep_", "", 1)).strip()
                    start_dt = self.safe_parse_timestamp(data.get("deploymentStart"))
                    end_dt = self.safe_parse_timestamp(data.get("deploymentEnd"))
                    if camera_id and start_dt is not None and end_dt is not None:
                        effort_days[camera_id] = max((end_dt.date() - start_dt.date()).days + 1, 1)
                return effort_days

        camera_timestamps = defaultdict(list)
        for record in self.get_all_records():
            metadata = record.get("metadata", {}) or {}
            camera_id = str(metadata.get("camera_id", "")).strip()
            parsed_time = self.safe_parse_timestamp(metadata.get("timestamp"))
            if camera_id and parsed_time is not None:
                camera_timestamps[camera_id].append(parsed_time)

        effort_days = {}
        for camera_id, timestamps in camera_timestamps.items():
            start_dt = min(timestamps)
            end_dt = max(timestamps)
            effort_days[camera_id] = max((end_dt.date() - start_dt.date()).days + 1, 1)
        return effort_days

    def _independent_event_counts_by_species_camera(self, species_list=None, min_gap_minutes=30, include_unknown=False):
        if self._using_camtrap_backend():
            df = self._load_camtrap_event_df()
            if df.empty:
                return [], {}, {}
            df = df.copy()
            if "species" not in df.columns:
                df["species"] = df.get("scientificName", "").map(self._canonical_species_name)
            df["species"] = df["species"].map(self._canonical_species_name)
            camera_col = "camera_id" if "camera_id" in df.columns else "cameraID"
            if species_list is None:
                selected_species = [
                    species for species in df["species"].value_counts().index.tolist()
                    if include_unknown or str(species).lower() != "unknown"
                ]
            else:
                selected_species = [
                    self._canonical_species_name(species)
                    for species in species_list
                    if include_unknown or self._canonical_species_name(species).lower() != "unknown"
                ]
            df = df[df["species"].isin(selected_species)]
            grouped_counts = df.groupby(["species", camera_col]).size()
            counts = {species: defaultdict(int) for species in selected_species}
            for (species, camera_id), count in grouped_counts.items():
                counts[species][str(camera_id)] = int(count)
            event_times = {species: defaultdict(list) for species in selected_species}
            if "event_time" in df.columns:
                df["_event_dt"] = pd.to_datetime(df["event_time"], errors="coerce")
            elif "eventStart" in df.columns:
                df["_event_dt"] = pd.to_datetime(df["eventStart"], errors="coerce")
            else:
                df["_event_dt"] = pd.NaT
            for (species, camera_id), group in df.dropna(subset=["_event_dt"]).groupby(["species", camera_col]):
                event_times[species][str(camera_id)] = [ts.to_pydatetime() for ts in group["_event_dt"]]
            return selected_species, counts, event_times

        if species_list is None:
            species_counter = Counter()
            for record in self.get_all_records():
                metadata = record.get("metadata", {}) or {}
                species = self._canonical_species_name(metadata.get("species"))
                if not include_unknown and species.lower() == "unknown":
                    continue
                species_counter[species] += 1
            selected_species = [species for species, _ in species_counter.most_common()]
        else:
            selected_species = [
                self._canonical_species_name(species)
                for species in species_list
                if include_unknown or self._canonical_species_name(species).lower() != "unknown"
            ]

        min_gap_minutes = max(int(min_gap_minutes), 0)
        counts = {species: defaultdict(int) for species in selected_species}
        event_times = {species: defaultdict(list) for species in selected_species}
        for event in self._collect_exact_independent_event_rows(
            species_list=selected_species,
            include_unknown=include_unknown,
            min_gap_minutes=min_gap_minutes,
        ):
            species = event["species"]
            camera_id = str(event.get("camera_id", "")).strip()
            event_time = event.get("event_time")
            if not camera_id:
                continue
            counts[species][camera_id] += 1
            if event_time is not None:
                event_times[species][camera_id].append(event_time)
        return selected_species, counts, event_times

    def build_camera_species_richness(
        self,
        top_k=20,
        min_records_per_species=2,
        min_gap_minutes=30,
        min_rai_per_100_camera_days=1.0,
    ):
        """
        Summarize species richness and independent-event volume for each camera.
        Consecutive videos from the same species and camera within min_gap_minutes are merged into one independent event.
        Species-level camera RAI equals independent events divided by camera-days times 100.
        Defaults require at least 2 independent events and RAI >= 1.0 to reduce short accidental records.
        """
        min_records_per_species = max(int(min_records_per_species), 1)
        min_rai_per_100_camera_days = max(float(min_rai_per_100_camera_days), 0.0)
        camera_locations = self.get_camera_locations()
        effort_days = self._camera_effort_days_from_records()
        _, species_camera_counts, _ = self._independent_event_counts_by_species_camera(
            min_gap_minutes=min_gap_minutes,
            include_unknown=False,
        )

        camera_species_counts = defaultdict(dict)
        for species, camera_counts in species_camera_counts.items():
            for camera_id, count in camera_counts.items():
                camera_species_counts[camera_id][species] = int(count)

        rows = []
        for camera_id, species_counts in camera_species_counts.items():
            camera_effort = max(float(effort_days.get(camera_id, 1)), 1.0)
            species_rai = {
                species: count / camera_effort * 100.0
                for species, count in species_counts.items()
            }
            retained_counts = {
                species: count
                for species, count in species_counts.items()
                if count >= min_records_per_species and species_rai.get(species, 0.0) >= min_rai_per_100_camera_days
            }
            if not retained_counts:
                continue
            loc = camera_locations.get(camera_id, {})
            rows.append({
                "camera_id": camera_id,
                "species_richness": len(retained_counts),
                "total_records": int(sum(retained_counts.values())),
                "total_independent_events": int(sum(retained_counts.values())),
                "camera_effort_days": round(camera_effort, 3),
                "total_rai_per_100_camera_days": round(sum(species_rai[species] for species in retained_counts), 4),
                "species_counts": dict(sorted(retained_counts.items(), key=lambda item: item[1], reverse=True)),
                "species_rai_per_100_camera_days": {
                    species: round(species_rai[species], 4)
                    for species in sorted(retained_counts, key=lambda name: species_rai[name], reverse=True)
                },
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
            })

        rows.sort(
            key=lambda item: (
                item["species_richness"],
                item["total_rai_per_100_camera_days"],
                item["total_independent_events"],
            ),
            reverse=True,
        )
        return {
            "camera_count": len(rows),
            "top_k": int(top_k),
            "min_records_per_species": min_records_per_species,
            "min_gap_minutes": max(int(min_gap_minutes), 0),
            "min_rai_per_100_camera_days": min_rai_per_100_camera_days,
            "richness_by_camera": rows[:max(int(top_k), 1)],
            "note": (
                "Camera species richness is based on independent events, not raw videos. "
                "RAI fields standardize those events by camera working days; richness remains a detected-species count, "
                "not a full occupancy-model biodiversity estimate."
            ),
        }

    def build_species_cooccurrence_matrix(self, species_list=None, top_k=10, min_camera_records=1, min_gap_minutes=30):
        """
        Build a species-by-species camera-level co-occurrence matrix.
        Cells store shared-camera counts and Jaccard overlap.
        Species presence at a camera is based on independent-event counts, not raw video counts.
        """
        selected_species = self._resolve_species_list(
            species_list=species_list,
            top_k=top_k,
            include_unknown=False,
        )
        min_camera_records = max(int(min_camera_records), 1)
        _, species_camera_counts, _ = self._independent_event_counts_by_species_camera(
            species_list=selected_species,
            min_gap_minutes=min_gap_minutes,
            include_unknown=False,
        )

        species_camera_sets = {
            species: {
                camera_id
                for camera_id, count in camera_counts.items()
                if count >= min_camera_records
            }
            for species, camera_counts in species_camera_counts.items()
        }

        matrix = []
        pairwise = []
        for species_a in selected_species:
            row = []
            cameras_a = species_camera_sets.get(species_a, set())
            for species_b in selected_species:
                cameras_b = species_camera_sets.get(species_b, set())
                shared = len(cameras_a & cameras_b)
                union = len(cameras_a | cameras_b)
                jaccard = shared / union if union else 0.0
                row.append({
                    "species": species_b,
                    "shared_camera_count": shared,
                    "jaccard": round(jaccard, 4),
                })
                if species_a < species_b:
                    pairwise.append({
                        "species_a": species_a,
                        "species_b": species_b,
                        "shared_camera_count": shared,
                        "jaccard": round(jaccard, 4),
                        "camera_count_a": len(cameras_a),
                        "camera_count_b": len(cameras_b),
                    })
            matrix.append({
                "species": species_a,
                "camera_count": len(cameras_a),
                "values": row,
            })

        pairwise.sort(key=lambda item: (item["jaccard"], item["shared_camera_count"]), reverse=True)
        return {
            "species": selected_species,
            "top_k": int(top_k),
            "min_camera_records": min_camera_records,
            "min_gap_minutes": max(int(min_gap_minutes), 0),
            "matrix": matrix,
            "top_cooccurring_pairs": pairwise[:20],
            "note": (
                "Co-occurrence is camera-level overlap based on independent events: two species are associated when both appeared at the same camera. "
                "This does not prove direct interaction or simultaneous use."
            ),
        }

    def _camera_set_for_species(self, species, min_camera_records=1, min_gap_minutes=30):
        min_camera_records = max(int(min_camera_records), 1)
        _, species_camera_counts, _ = self._independent_event_counts_by_species_camera(
            species_list=[species],
            min_gap_minutes=min_gap_minutes,
            include_unknown=False,
        )
        camera_counts = species_camera_counts.get(self._canonical_species_name(species), defaultdict(int))
        return {
            camera_id
            for camera_id, count in camera_counts.items()
            if count >= min_camera_records
        }, dict(camera_counts)

    def _classify_overlap(self, value, high=0.65, moderate=0.35):
        if value is None:
            return "unavailable"
        if value >= high:
            return "high"
        if value >= moderate:
            return "moderate"
        return "low"

    def _same_camera_temporal_proximity(
        self,
        species_a,
        species_b,
        temporal_window_hours=24,
        min_gap_minutes=30,
        sample_limit=20,
    ):
        window_seconds = max(float(temporal_window_hours), 0.0) * 3600.0
        if window_seconds <= 0:
            return {
                "temporal_window_hours": temporal_window_hours,
                "proximate_event_pair_count": 0,
                "camera_count_with_proximity": 0,
                "median_nearest_gap_hours": None,
                "nearest_event_samples": [],
            }

        def _events_by_camera(species):
            event_result = self.build_independent_events(
                species=species,
                by_camera=True,
                min_gap_minutes=min_gap_minutes,
            )
            grouped = defaultdict(list)
            for event in event_result.get("events", []):
                camera_id = str(event.get("camera_id", "")).strip()
                event_time = self.safe_parse_timestamp(event.get("event_time"))
                if camera_id and event_time is not None:
                    grouped[camera_id].append(event_time)
            for values in grouped.values():
                values.sort()
            return grouped, event_result.get("event_count", 0)

        events_a, event_count_a = _events_by_camera(species_a)
        events_b, event_count_b = _events_by_camera(species_b)
        shared_cameras = sorted(set(events_a) & set(events_b))
        proximate_count = 0
        cameras_with_proximity = set()
        nearest_gaps = []
        samples = []

        for camera_id in shared_cameras:
            values_a = events_a[camera_id]
            values_b = events_b[camera_id]
            pointer = 0
            for time_a in values_a:
                while pointer < len(values_b) and (time_a - values_b[pointer]).total_seconds() > window_seconds:
                    pointer += 1

                local_pointer = pointer
                nearest_gap = None
                nearest_time = None
                while local_pointer < len(values_b):
                    delta_seconds = (values_b[local_pointer] - time_a).total_seconds()
                    if delta_seconds > window_seconds:
                        break
                    gap_hours = abs(delta_seconds) / 3600.0
                    if gap_hours <= temporal_window_hours:
                        proximate_count += 1
                        cameras_with_proximity.add(camera_id)
                        if nearest_gap is None or gap_hours < nearest_gap:
                            nearest_gap = gap_hours
                            nearest_time = values_b[local_pointer]
                    local_pointer += 1

                if nearest_gap is not None:
                    nearest_gaps.append(nearest_gap)
                    if len(samples) < sample_limit:
                        samples.append({
                            "camera_id": camera_id,
                            "species_a_time": time_a.strftime("%Y-%m-%d %H:%M:%S"),
                            "species_b_time": nearest_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "gap_hours": round(nearest_gap, 3),
                        })

        return {
            "temporal_window_hours": temporal_window_hours,
            "event_count_a": event_count_a,
            "event_count_b": event_count_b,
            "shared_camera_count_with_events": len(shared_cameras),
            "proximate_event_pair_count": proximate_count,
            "camera_count_with_proximity": len(cameras_with_proximity),
            "median_nearest_gap_hours": round(float(median(nearest_gaps)), 3) if nearest_gaps else None,
            "nearest_event_samples": samples,
        }

    def analyze_species_pair_relationship(
        self,
        species_a,
        species_b,
        min_camera_records=1,
        min_gap_minutes=30,
        temporal_window_hours=24,
        smooth_sigma=1.25,
        bootstrap_iters=500,
        permutation_iters=0,
    ):
        """
        Analyze descriptive relationships between two species: camera-level co-occurrence, 24-hour activity overlap, and near-time same-camera events within temporal_window_hours.
        """
        species_a = self._canonical_species_name(species_a)
        species_b = self._canonical_species_name(species_b)
        cameras_a, camera_counts_a = self._camera_set_for_species(
            species_a,
            min_camera_records=min_camera_records,
            min_gap_minutes=min_gap_minutes,
        )
        cameras_b, camera_counts_b = self._camera_set_for_species(
            species_b,
            min_camera_records=min_camera_records,
            min_gap_minutes=min_gap_minutes,
        )
        shared_cameras = sorted(cameras_a & cameras_b)
        union_cameras = cameras_a | cameras_b
        jaccard = len(shared_cameras) / len(union_cameras) if union_cameras else 0.0

        activity_overlap = self.estimate_activity_overlap(
            species_a=species_a,
            species_b=species_b,
            min_gap_minutes=min_gap_minutes,
            smooth_sigma=smooth_sigma,
            bootstrap_iters=bootstrap_iters,
            permutation_iters=permutation_iters,
        )
        delta_hat = activity_overlap.get("delta_hat")
        proximity = self._same_camera_temporal_proximity(
            species_a=species_a,
            species_b=species_b,
            temporal_window_hours=temporal_window_hours,
            min_gap_minutes=min_gap_minutes,
        )

        spatial_level = self._classify_overlap(jaccard, high=0.10, moderate=0.03)
        temporal_level = self._classify_overlap(delta_hat, high=0.70, moderate=0.45)
        proximity_level = self._classify_overlap(
            proximity["camera_count_with_proximity"] / len(shared_cameras) if shared_cameras else None,
            high=0.50,
            moderate=0.20,
        )

        relationship_score = 0.0
        weight_sum = 0.0
        if union_cameras:
            relationship_score += min(jaccard / 0.15, 1.0) * 0.45
            weight_sum += 0.45
        if delta_hat is not None:
            relationship_score += delta_hat * 0.35
            weight_sum += 0.35
        if shared_cameras:
            relationship_score += min(proximity["camera_count_with_proximity"] / len(shared_cameras), 1.0) * 0.20
            weight_sum += 0.20
        relationship_score = relationship_score / weight_sum if weight_sum else 0.0

        return {
            "species_a": species_a,
            "species_b": species_b,
            "record_count_a": sum(camera_counts_a.values()),
            "record_count_b": sum(camera_counts_b.values()),
            "camera_count_a": len(cameras_a),
            "camera_count_b": len(cameras_b),
            "shared_camera_count": len(shared_cameras),
            "shared_camera_ids": shared_cameras[:100],
            "camera_jaccard": round(jaccard, 4),
            "spatial_overlap_level": spatial_level,
            "activity_overlap_delta": delta_hat,
            "activity_overlap_ci": {
                "low": activity_overlap.get("delta_ci_low"),
                "high": activity_overlap.get("delta_ci_high"),
            },
            "activity_overlap_estimator_id": activity_overlap.get(
                "estimator_id"
            ),
            "activity_overlap_estimator_version": activity_overlap.get(
                "estimator_version"
            ),
            "activity_overlap_source_events_sha256": activity_overlap.get(
                "source_events_sha256"
            ),
            "activity_overlap_result_bundle_id": activity_overlap.get(
                "overlap_result_bundle_id"
            ),
            "activity_overlap_result_bundle_sha256": activity_overlap.get(
                "overlap_result_bundle_sha256"
            ),
            "activity_overlap_result_sha256": activity_overlap.get(
                "overlap_result_sha256"
            ),
            "activity_overlap_result_reused": activity_overlap.get(
                "overlap_result_reused"
            ),
            "temporal_overlap_level": temporal_level,
            "activity_overlap_summary": {
                "valid_event_count_a": activity_overlap.get("valid_event_count_a"),
                "valid_event_count_b": activity_overlap.get("valid_event_count_b"),
                "peak_hours_a": activity_overlap.get("peak_hours_a", [])[:3],
                "peak_hours_b": activity_overlap.get("peak_hours_b", [])[:3],
                "permutation_p_value": activity_overlap.get("permutation_p_value"),
                "supports_temporal_separation": activity_overlap.get("supports_temporal_separation"),
                "estimator_id": activity_overlap.get("estimator_id"),
                "estimator_version": activity_overlap.get("estimator_version"),
                "source_events_sha256": activity_overlap.get(
                    "source_events_sha256"
                ),
                "overlap_result_bundle_id": activity_overlap.get(
                    "overlap_result_bundle_id"
                ),
                "overlap_result_bundle_sha256": activity_overlap.get(
                    "overlap_result_bundle_sha256"
                ),
                "overlap_result_sha256": activity_overlap.get(
                    "overlap_result_sha256"
                ),
                "overlap_result_reused": activity_overlap.get(
                    "overlap_result_reused"
                ),
            },
            "same_camera_temporal_proximity": proximity,
            "proximity_level": proximity_level,
            "relationship_score": round(float(relationship_score), 4),
            "relationship_label": self._classify_overlap(relationship_score, high=0.65, moderate=0.35),
            "note": (
                "This is a descriptive species relationship summary combining camera co-occurrence, diel activity overlap, "
                "and near-time same-camera events. It suggests association patterns but does not prove ecological interaction."
            ),
        }

    def summarize_species_relationship_network(
        self,
        species_list=None,
        top_k=10,
        min_camera_records=1,
        min_shared_cameras=1,
        min_gap_minutes=30,
        temporal_window_hours=24,
        include_temporal_overlap=True,
        max_pairs=12,
    ):
        """
        Summarize a multi-species relationship network. Nodes are species; edges represent spatial and temporal relationships between species pairs.
        """
        selected_species = self._resolve_species_list(
            species_list=species_list,
            top_k=top_k,
            include_unknown=False,
        )
        cooccurrence = self.build_species_cooccurrence_matrix(
            species_list=selected_species,
            top_k=top_k,
            min_camera_records=min_camera_records,
            min_gap_minutes=min_gap_minutes,
        )
        candidate_pairs = [
            pair for pair in cooccurrence.get("top_cooccurring_pairs", [])
            if pair.get("shared_camera_count", 0) >= min_shared_cameras
        ][:max(int(max_pairs), 1)]

        composition = self.summarize_community_composition(top_k=max(int(top_k), 1), include_unknown=False)
        abundance_by_species = {
            row["species"]: row["record_count"]
            for row in composition.get("top_species", [])
        }

        edges = []
        for pair in candidate_pairs:
            if include_temporal_overlap:
                relationship = self.analyze_species_pair_relationship(
                    pair["species_a"],
                    pair["species_b"],
                    min_camera_records=min_camera_records,
                    min_gap_minutes=min_gap_minutes,
                    temporal_window_hours=temporal_window_hours,
                    bootstrap_iters=0,
                    permutation_iters=0,
                )
                edge = {
                    "species_a": relationship["species_a"],
                    "species_b": relationship["species_b"],
                    "shared_camera_count": relationship["shared_camera_count"],
                    "camera_jaccard": relationship["camera_jaccard"],
                    "activity_overlap_delta": relationship["activity_overlap_delta"],
                    "relationship_score": relationship["relationship_score"],
                    "relationship_label": relationship["relationship_label"],
                    "spatial_overlap_level": relationship["spatial_overlap_level"],
                    "temporal_overlap_level": relationship["temporal_overlap_level"],
                    "proximity_level": relationship["proximity_level"],
                }
            else:
                edge = {
                    "species_a": pair["species_a"],
                    "species_b": pair["species_b"],
                    "shared_camera_count": pair["shared_camera_count"],
                    "camera_jaccard": pair["jaccard"],
                    "activity_overlap_delta": None,
                    "relationship_score": round(min(pair["jaccard"] / 0.15, 1.0), 4),
                    "relationship_label": self._classify_overlap(pair["jaccard"], high=0.10, moderate=0.03),
                    "spatial_overlap_level": self._classify_overlap(pair["jaccard"], high=0.10, moderate=0.03),
                    "temporal_overlap_level": "not_computed",
                    "proximity_level": "not_computed",
                }
            edges.append(edge)

        edges.sort(key=lambda item: (item["relationship_score"], item["shared_camera_count"]), reverse=True)
        nodes = []
        for species in selected_species:
            cameras, camera_counts = self._camera_set_for_species(
                species,
                min_camera_records=min_camera_records,
                min_gap_minutes=min_gap_minutes,
            )
            nodes.append({
                "species": species,
                "record_count": abundance_by_species.get(species, sum(camera_counts.values())),
                "camera_count": len(cameras),
            })

        return {
            "species": selected_species,
            "nodes": nodes,
            "edges": edges,
            "top_relationships": edges[:10],
            "min_camera_records": min_camera_records,
            "min_shared_cameras": min_shared_cameras,
            "include_temporal_overlap": bool(include_temporal_overlap),
            "note": (
                "Relationship network edges combine camera co-occurrence and optional diel activity overlap. "
                "Use this as an exploratory ecological relationship map, not causal interaction evidence."
            ),
        }

    def analyze_community_ecology(
        self,
        species_list=None,
        top_k=8,
        min_camera_records=1,
        min_records_per_species=2,
        min_gap_minutes=30,
        min_rai_per_100_camera_days=1.0,
        include_relationships=True,
        max_pairs=12,
    ):
        """
        Integrated multi-species ecological summary: community composition, camera richness, spatial co-occurrence, day-night activity guilds, and an optional relationship network.
        """
        top_k = max(int(top_k), 3)
        composition = self.summarize_community_composition(
            top_k=top_k,
            include_unknown=False,
            abundance_metric="rai",
        )
        if species_list is None:
            selected_species = [
                row["species"]
                for row in composition.get("top_species", [])
                if float(row.get("metric_value", 0.0)) > 0
            ][:top_k]
        else:
            selected_species = self._resolve_species_list(
                species_list=species_list,
                top_k=top_k,
                include_unknown=False,
            )
        richness = self.build_camera_species_richness(
            top_k=max(top_k, 10),
            min_records_per_species=min_records_per_species,
            min_gap_minutes=min_gap_minutes,
            min_rai_per_100_camera_days=min_rai_per_100_camera_days,
        )
        cooccurrence = self.build_species_cooccurrence_matrix(
            species_list=selected_species,
            top_k=top_k,
            min_camera_records=min_camera_records,
            min_gap_minutes=min_gap_minutes,
        )
        activity = self.compare_multi_species_hourly_activity(
            species_list=selected_species,
            top_k=top_k,
            min_gap_minutes=min_gap_minutes,
        )
        guilds = self.build_species_activity_guilds(
            species_list=selected_species,
            top_k=top_k,
            min_gap_minutes=min_gap_minutes,
        )
        relationships = None
        if include_relationships:
            relationships = self.summarize_species_relationship_network(
                species_list=selected_species,
                top_k=top_k,
                min_camera_records=min_camera_records,
                min_shared_cameras=1,
                min_gap_minutes=min_gap_minutes,
                temporal_window_hours=24,
                include_temporal_overlap=True,
                max_pairs=max_pairs,
            )

        return {
            "selected_species": selected_species,
            "community": {
                "total_record_count": composition.get("total_record_count"),
                "species_richness": composition.get("species_richness"),
                "shannon_index": composition.get("shannon_index"),
                "simpson_index": composition.get("simpson_index"),
                "top_species": composition.get("top_species", [])[:top_k],
            },
            "camera_richness_hotspots": richness.get("richness_by_camera", [])[:5],
            "top_spatial_cooccurrence_pairs": cooccurrence.get("top_cooccurring_pairs", [])[:5],
            "activity_profiles": activity.get("profiles", [])[:top_k],
            "activity_guilds": guilds.get("activity_guilds", []),
            "activity_guild_counts": guilds.get("guild_counts", {}),
            "top_relationships": (relationships or {}).get("top_relationships", [])[:5],
            "include_relationships": bool(include_relationships),
            "note": (
                "This community ecology summary combines record-based composition, camera-level spatial structure, "
                "independent-event/RAI corrected composition, real clock-time diel activity, and exploratory species relationships. It is intended for hypothesis "
                "generation and figure drafting, not causal inference."
            ),
        }

    def infer_complete_community(
        self,
        species_list=None,
        top_k=None,
        min_independent_events=5,
        min_detected_cameras=2,
        min_gap_minutes=30,
        accumulation_permutations=250,
        bootstrap_iters=500,
        association_fdr_alpha=0.05,
        max_beta_pairs=100000,
        fit_detection_corrected_occupancy=True,
        occupancy_species_limit=None,
        occasion_days=14,
        min_active_occasions=2,
        occupancy_bootstrap_iters=300,
        random_seed=42,
    ):
        """
        Dataset-wide community inference from independent camera-trap events.

        The tool estimates observed and Chao2 richness, Hill diversity, camera
        alpha diversity, Jaccard/Bray-Curtis beta diversity, species
        accumulation, conditional shared-camera associations with BH-FDR
        correction, and (when deployment histories support it) repeated-survey
        detection-corrected occupancy for multiple species.
        """
        if not self._using_camtrap_backend():
            raise ValueError(
                "Complete community inference requires an active CamtrapDP "
                "package so cameras, deployments, and independent events share "
                "one standardized backend."
            )
        events = self._load_camtrap_event_df(
            min_gap_minutes=min_gap_minutes
        ).copy()
        if events.empty:
            raise ValueError(
                "No independent events are available for community inference."
            )
        if "species" not in events:
            events["species"] = events.get(
                "scientificName",
                "Unknown",
            )
        events["species"] = events["species"].map(
            self._canonical_species_name
        )
        events["camera_id"] = events["camera_id"].astype(str)
        events = events.loc[
            (events["species"] != "Unknown")
            & events["camera_id"].str.strip().ne("")
        ].copy()
        if events.empty:
            raise ValueError(
                "No identified species-camera events are available."
            )

        event_counts = events["species"].value_counts()
        camera_counts = events.groupby("species")["camera_id"].nunique()
        eligible = [
            species
            for species in event_counts.index
            if int(event_counts[species]) >= max(
                int(min_independent_events),
                1,
            )
            and int(camera_counts.get(species, 0)) >= max(
                int(min_detected_cameras),
                1,
            )
        ]
        if species_list is not None:
            requested = self._resolve_species_list(
                species_list=species_list,
                top_k=max(len(eligible), 1),
                include_unknown=False,
            )
            eligible_set = set(eligible)
            selected_species = [
                species
                for species in requested
                if species in eligible_set
            ]
        else:
            # Complete-community inference must not inherit a display/ranking
            # limit.  Figure tools may display a bounded subset, but the
            # inferential matrix always contains every eligible species.
            selected_species = list(eligible)
        if len(selected_species) < 2:
            raise ValueError(
                "At least two eligible species are required for community "
                "inference after applying event and camera thresholds."
            )

        selected_events = events.loc[
            events["species"].isin(selected_species)
        ].copy()
        count_matrix = pd.crosstab(
            selected_events["camera_id"],
            selected_events["species"],
        ).reindex(columns=selected_species, fill_value=0)
        presence = (count_matrix > 0).astype(int)
        site_count = int(len(count_matrix))
        species_count = int(len(selected_species))
        total_events = int(count_matrix.to_numpy().sum())

        def _hill_numbers(values):
            values = np.asarray(values, dtype=float)
            total = float(values.sum())
            positive = values[values > 0]
            if total <= 0 or not len(positive):
                return {
                    "q0_richness": 0.0,
                    "q1_exponential_shannon": 0.0,
                    "q2_inverse_simpson": 0.0,
                    "shannon_entropy": 0.0,
                    "simpson_concentration": 0.0,
                }
            proportions = positive / total
            shannon = -float(
                np.sum(proportions * np.log(proportions))
            )
            concentration = float(np.sum(proportions ** 2))
            return {
                "q0_richness": float(len(positive)),
                "q1_exponential_shannon": float(np.exp(shannon)),
                "q2_inverse_simpson": (
                    float(1.0 / concentration)
                    if concentration > 0
                    else 0.0
                ),
                "shannon_entropy": shannon,
                "simpson_concentration": concentration,
            }

        def _chao2(presence_frame):
            occurrence = presence_frame.sum(axis=0).to_numpy(dtype=int)
            observed = int((occurrence > 0).sum())
            q1 = int((occurrence == 1).sum())
            q2 = int((occurrence == 2).sum())
            if q2 > 0:
                estimate = observed + (
                    q1 * q1
                    / (2.0 * q2)
                )
            else:
                estimate = observed + q1 * max(q1 - 1, 0) / 2.0
            return {
                "observed_richness": observed,
                "singleton_species_q1": q1,
                "doubleton_species_q2": q2,
                "chao2_richness": float(estimate),
            }

        gamma_hill = _hill_numbers(
            count_matrix.sum(axis=0).to_numpy()
        )
        richness = _chao2(presence)
        per_camera_rows = []
        for camera_id, row in count_matrix.iterrows():
            hill = _hill_numbers(row.to_numpy())
            per_camera_rows.append({
                "camera_id": str(camera_id),
                "independent_event_count": int(row.sum()),
                **hill,
            })
        alpha_frame = pd.DataFrame(per_camera_rows)
        alpha_summary = {
            "camera_count": site_count,
            "mean_species_richness": round(float(
                alpha_frame["q0_richness"].mean()
            ), 6),
            "median_species_richness": round(float(
                alpha_frame["q0_richness"].median()
            ), 6),
            "mean_hill_q1": round(float(
                alpha_frame["q1_exponential_shannon"].mean()
            ), 6),
            "mean_hill_q2": round(float(
                alpha_frame["q2_inverse_simpson"].mean()
            ), 6),
        }

        rng = np.random.default_rng(int(random_seed))
        possible_pair_count = site_count * max(site_count - 1, 0) // 2
        requested_pair_count = min(
            possible_pair_count,
            max(int(max_beta_pairs), 1),
        )
        jaccard_values = []
        bray_values = []
        sorensen_values = []
        turnover_values = []
        nestedness_values = []
        if site_count >= 2:
            if possible_pair_count <= requested_pair_count:
                pairs = [
                    (i, j)
                    for i in range(site_count)
                    for j in range(i + 1, site_count)
                ]
            else:
                pair_set = set()
                while len(pair_set) < requested_pair_count:
                    i, j = rng.integers(0, site_count, size=2)
                    if i != j:
                        pair_set.add(
                            (int(min(i, j)), int(max(i, j)))
                        )
                pairs = sorted(pair_set)
            presence_values = presence.to_numpy(dtype=bool)
            count_values = count_matrix.to_numpy(dtype=float)
            for i, j in pairs:
                intersection = float(
                    np.logical_and(
                        presence_values[i],
                        presence_values[j],
                    ).sum()
                )
                union = float(
                    np.logical_or(
                        presence_values[i],
                        presence_values[j],
                    ).sum()
                )
                jaccard_values.append(
                    1.0 - intersection / union
                    if union
                    else 0.0
                )
                only_i = float(
                    np.logical_and(
                        presence_values[i],
                        np.logical_not(presence_values[j]),
                    ).sum()
                )
                only_j = float(
                    np.logical_and(
                        np.logical_not(presence_values[i]),
                        presence_values[j],
                    ).sum()
                )
                sorensen_denominator = (
                    2.0 * intersection + only_i + only_j
                )
                sorensen = (
                    (only_i + only_j) / sorensen_denominator
                    if sorensen_denominator
                    else 0.0
                )
                minimum_unique = min(only_i, only_j)
                turnover_denominator = intersection + minimum_unique
                turnover = (
                    minimum_unique / turnover_denominator
                    if turnover_denominator
                    else 0.0
                )
                sorensen_values.append(float(sorensen))
                turnover_values.append(float(turnover))
                nestedness_values.append(float(max(
                    sorensen - turnover,
                    0.0,
                )))
                denominator = float(
                    (count_values[i] + count_values[j]).sum()
                )
                bray_values.append(
                    float(
                        np.abs(
                            count_values[i] - count_values[j]
                        ).sum()
                    )
                    / denominator
                    if denominator
                    else 0.0
                )
        beta_summary = {
            "camera_pair_count": len(jaccard_values),
            "pair_sampling": (
                "all_pairs"
                if len(jaccard_values) == possible_pair_count
                else "reproducible_random_pairs"
            ),
            "mean_jaccard_dissimilarity": (
                round(float(np.mean(jaccard_values)), 6)
                if jaccard_values
                else None
            ),
            "median_jaccard_dissimilarity": (
                round(float(np.median(jaccard_values)), 6)
                if jaccard_values
                else None
            ),
            "mean_bray_curtis_dissimilarity": (
                round(float(np.mean(bray_values)), 6)
                if bray_values
                else None
            ),
            "median_bray_curtis_dissimilarity": (
                round(float(np.median(bray_values)), 6)
                if bray_values
                else None
            ),
            "mean_sorensen_dissimilarity": (
                round(float(np.mean(sorensen_values)), 6)
                if sorensen_values
                else None
            ),
            "mean_turnover_component": (
                round(float(np.mean(turnover_values)), 6)
                if turnover_values
                else None
            ),
            "mean_nestedness_component": (
                round(float(np.mean(nestedness_values)), 6)
                if nestedness_values
                else None
            ),
            "partition_identity": (
                "Sørensen dissimilarity = turnover + nestedness-resultant "
                "dissimilarity"
            ),
        }

        accumulation_permutations = max(
            int(accumulation_permutations),
            1,
        )
        steps = np.unique(np.linspace(
            1,
            site_count,
            min(site_count, 25),
            dtype=int,
        ))
        accumulation = {int(step): [] for step in steps}
        presence_values = presence.to_numpy(dtype=int)
        for _ in range(accumulation_permutations):
            order = rng.permutation(site_count)
            cumulative = np.cumsum(
                presence_values[order],
                axis=0,
            )
            for step in steps:
                accumulation[int(step)].append(float(
                    (cumulative[int(step) - 1] > 0).sum()
                ))
        accumulation_curve = []
        for step in steps:
            estimates = np.asarray(accumulation[int(step)])
            accumulation_curve.append({
                "camera_count": int(step),
                "mean_observed_species": round(
                    float(estimates.mean()),
                    6,
                ),
                "ci_low": round(
                    float(np.quantile(estimates, 0.025)),
                    6,
                ),
                "ci_high": round(
                    float(np.quantile(estimates, 0.975)),
                    6,
                ),
            })

        association_rows = []
        n_sites = int(len(presence))
        for i, species_a in enumerate(selected_species):
            detected_a = presence[species_a].to_numpy(dtype=int)
            n_a = int(detected_a.sum())
            for species_b in selected_species[i + 1:]:
                detected_b = presence[species_b].to_numpy(dtype=int)
                n_b = int(detected_b.sum())
                shared = int((detected_a * detected_b).sum())
                expected = n_a * n_b / max(n_sites, 1)
                variance = (
                    n_b
                    * (n_a / max(n_sites, 1))
                    * (1.0 - n_a / max(n_sites, 1))
                    * (
                        (n_sites - n_b) / max(n_sites - 1, 1)
                    )
                )
                ses = (
                    (shared - expected) / math.sqrt(variance)
                    if variance > 0
                    else 0.0
                )
                support, probabilities = (
                    _conditional_shared_camera_distribution(
                        n_sites,
                        n_a,
                        n_b,
                    )
                )
                p_upper = float(
                    probabilities[support >= shared].sum()
                )
                p_lower = float(
                    probabilities[support <= shared].sum()
                )
                p_two_sided = min(
                    1.0,
                    2.0 * min(p_upper, p_lower),
                )
                association_rows.append({
                    "species_a": species_a,
                    "species_b": species_b,
                    "camera_count_a": n_a,
                    "camera_count_b": n_b,
                    "shared_camera_count": shared,
                    "expected_shared_camera_count": round(
                        expected,
                        6,
                    ),
                    "standardized_effect_size": round(
                        float(ses),
                        6,
                    ),
                    "p_two_sided": p_two_sided,
                    "direction": (
                        "positive"
                        if shared > expected
                        else "negative"
                        if shared < expected
                        else "neutral"
                    ),
                })
        if association_rows:
            order = np.argsort([
                row["p_two_sided"]
                for row in association_rows
            ])
            adjusted = np.ones(len(association_rows), dtype=float)
            running = 1.0
            for reverse_rank, index in enumerate(order[::-1], start=1):
                rank = len(association_rows) - reverse_rank + 1
                candidate = (
                    association_rows[int(index)]["p_two_sided"]
                    * len(association_rows)
                    / rank
                )
                running = min(running, candidate)
                adjusted[int(index)] = min(running, 1.0)
            for row, q_value in zip(association_rows, adjusted):
                row["fdr_q_value"] = round(float(q_value), 6)
                row["fdr_significant"] = bool(
                    q_value <= float(association_fdr_alpha)
                )
        association_rows.sort(
            key=lambda row: (
                row.get("fdr_q_value", 1.0),
                -abs(row["standardized_effect_size"]),
            )
        )

        species_summary = []
        for species in selected_species:
            species_summary.append({
                "species": species,
                "independent_event_count": int(
                    event_counts.get(species, 0)
                ),
                "detected_camera_count": int(
                    camera_counts.get(species, 0)
                ),
                "camera_detection_frequency": round(
                    float(camera_counts.get(species, 0))
                    / max(site_count, 1),
                    6,
                ),
            })
        species_summary.sort(
            key=lambda row: (
                -row["independent_event_count"],
                row["species"],
            )
        )

        bootstrap_iters = max(int(bootstrap_iters), 0)
        bootstrap_rows = []
        for _ in range(bootstrap_iters):
            indices = rng.integers(0, site_count, size=site_count)
            sampled_counts = count_matrix.iloc[indices]
            sampled_presence = (sampled_counts > 0).astype(int)
            sampled_hill = _hill_numbers(
                sampled_counts.sum(axis=0).to_numpy()
            )
            bootstrap_rows.append({
                **sampled_hill,
                **_chao2(sampled_presence),
            })
        community_intervals = {}
        if bootstrap_rows:
            bootstrap_frame = pd.DataFrame(bootstrap_rows)
            for metric in (
                "observed_richness",
                "chao2_richness",
                "q1_exponential_shannon",
                "q2_inverse_simpson",
            ):
                community_intervals[metric] = {
                    "ci_low": round(float(
                        bootstrap_frame[metric].quantile(0.025)
                    ), 6),
                    "ci_high": round(float(
                        bootstrap_frame[metric].quantile(0.975)
                    ), 6),
                }

        occupancy_models = []
        if fit_detection_corrected_occupancy:
            occupancy_candidates = species_summary
            if occupancy_species_limit is not None:
                limit = int(occupancy_species_limit)
                if limit > 0:
                    occupancy_candidates = species_summary[:limit]
            for item in occupancy_candidates:
                model = self.fit_single_season_occupancy_model(
                    species=item["species"],
                    occasion_days=occasion_days,
                    min_gap_minutes=min_gap_minutes,
                    min_active_occasions=min_active_occasions,
                    bootstrap_iters=occupancy_bootstrap_iters,
                    random_seed=random_seed,
                    allow_inferred_effort=False,
                )
                occupancy_models.append(model)
        fitted_occupancy = [
            model
            for model in occupancy_models
            if model.get("model_status") == "fitted"
        ]

        return {
            "status": "success",
            "analysis_scope": "dataset_wide_eligible_species",
            "community_inference_status": (
                "complete_diversity_association_and_detection_corrected_occupancy"
                if fitted_occupancy
                else "complete_diversity_and_conditional_association"
            ),
            "source_species_count": int(event_counts.shape[0]),
            "eligible_species_count": len(eligible),
            "analysed_species_count": species_count,
            "excluded_species_count": int(
                event_counts.shape[0] - len(selected_species)
            ),
            "camera_count": site_count,
            "independent_event_count": total_events,
            "species": selected_species,
            "species_summary": species_summary,
            "gamma_diversity": {
                **richness,
                **{
                    key: round(float(value), 6)
                    for key, value in gamma_hill.items()
                },
                "bootstrap_intervals": community_intervals,
            },
            "alpha_diversity": alpha_summary,
            "beta_diversity": beta_summary,
            "species_accumulation": accumulation_curve,
            "conditional_camera_associations": association_rows,
            "association_model": {
                "null_distribution": (
                    "hypergeometric conditional on each species' "
                    "camera-detection frequency"
                ),
                "multiple_testing": "Benjamini-Hochberg FDR",
                "fdr_alpha": float(association_fdr_alpha),
            },
            "occupancy_models": occupancy_models,
            "detection_corrected_occupancy_species_count": len(
                fitted_occupancy
            ),
            "occupancy_effort_basis": self._camera_effort_basis(),
            "parameters": {
                "min_independent_events": int(min_independent_events),
                "min_detected_cameras": int(min_detected_cameras),
                "min_gap_minutes": int(min_gap_minutes),
                "accumulation_permutations": accumulation_permutations,
                "bootstrap_iters": bootstrap_iters,
                "random_seed": int(random_seed),
            },
            "method_references": [
                (
                    "Hill (1973) Ecology 54:427-432; "
                    "doi:10.2307/1934352"
                ),
                (
                    "Colwell and Coddington (1994) Philosophical "
                    "Transactions of the Royal Society B 345:101-118; "
                    "doi:10.1098/rstb.1994.0091"
                ),
                (
                    "Bray and Curtis (1957) Ecological Monographs "
                    "27:325-349; doi:10.2307/1942268"
                ),
                (
                    "Baselga (2010) Global Ecology and Biogeography "
                    "19:134-143; doi:10.1111/j.1466-8238.2009.00490.x"
                ),
                (
                    "Gotelli (2000) Ecology 81:2606-2621; "
                    "doi:10.1890/0012-9658(2000)081"
                    "[2606:NMAOSC]2.0.CO;2"
                ),
                (
                    "MacKenzie et al. (2002) Ecology 83:2248-2255; "
                    "doi:10.1890/0012-9658(2002)083[2248:ESORWD]2.0.CO;2"
                ),
                (
                    "Dorazio et al. (2006) Ecology 87:842-854; "
                    "doi:10.1890/0012-9658(2006)87[842:ESRAAB]2.0.CO;2"
                ),
                (
                    "Benjamini and Hochberg (1995) Journal of the Royal "
                    "Statistical Society B 57:289-300; "
                    "doi:10.1111/j.2517-6161.1995.tb02031.x"
                ),
            ],
            "interpretation": (
                "The community result uses every species meeting prespecified "
                "event and camera thresholds, not a hard-coded focal list. "
                "Shared-camera tests condition on marginal detection frequencies; "
                "detection-corrected occupancy is added when documented repeated-"
                "survey effort is available in CamtrapDP."
            ),
        }

    def analyze_spatial_distribution(self, species, center_camera_id=None, radius_km=None, top_k=5):
        """
        Integrated spatial-distribution analysis:
        1. detection map
        2. hotspot
        3. naive occupancy
        4. detection-corrected occupancy model
        4. If a center camera and radius are provided, analyze that local radius.
        """
        species = species.strip()

        detection_map = self.build_detection_map(species)
        hotspot_rank = self.build_hotspot_rank(species, top_k=top_k)
        occupancy = self.build_occupancy_summary(species)
        occupancy_model = self.fit_single_season_occupancy_model(
            species=species,
            occasion_days=14,
            min_gap_minutes=30,
            min_active_occasions=2,
            bootstrap_iters=0,
        )

        result = {
            "species": species,
            "total_observations": sum(item["count"] for item in detection_map),
            "detection_map": detection_map,
            "hotspot_rank": hotspot_rank,
            "occupancy_summary": occupancy,
            "occupancy_model_summary": occupancy_model,
        }

        if center_camera_id is not None and radius_km is not None:
            nearby = self.get_cameras_in_radius(center_camera_id, radius_km)
            nearby_ids = set([str(center_camera_id)] + [x["camera_id"] for x in nearby])

            local_detection = [item for item in detection_map if item["camera_id"] in nearby_ids]

            result["local_analysis"] = {
                "center_camera_id": str(center_camera_id),
                "radius_km": radius_km,
                "nearby_cameras": nearby,
                "nearby_detection_map": local_detection,
                "local_hotspots_top5": local_detection[:top_k],
                "nearby_total_observations": sum(item["count"] for item in local_detection)
            }

        return result
