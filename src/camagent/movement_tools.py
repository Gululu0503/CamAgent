import math
from collections import Counter

from .base_tools import BaseTools


class MovementTools(BaseTools):
    def build_time_sorted_camera_sequence(self, species="Siberian tiger", min_gap_minutes=30):
        """
        Return the time-sorted independent-event sequence.
        """
        event_result = self.build_independent_events(
            species=species,
            by_camera=True,
            min_gap_minutes=min_gap_minutes,
        )
        camera_locations = self.get_camera_locations()
        result = []
        for event in event_result.get("events", []):
            parsed_time = self.safe_parse_timestamp(event.get("event_time"))
            camera_id = str(event.get("camera_id", "")).strip()
            loc = camera_locations.get(camera_id, {})
            lat = event.get("latitude") if event.get("latitude") is not None else loc.get("latitude")
            lon = event.get("longitude") if event.get("longitude") is not None else loc.get("longitude")
            if parsed_time is None or not camera_id or lat is None or lon is None:
                continue
            result.append({
                "id": event.get("event_id"),
                "camera_id": camera_id,
                "timestamp": event.get("event_time"),
                "parsed_time": parsed_time,
                "latitude": float(lat),
                "longitude": float(lon),
            })
        result.sort(key=lambda x: (x["parsed_time"], x["camera_id"], str(x.get("id", ""))))
        return result

    def analyze_movement_steps(self, species="Siberian tiger", max_time_gap_hours=72, max_distance_km=100, min_gap_minutes=30):
        """
        Analyze potential transition edges between temporally adjacent observations.
        Notes:
        1. The result is not an individual trajectory.
        2. Different independent events may come from different individuals.
        3. The result only constructs a potential camera-to-camera transition network under temporal and spatial proximity constraints.
        """
        seq = self.build_time_sorted_camera_sequence(species, min_gap_minutes=min_gap_minutes)
        steps = []

        for i in range(1, len(seq)):
            prev_r = seq[i - 1]
            curr_r = seq[i]

            if prev_r["camera_id"] == curr_r["camera_id"]:
                continue

            time_gap = (curr_r["parsed_time"] - prev_r["parsed_time"]).total_seconds() / 3600.0
            if time_gap <= 0 or time_gap > max_time_gap_hours:
                continue

            dist = self.calculate_distance_km(
                prev_r["latitude"], prev_r["longitude"],
                curr_r["latitude"], curr_r["longitude"]
            )

            if dist > max_distance_km:
                continue

            lat1 = math.radians(prev_r["latitude"])
            lat2 = math.radians(curr_r["latitude"])
            delta_lon = math.radians(curr_r["longitude"] - prev_r["longitude"])
            y = math.sin(delta_lon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
            bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
            compass = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
            direction = compass[int((bearing + 22.5) // 45) % 8]

            steps.append({
                "from_camera": prev_r["camera_id"],
                "to_camera": curr_r["camera_id"],
                "from_time": prev_r["timestamp"],
                "to_time": curr_r["timestamp"],
                "time_gap_hours": round(time_gap, 3),
                "distance_km": round(dist, 3),
                "speed_km_h": round(dist / time_gap, 4),
                "bearing_degrees": round(bearing, 2),
                "direction": direction,
                "inference_level": "camera_detection_adjacency_hypothesis",
                "identity_evidence": "none",
                "warning": "Adjacent detections may represent different individuals; this is not an animal movement step."
            })

        return steps

    def summarize_movement_distance(self, species="Siberian tiger", max_time_gap_hours=72, max_distance_km=100, steps=None):
        """
        Summarize potential dispersal-distance distribution.
        """
        if steps is None:
            steps = self.analyze_movement_steps(
                species,
                max_time_gap_hours=max_time_gap_hours,
                max_distance_km=max_distance_km
            )

        if not steps:
            return {
                "species": species,
                "step_count": 0,
                "mean_distance_km": None,
                "max_distance_km": None,
                "min_distance_km": None
            }

        distances = sorted(float(x["distance_km"]) for x in steps)

        def percentile(fraction):
            if not distances:
                return None
            position = (len(distances) - 1) * float(fraction)
            lower = int(math.floor(position))
            upper = int(math.ceil(position))
            if lower == upper:
                return distances[lower]
            weight = position - lower
            return distances[lower] * (1.0 - weight) + distances[upper] * weight

        return {
            "species": species,
            "step_count": len(steps),
            "mean_distance_km": round(sum(distances) / len(distances), 3),
            "max_distance_km": round(max(distances), 3),
            "min_distance_km": round(min(distances), 3),
            "distance_quantiles_km": {
                "p25": round(percentile(0.25), 3),
                "median": round(percentile(0.50), 3),
                "p75": round(percentile(0.75), 3),
                "p90": round(percentile(0.90), 3),
            },
            "distance_method": "Haversine great-circle distance between camera coordinates",
            "interpretation_boundary": (
                "These distances connect temporally adjacent camera detections "
                "without individual identity evidence; they are adjacency "
                "hypotheses, not verified animal movement steps."
            ),
        }

    def summarize_movement_direction(self, species="Siberian tiger", max_time_gap_hours=72, steps=None, max_distance_km=100):
        """
        Summarize potential dispersal-direction distribution.
        """
        if steps is None:
            steps = self.analyze_movement_steps(
                species,
                max_time_gap_hours=max_time_gap_hours,
                max_distance_km=max_distance_km
            )

        direction_counter = Counter(x["direction"] for x in steps)

        total = sum(direction_counter.values())
        result = []
        for d, c in direction_counter.items():
            result.append({
                "direction": d,
                "count": c,
                "ratio": round(c / total, 4) if total > 0 else 0.0
            })

        result.sort(key=lambda x: x["count"], reverse=True)
        return result

    def infer_transition_corridors(self, species="Siberian tiger", max_time_gap_hours=72, max_distance_km=100, top_k=20, steps=None):
        """
        Identify potential transition corridors by counting the most common camera-pair transition edges.
        These are only potential transition edges, not confirmed migration corridors or individual movement paths.
        """
        if steps is None:
            steps = self.analyze_movement_steps(
                species,
                max_time_gap_hours=max_time_gap_hours,
                max_distance_km=max_distance_km
            )

        edge_counter = Counter()

        for s in steps:
            edge = (s["from_camera"], s["to_camera"])
            edge_counter[edge] += 1

        result = []
        for (cam_a, cam_b), count in edge_counter.most_common(top_k):
            result.append({
                "from_camera": cam_a,
                "to_camera": cam_b,
                "count": count,
                "inference_level": "potential_transition_edge"
            })

        return result

    def infer_migration_corridors(self, species="Siberian tiger", max_time_gap_hours=72, max_distance_km=100, top_k=20, steps=None):
        """
        Backward-compatible alias.
        New code should use infer_transition_corridors.
        """
        return self.infer_transition_corridors(
            species=species,
            max_time_gap_hours=max_time_gap_hours,
            max_distance_km=max_distance_km,
            top_k=top_k,
            steps=steps
        )

    def infer_longest_transition_steps(self, species="Siberian tiger", max_time_gap_hours=72, max_distance_km=100, top_k=20, steps=None):
        """
        Return the largest potential transition steps in a simplified form.
        This only finds the longest adjacent observation pairs and is not full path reconstruction.
        """
        if steps is None:
            steps = self.analyze_movement_steps(
                species,
                max_time_gap_hours=max_time_gap_hours,
                max_distance_km=max_distance_km
            )

        ranked = sorted(steps, key=lambda x: x["distance_km"], reverse=True)
        return ranked[:top_k]

    def infer_longest_movement_paths(self, species="Siberian tiger", max_time_gap_hours=72, max_distance_km=100, top_k=20, steps=None):
        """
        Backward-compatible alias.
        New code should use infer_longest_transition_steps.
        """
        return self.infer_longest_transition_steps(
            species=species,
            max_time_gap_hours=max_time_gap_hours,
            max_distance_km=max_distance_km,
            top_k=top_k,
            steps=steps
        )

    def analyze_animal_movement_pattern(self, species="Siberian tiger", max_time_gap_hours=72, max_distance_km=100):
        """
        Integrated animal movement-pattern analysis.
        The output is a conservative potential local-movement summary, not a confirmed individual migration trajectory.
        """
        steps = self.analyze_movement_steps(
            species,
            max_time_gap_hours=max_time_gap_hours,
            max_distance_km=max_distance_km
        )

        return {
            "note": (
    "This module summarizes potential transition patterns derived from temporally adjacent records. "
    "It is not an individual trajectory reconstruction and should not be interpreted as confirmed migration routes."
),
            "species": species,
            "movement_step_count": len(steps),
            "movement_distance_summary": self.summarize_movement_distance(
                species=species,
                max_time_gap_hours=max_time_gap_hours,
                max_distance_km=max_distance_km,
                steps=steps
            ),
            "movement_direction_summary": self.summarize_movement_direction(
                species=species,
                max_time_gap_hours=max_time_gap_hours,
                max_distance_km=max_distance_km,
                steps=steps
            ),
            "top_transition_corridors": self.infer_transition_corridors(
                species=species,
                max_time_gap_hours=max_time_gap_hours,
                max_distance_km=max_distance_km,
                top_k=20,
                steps=steps
            ),
            "longest_transition_steps": self.infer_longest_transition_steps(
                species=species,
                max_time_gap_hours=max_time_gap_hours,
                max_distance_km=max_distance_km,
                top_k=20,
                steps=steps
            )
        }
