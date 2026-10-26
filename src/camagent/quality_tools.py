from collections import defaultdict

from .base_tools import BaseTools

QUALITY_SCORE_WEIGHTS = {
    "missing_behavior_ratio": 25,
    "invalid_timestamp_ratio": 35,
    "invalid_individuals_ratio": 25,
    "duplicate_burden_ratio": 15,
}
QUALITY_SCORE_PRIOR_RECORDS = 5
MISSING_BEHAVIOR_VALUES = {
    "",
    "unknown",
    "not assessed",
    "not_assessed",
    "na",
    "n/a",
    "none",
    "null",
    "unknown",
    "not evaluated",
}


class QualityTools(BaseTools):
    def analyze_camera_data_quality(self, species="Siberian tiger", top_k=10):
        """
        Analyze which cameras have poorer data quality.
        Considers missing behavior, invalid timestamps, invalid individuals, and high duplicate burden.
        quality_score is a heuristic engineering score, not a standard ecological metric.
        Returns the lowest-scoring cameras by default to keep agent output readable.
        Pass top_k=None to return all camera rows.
        """
        records = self.query_by_species(species)

        camera_stats = defaultdict(lambda: {
            "total_records": 0,
            "missing_behavior": 0,
            "invalid_timestamp": 0,
            "invalid_individuals": 0,
            "duplicate_burden": 0
        })

        # Accumulate base issues first
        for r in records:
            meta = r["metadata"]
            cam = str(meta.get("camera_id", "")).strip()
            if not cam:
                cam = "UNKNOWN_CAMERA"

            camera_stats[cam]["total_records"] += 1

            behavior = str(meta.get("behavior", "")).strip()
            if behavior.lower() in MISSING_BEHAVIOR_VALUES and not self._using_camtrap_backend():
                camera_stats[cam]["missing_behavior"] += 1

            dt = self.safe_parse_timestamp(meta.get("timestamp"))
            if dt is None:
                camera_stats[cam]["invalid_timestamp"] += 1

            individuals = meta.get("individuals")
            try:
                if individuals is None or int(individuals) <= 0:
                    camera_stats[cam]["invalid_individuals"] += 1
            except (TypeError, ValueError):
                camera_stats[cam]["invalid_individuals"] += 1

        # Add duplicate burden
        dup_groups = self.detect_duplicate_like_records(species, max_groups=None)["duplicate_groups"]
        for item in dup_groups:
            cam = item["camera_id"]
            camera_stats[cam]["duplicate_burden"] += max(item["duplicate_count"]-1,0)

        result = []
        for cam, stat in camera_stats.items():
            total = stat["total_records"]

            missing_behavior_ratio = stat["missing_behavior"] / total if total > 0 else 0.0
            invalid_timestamp_ratio = stat["invalid_timestamp"] / total if total > 0 else 0.0
            invalid_individuals_ratio = stat["invalid_individuals"] / total if total > 0 else 0.0
            duplicate_burden_ratio = stat["duplicate_burden"] / total if total > 0 else 0.0

            # Use lightly shrunken ratios for scoring so very small cameras are
            # not ranked only by one unstable record-level issue.
            score_denominator = total + QUALITY_SCORE_PRIOR_RECORDS
            score_missing_behavior_ratio = stat["missing_behavior"] / score_denominator if score_denominator > 0 else 0.0
            score_invalid_timestamp_ratio = stat["invalid_timestamp"] / score_denominator if score_denominator > 0 else 0.0
            score_invalid_individuals_ratio = stat["invalid_individuals"] / score_denominator if score_denominator > 0 else 0.0
            score_duplicate_burden_ratio = stat["duplicate_burden"] / score_denominator if score_denominator > 0 else 0.0

            # Weighted penalty
            quality_score = 100.0
            quality_score -= score_missing_behavior_ratio * QUALITY_SCORE_WEIGHTS["missing_behavior_ratio"]
            quality_score -= score_invalid_timestamp_ratio * QUALITY_SCORE_WEIGHTS["invalid_timestamp_ratio"]
            quality_score -= score_invalid_individuals_ratio * QUALITY_SCORE_WEIGHTS["invalid_individuals_ratio"]
            quality_score -= score_duplicate_burden_ratio * QUALITY_SCORE_WEIGHTS["duplicate_burden_ratio"]
            quality_score = round(max(0.0, quality_score), 2)

            result.append({
                "camera_id": cam,
                "total_records": total,
                "missing_behavior": stat["missing_behavior"],
                "invalid_timestamp": stat["invalid_timestamp"],
                "invalid_individuals": stat["invalid_individuals"],
                "duplicate_burden": stat["duplicate_burden"],
                "missing_behavior_ratio": round(missing_behavior_ratio, 4),
                "invalid_timestamp_ratio": round(invalid_timestamp_ratio, 4),
                "invalid_individuals_ratio": round(invalid_individuals_ratio, 4),
                "duplicate_burden_ratio": round(duplicate_burden_ratio, 4),
                "score_missing_behavior_ratio": round(score_missing_behavior_ratio, 4),
                "score_invalid_timestamp_ratio": round(score_invalid_timestamp_ratio, 4),
                "score_invalid_individuals_ratio": round(score_invalid_individuals_ratio, 4),
                "score_duplicate_burden_ratio": round(score_duplicate_burden_ratio, 4),
                "quality_score": quality_score,
                "quality_score_type": "heuristic",
                "quality_score_prior_records": QUALITY_SCORE_PRIOR_RECORDS,
                "quality_score_weights": dict(QUALITY_SCORE_WEIGHTS),
                "quality_score_note": "Evidence-adjusted heuristic engineering score; not a formal ecological data-quality metric."
            })

        # Sort by ascending quality score, then descending issue burden
        result.sort(
            key=lambda x: (
                x["quality_score"],
                -(x["missing_behavior"] + x["invalid_timestamp"] + x["invalid_individuals"] + x["duplicate_burden"]),
                -x["total_records"]
            )
        )
        if top_k is None:
            return result
        return result[:max(int(top_k), 0)]
    def check_missing_fields(self, species="Siberian tiger"):
        """
        Check missing fields.
        """
        records = self.query_by_species(species)

        required_fields = [
            "camera_id", "timestamp", "latitude", "longitude",
            "species", "individuals"
        ]
        if not self._using_camtrap_backend():
            required_fields.append("behavior")

        missing_counter = {field: 0 for field in required_fields}
        total = len(records)

        for r in records:
            meta = r["metadata"]
            for field in required_fields:
                value = meta.get(field)
                if value is None or str(value).strip() == "":
                    missing_counter[field] += 1

        return {
            "species": species,
            "total_records": total,
            "missing_counts": missing_counter,
            "missing_ratios": {
                k: round(v / total, 4) if total > 0 else 0.0
                for k, v in missing_counter.items()
            }
        }

    def detect_invalid_timestamps(self, species="Siberian tiger"):
        """
        Detect unparseable timestamps and clearly invalid times.
        """
        records = self.query_by_species(species)

        invalid_records = []
        suspicious_records = []

        for r in records:
            meta = r["metadata"]
            ts = meta.get("timestamp")
            dt = self.safe_parse_timestamp(ts)

            if dt is None:
                invalid_records.append({
                    "id": r["id"],
                    "camera_id": meta.get("camera_id"),
                    "timestamp": ts
                })
            else:
                if dt.year < 2000 or dt.year > 2035:
                    suspicious_records.append({
                        "id": r["id"],
                        "camera_id": meta.get("camera_id"),
                        "timestamp": ts
                    })

        return {
            "species": species,
            "invalid_timestamp_count": len(invalid_records),
            "invalid_timestamp_records": invalid_records[:50],
            "suspicious_timestamp_count": len(suspicious_records),
            "suspicious_timestamp_records": suspicious_records[:50]
        }

    def detect_label_noise(self, species="Siberian tiger"):
        """
        Check record-level label/schema consistency.

        This check does not compare labels with held-out expert-reviewed images
        and therefore cannot estimate taxonomic misclassification error.
        """
        records = self.query_by_species(species)

        noise_records = []

        for r in records:
            meta = r["metadata"]
            sp = str(meta.get("species", "")).strip()
            behavior = str(meta.get("behavior", "")).strip()
            individuals = meta.get("individuals")

            bad_reason = []

            if sp == "" or sp.lower() == "unknown":
                bad_reason.append("species_missing_or_unknown")

            if (behavior == "" or behavior.lower() == "unknown") and not self._using_camtrap_backend():
                bad_reason.append("behavior_missing_or_unknown")

            try:
                if individuals is None or int(individuals) <= 0:
                    bad_reason.append("invalid_individuals")
            except (TypeError, ValueError):
                bad_reason.append("invalid_individuals")

            if bad_reason:
                noise_records.append({
                    "id": r["id"],
                    "camera_id": meta.get("camera_id"),
                    "timestamp": meta.get("timestamp"),
                    "reasons": bad_reason
                })

        return {
            "species": species,
            "label_noise_count": len(noise_records),
            "label_noise_records": noise_records[:100],
            "check_scope": "record_schema_consistency_only",
            "image_label_validation_status": "not_evaluated",
            "note": (
                "The check flags missing/unknown species labels and invalid "
                "record fields. It does not validate species identifications "
                "against held-out expert-reviewed images."
            ),
        }

    def detect_duplicate_like_records(self, species="Siberian tiger", max_groups=100):
        """
        Check duplicate-like records with the same camera_id, timestamp, and species.
        """
        records = self.query_by_species(species)
        counter = defaultdict(list)

        for r in records:
            meta = r["metadata"]
            key = (
                str(meta.get("camera_id", "")).strip(),
                str(meta.get("timestamp", "")).strip(),
                str(meta.get("species", "")).strip().lower()
            )
            counter[key].append(r["id"])

        duplicates = []
        for key, ids in counter.items():
            if len(ids) > 1:
                duplicates.append({
                    "camera_id": key[0],
                    "timestamp": key[1],
                    "species": key[2],
                    "duplicate_count": len(ids),
                    "record_ids": ids[:20]
                })

        duplicates.sort(key=lambda x: x["duplicate_count"], reverse=True)

        return {
            "species": species,
            "duplicate_groups_count": len(duplicates),
            "duplicate_groups": duplicates if max_groups is None else duplicates[:max(int(max_groups), 0)]
        }


    def summarize_camera_record_volume(self, species="Siberian tiger", top_k=20):
        """
        Summarize cameras by record volume, including the highest and lowest volumes.
        """
        records = self.query_by_species(species)
        counter = defaultdict(int)

        for r in records:
            cam = str(r["metadata"].get("camera_id", "")).strip()
            if cam:
                counter[cam] += 1

        result = [{"camera_id": cam, "count": count} for cam, count in counter.items()]
        result.sort(key=lambda x: x["count"])
        return {
            "lowest_record_cameras": result[:top_k],
            "highest_record_cameras": list(reversed(result))[:top_k]
        }

    def summarize_duplicate_burden_by_camera(self, species="Siberian tiger", top_k=20):
        """
        Summarize duplicate burden for each camera using the same definition as analyze_camera_data_quality.
        duplicate burden = sum(max(duplicate_count - 1, 0))
        """
        dup = self.detect_duplicate_like_records(species)["duplicate_groups"]
        counter = defaultdict(int)

        for item in dup:
            counter[item["camera_id"]] += max(item["duplicate_count"] - 1, 0)

        result = [{"camera_id": cam, "duplicate_burden": burden} for cam, burden in counter.items()]
        result.sort(key=lambda x: x["duplicate_burden"], reverse=True)
        return result[:top_k]

    def analyze_data_quality(self, species="Siberian tiger"):
        """
        Entry point for integrated data-quality analysis.
        """
        return {
            "missing_fields": self.check_missing_fields(species),
            "invalid_timestamps": self.detect_invalid_timestamps(species),
            "label_noise": self.detect_label_noise(species),
            "duplicate_like_records": self.detect_duplicate_like_records(species),
            "camera_quality": self.analyze_camera_data_quality(species, top_k=10),
            "camera_record_volume": self.summarize_camera_record_volume(species),
            "duplicate_burden_by_camera": self.summarize_duplicate_burden_by_camera(species)
        }
