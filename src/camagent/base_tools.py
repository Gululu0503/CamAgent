import csv
import hashlib
import json
import math
import mimetypes
import re
import shutil
import uuid
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .*doesn't match a supported version!",
    category=Warning,
)

try:
    import chromadb
except ModuleNotFoundError:
    chromadb = None

from .paths import (
    CAMTRAP_EXAMPLE_DIR,
    CAMTRAP_ACTIVE_DATASET,
    CAMTRAP_EXPORTS_DIR,
    CAMTRAP_IMPORTS_DIR,
    DETECTION_INPUT_DIR,
    DETECTION_RECORDS_DIR,
    MEDIA_EXPORTS_DIR,
    OVERLAP_RESULT_BUNDLES_DIR,
    TIGER_JSON_DIR,
    TIGER_VIDEO_DIR,
    VECTOR_DB_DIR,
)
from .detection_inference import (
    run_speciesnet_on_image,
    run_speciesnet_on_video,
    write_speciesnet_sidecar,
)
from .timestamp_inference import extract_video_timestamp_with_paddleocr
from .camtrap_conversion import convert_species_csv_directory_to_camtrap_dp
from .ecology_figures import run_ecology_figure_suite
from .perception_evaluation import run_perception_benchmark


DEFAULT_COLLECTION_NAME = "tiger_observations_2024"


class BaseTools:
    def __init__(self, db_path=None, collection_name=DEFAULT_COLLECTION_NAME):
        db_path = str(VECTOR_DB_DIR if db_path is None else db_path)
        self.client = None
        self.collection = None
        self._json_records_cache = None
        self._json_data_dir = Path(TIGER_JSON_DIR)
        self._prefer_json_backend = False
        self._camtrap_package_dir = None
        self._prefer_camtrap_backend = False
        self._camtrap_records_df_cache = None
        self._camtrap_deployments_df_cache = None
        self._camtrap_event_df_cache = None
        self._camtrap_event_df_by_gap = {}
        self._media_path_cache = {}
        # Activity-overlap estimates are persisted separately from figures so
        # later plot tools and later Agent runs can reuse one exact numerical
        # result instead of silently estimating the same pair again.
        self._overlap_result_bundle_dir = Path(OVERLAP_RESULT_BUNDLES_DIR)
        self._overlap_result_memory_cache = {}

        if CAMTRAP_ACTIVE_DATASET.exists():
            self._activate_default_camtrap_backend()

        # Prefer a validated active CamtrapDP package; otherwise fall back to
        # the explicitly configured JSON or collection backend.
        if chromadb is None:
            return

        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.client = None
            self.collection = None

    def _activate_default_camtrap_backend(self):
        if not CAMTRAP_ACTIVE_DATASET.exists():
            raise FileNotFoundError(
                f"Current CamtrapDP data state is missing: {CAMTRAP_ACTIVE_DATASET}. "
                "Load or convert a CamtrapDP dataset before running the agent."
            )
        try:
            payload = json.loads(CAMTRAP_ACTIVE_DATASET.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Could not read current CamtrapDP data state: {CAMTRAP_ACTIVE_DATASET}") from exc
        package_dir = Path(payload.get("package_dir", ""))
        if not package_dir.is_absolute():
            package_dir = (CAMTRAP_ACTIVE_DATASET.parent / package_dir).resolve()
        datapackage_file = package_dir / "datapackage.json"
        if not datapackage_file.exists():
            raise FileNotFoundError(
                f"Current CamtrapDP datapackage is invalid or missing: {datapackage_file}. "
                "The agent no longer falls back to demo/example/legacy JSON or Chroma data."
            )
        self._camtrap_package_dir = package_dir
        self._prefer_camtrap_backend = True
        self._prefer_json_backend = False

    def set_json_data_dir(self, json_data_dir):
        json_data_dir = Path(json_data_dir)
        if not json_data_dir.exists():
            raise FileNotFoundError(f"JSON data directory not found: {json_data_dir}")
        self._json_data_dir = json_data_dir
        self._json_records_cache = None
        self._prefer_json_backend = True
        self._prefer_camtrap_backend = False
        return {
            "json_data_dir": str(self._json_data_dir),
            "record_count": len(self._load_records_from_json()),
            "note": "JSON-backed data source has been switched for subsequent analysis calls.",
        }

    def set_camtrap_data_dir(self, datapackage_path, activate=True, persist_active=True, write_validation=True):
        datapackage_path = self._resolve_camtrap_datapackage_path(datapackage_path)
        package_dir = (
            datapackage_path
            if datapackage_path.is_dir()
            else datapackage_path.parent
        ).expanduser().resolve()
        datapackage_file = package_dir / "datapackage.json"
        if not datapackage_file.exists():
            raise FileNotFoundError(f"CamtrapDP datapackage not found: {datapackage_file}")

        self._camtrap_package_dir = package_dir
        validation_report = self._validate_camtrap_package(package_dir)
        if write_validation:
            self._write_camtrap_conversion_validation(package_dir, validation_report)
        self._prefer_camtrap_backend = bool(activate)
        if activate:
            self._prefer_json_backend = False
        if activate and persist_active:
            CAMTRAP_ACTIVE_DATASET.parent.mkdir(parents=True, exist_ok=True)
            CAMTRAP_ACTIVE_DATASET.write_text(
                json.dumps(
                    {
                        "package_dir": str(package_dir),
                        "datapackage_path": str(datapackage_file),
                        "activated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "backend": "camtrap_dp_tabular",
                        "validation_status": validation_report.get("status"),
                        "validation_issue_count": validation_report.get("issue_count", 0),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        self._camtrap_records_df_cache = None
        self._camtrap_deployments_df_cache = None
        self._camtrap_event_df_cache = None
        self._camtrap_event_df_by_gap = {}
        report = self._load_camtrap_conversion_report()
        record_count = report.get("kept_media_records")
        event_count = report.get("independent_events")
        if record_count is None:
            record_count = int(len(self._load_camtrap_records_df()))
        if event_count is None:
            event_count = int(len(self._load_camtrap_event_df()))
        return {
            "backend": "camtrap_dp_tabular",
            "package_dir": str(package_dir),
            "datapackage_path": str(datapackage_file),
            "record_count": int(record_count),
            "event_count": int(event_count),
            "active": bool(activate),
            "validation": validation_report,
            "current_data_state": str(CAMTRAP_ACTIVE_DATASET) if activate and persist_active else "",
            "note": (
                "CamtrapDP tabular backend is active and recorded in session state; records are read from standard package "
                "tables without creating per-record JSON. Validation issues are recorded in conversion_report.json when present."
            ),
        }

    def use_collection_backend(self):
        self._prefer_camtrap_backend = False
        self._prefer_json_backend = False
        return {
            "backend": "collection" if self.collection is not None else "json_fallback",
            "note": (
                "Collection backend is active. The normal flow is recognition -> JSON -> Chroma DB -> agent analysis."
            ),
        }

    def _get_collection_records(self):
        if self.collection is None:
            return []

        results = self.collection.get(include=["documents", "metadatas"])
        records = []

        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        for i in range(len(ids)):
            meta = metas[i] if i < len(metas) else {}
            doc = docs[i] if i < len(docs) else ""
            records.append({
                "id": ids[i],
                "document": doc,
                "metadata": meta,
            })
        return records

    def _load_records_from_json(self):
        if self._json_records_cache is not None:
            return self._json_records_cache

        records = []
        if not self._json_data_dir.exists():
            self._json_records_cache = records
            return records

        for json_path in sorted(self._json_data_dir.glob("*.json")):
            try:
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            records.append({
                "id": json_path.stem,
                "document": metadata.get("ecological_report", ""),
                "metadata": metadata,
            })

        self._json_records_cache = records
        return records

    def _using_camtrap_backend(self):
        return bool(self._prefer_camtrap_backend and self._camtrap_package_dir is not None)

    def _load_camtrap_conversion_report(self):
        if self._camtrap_package_dir is None:
            return {}
        report_path = Path(self._camtrap_package_dir) / "conversion_report.json"
        if not report_path.exists():
            return {}
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _camtrap_source_timezone(self):
        """Return the field-clock timezone declared by the active conversion."""
        report = self._load_camtrap_conversion_report()
        timezone_name = str(report.get("source_timezone", "")).strip()
        return timezone_name or "UTC"

    def _parse_camtrap_timestamp_series(self, values, *, local_clock=True):
        """Parse stored timestamps and optionally restore the declared field clock."""
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        if not local_clock:
            return parsed
        try:
            return parsed.dt.tz_convert(self._camtrap_source_timezone()).dt.tz_localize(None)
        except (KeyError, TypeError, ValueError):
            return parsed.dt.tz_convert("UTC").dt.tz_localize(None)

    def _validate_camtrap_package(self, package_dir):
        package_dir = Path(package_dir)
        issues = []
        table_counts = {}

        def _table_path(resource_name, fallback_name):
            datapackage_file = package_dir / "datapackage.json"
            if datapackage_file.exists():
                try:
                    package = json.loads(datapackage_file.read_text(encoding="utf-8"))
                    for resource in package.get("resources", []):
                        if resource.get("name") == resource_name:
                            return package_dir / resource.get("path", fallback_name)
                except Exception as exc:
                    issues.append({
                        "severity": "warning",
                        "table": "datapackage",
                        "field": "resources",
                        "message": f"Could not fully parse datapackage resources: {exc}",
                    })
            return package_dir / fallback_name

        required_tables = {
            "deployments": ("deployments.csv", {"deploymentID", "deploymentStart", "deploymentEnd"}),
            "observations": ("observations.csv", {"observationID", "deploymentID", "observationType", "scientificName"}),
        }
        optional_tables = {
            "media": ("media.csv", {"mediaID", "deploymentID", "timestamp", "filePath"}),
        }

        loaded_samples = {}
        for table_name, (fallback, required_cols) in {**required_tables, **optional_tables}.items():
            path = _table_path(table_name, fallback)
            if not path.exists():
                severity = "error" if table_name in required_tables else "warning"
                issues.append({
                    "severity": severity,
                    "table": table_name,
                    "field": "",
                    "message": f"{table_name} table was not found at {path}.",
                })
                loaded_samples[table_name] = pd.DataFrame()
                continue
            try:
                sample = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=500)
                loaded_samples[table_name] = sample
                table_counts[table_name] = int(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1)
            except Exception as exc:
                issues.append({
                    "severity": "error",
                    "table": table_name,
                    "field": "",
                    "message": f"Could not read {table_name}: {exc}",
                })
                loaded_samples[table_name] = pd.DataFrame()
                continue

            missing = sorted(required_cols - set(sample.columns))
            for field in missing:
                issues.append({
                    "severity": "error",
                    "table": table_name,
                    "field": field,
                    "message": f"Required CamtrapDP field `{field}` is missing from {table_name}.",
                })

        deployments = loaded_samples.get("deployments", pd.DataFrame())
        if not deployments.empty:
            for col in ["deploymentStart", "deploymentEnd"]:
                if col in deployments.columns:
                    parsed = pd.to_datetime(deployments[col], errors="coerce", utc=True)
                    bad = int(parsed.isna().sum())
                    if bad:
                        issues.append({
                            "severity": "warning",
                            "table": "deployments",
                            "field": col,
                            "message": f"{bad} sampled deployment rows have unparseable {col} values.",
                        })
            for col in ["latitude", "longitude"]:
                if col in deployments.columns:
                    numeric = pd.to_numeric(deployments[col], errors="coerce")
                    bad = int(numeric.isna().sum())
                    if bad:
                        issues.append({
                            "severity": "warning",
                            "table": "deployments",
                            "field": col,
                            "message": f"{bad} sampled deployment rows have missing or nonnumeric {col}.",
                        })

        media = loaded_samples.get("media", pd.DataFrame())
        if not media.empty and "timestamp" in media.columns:
            parsed = pd.to_datetime(media["timestamp"], errors="coerce", utc=True)
            bad = int(parsed.isna().sum())
            if bad:
                issues.append({
                    "severity": "warning",
                    "table": "media",
                    "field": "timestamp",
                    "message": f"{bad} sampled media rows have unparseable timestamp values.",
                })

        observations = loaded_samples.get("observations", pd.DataFrame())
        if not observations.empty:
            if "observationType" in observations.columns:
                animal_share = float(observations["observationType"].astype(str).str.lower().eq("animal").mean())
                if animal_share == 0:
                    issues.append({
                        "severity": "warning",
                        "table": "observations",
                        "field": "observationType",
                        "message": "Sampled observations contain no `animal` rows, so ecological tools may return empty results.",
                    })
            for col in ["eventStart", "eventEnd"]:
                if col in observations.columns:
                    nonempty = observations[col].astype(str).str.strip() != ""
                    if nonempty.any():
                        parsed = pd.to_datetime(observations.loc[nonempty, col], errors="coerce", utc=True)
                        bad = int(parsed.isna().sum())
                        if bad:
                            issues.append({
                                "severity": "warning",
                                "table": "observations",
                                "field": col,
                                "message": f"{bad} sampled nonempty observation rows have unparseable {col} values.",
                            })

        error_count = sum(1 for item in issues if item.get("severity") == "error")
        warning_count = sum(1 for item in issues if item.get("severity") == "warning")
        return {
            "status": "error" if error_count else "warning" if warning_count else "ok",
            "issue_count": len(issues),
            "error_count": error_count,
            "warning_count": warning_count,
            "issues": issues[:50],
            "table_counts": table_counts,
            "note": (
                "Validation checks CamtrapDP basic fields, parseable dates, and usable sampled coordinates. "
                "A warning does not always block loading, but it is recorded so downstream analyses remain traceable."
            ),
        }

    def _write_camtrap_conversion_validation(self, package_dir, validation_report):
        package_dir = Path(package_dir)
        report_path = package_dir / "conversion_report.json"
        payload = {}
        if report_path.exists():
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        payload.setdefault("output_dir", str(package_dir))
        payload.setdefault("datapackage_path", str(package_dir / "datapackage.json"))
        payload["camtrap_validation"] = validation_report
        payload["active_backend_trace"] = {
            "backend": "camtrap_dp_tabular",
            "package_dir": str(package_dir),
            "datapackage_path": str(package_dir / "datapackage.json"),
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "purpose": "Prevent downstream question answering and plotting from falling back to an old JSON or Chroma dataset.",
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _camtrap_table_path(self, resource_name, fallback_name):
        package_dir = Path(self._camtrap_package_dir)
        datapackage_file = package_dir / "datapackage.json"
        if datapackage_file.exists():
            try:
                package = json.loads(datapackage_file.read_text(encoding="utf-8"))
                for resource in package.get("resources", []):
                    if resource.get("name") == resource_name:
                        return package_dir / resource.get("path", fallback_name)
            except Exception:
                pass
        return package_dir / fallback_name

    def _load_camtrap_deployments_df(self):
        if self._camtrap_deployments_df_cache is not None:
            return self._camtrap_deployments_df_cache
        if self._camtrap_package_dir is None:
            self._camtrap_deployments_df_cache = pd.DataFrame()
            return self._camtrap_deployments_df_cache

        path = self._camtrap_table_path("deployments", "deployments.csv")
        if not path.exists():
            self._camtrap_deployments_df_cache = pd.DataFrame()
            return self._camtrap_deployments_df_cache
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in ["latitude", "longitude"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        self._camtrap_deployments_df_cache = df
        return df

    def _load_camtrap_records_df(self):
        if self._camtrap_records_df_cache is not None:
            return self._camtrap_records_df_cache
        if self._camtrap_package_dir is None:
            self._camtrap_records_df_cache = pd.DataFrame()
            return self._camtrap_records_df_cache

        package_dir = Path(self._camtrap_package_dir)
        media_path = self._camtrap_table_path("media", "media.csv")
        observations_path = self._camtrap_table_path("observations", "observations.csv")
        if not media_path.exists() or not observations_path.exists():
            self._camtrap_records_df_cache = pd.DataFrame()
            return self._camtrap_records_df_cache

        media_cols = [
            "mediaID", "deploymentID", "timestamp", "filePath",
            "fileName", "fileMediatype", "captureMethod", "mediaComments",
        ]
        obs_cols = [
            "observationID", "deploymentID", "mediaID", "eventID", "eventStart",
            "eventEnd", "observationLevel", "observationType", "scientificName",
            "count", "behavior", "classificationProbability", "observationTags",
            "observationComments",
        ]
        media = pd.read_csv(media_path, dtype=str, keep_default_na=False, usecols=lambda c: c in media_cols)
        obs = pd.read_csv(observations_path, dtype=str, keep_default_na=False, usecols=lambda c: c in obs_cols)
        obs = obs[
            (obs.get("observationType", "").astype(str).str.lower() == "animal")
            & (obs.get("observationLevel", "").astype(str).str.lower() == "media")
        ].copy()

        df = obs.merge(media, on=["mediaID", "deploymentID"], how="left", suffixes=("", "_media"))
        deployments = self._load_camtrap_deployments_df()
        if not deployments.empty:
            deployment_cols = [
                col for col in ["deploymentID", "cameraID", "locationID", "latitude", "longitude"]
                if col in deployments.columns
            ]
            df = df.merge(deployments[deployment_cols], on="deploymentID", how="left")
        else:
            df["cameraID"] = df["deploymentID"].astype(str).str.replace(r"^dep_", "", regex=True)
            df["latitude"] = None
            df["longitude"] = None

        df["timestamp"] = df["timestamp"].where(df["timestamp"].astype(str).str.strip() != "", df["eventStart"])
        df["species"] = df["scientificName"].map(self._canonical_species_name)
        df["record_id"] = df["observationID"].where(df["observationID"].astype(str).str.strip() != "", df["mediaID"])
        df["filename"] = df["fileName"].where(df["fileName"].astype(str).str.strip() != "", df["filePath"].astype(str).map(lambda x: Path(x).name))
        comments = df.get("mediaComments", pd.Series("", index=df.index)).astype(str)
        df["timestamp_source"] = comments.str.extract(r"(?:^|;\s*)timestamp_source=([^;]+)", expand=False).fillna("camtrap_dp_unspecified").str.strip()
        source_recordtime = comments.str.extract(r"(?:^|;\s*)source_recordtime=([^;]+)", expand=False).fillna("").str.strip()
        explicit_clock = source_recordtime.str.contains(r"\d{1,2}:\d{2}", regex=True, na=False)
        declared_clock = comments.str.extract(r"(?:^|;\s*)timestamp_has_observed_clock=([^;]+)", expand=False)
        declared_synthetic = comments.str.extract(r"(?:^|;\s*)timestamp_is_synthetic=([^;]+)", expand=False)
        df["timestamp_has_observed_clock"] = declared_clock.fillna(explicit_clock.astype(str)).astype(str).str.lower().isin({"1", "true", "yes"})
        df["timestamp_is_synthetic"] = declared_synthetic.fillna((~df["timestamp_has_observed_clock"]).astype(str)).astype(str).str.lower().isin({"1", "true", "yes"})
        df["package_dir"] = str(package_dir)
        for col in ["latitude", "longitude", "count", "classificationProbability"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        self._camtrap_records_df_cache = df
        return df

    def _load_camtrap_event_df(self, min_gap_minutes=None):
        """Load cached events or deterministically rebuild them for a requested gap."""
        if self._camtrap_package_dir is None:
            self._camtrap_event_df_cache = pd.DataFrame()
            return self._camtrap_event_df_cache

        report = self._load_camtrap_conversion_report()
        cached_gap = report.get("filters", {}).get("min_gap_minutes", 30)
        try:
            cached_gap = float(cached_gap)
        except (TypeError, ValueError):
            cached_gap = 30.0
        requested_gap = cached_gap if min_gap_minutes is None else max(float(min_gap_minutes), 0.0)
        cache_key = round(requested_gap, 9)
        if cache_key in self._camtrap_event_df_by_gap:
            return self._camtrap_event_df_by_gap[cache_key]

        events_path = Path(self._camtrap_package_dir) / "agent_independent_events.csv"
        if events_path.exists() and math.isclose(requested_gap, cached_gap):
            df = pd.read_csv(events_path, dtype=str, keep_default_na=False)
            for col in ["latitude", "longitude", "event_record_count", "observed_clock_time_records", "synthetic_time_records"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "event_time" in df.columns:
                df["event_time_dt"] = self._parse_camtrap_timestamp_series(
                    df["event_time"],
                    local_clock=True,
                )
            if "has_observed_clock_time" not in df.columns:
                records = self._load_camtrap_records_df()
                if not records.empty and "eventID" in records.columns:
                    provenance = records.groupby("eventID", sort=False).agg(
                        observed_clock_time_records=("timestamp_has_observed_clock", "sum"),
                        synthetic_time_records=("timestamp_is_synthetic", "sum"),
                    )
                    df = df.merge(provenance, left_on="event_id", right_index=True, how="left", suffixes=("", "_derived"))
                    for col in ["observed_clock_time_records", "synthetic_time_records"]:
                        derived = f"{col}_derived"
                        if derived in df.columns:
                            df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(pd.to_numeric(df[derived], errors="coerce")).fillna(0)
                            df = df.drop(columns=[derived])
                if "observed_clock_time_records" not in df.columns:
                    df["observed_clock_time_records"] = 0
                if "synthetic_time_records" not in df.columns:
                    df["synthetic_time_records"] = 0
                df["has_observed_clock_time"] = pd.to_numeric(df["observed_clock_time_records"], errors="coerce").fillna(0).gt(0)
            else:
                df["has_observed_clock_time"] = df["has_observed_clock_time"].astype(str).str.lower().isin({"1", "true", "yes"})
            df["event_gap_minutes"] = requested_gap
            self._camtrap_event_df_cache = df
            self._camtrap_event_df_by_gap[cache_key] = df
            return df

        records = self._load_camtrap_records_df()
        if not records.empty:
            work = records.copy()
            work["event_time_dt"] = self._parse_camtrap_timestamp_series(
                work["timestamp"],
                local_clock=True,
            )
            work = work.dropna(subset=["event_time_dt", "cameraID", "species"])
            work = work.sort_values(["species", "cameraID", "event_time_dt", "record_id"])
            gap = work.groupby(["species", "cameraID"])["event_time_dt"].diff().dt.total_seconds().div(60)
            work["_event_seq"] = (gap.isna() | gap.gt(requested_gap)).groupby([work["species"], work["cameraID"]]).cumsum().astype(int)
            grouped = work.groupby(["species", "cameraID", "_event_seq"], sort=False)
            df = grouped.agg(
                deploymentID=("deploymentID", "first"),
                scientificName=("scientificName", "first"),
                event_time_start=("event_time_dt", "min"),
                event_time_end=("event_time_dt", "max"),
                event_record_count=("record_id", "size"),
                latitude=("latitude", "median"),
                longitude=("longitude", "median"),
                observed_clock_time_records=("timestamp_has_observed_clock", "sum"),
                synthetic_time_records=("timestamp_is_synthetic", "sum"),
            ).reset_index()
            df["event_time_dt"] = df["event_time_start"] + (df["event_time_end"] - df["event_time_start"]) / 2
            df["event_time"] = df["event_time_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df["start_time"] = df["event_time_start"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df["end_time"] = df["event_time_end"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df["camera_id"] = df["cameraID"].astype(str)
            df["event_id"] = [f"derived_{stable}" for stable in pd.util.hash_pandas_object(df[["species", "cameraID", "_event_seq"]].astype(str), index=False).astype("uint64").map(lambda value: f"{int(value):016x}")]
            df["has_observed_clock_time"] = pd.to_numeric(df["observed_clock_time_records"], errors="coerce").fillna(0).gt(0)
            df["event_gap_minutes"] = requested_gap
            self._camtrap_event_df_by_gap[cache_key] = df
            return df

        observations_path = self._camtrap_table_path("observations", "observations.csv")
        if not observations_path.exists():
            self._camtrap_event_df_cache = pd.DataFrame()
            return self._camtrap_event_df_cache
        obs = pd.read_csv(observations_path, dtype=str, keep_default_na=False)
        obs = obs[
            (obs.get("observationType", "").astype(str).str.lower() == "animal")
            & (obs.get("observationLevel", "").astype(str).str.lower() == "event")
        ].copy()
        if obs.empty:
            self._camtrap_event_df_cache = pd.DataFrame()
            return self._camtrap_event_df_cache
        deployments = self._load_camtrap_deployments_df()
        if not deployments.empty:
            obs = obs.merge(
                deployments[[c for c in ["deploymentID", "cameraID", "latitude", "longitude"] if c in deployments.columns]],
                on="deploymentID",
                how="left",
            )
        obs["species"] = obs["scientificName"].map(self._canonical_species_name)
        obs["event_time"] = obs["eventStart"]
        obs["event_time_dt"] = self._parse_camtrap_timestamp_series(
            obs["event_time"],
            local_clock=True,
        )
        obs["event_record_count"] = pd.to_numeric(obs.get("count", 1), errors="coerce").fillna(1)
        obs["observed_clock_time_records"] = 0
        obs["synthetic_time_records"] = 0
        obs["has_observed_clock_time"] = False
        obs["event_gap_minutes"] = requested_gap
        self._camtrap_event_df_cache = obs
        self._camtrap_event_df_by_gap[cache_key] = obs
        return obs

    def _camera_effort_days_from_records(self):
        deployments = self._load_camtrap_deployments_df()
        effort_days = {}
        if not deployments.empty:
            camera_col = "cameraID" if "cameraID" in deployments.columns else "deploymentID"
            start = pd.to_datetime(deployments.get("deploymentStart"), errors="coerce")
            end = pd.to_datetime(deployments.get("deploymentEnd"), errors="coerce")
            valid = deployments.assign(_start=start, _end=end).dropna(subset=[camera_col, "_start", "_end"])
            for camera_id, group in valid.groupby(camera_col):
                days = ((group["_end"] - group["_start"]).dt.total_seconds() / 86400).clip(lower=0)
                effort_days[str(camera_id)] = max(float(sum(math.ceil(day) for day in days)), 1.0)
            if effort_days:
                return effort_days

        records = self._load_camtrap_records_df()
        if records.empty or "timestamp" not in records.columns:
            return effort_days
        camera_col = "cameraID" if "cameraID" in records.columns else "deploymentID"
        work = records.copy()
        work["_timestamp"] = self._parse_camtrap_timestamp_series(
            work["timestamp"],
            local_clock=True,
        )
        work = work.dropna(subset=[camera_col, "_timestamp"])
        for camera_id, group in work.groupby(camera_col):
            start_dt = group["_timestamp"].min()
            end_dt = group["_timestamp"].max()
            if pd.isna(start_dt) or pd.isna(end_dt):
                continue
            effort_days[str(camera_id)] = max((end_dt.date() - start_dt.date()).days + 1, 1)
        return effort_days

    def _camera_effort_basis(self):
        if self._using_camtrap_backend():
            deployments = self._load_camtrap_deployments_df()
            if (
                not deployments.empty
                and "deploymentComments" in deployments.columns
                and deployments["deploymentComments"].astype(str).str.contains("approximated", case=False, na=False).any()
            ):
                return "detection_span_proxy"
            return "documented_deployment_intervals"
        return "record_span_proxy"

    def _records_from_camtrap_frame(self, df):
        records = []
        if df is None or df.empty:
            return records
        for row in df.itertuples(index=False):
            row_data = row._asdict()
            metadata = {
                "filename": str(row_data.get("filename", "")),
                "camera_id": str(row_data.get("cameraID", "") or str(row_data.get("deploymentID", "")).replace("dep_", "", 1)),
                "latitude": row_data.get("latitude"),
                "longitude": row_data.get("longitude"),
                "timestamp": str(row_data.get("timestamp", "")),
                "timestamp_original_date": str(row_data.get("timestamp", ""))[:10],
                "timestamp_source": str(row_data.get("timestamp_source", "camtrap_dp_unspecified")),
                "timestamp_is_synthetic": bool(row_data.get("timestamp_is_synthetic", False)),
                "timestamp_has_observed_clock": bool(row_data.get("timestamp_has_observed_clock", False)),
                "species": str(row_data.get("species", "")) or str(row_data.get("scientificName", "")),
                "scientific_name": str(row_data.get("scientificName", "")),
                "confidence": row_data.get("classificationProbability", ""),
                "individuals": int(row_data.get("count")) if pd.notna(row_data.get("count")) else 1,
                "behavior": str(row_data.get("behavior", "")) or "Unknown",
                "special_observations": str(row_data.get("observationComments", "")),
                "ecological_report": "",
                "camtrap_package_name": Path(self._camtrap_package_dir).name if self._camtrap_package_dir else "",
                "camtrap_package_dir": str(row_data.get("package_dir", self._camtrap_package_dir or "")),
                "camtrap_datapackage_path": str(Path(self._camtrap_package_dir) / "datapackage.json") if self._camtrap_package_dir else "",
                "camtrap_deployment_id": str(row_data.get("deploymentID", "")),
                "camtrap_media_id": str(row_data.get("mediaID", "")),
                "camtrap_observation_id": str(row_data.get("observationID", "")),
                "camtrap_event_id": str(row_data.get("eventID", "")),
                "camtrap_observation_level": str(row_data.get("observationLevel", "")),
                "camtrap_observation_type": str(row_data.get("observationType", "")),
                "camtrap_capture_method": str(row_data.get("captureMethod", "")),
                "camtrap_file_path": str(row_data.get("filePath", "")),
                "camtrap_file_public": False,
                "camtrap_file_mediatype": str(row_data.get("fileMediatype", "")),
                "camtrap_event_start": str(row_data.get("eventStart", "")),
                "camtrap_event_end": str(row_data.get("eventEnd", "")),
                "camtrap_observation_tags": str(row_data.get("observationTags", "")),
                "camtrap_observation_comments": str(row_data.get("observationComments", "")),
                "camtrap_classification_method": "human",
                "camtrap_classified_by": "0525_csv_conversion",
                "camtrap_classification_timestamp": "",
            }
            records.append({
                "id": str(row_data.get("record_id", "")),
                "document": "",
                "metadata": metadata,
            })
        return records

    def _make_unique_dir(self, parent_dir, stem):
        parent_dir = Path(parent_dir)
        parent_dir.mkdir(parents=True, exist_ok=True)

        safe_stem = self._slugify_text(stem or "dataset")
        candidate = parent_dir / safe_stem
        if not candidate.exists():
            return candidate

        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = parent_dir / f"{safe_stem}_{suffix}"
        if not candidate.exists():
            return candidate

        return parent_dir / f"{safe_stem}_{suffix}_{uuid.uuid4().hex[:8]}"

    def _slugify_text(self, text):
        text = str(text or "").strip().lower()
        pieces = []
        for char in text:
            if char.isalnum():
                pieces.append(char)
            elif char in {"-", "_"}:
                pieces.append(char)
            else:
                pieces.append("_")

        slug = "".join(pieces).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug or "dataset"

    def _read_csv_rows(self, csv_path):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _write_csv_rows(self, csv_path, fieldnames, rows):
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})

    def _stringify_bool(self, value):
        return "true" if self._is_truthy_flag(value) else "false"

    def _normalize_numeric_text(self, value):
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        return text

    def _to_camtrap_iso(self, dt):
        if dt is None:
            return ""
        if dt.tzinfo is not None:
            return dt.isoformat()
        return dt.isoformat(timespec="seconds") + "Z"

    def _source_datetime_to_utc(self, value, source_timezone="UTC"):
        """Interpret a naive field-clock value in source_timezone and return UTC."""
        if value is None or not str(value).strip():
            return None
        try:
            timestamp = pd.Timestamp(value)
        except Exception:
            return None
        if pd.isna(timestamp):
            return None
        try:
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(
                    str(source_timezone or "UTC"),
                    ambiguous="raise",
                    nonexistent="raise",
                )
            timestamp = timestamp.tz_convert("UTC")
        except (KeyError, TypeError, ValueError):
            return None
        return timestamp.to_pydatetime()

    def _guess_media_mimetype(self, file_name, fallback="image/jpeg"):
        guessed, _ = mimetypes.guess_type(str(file_name or ""))
        return guessed or fallback

    def _normalize_probability_text(self, value):
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return str(float(text))
        except Exception:
            return ""

    def _normalize_species_text(self, text):
        return str(text or "").strip().lower()

    def _canonical_species_name(self, text):
        raw = str(text or "").strip()
        if not raw:
            return "Unknown"

        key = raw.lower()
        canonical_map = {
            "tiger": "Siberian tiger",
            "siberian tiger": "Siberian tiger",
            "amur tiger": "Siberian tiger",
            "panthera tigris altaica": "Siberian tiger",
            "wild boar": "Wild boar",
            "boar": "Wild boar",
            "sus scrofa": "Wild boar",
            "red fox": "Red fox",
            "fox": "Red fox",
            "vulpes vulpes": "Red fox",
            "amur leopard": "Amur leopard",
            "leopard": "Amur leopard",
            "panthera pardus orientalis": "Amur leopard",
            "leopard cat": "Leopard cat",
            "prionailurus bengalensis": "Leopard cat",
            "black bear": "Black bear",
            "asian black bear": "Black bear",
            "ursus thibetanus": "Black bear",
            "siberian roe deer": "Roe deer",
            "roe deer": "Roe deer",
            "capreolus pygargus": "Roe deer",
            "sika deer": "Sika deer",
            "cervus nippon": "Sika deer",
            "musk deer": "Musk deer",
            "moschus moschiferus": "Musk deer",
            "raccoon dog": "Raccoon dog",
            "nyctereutes procyonoides": "Raccoon dog",
            "amur hedgehog": "Amur hedgehog",
            "hedgehog": "Amur hedgehog",
            "erinaceus amurensis": "Amur hedgehog",
            "manchurian hare": "Manchurian hare",
            "hare": "Manchurian hare",
            "lepus mandshuricus": "Manchurian hare",
            "badger": "Badger",
            "meles leucurus": "Badger",
            "chipmunk": "Chipmunk",
            "siberian chipmunk": "Chipmunk",
            "eutamias sibiricus": "Chipmunk",
            "squirrel": "Squirrel",
            "sciurus vulgaris": "Squirrel",
            "dog": "Dog",
            "domestic dog": "Dog",
            "canis lupus familiaris": "Dog",
            "cattle": "Cattle",
            "cow": "Cattle",
            "bos taurus": "Cattle",
            "horse": "Horse",
            "equus ferus caballus": "Horse",
            "sheep": "Sheep",
            "ovis aries": "Sheep",
            "bird": "Bird",
            "aves sp.": "Bird",
        }
        return canonical_map.get(key, raw[:1].upper() + raw[1:] if raw.islower() else raw)

    def _expand_species_query_tokens(self, species):
        raw = self._normalize_species_text(species)
        if not raw:
            return set()

        alias_groups = [
            {
                "keys": {"siberian tiger", "amur tiger", "tiger", "panthera tigris altaica"},
                "tokens": {"siberian tiger", "amur tiger", "tiger", "panthera tigris altaica"},
            },
            {
                "keys": {"amur leopard", "leopard", "panthera pardus orientalis"},
                "tokens": {"amur leopard", "leopard", "panthera pardus orientalis"},
            },
            {
                "keys": {"wild boar", "boar", "sus scrofa"},
                "tokens": {"wild boar", "boar", "sus scrofa"},
            },
            {
                "keys": {"roe deer", "siberian roe deer", "capreolus pygargus"},
                "tokens": {"roe deer", "siberian roe deer", "capreolus pygargus"},
            },
            {
                "keys": {"sika deer", "cervus nippon"},
                "tokens": {"sika deer", "cervus nippon"},
            },
            {
                "keys": {"musk deer", "moschus moschiferus"},
                "tokens": {"musk deer", "moschus moschiferus"},
            },
            {
                "keys": {"black bear", "asian black bear", "ursus thibetanus"},
                "tokens": {"black bear", "asian black bear", "ursus thibetanus"},
            },
            {
                "keys": {"leopard cat", "prionailurus bengalensis"},
                "tokens": {"leopard cat", "prionailurus bengalensis"},
            },
            {
                "keys": {"red fox", "fox", "vulpes vulpes"},
                "tokens": {"red fox", "fox", "vulpes vulpes"},
            },
            {
                "keys": {"raccoon dog", "nyctereutes procyonoides"},
                "tokens": {"raccoon dog", "nyctereutes procyonoides"},
            },
            {
                "keys": {"badger", "meles leucurus"},
                "tokens": {"badger", "meles leucurus"},
            },
            {
                "keys": {"amur hedgehog", "hedgehog", "erinaceus amurensis"},
                "tokens": {"amur hedgehog", "hedgehog", "erinaceus amurensis"},
            },
            {
                "keys": {"manchurian hare", "hare", "lepus mandshuricus"},
                "tokens": {"manchurian hare", "hare", "lepus mandshuricus"},
            },
            {
                "keys": {"chipmunk", "siberian chipmunk", "eutamias sibiricus"},
                "tokens": {"chipmunk", "siberian chipmunk", "eutamias sibiricus"},
            },
            {
                "keys": {"squirrel", "sciurus vulgaris"},
                "tokens": {"squirrel", "sciurus vulgaris"},
            },
            {
                "keys": {"dog", "domestic dog", "canis lupus familiaris"},
                "tokens": {"dog", "domestic dog", "canis lupus familiaris"},
            },
            {
                "keys": {"cattle", "cow", "cows", "bos taurus"},
                "tokens": {"cattle", "cow", "cows", "bos taurus"},
            },
            {
                "keys": {"horse", "equus ferus caballus"},
                "tokens": {"horse", "equus ferus caballus"},
            },
            {
                "keys": {"sheep", "ovis aries"},
                "tokens": {"sheep", "ovis aries"},
            },
            {
                "keys": {"duck", "ducks", "anas", "mallard"},
                "tokens": {"duck", "ducks", "anas", "mallard"},
            },
            {
                "keys": {"anas platyrhynchos", "mallard"},
                "tokens": {"anas platyrhynchos", "anas", "mallard", "duck"},
            },
            {
                "keys": {"anas strepera", "gadwall"},
                "tokens": {"anas strepera", "anas", "gadwall", "duck"},
            },
            {
                "keys": {"ardea", "heron"},
                "tokens": {"ardea", "heron"},
            },
        ]

        tokens = {raw}
        for group in alias_groups:
            if raw in group["keys"]:
                tokens.update(group["tokens"])

        if "duck" in raw:
            tokens.update({"anas", "duck"})
        if "tiger" in raw:
            tokens.update({"siberian tiger", "tiger", "panthera tigris altaica"})
        if "leopard" in raw and "cat" not in raw:
            tokens.update({"amur leopard", "leopard", "panthera pardus orientalis"})
        if raw in {"deer", "deers", "cervidae"}:
            tokens.update({"roe deer", "sika deer", "musk deer"})
        return {token for token in tokens if token}

    def _species_matches_query(self, metadata, species_query):
        if not species_query:
            return True

        tokens = self._expand_species_query_tokens(species_query)
        if not tokens:
            return True

        species_text = self._normalize_species_text(metadata.get("species"))
        scientific_text = self._normalize_species_text(metadata.get("scientific_name"))
        searchable = [species_text, scientific_text]

        for token in tokens:
            for text in searchable:
                if not text:
                    continue
                if token == "leopard":
                    if text in {"leopard", "amur leopard"}:
                        return True
                    continue
                if token == text or token in text:
                    return True
        return False

    def _coerce_float(self, value):
        try:
            return float(value)
        except Exception:
            return None

    def _normalize_bbox(self, bbox_value=None, metadata=None):
        if isinstance(bbox_value, str):
            text = bbox_value.strip()
            if text:
                try:
                    bbox_value = json.loads(text)
                except Exception:
                    bbox_value = None

        if isinstance(bbox_value, (list, tuple)) and len(bbox_value) == 4:
            numeric = [self._coerce_float(x) for x in bbox_value]
            if all(x is not None for x in numeric):
                return [round(x, 4) for x in numeric]

        metadata = metadata or {}
        x = self._coerce_float(metadata.get("bbox_x"))
        y = self._coerce_float(metadata.get("bbox_y"))
        width = self._coerce_float(metadata.get("bbox_width"))
        height = self._coerce_float(metadata.get("bbox_height"))
        if None not in {x, y, width, height}:
            return [round(x, 4), round(y, 4), round(x + width, 4), round(y + height, 4)]
        return []

    def _extract_confidence_value(self, metadata):
        metadata = metadata or {}
        candidates = [
            metadata.get("confidence"),
            metadata.get("classificationProbability"),
            metadata.get("camtrap_classification_probability"),
        ]
        for value in candidates:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            try:
                numeric = float(text)
                if math.isnan(numeric) or math.isinf(numeric):
                    continue
                return numeric
            except Exception:
                mapped = {
                    "high": 0.9,
                    "medium": 0.6,
                    "low": 0.3,
                }.get(text.lower())
                if mapped is not None:
                    return mapped
        return None

    def _extract_file_reference(self, metadata):
        metadata = metadata or {}
        file_path = (
            str(metadata.get("media_abspath", "")).strip()
            or str(metadata.get("camtrap_file_path", "")).strip()
        )
        if file_path:
            maybe_path = Path(file_path)
            is_url = "://" in file_path or file_path.startswith("https:/") or file_path.startswith("http:/")
            if (not maybe_path.is_absolute()) and (not is_url):
                package_dir = str(metadata.get("camtrap_package_dir", "")).strip()
                if package_dir:
                    maybe_path = Path(package_dir) / file_path
            file_path = file_path if is_url else str(maybe_path)

        if not file_path:
            file_path = self._infer_local_media_path(metadata)

        filename = (
            str(metadata.get("filename", "")).strip()
            or (Path(file_path).name if file_path else "")
        )
        return filename, file_path

    def _infer_local_media_path(self, metadata):
        metadata = metadata or {}
        filename = str(metadata.get("filename", "")).strip()
        if not filename:
            return ""

        cached = self._media_path_cache.get(filename)
        if cached is not None:
            return cached

        candidate_paths = []
        date_text = (
            str(metadata.get("timestamp_original_date", "")).strip()
            or str(metadata.get("timestamp", "")).strip()[:10]
        )
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text):
            candidate_paths.append(Path(TIGER_VIDEO_DIR) / date_text / filename)

        stem_match = re.match(r"(\d{4}-\d{2}-\d{2})_", filename)
        if stem_match:
            candidate_paths.append(Path(TIGER_VIDEO_DIR) / stem_match.group(1) / filename)

        candidate_paths.append(Path(TIGER_VIDEO_DIR) / filename)
        for candidate in candidate_paths:
            if candidate.exists():
                resolved = str(candidate.resolve())
                self._media_path_cache[filename] = resolved
                return resolved

        self._media_path_cache[filename] = ""
        return ""

    def _resolve_camtrap_datapackage_path(self, datapackage_path):
        candidate_text = str(datapackage_path or "").strip()
        if not candidate_text:
            if CAMTRAP_EXAMPLE_DIR.exists():
                return Path(CAMTRAP_EXAMPLE_DIR)
            raise FileNotFoundError("CamtrapDP datapackage path is empty.")

        candidate = Path(candidate_text)
        if candidate.exists():
            return candidate

        lowered = candidate_text.lower()
        if (
            lowered in {"example", "camtrap_example", "camtrapdp_example"}
            or "example" in lowered
            or ("example" in lowered and "dataset" in lowered)
        ) and CAMTRAP_EXAMPLE_DIR.exists():
            return Path(CAMTRAP_EXAMPLE_DIR)

        if candidate_text in {"./datapackage.json", "datapackage.json", "example dataset path or directory"} and CAMTRAP_EXAMPLE_DIR.exists():
            return Path(CAMTRAP_EXAMPLE_DIR)

        fallback_candidates = [
            Path(CAMTRAP_EXAMPLE_DIR.parent) / candidate_text,
            Path(CAMTRAP_EXAMPLE_DIR.parent) / candidate_text / "datapackage.json",
        ]
        for fallback in fallback_candidates:
            if fallback.exists():
                return fallback

        return candidate

    def _resolve_media_dir(self, media_dir):
        media_text = str(media_dir or "").strip()
        if not media_text:
            return Path(DETECTION_INPUT_DIR)

        candidate = Path(media_text)
        if candidate.exists():
            return candidate

        if media_text.lower() in {"input", "default"}:
            return Path(DETECTION_INPUT_DIR)

        return candidate

    def _hour_in_window(self, hour, start_hour=None, end_hour=None):
        if start_hour is None and end_hour is None:
            return True
        if hour is None:
            return False

        start = 0 if start_hour is None else int(start_hour) % 24
        end = 23 if end_hour is None else int(end_hour) % 24

        if start <= end:
            return start <= hour <= end
        return hour >= start or hour <= end

    def _infer_camera_id_from_path(self, media_path):
        candidates = [media_path.stem, media_path.parent.name, media_path.parent.parent.name]
        for candidate in candidates:
            candidate_text = str(candidate or "").strip()
            if not candidate_text:
                continue
            match = re.search(r"(?<!\d)(\d{5,6})(?!\d)", candidate_text)
            if match:
                return match.group(1)
            digits = "".join(ch for ch in candidate_text if ch.isdigit())
            if 5 <= len(digits) <= 8:
                return digits
        return media_path.parent.name or "unknown_camera"

    def _infer_timestamp_from_media_path(self, media_path):
        stem = media_path.stem
        match = re.search(r"(\d{8})[_-]?(\d{6})", stem)
        if match:
            try:
                return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
            except Exception:
                pass
        patterns = [
            ("%Y%m%d_%H%M%S", stem),
            ("%Y%m%d%H%M%S", stem),
        ]
        parts = stem.replace("-", "_").split("_")
        if len(parts) >= 2:
            patterns.append(("%Y%m%d_%H%M%S", f"{parts[0]}_{parts[1]}"))

        for fmt, candidate in patterns:
            try:
                return datetime.strptime(candidate, fmt)
            except Exception:
                continue

        try:
            return datetime.fromtimestamp(media_path.stat().st_mtime)
        except Exception:
            return None

    def _build_record_preview(self, record):
        metadata = record.get("metadata", {}) or {}
        parsed_time = self.safe_parse_timestamp(metadata.get("timestamp"))
        confidence = self._extract_confidence_value(metadata)
        filename, file_path = self._extract_file_reference(metadata)

        return {
            "id": record.get("id"),
            "species": self._canonical_species_name(metadata.get("species")),
            "scientific_name": str(metadata.get("scientific_name", "")).strip(),
            "camera_id": str(metadata.get("camera_id", "")).strip(),
            "timestamp": str(metadata.get("timestamp", "")).strip(),
            "timestamp_source": str(
                metadata.get("timestamp_source", "")
            ).strip(),
            "timestamp_ocr_score": self._coerce_float(
                metadata.get("timestamp_ocr_score")
            ),
            "timestamp_ocr_vote_count": self._coerce_float(
                metadata.get("timestamp_ocr_vote_count")
            ),
            "date": parsed_time.strftime("%Y-%m-%d") if parsed_time else "",
            "hour": parsed_time.hour if parsed_time else None,
            "filename": filename,
            "file_path": file_path,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "perception_model_version": str(
                metadata.get("perception_model_version", "")
            ).strip(),
            "prediction_source": str(
                metadata.get("prediction_source", "")
            ).strip(),
            "frame_offset_seconds": self._coerce_float(
                metadata.get("camtrap_source_frame_offset_seconds")
            ),
            "bbox": self._normalize_bbox(metadata.get("bbox"), metadata=metadata),
            "latitude": self._coerce_float(metadata.get("latitude")),
            "longitude": self._coerce_float(metadata.get("longitude")),
        }

    def summarize_species_inventory(self, top_k=50):
        if self._using_camtrap_backend():
            report = self._load_camtrap_conversion_report()
            if report.get("species_summary"):
                rows = []
                date_values = []
                for species, summary in report["species_summary"].items():
                    row = {
                        "species": species,
                        "record_count": int(summary.get("media_records", 0)),
                        "date_start": summary.get("date_start"),
                        "date_end": summary.get("date_end"),
                    }
                    rows.append(row)
                    if summary.get("date_start"):
                        date_values.append(summary.get("date_start"))
                    if summary.get("date_end"):
                        date_values.append(summary.get("date_end"))
                rows.sort(key=lambda item: item["record_count"], reverse=True)
                return {
                    "species_count": len(rows),
                    "total_record_count": int(sum(row["record_count"] for row in rows)),
                    "date_start": min(date_values) if date_values else None,
                    "date_end": max(date_values) if date_values else None,
                    "species_inventory": rows[:max(int(top_k), 1)],
                    "note": "Species inventory and time range are summarized from the current imported CamtrapDP conversion report.",
                }

            df = self._load_camtrap_records_df()
            if df.empty:
                return {
                    "species_count": 0,
                    "total_record_count": 0,
                    "species_inventory": [],
                    "note": "Species inventory is summarized from the current imported CamtrapDP tabular backend.",
                }
            counts = df["species"].fillna("Unknown").map(self._canonical_species_name).value_counts()
            timestamps = pd.to_datetime(df.get("timestamp"), errors="coerce", utc=True).dropna()
            ranked = [
                {"species": str(species), "record_count": int(count)}
                for species, count in counts.head(max(int(top_k), 1)).items()
            ]
            return {
                "species_count": int(counts.shape[0]),
                "total_record_count": int(counts.sum()),
                "date_start": timestamps.min().strftime("%Y-%m-%dT%H:%M:%SZ") if not timestamps.empty else None,
                "date_end": timestamps.max().strftime("%Y-%m-%dT%H:%M:%SZ") if not timestamps.empty else None,
                "species_inventory": ranked,
                "note": "Species inventory and time range are summarized from the current imported CamtrapDP tabular backend.",
            }

        counter = Counter()
        for record in self.get_all_records():
            metadata = record.get("metadata", {}) or {}
            species = self._canonical_species_name(metadata.get("species"))
            counter[species] += 1

        ranked = [
            {"species": species, "record_count": count}
            for species, count in counter.most_common(max(int(top_k), 1))
        ]
        return {
            "species_count": len(counter),
            "total_record_count": sum(counter.values()),
            "species_inventory": ranked,
            "note": "Species inventory is summarized from the currently active standardized data source.",
        }

    def evaluate_perception_benchmark(
        self,
        annotation_path,
        output_dir,
        media_root=None,
        media_path_column="media_path",
        true_label_column="true_label",
        predicted_label_column="predicted_label",
        confidence_column="confidence",
        group_column="camera_id",
        sample_id_column="sample_id",
        run_inference=True,
        country=None,
        admin1_region=None,
        model=None,
        video_frame_interval_seconds=1.0,
        max_video_frames=12,
        max_samples=None,
        label_aliases=None,
        confidence_threshold=0.0,
        calibration_bins=10,
        bootstrap_iters=1000,
        random_seed=42,
        camera_disjoint_verified=False,
        human_review_status="unverified",
        split_note=None,
        development_camera_ids_path=None,
        annotation_manifest_path=None,
        minimum_class_support=20,
        resume=True,
    ):
        """Run and score a generic, media-linked perception benchmark."""
        return run_perception_benchmark(
            annotation_path=annotation_path,
            output_dir=output_dir,
            media_root=media_root,
            media_path_column=media_path_column,
            true_label_column=true_label_column,
            predicted_label_column=predicted_label_column,
            confidence_column=confidence_column,
            group_column=group_column,
            sample_id_column=sample_id_column,
            run_inference=run_inference,
            country=country,
            admin1_region=admin1_region,
            model=model,
            video_frame_interval_seconds=video_frame_interval_seconds,
            max_video_frames=max_video_frames,
            max_samples=max_samples,
            label_aliases=label_aliases,
            confidence_threshold=confidence_threshold,
            calibration_bins=calibration_bins,
            bootstrap_iters=bootstrap_iters,
            random_seed=random_seed,
            camera_disjoint_verified=camera_disjoint_verified,
            human_review_status=human_review_status,
            split_note=split_note,
            development_camera_ids_path=development_camera_ids_path,
            annotation_manifest_path=annotation_manifest_path,
            minimum_class_support=minimum_class_support,
            resume=resume,
        )

    def _resolve_species_list(self, species_list=None, top_k=10, include_unknown=False):
        if isinstance(species_list, str):
            requested = [item.strip() for item in re.split(r"[,，;；]", species_list) if item.strip()]
        elif isinstance(species_list, (list, tuple, set)):
            requested = [str(item).strip() for item in species_list if str(item).strip()]
        else:
            requested = []

        if requested:
            return requested

        inventory = self.summarize_species_inventory(top_k=max(int(top_k), 1) + 5)
        resolved = []
        for item in inventory.get("species_inventory", []):
            species = item.get("species")
            if not include_unknown and str(species).strip().lower() == "unknown":
                continue
            resolved.append(species)
            if len(resolved) >= max(int(top_k), 1):
                break
        return resolved

    def _collect_exact_independent_event_rows(self, species_list=None, include_unknown=False, min_gap_minutes=30):
        if self._using_camtrap_backend():
            df = self._load_camtrap_event_df(min_gap_minutes=min_gap_minutes)
            if df.empty:
                return []
            allowed_species = None
            if species_list is not None:
                allowed_species = {
                    self._canonical_species_name(species)
                    for species in species_list
                    if include_unknown or self._canonical_species_name(species).lower() != "unknown"
                }
            rows = []
            for item in df.itertuples(index=False):
                row = item._asdict()
                species = self._canonical_species_name(row.get("species") or row.get("scientificName"))
                if not include_unknown and species.lower() == "unknown":
                    continue
                if allowed_species is not None and species not in allowed_species:
                    continue
                event_dt = row.get("event_time_dt")
                if pd.isna(event_dt):
                    event_dt = self.safe_parse_timestamp(row.get("event_time"))
                if event_dt is None or pd.isna(event_dt):
                    continue
                camera_id = str(row.get("camera_id") or row.get("cameraID") or str(row.get("deploymentID", "")).replace("dep_", "", 1)).strip()
                if not camera_id:
                    continue
                rows.append({
                    "species": species,
                    "camera_id": camera_id,
                    "event_time": event_dt.to_pydatetime() if hasattr(event_dt, "to_pydatetime") else event_dt,
                    "start_time": self.safe_parse_timestamp(row.get("start_time") or row.get("eventStart")) or (event_dt.to_pydatetime() if hasattr(event_dt, "to_pydatetime") else event_dt),
                    "end_time": self.safe_parse_timestamp(row.get("end_time") or row.get("eventEnd")) or (event_dt.to_pydatetime() if hasattr(event_dt, "to_pydatetime") else event_dt),
                    "latitude": self._coerce_float(row.get("latitude")),
                    "longitude": self._coerce_float(row.get("longitude")),
                    "event_record_count": int(row.get("event_record_count") or 1),
                    "record_ids": [],
                })
            return rows

        min_gap_minutes = max(int(min_gap_minutes), 0)
        allowed_species = None
        if species_list is not None:
            allowed_species = {
                self._canonical_species_name(species)
                for species in species_list
                if include_unknown or self._canonical_species_name(species).lower() != "unknown"
            }

        grouped = defaultdict(list)
        for record in self.get_all_records():
            metadata = record.get("metadata", {}) or {}
            species = self._canonical_species_name(metadata.get("species"))
            if not include_unknown and species.lower() == "unknown":
                continue
            if allowed_species is not None and species not in allowed_species:
                continue

            camera_id = str(metadata.get("camera_id", "")).strip()
            parsed_time = self.safe_parse_timestamp(metadata.get("timestamp"))
            if not camera_id or parsed_time is None:
                continue
            grouped[(species, camera_id)].append({
                "species": species,
                "camera_id": camera_id,
                "parsed_time": parsed_time,
                "record_id": record.get("id"),
            })

        event_rows = []
        for (species, camera_id), records in grouped.items():
            records.sort(key=lambda item: (item["parsed_time"], str(item.get("record_id", ""))))
            current = []
            for record in records:
                if not current:
                    current = [record]
                    continue
                gap_minutes = (record["parsed_time"] - current[-1]["parsed_time"]).total_seconds() / 60.0
                if gap_minutes <= min_gap_minutes:
                    current.append(record)
                else:
                    event_rows.append(self._summarize_exact_event(species, camera_id, current))
                    current = [record]
            if current:
                event_rows.append(self._summarize_exact_event(species, camera_id, current))
        return event_rows

    def _summarize_exact_event(self, species, camera_id, event_records):
        start_dt = event_records[0]["parsed_time"]
        end_dt = event_records[-1]["parsed_time"]
        midpoint_dt = start_dt + (end_dt - start_dt) / 2
        return {
            "species": species,
            "camera_id": camera_id,
            "event_time": midpoint_dt,
            "start_time": start_dt,
            "end_time": end_dt,
            "event_record_count": len(event_records),
            "record_ids": [item.get("record_id") for item in event_records],
        }

    def summarize_community_composition(
        self,
        top_k=20,
        include_unknown=False,
        min_gap_minutes=30,
        abundance_metric="rai",
        min_effort_days_for_rai=7,
        min_observed_span_days_for_rai=7,
        min_independent_events_for_rai=5,
    ):
        """
        Summarize community composition with ecological effort correction.

        Raw videos are first collapsed into independent events per species and camera.
        Relative abundance defaults to RAI: independent events per 100 camera-days
        over cameras where the species was observed. Raw record counts are retained
        in the output only as provenance.
        """
        if self._using_camtrap_backend():
            report = self._load_camtrap_conversion_report()
            raw_counter = Counter()
            if report.get("species_summary"):
                raw_counter.update({
                    species: int(summary.get("media_records", 0))
                    for species, summary in report["species_summary"].items()
                })
            else:
                records_df = self._load_camtrap_records_df()
                raw_counter = Counter(records_df["species"].map(self._canonical_species_name)) if not records_df.empty else Counter()
            event_df = self._load_camtrap_event_df()
            if not raw_counter and event_df.empty:
                return {
                    "total_record_count": 0,
                    "total_source_record_count": 0,
                    "species_richness": 0,
                    "shannon_index": 0.0,
                    "simpson_index": 0.0,
                    "top_species": [],
                    "include_unknown": bool(include_unknown),
                    "min_gap_minutes": max(int(min_gap_minutes), 0),
                    "abundance_metric": abundance_metric,
                    "note": "No CamtrapDP tabular records are available.",
                }

            if not include_unknown:
                raw_counter = Counter({k: v for k, v in raw_counter.items() if str(k).lower() != "unknown"})

            camera_effort_days = self._camera_effort_days_from_records()
            independent_counter = Counter()
            species_cameras = defaultdict(set)
            species_event_times = defaultdict(list)
            if not event_df.empty:
                df = event_df.copy()
                if "species" not in df.columns:
                    df["species"] = df.get("scientificName", "").map(self._canonical_species_name)
                df["species"] = df["species"].map(self._canonical_species_name)
                if not include_unknown:
                    df = df[df["species"].str.lower() != "unknown"]
                camera_col = "camera_id" if "camera_id" in df.columns else "cameraID"
                independent_counter.update(df["species"].value_counts().to_dict())
                for species, group in df.groupby("species"):
                    species_cameras[species] = set(group[camera_col].astype(str))
                    times = pd.to_datetime(group.get("event_time", group.get("eventStart")), errors="coerce").dropna()
                    species_event_times[species] = [ts.to_pydatetime() for ts in times]

            species_seen = set(raw_counter) | set(independent_counter)
            metric_rows = []
            abundance_metric = str(abundance_metric or "rai").strip().lower()
            total_effort_days = float(sum(camera_effort_days.values()))
            total_effort_days = max(total_effort_days, 1.0 if sum(independent_counter.values()) else 0.0)
            for species in sorted(species_seen):
                event_count = int(independent_counter.get(species, 0))
                raw_count = int(raw_counter.get(species, 0))
                effort_days = total_effort_days
                rai = (event_count / effort_days * 100.0) if effort_days > 0 else 0.0
                times = species_event_times.get(species, [])
                observed_span_days = (max(times).date() - min(times).date()).days + 1 if times else 0
                low_effort_flag = bool(
                    event_count
                    and (
                        observed_span_days < int(min_observed_span_days_for_rai)
                        or effort_days < float(min_effort_days_for_rai)
                        or event_count < int(min_independent_events_for_rai)
                    )
                )
                if abundance_metric == "independent_events":
                    metric_value = float(event_count)
                elif abundance_metric == "raw_records":
                    metric_value = float(raw_count)
                else:
                    metric_value = 0.0 if low_effort_flag else float(rai)
                metric_rows.append({
                    "species": species,
                    "record_count": event_count,
                    "independent_event_count": event_count,
                    "source_record_count": raw_count,
                    "camera_count": len(species_cameras.get(species, set())),
                    "camera_effort_days": round(effort_days, 3),
                    "rai_per_100_camera_days": round(rai, 4),
                    "observed_span_days": int(observed_span_days),
                    "low_effort_flag": low_effort_flag,
                    "excluded_from_adjusted_abundance": bool(abundance_metric == "rai" and low_effort_flag),
                    "metric_value": metric_value,
                })

            metric_rows.sort(key=lambda item: (item["metric_value"], item["independent_event_count"]), reverse=True)
            total_metric = sum(row["metric_value"] for row in metric_rows)
            proportions = [row["metric_value"] / total_metric for row in metric_rows if total_metric > 0]
            shannon_index = -sum(p * math.log(p) for p in proportions if p > 0)
            simpson_index = 1.0 - sum(p * p for p in proportions)
            species_rows = []
            for row in metric_rows[:max(int(top_k), 1)]:
                row = row.copy()
                row["relative_abundance"] = round(row["metric_value"] / total_metric, 4) if total_metric else 0.0
                species_rows.append(row)

            return {
                "total_record_count": int(sum(independent_counter.values())),
                "total_source_record_count": int(sum(raw_counter.values())),
                "species_richness": len([row for row in metric_rows if row["independent_event_count"] > 0]),
                "shannon_index": round(shannon_index, 4),
                "simpson_index": round(simpson_index, 4),
                "top_species": species_rows,
                "include_unknown": bool(include_unknown),
                "min_gap_minutes": max(int(min_gap_minutes), 0),
                "abundance_metric": abundance_metric,
                "excluded_low_effort_species_count": sum(1 for row in metric_rows if row.get("excluded_from_adjusted_abundance")),
                "total_camera_effort_days": round(total_effort_days, 3),
                "effort_basis": self._camera_effort_basis(),
                "rai_inference_status": "exploratory" if self._camera_effort_basis() == "detection_span_proxy" else "supported_by_deployment_intervals",
                "note": (
                    "Community composition is computed from CamtrapDP media records and independent events; RAI uses camera deployment spans. "
                    + ("Because those spans were inferred from detections, RAI is exploratory until camera operation logs are supplied." if self._camera_effort_basis() == "detection_span_proxy" else "Deployment intervals are documented independently of detections.")
                ),
            }

        raw_counter = Counter()
        camera_timestamps = defaultdict(list)
        species_seen = set()

        for record in self.get_all_records():
            metadata = record.get("metadata", {}) or {}
            species = self._canonical_species_name(metadata.get("species"))
            if not include_unknown and species.lower() == "unknown":
                continue

            camera_id = str(metadata.get("camera_id", "")).strip()
            parsed_time = self.safe_parse_timestamp(metadata.get("timestamp"))
            raw_counter[species] += 1
            species_seen.add(species)
            if camera_id and parsed_time is not None:
                camera_timestamps[camera_id].append(parsed_time)

        camera_effort_days = {}
        for camera_id, timestamps in camera_timestamps.items():
            if not timestamps:
                continue
            start_dt = min(timestamps)
            end_dt = max(timestamps)
            camera_effort_days[camera_id] = max((end_dt.date() - start_dt.date()).days + 1, 1)

        min_gap_minutes = max(int(min_gap_minutes), 0)
        independent_counter = Counter()
        species_cameras = defaultdict(set)
        species_event_times = defaultdict(list)
        for event in self._collect_exact_independent_event_rows(
            species_list=sorted(species_seen),
            include_unknown=include_unknown,
            min_gap_minutes=min_gap_minutes,
        ):
            species = event["species"]
            camera_id = str(event.get("camera_id", "")).strip()
            event_time = event.get("event_time")
            independent_counter[species] += 1
            if camera_id:
                species_cameras[species].add(camera_id)
            if event_time is not None:
                species_event_times[species].append(event_time)

        abundance_metric = str(abundance_metric or "rai").strip().lower()
        min_effort_days_for_rai = max(float(min_effort_days_for_rai), 0.0)
        min_observed_span_days_for_rai = max(int(min_observed_span_days_for_rai), 0)
        min_independent_events_for_rai = max(int(min_independent_events_for_rai), 1)

        total_effort_days = float(sum(camera_effort_days.values()))
        total_effort_days = max(total_effort_days, 1.0 if sum(independent_counter.values()) else 0.0)
        metric_rows = []
        for species in sorted(species_seen):
            event_count = int(independent_counter.get(species, 0))
            raw_count = int(raw_counter.get(species, 0))
            effort_days = total_effort_days
            rai = (event_count / effort_days * 100.0) if effort_days > 0 else 0.0
            times = species_event_times.get(species, [])
            observed_span_days = (max(times).date() - min(times).date()).days + 1 if times else 0
            low_effort_flag = bool(
                event_count
                and (
                    observed_span_days < min_observed_span_days_for_rai
                    or effort_days < min_effort_days_for_rai
                    or event_count < min_independent_events_for_rai
                )
            )
            if abundance_metric == "independent_events":
                metric_value = float(event_count)
            elif abundance_metric == "raw_records":
                metric_value = float(raw_count)
            else:
                metric_value = 0.0 if low_effort_flag else float(rai)
            metric_rows.append({
                "species": species,
                "record_count": event_count,
                "independent_event_count": event_count,
                "source_record_count": raw_count,
                "camera_count": len(species_cameras.get(species, set())),
                "camera_effort_days": round(effort_days, 3),
                "rai_per_100_camera_days": round(rai, 4),
                "observed_span_days": int(observed_span_days),
                "low_effort_flag": low_effort_flag,
                "excluded_from_adjusted_abundance": bool(abundance_metric == "rai" and low_effort_flag),
                "metric_value": metric_value,
            })

        metric_rows.sort(key=lambda item: (item["metric_value"], item["independent_event_count"]), reverse=True)
        total_metric = sum(row["metric_value"] for row in metric_rows)
        proportions = [row["metric_value"] / total_metric for row in metric_rows if total_metric > 0]
        shannon_index = -sum(p * math.log(p) for p in proportions if p > 0)
        simpson_index = 1.0 - sum(p * p for p in proportions)

        species_rows = []
        for row in metric_rows[:max(int(top_k), 1)]:
            row = row.copy()
            row["relative_abundance"] = round(row["metric_value"] / total_metric, 4) if total_metric else 0.0
            species_rows.append(row)

        return {
            "total_record_count": int(sum(independent_counter.values())),
            "total_source_record_count": int(sum(raw_counter.values())),
            "species_richness": len([row for row in metric_rows if row["independent_event_count"] > 0]),
            "shannon_index": round(shannon_index, 4),
            "simpson_index": round(simpson_index, 4),
            "top_species": species_rows,
            "include_unknown": bool(include_unknown),
            "min_gap_minutes": min_gap_minutes,
            "abundance_metric": abundance_metric,
            "min_effort_days_for_rai": min_effort_days_for_rai,
            "min_observed_span_days_for_rai": min_observed_span_days_for_rai,
            "min_independent_events_for_rai": min_independent_events_for_rai,
            "excluded_low_effort_species_count": sum(
                1 for row in metric_rows if row.get("excluded_from_adjusted_abundance")
            ),
            "total_camera_effort_days": round(total_effort_days, 3),
            "effort_basis": self._camera_effort_basis(),
            "rai_inference_status": "exploratory",
            "note": (
                "Community composition is corrected by collapsing same-species same-camera records within "
                f"{min_gap_minutes} minutes into independent events. Relative abundance defaults to RAI "
                "(independent events per 100 camera-days). Species below the minimum effort/span/event-count "
                "threshold are retained for richness but excluded from adjusted abundance to avoid inflating "
                "short or tiny samples."
            ),
        }

    def _get_filtered_records(
        self,
        species=None,
        camera_id=None,
        start_date=None,
        end_date=None,
        start_hour=None,
        end_hour=None,
        center_camera_id=None,
        radius_km=None,
        min_confidence=None,
        limit=None,
        sort_by="timestamp_asc",
        allow_collection_fallback=True,
    ):
        if self._using_camtrap_backend():
            active_backend = "camtrap_dp_tabular"
        elif self._prefer_json_backend:
            active_backend = "json"
        else:
            active_backend = "collection"
        species_text = str(species or "").strip().lower()
        camera_text = str(camera_id or "").strip()
        start_dt = self.safe_parse_timestamp(start_date) if start_date else None
        end_dt = self.safe_parse_timestamp(end_date) if end_date else None
        min_conf = float(min_confidence) if min_confidence is not None else None
        radius_value = float(radius_km) if radius_km is not None else None
        camera_locations = self.get_camera_locations()
        center_location = None
        if center_camera_id:
            center_location = camera_locations.get(str(center_camera_id).strip())

        if self._using_camtrap_backend():
            df = self._load_camtrap_records_df()
            if df.empty:
                return {
                    "matches": [],
                    "matched_count_total": 0,
                    "backend_used": active_backend,
                    "fallback_used": False,
                    "active_backend": active_backend,
                }

            work = df.copy()
            work["_timestamp"] = self._parse_camtrap_timestamp_series(
                work["timestamp"],
                local_clock=True,
            )

            if species_text:
                tokens = self._expand_species_query_tokens(species_text)
                species_mask = work["species"].astype(str).str.lower().isin(tokens)
                if "scientificName" in work.columns:
                    species_mask = species_mask | work["scientificName"].astype(str).str.lower().isin(tokens)
                work = work[species_mask]

            if camera_text:
                work = work[work["cameraID"].astype(str) == camera_text]

            if start_dt is not None:
                work = work[work["_timestamp"].notna() & (work["_timestamp"] >= start_dt)]
            if end_dt is not None:
                work = work[work["_timestamp"].notna() & (work["_timestamp"] <= end_dt)]

            if start_hour is not None or end_hour is not None:
                start = 0 if start_hour is None else int(start_hour) % 24
                end = 23 if end_hour is None else int(end_hour) % 24
                hours = work["_timestamp"].dt.hour
                if start <= end:
                    hour_mask = hours.between(start, end)
                else:
                    hour_mask = (hours >= start) | (hours <= end)
                work = work[work["_timestamp"].notna() & hour_mask]

            if min_conf is not None:
                work["_confidence"] = pd.to_numeric(work.get("classificationProbability", ""), errors="coerce")
                work = work[work["_confidence"].notna() & (work["_confidence"] >= min_conf)]
            elif sort_by == "confidence_desc":
                work["_confidence"] = pd.to_numeric(work.get("classificationProbability", ""), errors="coerce")

            if center_location is not None and radius_value is not None:
                distances = []
                keep = []
                for row in work.itertuples(index=False):
                    row_data = row._asdict()
                    lat = self._coerce_float(row_data.get("latitude"))
                    lon = self._coerce_float(row_data.get("longitude"))
                    camera_loc = camera_locations.get(str(row_data.get("cameraID", "")).strip(), {})
                    if lat is None:
                        lat = camera_loc.get("latitude")
                    if lon is None:
                        lon = camera_loc.get("longitude")
                    if lat is None or lon is None:
                        distances.append(None)
                        keep.append(False)
                        continue
                    distance = self.calculate_distance_km(
                        center_location["latitude"],
                        center_location["longitude"],
                        lat,
                        lon,
                    )
                    distances.append(round(distance, 4))
                    keep.append(distance <= radius_value)
                work = work.assign(_distance_km=distances)
                work = work[keep]

            matched_count = int(len(work))
            if sort_by == "confidence_desc":
                work = work.sort_values(
                    by=["_confidence", "_timestamp", "record_id"],
                    ascending=[False, True, True],
                    na_position="last",
                )
            elif sort_by == "timestamp_desc":
                work = work.sort_values(by=["_timestamp", "record_id"], ascending=[False, False], na_position="last")
            else:
                work = work.sort_values(by=["_timestamp", "record_id"], ascending=[True, True], na_position="last")

            returned_df = work.head(max(int(limit), 0)) if limit is not None else work
            records = self._records_from_camtrap_frame(returned_df)
            distances = returned_df["_distance_km"].tolist() if "_distance_km" in returned_df.columns else [None] * len(records)
            matched = []
            for record, distance_km in zip(records, distances):
                metadata = record.get("metadata", {}) or {}
                matched.append({
                    "record": record,
                    "parsed_time": self.safe_parse_timestamp(metadata.get("timestamp")),
                    "confidence": self._extract_confidence_value(metadata),
                    "distance_km": distance_km,
                    "backend_used": active_backend,
                })
            return {
                "matches": matched,
                "matched_count_total": matched_count,
                "backend_used": active_backend,
                "fallback_used": False,
                "active_backend": active_backend,
            }

        def _filter_from_records(records, backend_label):
            matched_rows = []
            for record in records:
                metadata = record.get("metadata", {}) or {}
                record_camera = str(metadata.get("camera_id", "")).strip()

                if species_text and not self._species_matches_query(metadata, species_text):
                    continue
                if camera_text and record_camera != camera_text:
                    continue

                parsed_time = self.safe_parse_timestamp(metadata.get("timestamp"))
                if start_dt and (parsed_time is None or parsed_time < start_dt):
                    continue
                if end_dt and (parsed_time is None or parsed_time > end_dt):
                    continue
                if (start_hour is not None or end_hour is not None) and not self._hour_in_window(
                    parsed_time.hour if parsed_time else None,
                    start_hour=start_hour,
                    end_hour=end_hour,
                ):
                    continue

                confidence = self._extract_confidence_value(metadata)
                if min_conf is not None and (confidence is None or confidence < min_conf):
                    continue

                distance_km = None
                if center_location is not None and radius_value is not None:
                    lat = self._coerce_float(metadata.get("latitude"))
                    lon = self._coerce_float(metadata.get("longitude"))
                    if lat is None or lon is None:
                        camera_loc = camera_locations.get(record_camera, {})
                        lat = camera_loc.get("latitude")
                        lon = camera_loc.get("longitude")
                    if lat is None or lon is None:
                        continue
                    distance_km = self.calculate_distance_km(
                        center_location["latitude"],
                        center_location["longitude"],
                        lat,
                        lon,
                    )
                    if distance_km > radius_value:
                        continue

                matched_rows.append({
                    "record": record,
                    "parsed_time": parsed_time,
                    "confidence": confidence,
                    "distance_km": round(distance_km, 4) if distance_km is not None else None,
                    "backend_used": backend_label,
                })
            return matched_rows

        active_records = self.get_all_records()
        matched = _filter_from_records(active_records, active_backend)
        fallback_used = False
        if (
            not matched
            and allow_collection_fallback
            and active_backend == "json"
            and self.collection is not None
        ):
            matched = _filter_from_records(self._get_collection_records(), "collection_fallback")
            fallback_used = bool(matched)

        if sort_by == "confidence_desc":
            matched.sort(
                key=lambda item: (
                    -(item["confidence"] if item["confidence"] is not None else -1.0),
                    item["parsed_time"] or datetime.min,
                    str(item["record"].get("id", "")),
                )
            )
        elif sort_by == "timestamp_desc":
            matched.sort(
                key=lambda item: (
                    item["parsed_time"] or datetime.min,
                    str(item["record"].get("id", "")),
                ),
                reverse=True,
            )
        else:
            matched.sort(
                key=lambda item: (
                    item["parsed_time"] or datetime.min,
                    str(item["record"].get("id", "")),
                )
            )

        returned = matched[: max(int(limit), 0)] if limit is not None else matched
        return {
            "matches": returned,
            "matched_count_total": len(matched),
            "backend_used": matched[0]["backend_used"] if matched else active_backend,
            "fallback_used": fallback_used,
            "active_backend": active_backend,
        }

    def get_all_records(self):
        """
        Return all records from the current database or explicitly selected backend.
        """
        if self._using_camtrap_backend():
            return self._records_from_camtrap_frame(self._load_camtrap_records_df())

        if self._prefer_json_backend:
            return self._load_records_from_json()

        records = self._get_collection_records()
        if records:
            return records

        json_records = self._load_records_from_json()
        if json_records:
            return json_records

        raise RuntimeError(
            f"No records are available from Chroma collection {DEFAULT_COLLECTION_NAME!r} "
            f"or JSON directory {self._json_data_dir}. Run the JSON-to-DB sync before starting the agent."
        )

    def query_by_species(self, species):
        """
        Filter records by species, case-insensitively.
        """
        species = species.strip().lower()
        if self._using_camtrap_backend():
            df = self._load_camtrap_records_df()
            if df.empty:
                return []
            tokens = self._expand_species_query_tokens(species)
            mask = df["species"].astype(str).str.lower().isin(tokens)
            if "scientificName" in df.columns:
                mask = mask | df["scientificName"].astype(str).str.lower().isin(tokens)
            return self._records_from_camtrap_frame(df[mask].copy())

        all_records = self.get_all_records()

        matched = []
        for r in all_records:
            if self._species_matches_query(r["metadata"], species):
                matched.append(r)

        return matched

    def query_by_camera(self, camera_id):
        """
        Filter records by camera identifier.
        """
        camera_id = str(camera_id).strip()
        if self._using_camtrap_backend():
            df = self._load_camtrap_records_df()
            if df.empty:
                return []
            return self._records_from_camtrap_frame(df[df["cameraID"].astype(str) == camera_id].copy())

        all_records = self.get_all_records()

        matched = []
        for r in all_records:
            meta_camera = str(r["metadata"].get("camera_id", "")).strip()
            if meta_camera == camera_id:
                matched.append(r)

        return matched

    def filter_records(
        self,
        species=None,
        camera_id=None,
        start_date=None,
        end_date=None,
        start_hour=None,
        end_hour=None,
        center_camera_id=None,
        radius_km=None,
        min_confidence=None,
        limit=100,
        sort_by="timestamp_asc",
    ):
        filtered_ctx = self._get_filtered_records(
            species=species,
            camera_id=camera_id,
            start_date=start_date,
            end_date=end_date,
            start_hour=start_hour,
            end_hour=end_hour,
            center_camera_id=center_camera_id,
            radius_km=radius_km,
            min_confidence=min_confidence,
            limit=None,
            sort_by=sort_by,
        )
        matched = filtered_ctx["matches"]
        returned = matched[: max(int(limit), 0)] if limit is not None else matched

        preview = []
        for item in returned:
            row = self._build_record_preview(item["record"])
            if item["distance_km"] is not None:
                row["distance_km"] = item["distance_km"]
            preview.append(row)

        return {
            "filters": {
                "species": species,
                "camera_id": camera_id,
                "start_date": start_date,
                "end_date": end_date,
                "start_hour": start_hour,
                "end_hour": end_hour,
                "center_camera_id": center_camera_id,
                "radius_km": radius_km,
                "min_confidence": min_confidence,
                "sort_by": sort_by,
            },
            "matched_count": filtered_ctx["matched_count_total"],
            "returned_count": len(preview),
            "backend_used": filtered_ctx["backend_used"],
            "fallback_used": filtered_ctx["fallback_used"],
            "active_backend": filtered_ctx["active_backend"],
            "records": preview,
            "note": (
                "These records were filtered from the active standardized data source. "
                "The same filtering logic is shared by browse and evidence lookup tools."
            ),
        }

    def browse_records(
        self,
        species=None,
        camera_id=None,
        start_date=None,
        end_date=None,
        start_hour=None,
        end_hour=None,
        center_camera_id=None,
        radius_km=None,
        min_confidence=None,
        limit=20,
        sort_by="timestamp_asc",
    ):
        filtered = self.filter_records(
            species=species,
            camera_id=camera_id,
            start_date=start_date,
            end_date=end_date,
            start_hour=start_hour,
            end_hour=end_hour,
            center_camera_id=center_camera_id,
            radius_km=radius_km,
            min_confidence=min_confidence,
            limit=limit,
            sort_by=sort_by,
        )
        filtered["note"] = (
            "Browse mode returns sample-level records for manual inspection, closer to a media browser than a summary analysis."
        )
        return filtered

    def export_media_by_species(
        self,
        species=None,
        output_dir=None,
        use_symlinks=True,
        min_confidence=None,
        limit=None,
    ):
        filtered_ctx = self._get_filtered_records(
            species=species,
            min_confidence=min_confidence,
            limit=None,
            sort_by="timestamp_asc",
        )
        matched = filtered_ctx["matches"]
        if limit is not None:
            matched = matched[: max(int(limit), 0)]

        export_root = self._make_unique_dir(
            MEDIA_EXPORTS_DIR,
            self._slugify_text(species or "all_species_media"),
        ) if output_dir is None else Path(output_dir)
        export_root.mkdir(parents=True, exist_ok=True)

        manifest_rows = []
        exported_count = 0
        skipped_count = 0
        species_dirs = {}

        for item in matched:
            record = item["record"]
            preview = self._build_record_preview(record)
            species_name = preview["species"] or "Unknown"
            species_dir = export_root / self._slugify_text(species_name)
            species_dir.mkdir(parents=True, exist_ok=True)
            species_dirs[species_name] = str(species_dir)

            source_path = preview["file_path"]
            export_path = ""
            status = "skipped_missing_local_media"

            if source_path and Path(source_path).exists():
                src = Path(source_path)
                target_name = src.name
                destination = species_dir / target_name
                if destination.exists():
                    destination = species_dir / f"{self._slugify_text(str(record.get('id', 'record')))}_{src.name}"

                if use_symlinks:
                    destination.symlink_to(src)
                else:
                    shutil.copy2(src, destination)

                export_path = str(destination)
                status = "exported_symlink" if use_symlinks else "exported_copy"
                exported_count += 1
            else:
                skipped_count += 1

            manifest_rows.append({
                "id": preview["id"],
                "species": species_name,
                "camera_id": preview["camera_id"],
                "timestamp": preview["timestamp"],
                "filename": preview["filename"],
                "source_path": source_path,
                "export_path": export_path,
                "status": status,
                "confidence": preview["confidence"],
                "latitude": preview["latitude"],
                "longitude": preview["longitude"],
            })

        manifest_path = export_root / "media_manifest.csv"
        self._write_csv_rows(
            manifest_path,
            [
                "id", "species", "camera_id", "timestamp", "filename",
                "source_path", "export_path", "status", "confidence",
                "latitude", "longitude",
            ],
            manifest_rows,
        )

        return {
            "output_dir": str(export_root),
            "manifest_path": str(manifest_path),
            "matched_count": len(matched),
            "exported_count": exported_count,
            "skipped_count": skipped_count,
            "backend_used": filtered_ctx["backend_used"],
            "fallback_used": filtered_ctx["fallback_used"],
            "use_symlinks": bool(use_symlinks),
            "species_dirs": species_dirs,
            "note": (
                "Media were organized by species for local browsing or handoff. "
                "Records without resolvable local media paths are preserved in the manifest with skip status."
            ),
        }

    def evidence_lookup(
        self,
        species=None,
        camera_id=None,
        start_date=None,
        end_date=None,
        start_hour=None,
        end_hour=None,
        center_camera_id=None,
        radius_km=None,
        min_confidence=None,
        limit=5,
        sort_by="confidence_desc",
    ):
        filtered_ctx = self._get_filtered_records(
            species=species,
            camera_id=camera_id,
            start_date=start_date,
            end_date=end_date,
            start_hour=start_hour,
            end_hour=end_hour,
            center_camera_id=center_camera_id,
            radius_km=radius_km,
            min_confidence=min_confidence,
            limit=None,
            sort_by=sort_by,
        )
        matched = filtered_ctx["matches"]
        returned = matched[: max(int(limit), 0)] if limit is not None else matched

        evidence_samples = []
        for item in returned:
            preview = self._build_record_preview(item["record"])
            preview["distance_km"] = item["distance_km"]
            evidence_samples.append(preview)

        return {
            "filters": {
                "species": species,
                "camera_id": camera_id,
                "start_date": start_date,
                "end_date": end_date,
                "start_hour": start_hour,
                "end_hour": end_hour,
                "center_camera_id": center_camera_id,
                "radius_km": radius_km,
                "min_confidence": min_confidence,
                "sort_by": sort_by,
            },
            "matched_count": filtered_ctx["matched_count_total"],
            "returned_count": len(evidence_samples),
            "backend_used": filtered_ctx["backend_used"],
            "fallback_used": filtered_ctx["fallback_used"],
            "active_backend": filtered_ctx["active_backend"],
            "evidence_samples": evidence_samples,
            "note": (
                "Evidence lookup links analysis-ready records back to file-level media references, so agent conclusions can be traced to concrete images or videos."
            ),
        }

    def safe_parse_timestamp(self, timestamp_str):
        """
        Parse a timestamp string into a datetime; return None on failure.
        Common project timestamp formats are supported.
        """
        if timestamp_str is None:
            return None

        ts = str(timestamp_str).strip()
        if not ts:
            return None

        candidate_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M",
            "%Y%m%d_%H%M%S",
            "%Y%m%d%H%M%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d"
        ]

        for fmt in candidate_formats:
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                pass

        iso_candidate = ts.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(iso_candidate)
            if parsed.tzinfo is not None:
                if self._using_camtrap_backend():
                    try:
                        parsed = parsed.astimezone(
                            ZoneInfo(self._camtrap_source_timezone())
                        )
                    except (KeyError, ValueError):
                        parsed = parsed.astimezone(timezone.utc)
                return parsed.replace(tzinfo=None)
            return parsed
        except ValueError:
            pass

        return None

    def get_species_records_with_parsed_time(self, species):
        """
        Return records for a species with an added parsed_time field.
        """
        records = self.query_by_species(species)
        result = []

        for r in records:
            meta = r["metadata"]
            ts = meta.get("timestamp")
            parsed_dt = self.safe_parse_timestamp(ts)

            new_r = {
                "id": r["id"],
                "document": r["document"],
                "metadata": meta,
                "parsed_time": parsed_dt
            }
            result.append(new_r)

        return result

    def _has_explicit_clock_time(self, timestamp_str):
        if timestamp_str is None:
            return False
        return ":" in str(timestamp_str).strip()

    def _is_truthy_flag(self, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def is_synthetic_timestamp_metadata(self, metadata):
        metadata = metadata or {}
        if self._is_truthy_flag(metadata.get("timestamp_is_synthetic")):
            return True

        source = str(metadata.get("timestamp_source", "")).strip().lower()
        return source.startswith("synthetic")

    def summarize_timestamp_sources(self, records):
        summary = {
            "valid_time_records": 0,
            "records_with_clock_time": 0,
            "synthetic_time_records": 0,
            "observed_time_records": 0,
            "synthetic_clock_time_records": 0,
            "observed_clock_time_records": 0,
            "timestamp_source_breakdown": {},
        }

        for record in records:
            parsed_dt = record.get("parsed_time")
            if parsed_dt is None:
                continue

            metadata = record.get("metadata", {}) or {}
            source = str(metadata.get("timestamp_source", "")).strip() or "unspecified"
            raw_timestamp = metadata.get("timestamp")
            has_clock_time = self._has_explicit_clock_time(raw_timestamp)
            is_synthetic = self.is_synthetic_timestamp_metadata(metadata)

            summary["valid_time_records"] += 1
            summary["timestamp_source_breakdown"][source] = (
                summary["timestamp_source_breakdown"].get(source, 0) + 1
            )

            if is_synthetic:
                summary["synthetic_time_records"] += 1
            else:
                summary["observed_time_records"] += 1

            if has_clock_time:
                summary["records_with_clock_time"] += 1
                if is_synthetic:
                    summary["synthetic_clock_time_records"] += 1
                else:
                    summary["observed_clock_time_records"] += 1

        total_valid = summary["valid_time_records"]
        total_clock = summary["records_with_clock_time"]
        summary["clock_time_ratio"] = round(total_clock / total_valid, 4) if total_valid > 0 else 0.0
        summary["synthetic_time_ratio"] = round(
            summary["synthetic_time_records"] / total_valid,
            4,
        ) if total_valid > 0 else 0.0
        summary["observed_time_ratio"] = round(
            summary["observed_time_records"] / total_valid,
            4,
        ) if total_valid > 0 else 0.0
        summary["synthetic_clock_time_ratio"] = round(
            summary["synthetic_clock_time_records"] / total_valid,
            4,
        ) if total_valid > 0 else 0.0
        summary["observed_clock_time_ratio"] = round(
            summary["observed_clock_time_records"] / total_valid,
            4,
        ) if total_valid > 0 else 0.0
        return summary

    def build_independent_events(
        self,
        species="Siberian tiger",
        by_camera=True,
        min_gap_minutes=30
    ):
        """
        Merge consecutive records into independent events.
        By default, records from the same species and camera are grouped until the time gap exceeds min_gap_minutes.
        """
        if self._using_camtrap_backend():
            species = self._canonical_species_name(species)
            if not by_camera:
                raise ValueError("CamtrapDP independent events must be constructed by species and camera.")
            min_gap_minutes = max(float(min_gap_minutes), 0.0)
            event_df = self._load_camtrap_event_df(min_gap_minutes=min_gap_minutes)
            events = []
            if not event_df.empty:
                df = event_df.copy()
                if "species" not in df.columns:
                    df["species"] = df.get("scientificName", "").map(self._canonical_species_name)
                df["species"] = df["species"].map(self._canonical_species_name)
                camera_col = "camera_id" if "camera_id" in df.columns else "cameraID"
                df = df[df["species"] == species].copy()
                if "event_time_dt" not in df.columns:
                    if "event_time" in df.columns:
                        df["event_time_dt"] = pd.to_datetime(df["event_time"], errors="coerce")
                    else:
                        df["event_time_dt"] = pd.to_datetime(df.get("eventStart"), errors="coerce")
                df = df.dropna(subset=["event_time_dt"]).sort_values([camera_col, "event_time_dt"])
                for idx, row in enumerate(df.itertuples(index=False), start=1):
                    data = row._asdict()
                    event_time = data.get("event_time_dt")
                    event_time = event_time.to_pydatetime() if hasattr(event_time, "to_pydatetime") else event_time
                    start_time = self.safe_parse_timestamp(data.get("start_time") or data.get("eventStart")) or event_time
                    end_time = self.safe_parse_timestamp(data.get("end_time") or data.get("eventEnd")) or event_time
                    camera_id = str(data.get(camera_col, "")).strip()
                    event_record_count = data.get("event_record_count", 1)
                    try:
                        event_record_count = int(event_record_count)
                    except Exception:
                        event_record_count = 1
                    observed_clock_count = int(float(data.get("observed_clock_time_records") or 0))
                    synthetic_time_count = int(float(data.get("synthetic_time_records") or 0))
                    has_observed_clock = bool(data.get("has_observed_clock_time")) and observed_clock_count > 0
                    events.append({
                        "event_id": str(data.get("event_id") or data.get("eventID") or f"{species}_{camera_id}_{idx:05d}"),
                        "species": species,
                        "camera_id": camera_id,
                        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(start_time, "strftime") else str(start_time),
                        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(end_time, "strftime") else str(end_time),
                        "event_time": event_time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(event_time, "strftime") else str(event_time),
                        "latitude": self._coerce_float(data.get("latitude")),
                        "longitude": self._coerce_float(data.get("longitude")),
                        "event_record_count": event_record_count,
                        "event_duration_minutes": round((end_time - start_time).total_seconds() / 60.0, 2) if hasattr(end_time, "__sub__") and hasattr(start_time, "strftime") else 0.0,
                        "records_with_clock_time": observed_clock_count,
                        "has_explicit_clock_time": has_observed_clock,
                        "observed_event_time": (event_time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(event_time, "strftime") else str(event_time)) if has_observed_clock else None,
                        "observed_clock_time_records": observed_clock_count,
                        "synthetic_clock_time_records": synthetic_time_count,
                        "has_observed_clock_time": has_observed_clock,
                        "individuals_max": None,
                        "behavior_mode": None,
                        "record_ids": [],
                    })
            report = self._load_camtrap_conversion_report()
            source_count = None
            if report.get("species_summary", {}).get(species):
                source_count = int(report["species_summary"][species].get("media_records", 0))
            if source_count is None:
                source_count = int(self._load_camtrap_records_df()["species"].map(self._canonical_species_name).eq(species).sum())
            return {
                "species": species,
                "by_camera": by_camera,
                "min_gap_minutes": min_gap_minutes,
                "source_record_count": source_count,
                "valid_record_count": source_count,
                "event_count": len(events),
                "events": events,
                "event_source": "cached" if math.isclose(float(min_gap_minutes), float(report.get("filters", {}).get("min_gap_minutes", 30))) else "recomputed_from_media",
                "note": (
                    "Independent events use the requested same-species, same-camera gap. "
                    "A matching conversion cache is reused; other thresholds are deterministically recomputed from media rows."
                ),
            }

        min_gap_minutes = max(int(min_gap_minutes), 0)
        records = self.get_species_records_with_parsed_time(species)
        camera_locations = self.get_camera_locations()

        valid_records = []
        for r in records:
            parsed_dt = r["parsed_time"]
            if parsed_dt is None:
                continue

            meta = r["metadata"]
            camera_id = str(meta.get("camera_id", "")).strip()
            if by_camera and not camera_id:
                continue

            lat = meta.get("latitude")
            lon = meta.get("longitude")
            camera_loc = camera_locations.get(camera_id, {})

            try:
                latitude = float(lat) if lat is not None else camera_loc.get("latitude")
                longitude = float(lon) if lon is not None else camera_loc.get("longitude")
            except Exception:
                latitude = camera_loc.get("latitude")
                longitude = camera_loc.get("longitude")

            valid_records.append({
                "id": r["id"],
                "document": r["document"],
                "metadata": meta,
                "parsed_time": parsed_dt,
                "camera_id": camera_id if by_camera else "ALL_CAMERAS",
                "latitude": latitude,
                "longitude": longitude,
                "has_clock_time": self._has_explicit_clock_time(meta.get("timestamp")),
                "is_synthetic_timestamp": self.is_synthetic_timestamp_metadata(meta),
            })

        valid_records.sort(
            key=lambda r: (
                r["camera_id"],
                r["parsed_time"],
                str(r["id"]),
            )
        )

        grouped = {}
        for record in valid_records:
            grouped.setdefault(record["camera_id"], []).append(record)

        event_rows = []
        for group_key, group_records in grouped.items():
            current_event = []

            for record in group_records:
                if not current_event:
                    current_event = [record]
                    continue

                prev_record = current_event[-1]
                gap_minutes = (record["parsed_time"] - prev_record["parsed_time"]).total_seconds() / 60.0
                if gap_minutes <= min_gap_minutes:
                    current_event.append(record)
                else:
                    event_rows.append(
                        self._summarize_independent_event(
                            species=species,
                            event_index=len(event_rows) + 1,
                            group_key=group_key,
                            event_records=current_event,
                        )
                    )
                    current_event = [record]

            if current_event:
                event_rows.append(
                    self._summarize_independent_event(
                        species=species,
                        event_index=len(event_rows) + 1,
                        group_key=group_key,
                        event_records=current_event,
                    )
                )

        return {
            "species": species,
            "by_camera": by_camera,
            "min_gap_minutes": min_gap_minutes,
            "source_record_count": len(records),
            "valid_record_count": len(valid_records),
            "event_count": len(event_rows),
            "events": event_rows,
            "note": (
                "Independent events are aggregated from temporally adjacent records. "
                "By default, records from the same camera within the gap threshold are merged."
            ),
        }

    def _summarize_independent_event(self, species, event_index, group_key, event_records):
        start_dt = event_records[0]["parsed_time"]
        end_dt = event_records[-1]["parsed_time"]
        midpoint_dt = start_dt + (end_dt - start_dt) / 2
        start_text = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_text = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        event_text = midpoint_dt.strftime("%Y-%m-%d %H:%M:%S")

        behavior_values = []
        individuals_values = []
        latitudes = []
        longitudes = []
        record_ids = []
        clock_time_count = 0
        observed_clock_time_count = 0
        synthetic_clock_time_count = 0
        observed_clock_times = []

        for record in event_records:
            meta = record["metadata"]
            record_ids.append(record["id"])

            behavior = str(meta.get("behavior", "")).strip()
            if behavior and behavior.lower() != "unknown":
                behavior_values.append(behavior)

            individuals = meta.get("individuals")
            try:
                individuals_int = int(individuals)
                if individuals_int > 0:
                    individuals_values.append(individuals_int)
            except Exception:
                pass

            if record["latitude"] is not None:
                latitudes.append(float(record["latitude"]))
            if record["longitude"] is not None:
                longitudes.append(float(record["longitude"]))
            if record["has_clock_time"]:
                clock_time_count += 1
                if record.get("is_synthetic_timestamp"):
                    synthetic_clock_time_count += 1
                else:
                    observed_clock_time_count += 1
                    observed_clock_times.append(record["parsed_time"])

        behavior_mode = None
        if behavior_values:
            behavior_mode = Counter(behavior_values).most_common(1)[0][0]

        latitude = round(sum(latitudes) / len(latitudes), 6) if latitudes else None
        longitude = round(sum(longitudes) / len(longitudes), 6) if longitudes else None
        duration_minutes = (end_dt - start_dt).total_seconds() / 60.0
        observed_event_text = None
        if observed_clock_times:
            observed_start = min(observed_clock_times)
            observed_end = max(observed_clock_times)
            observed_midpoint = observed_start + (observed_end - observed_start) / 2
            observed_event_text = observed_midpoint.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "event_id": f"{species}_{group_key}_{event_index:05d}",
            "species": species,
            "camera_id": str(group_key),
            "start_time": start_text,
            "end_time": end_text,
            "event_time": event_text,
            "latitude": latitude,
            "longitude": longitude,
            "event_record_count": len(event_records),
            "event_duration_minutes": round(duration_minutes, 2),
            "records_with_clock_time": clock_time_count,
            "has_explicit_clock_time": clock_time_count > 0,
            "observed_event_time": observed_event_text,
            "observed_clock_time_records": observed_clock_time_count,
            "synthetic_clock_time_records": synthetic_clock_time_count,
            "has_observed_clock_time": observed_clock_time_count > 0,
            "individuals_max": max(individuals_values) if individuals_values else None,
            "behavior_mode": behavior_mode,
            "record_ids": record_ids,
        }

    def get_valid_species_records(self, species):
        """
        Return records for a species with valid time and location fields.
        """
        records = self.get_species_records_with_parsed_time(species)
        valid = []

        for r in records:
            meta = r["metadata"]
            lat = meta.get("latitude")
            lon = meta.get("longitude")
            cam = meta.get("camera_id")

            if (
                    r["parsed_time"] is not None
                    and lat is not None
                    and lon is not None
                    and str(cam).strip() != ""
            ):
                valid.append(r)

        return valid

    def calculate_distance_km(self, lat1, lon1, lat2, lon2):
        """
        Haversine distance in kilometers.
        """
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])

        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
                math.sin(dlat / 2) ** 2
                + math.cos(math.radians(lat1))
                * math.cos(math.radians(lat2))
                * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def get_camera_locations(self):
        """
        Extract all camera locations from records.
        If a camera_id has conflicting coordinates, keep the first location and record the conflict.
        """
        if self._using_camtrap_backend():
            deployments = self._load_camtrap_deployments_df()
            camera_locations = {}
            if deployments.empty:
                return camera_locations
            for row in deployments.itertuples(index=False):
                data = row._asdict()
                camera_id = str(data.get("cameraID") or str(data.get("deploymentID", "")).replace("dep_", "", 1)).strip()
                lat = self._coerce_float(data.get("latitude"))
                lon = self._coerce_float(data.get("longitude"))
                if camera_id and lat is not None and lon is not None:
                    camera_locations[camera_id] = {"latitude": lat, "longitude": lon}
            return camera_locations

        all_records = self.get_all_records()
        camera_locations = {}
        conflicts = []

        for r in all_records:
            meta = r["metadata"]
            camera_id = str(meta.get("camera_id", "")).strip()
            lat = meta.get("latitude")
            lon = meta.get("longitude")

            if not camera_id or lat is None or lon is None:
                continue

            lat = float(lat)
            lon = float(lon)

            if camera_id not in camera_locations:
                camera_locations[camera_id] = {
                    "latitude": lat,
                    "longitude": lon
                }
            else:
                old_lat = camera_locations[camera_id]["latitude"]
                old_lon = camera_locations[camera_id]["longitude"]

                if abs(old_lat - lat) > 1e-6 or abs(old_lon - lon) > 1e-6:
                    conflicts.append({
                        "camera_id": camera_id,
                        "old_latitude": old_lat,
                        "old_longitude": old_lon,
                        "new_latitude": lat,
                        "new_longitude": lon
                    })

        return camera_locations

    def get_cameras_in_radius(self, center_camera_id, radius_km=50):
        """
        Return cameras within a radius of the center camera.
        """
        center_camera_id = str(center_camera_id).strip()
        camera_locations = self.get_camera_locations()

        if center_camera_id not in camera_locations:
            return []

        center = camera_locations[center_camera_id]
        result = []

        for cam_id, loc in camera_locations.items():
            if cam_id == center_camera_id:
                continue

            dist = self.calculate_distance_km(
                center["latitude"], center["longitude"],
                loc["latitude"], loc["longitude"]
            )

            if dist <= radius_km:
                result.append({
                    "camera_id": cam_id,
                    "distance_km": round(dist, 3),
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"]
                })

        result.sort(key=lambda x: x["distance_km"])
        return result

    def _build_camtrap_record_from_observation(
        self,
        observation,
        deployments_by_id,
        media_by_id,
        package_name,
        package_dir,
        datapackage_file,
    ):
        deployment_id = str(observation.get("deploymentID", "")).strip()
        media_id = str(observation.get("mediaID", "")).strip()
        deployment = deployments_by_id.get(deployment_id, {})
        media = media_by_id.get(media_id, {})

        scientific_name = str(observation.get("scientificName", "")).strip()
        count_text = str(observation.get("count", "")).strip()
        try:
            individuals = int(float(count_text)) if count_text else 1
        except Exception:
            individuals = 1

        timestamp_text = (
            str(media.get("timestamp", "")).strip()
            or str(observation.get("eventStart", "")).strip()
            or str(observation.get("eventEnd", "")).strip()
        )
        file_name = (
            str(media.get("fileName", "")).strip()
            or str(media.get("filePath", "")).strip().rsplit("/", 1)[-1]
            or f"{observation.get('observationID', 'observation')}.jpg"
        )
        observation_probability = str(observation.get("classificationProbability", "")).strip()

        special_notes = []
        for key in ["observationComments", "observationTags", "mediaComments"]:
            value = str(observation.get(key, "")).strip() or str(media.get(key, "")).strip()
            if value:
                special_notes.append(value)

        metadata = {
            "filename": file_name,
            "camera_id": str(deployment.get("cameraID", "")).strip() or deployment_id,
            "latitude": deployment.get("latitude"),
            "longitude": deployment.get("longitude"),
            "timestamp": timestamp_text,
            "timestamp_original_date": timestamp_text[:10] if timestamp_text else "",
            "timestamp_source": "camtrap_dp_import",
            "timestamp_is_synthetic": False,
            "species": scientific_name or "Unknown",
            "scientific_name": scientific_name or "",
            "confidence": observation_probability or "",
            "individuals": individuals,
            "behavior": str(observation.get("behavior", "")).strip() or "Unknown",
            "special_observations": " | ".join(special_notes),
            "ecological_report": "",
            "camtrap_package_name": package_name,
            "camtrap_package_dir": str(package_dir),
            "camtrap_datapackage_path": str(datapackage_file),
            "camtrap_deployment_id": deployment_id,
            "camtrap_media_id": media_id,
            "camtrap_observation_id": str(observation.get("observationID", "")).strip(),
            "camtrap_event_id": str(observation.get("eventID", "")).strip(),
            "camtrap_observation_level": str(observation.get("observationLevel", "")).strip(),
            "camtrap_observation_type": str(observation.get("observationType", "")).strip(),
            "camtrap_capture_method": str(media.get("captureMethod", "")).strip(),
            "camtrap_file_path": str(media.get("filePath", "")).strip(),
            "camtrap_file_public": self._is_truthy_flag(media.get("filePublic")),
            "camtrap_file_mediatype": str(media.get("fileMediatype", "")).strip(),
            "camtrap_event_start": str(observation.get("eventStart", "")).strip(),
            "camtrap_event_end": str(observation.get("eventEnd", "")).strip(),
            "camtrap_observation_tags": str(observation.get("observationTags", "")).strip(),
            "camtrap_observation_comments": str(observation.get("observationComments", "")).strip(),
            "camtrap_classification_method": str(observation.get("classificationMethod", "")).strip(),
            "camtrap_classified_by": str(observation.get("classifiedBy", "")).strip(),
            "camtrap_classification_timestamp": str(observation.get("classificationTimestamp", "")).strip(),
        }

        return {
            "id": metadata["camtrap_observation_id"] or media_id or f"{deployment_id}_{uuid.uuid4().hex[:8]}",
            "document": "",
            "metadata": metadata,
        }

    def import_camtrap_dp(
        self,
        datapackage_path=None,
        target_dir=None,
        prefer_media_level=True,
        include_blank=False,
        include_event_fallback=True,
    ):
        """
        Import a CamtrapDP package and convert it into JSON records readable by this system.
        Media-level animal observations are preferred by default to avoid event/media double counting.
        """
        datapackage_path = self._resolve_camtrap_datapackage_path(datapackage_path)
        if datapackage_path.is_dir():
            package_dir = datapackage_path
            datapackage_file = package_dir / "datapackage.json"
        else:
            datapackage_file = datapackage_path
            package_dir = datapackage_file.parent

        if not datapackage_file.exists():
            raise FileNotFoundError(f"CamtrapDP datapackage not found: {datapackage_file}")

        package = json.loads(datapackage_file.read_text(encoding="utf-8"))
        package_name = package.get("name") or datapackage_file.stem
        resources = {item.get("name"): item for item in package.get("resources", []) if isinstance(item, dict)}

        deployments_resource = resources.get("deployments")
        media_resource = resources.get("media")
        observations_resource = resources.get("observations")
        if deployments_resource is None or observations_resource is None:
            raise ValueError("CamtrapDP package must include at least deployments and observations resources.")

        deployments_rows = self._read_csv_rows(package_dir / deployments_resource.get("path", "deployments.csv"))
        media_rows = self._read_csv_rows(package_dir / media_resource.get("path", "media.csv")) if media_resource else []
        observations_rows = self._read_csv_rows(package_dir / observations_resource.get("path", "observations.csv"))

        deployments_by_id = {
            str(row.get("deploymentID", "")).strip(): row
            for row in deployments_rows
            if str(row.get("deploymentID", "")).strip()
        }
        media_by_id = {
            str(row.get("mediaID", "")).strip(): row
            for row in media_rows
            if str(row.get("mediaID", "")).strip()
        }

        selected_records = []
        media_backed_event_keys = set()
        imported_species = Counter()

        def _should_keep_observation(row):
            observation_type = str(row.get("observationType", "")).strip().lower()
            if include_blank:
                return observation_type in {"animal", "blank"} or not observation_type
            return observation_type == "animal"

        for row in observations_rows:
            if not _should_keep_observation(row):
                continue
            observation_level = str(row.get("observationLevel", "")).strip().lower()
            media_id = str(row.get("mediaID", "")).strip()
            if observation_level == "media" and media_id:
                record = self._build_camtrap_record_from_observation(
                    row,
                    deployments_by_id=deployments_by_id,
                    media_by_id=media_by_id,
                    package_name=package_name,
                    package_dir=package_dir,
                    datapackage_file=datapackage_file,
                )
                selected_records.append(record)
                media_backed_event_keys.add(
                    (
                        str(row.get("eventID", "")).strip(),
                        str(row.get("scientificName", "")).strip(),
                    )
                )
                imported_species[record["metadata"].get("species", "Unknown")] += 1

        event_fallback_count = 0
        if include_event_fallback:
            for row in observations_rows:
                if not _should_keep_observation(row):
                    continue
                observation_level = str(row.get("observationLevel", "")).strip().lower()
                if observation_level != "event":
                    continue

                event_key = (
                    str(row.get("eventID", "")).strip(),
                    str(row.get("scientificName", "")).strip(),
                )
                if prefer_media_level and event_key in media_backed_event_keys:
                    continue

                record = self._build_camtrap_record_from_observation(
                    row,
                    deployments_by_id=deployments_by_id,
                    media_by_id=media_by_id,
                    package_name=package_name,
                    package_dir=package_dir,
                    datapackage_file=datapackage_file,
                )
                selected_records.append(record)
                imported_species[record["metadata"].get("species", "Unknown")] += 1
                event_fallback_count += 1

        if target_dir is None:
            target_dir = self._make_unique_dir(CAMTRAP_IMPORTS_DIR, package_name)
        else:
            target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        for record in selected_records:
            output_path = target_dir / f"{self._slugify_text(record['id'])}.json"
            output_path.write_text(
                json.dumps(record["metadata"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return {
            "package_name": package_name,
            "package_id": package.get("id"),
            "source_datapackage": str(datapackage_file),
            "target_dir": str(target_dir),
            "deployment_count": len(deployments_rows),
            "media_count": len(media_rows),
            "observation_count": len(observations_rows),
            "imported_record_count": len(selected_records),
            "event_fallback_count": event_fallback_count,
            "prefer_media_level": bool(prefer_media_level),
            "include_blank": bool(include_blank),
            "species_breakdown": dict(imported_species.most_common()),
            "note": (
                "Imported CamtrapDP records were normalized into the agent's JSON record format. "
                "Media-level animal observations are preferred, and event-level observations are only used as fallback "
                "to avoid double-counting when both levels are present."
            ),
        }

    def load_camtrap_dp_dataset(
        self,
        datapackage_path=None,
        target_dir=None,
        prefer_media_level=True,
        include_blank=False,
        include_event_fallback=True,
        activate_json_backend=True,
        tabular_backend=True,
    ):
        if tabular_backend:
            backend_result = self.set_camtrap_data_dir(datapackage_path, activate=activate_json_backend)
            return {
                "package_name": Path(backend_result["package_dir"]).name,
                "source_datapackage": backend_result["datapackage_path"],
                "target_dir": backend_result["package_dir"],
                "imported_record_count": backend_result["record_count"],
                "event_count": backend_result["event_count"],
                "active_json_backend": False,
                "active_tabular_backend": bool(activate_json_backend),
                "backend_result": backend_result,
                "note": "CamtrapDP package activated through the tabular backend; no per-record JSON files were created.",
            }

        import_result = self.import_camtrap_dp(
            datapackage_path=datapackage_path,
            target_dir=target_dir,
            prefer_media_level=prefer_media_level,
            include_blank=include_blank,
            include_event_fallback=include_event_fallback,
        )

        backend_result = None
        if activate_json_backend:
            backend_result = self.set_json_data_dir(import_result["target_dir"])

        return {
            **import_result,
            "active_json_backend": bool(activate_json_backend),
            "backend_result": backend_result,
            "note": (
                "CamtrapDP package imported successfully and the standardized JSON dataset is ready for downstream browse, evidence, and ecological analysis tools."
                if activate_json_backend
                else "CamtrapDP package imported successfully; call set_json_data_dir to activate it for downstream analysis."
            ),
        }

    def _stable_hash_id(self, *parts, prefix="", length=16):
        text = "||".join(str(part) for part in parts)
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]
        return f"{prefix}{digest}" if prefix else digest

    def _read_tabular_input(self, table_path, sheet_name=0):
        table_path = Path(table_path)
        if not table_path.exists():
            raise FileNotFoundError(f"Tabular input not found: {table_path}")
        suffix = table_path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(table_path, sheet_name=sheet_name)
        if suffix in {".tsv", ".tab"}:
            return pd.read_csv(table_path, sep="\t")
        return pd.read_csv(table_path)

    def _parse_field_mapping(self, field_mapping):
        if field_mapping is None:
            return {}
        if isinstance(field_mapping, str):
            text = field_mapping.strip()
            if not text:
                return {}
            return json.loads(text)
        if isinstance(field_mapping, dict):
            return dict(field_mapping)
        raise TypeError("field_mapping must be a dict or a JSON object string.")

    def _guess_column(self, columns, candidates):
        lowered = {str(col).strip().lower(): col for col in columns}
        for candidate in candidates:
            key = candidate.lower()
            if key in lowered:
                return lowered[key]
        for col in columns:
            text = str(col).strip().lower()
            for candidate in candidates:
                if candidate.lower() in text:
                    return col
        return None

    def _resolve_table_mapping(self, df, field_mapping):
        mapping = self._parse_field_mapping(field_mapping)
        guesses = {
            "species": ["species", "scientificName", "scientific_name", "taxon", "common_name"],
            "scientific_name": ["scientificName", "scientific_name", "latin_name"],
            "camera_id": ["camera_id", "cameraID", "camera", "deploymentID", "locationID"],
            "timestamp": ["timestamp", "datetime", "date_time", "recordtime", "capture_time"],
            "latitude": ["latitude", "lat"],
            "longitude": ["longitude", "lon", "lng"],
            "media": ["media", "filePath", "file_path", "filename", "fileName", "videoname", "image", "video"],
            "count": ["count", "individual_count", "individuals"],
            "confidence": ["confidence", "classificationProbability", "probability", "score"],
            "behavior": ["behavior"],
        }
        resolved = {}
        for target, candidates in guesses.items():
            mapped = mapping.get(target)
            if mapped and mapped in df.columns:
                resolved[target] = mapped
            elif mapped:
                resolved[target] = mapped
            else:
                guessed = self._guess_column(df.columns, candidates)
                if guessed is not None:
                    resolved[target] = guessed
        return resolved

    def convert_table_to_camtrap_dp(
        self,
        table_path,
        field_mapping=None,
        output_dir=None,
        sheet_name=0,
        package_name=None,
        package_title=None,
        min_gap_minutes=30,
        activate_backend=True,
        persist_active=True,
        source_timezone="UTC",
    ):
        """
        Convert a simple camera-trap CSV/TSV/Excel table into a CamtrapDP package.

        Required semantic fields are species, camera_id, timestamp, latitude, and longitude.
        A media/file column is strongly recommended and will be synthesized when absent.
        """
        df = self._read_tabular_input(table_path, sheet_name=sheet_name)
        source_columns = [str(col) for col in df.columns]
        resolved = self._resolve_table_mapping(df, field_mapping)
        required = ["species", "camera_id", "timestamp", "latitude", "longitude"]
        issues = []
        for field in required:
            col = resolved.get(field)
            if not col or col not in df.columns:
                issues.append({
                    "severity": "error",
                    "field": field,
                    "message": f"Required semantic field `{field}` was not mapped to an input column.",
                })
        if issues:
            return {
                "status": "error",
                "source_table": str(table_path),
                "source_columns": source_columns,
                "field_mapping": resolved,
                "issues": issues,
                "note": "Conversion was not run because required fields were missing.",
            }

        work = pd.DataFrame({
            "species": df[resolved["species"]].astype(str).str.strip(),
            "camera_id": df[resolved["camera_id"]].astype(str).str.strip(),
            "timestamp_raw": df[resolved["timestamp"]],
            "latitude": pd.to_numeric(df[resolved["latitude"]], errors="coerce"),
            "longitude": pd.to_numeric(df[resolved["longitude"]], errors="coerce"),
        })
        sci_col = resolved.get("scientific_name")
        work["scientificName"] = (
            df[sci_col].astype(str).str.strip()
            if sci_col in df.columns
            else work["species"].map(self._canonical_species_name)
        )
        media_col = resolved.get("media")
        if media_col in df.columns:
            work["media_file"] = df[media_col].astype(str).str.strip()
        else:
            work["media_file"] = ""
            issues.append({
                "severity": "warning",
                "field": "media",
                "message": "No media/file column was mapped; synthetic media file names will be generated.",
            })
        count_col = resolved.get("count")
        work["count"] = pd.to_numeric(df[count_col], errors="coerce").fillna(1).clip(lower=1).astype(int) if count_col in df.columns else 1
        conf_col = resolved.get("confidence")
        work["classificationProbability"] = pd.to_numeric(df[conf_col], errors="coerce") if conf_col in df.columns else ""
        behavior_col = resolved.get("behavior")
        work["behavior"] = df[behavior_col].astype(str).str.strip() if behavior_col in df.columns else ""
        work["timestamp_dt"] = work["timestamp_raw"].map(
            lambda value: self._source_datetime_to_utc(
                value,
                source_timezone=source_timezone,
            )
        )
        work["timestamp_dt"] = pd.to_datetime(
            work["timestamp_dt"],
            errors="coerce",
            utc=True,
        )
        work["source_row"] = df.index.astype(int) + 2

        raw_rows = int(len(work))
        invalid_mask = (
            work["species"].eq("")
            | work["camera_id"].eq("")
            | work["timestamp_dt"].isna()
            | work["latitude"].isna()
            | work["longitude"].isna()
        )
        invalid_rows = int(invalid_mask.sum())
        if invalid_rows:
            issues.append({
                "severity": "warning",
                "field": "required_fields",
                "message": f"{invalid_rows} rows were dropped because required values were missing or unparseable.",
            })
        work = work.loc[~invalid_mask].copy()
        if work.empty:
            raise ValueError("No valid rows remain after parsing required camera-trap fields.")

        work["species"] = work["species"].map(lambda value: self._canonical_species_name(value))
        work["scientificName"] = work["scientificName"].where(work["scientificName"].astype(str).str.strip() != "", work["species"])
        work["deploymentID"] = "dep_" + work["camera_id"].map(self._slugify_text)
        work["mediaID"] = [
            self._stable_hash_id(table_path, row.source_row, row.camera_id, row.timestamp_dt, row.species, prefix="med_")
            for row in work.itertuples()
        ]
        duplicate_ids = work["mediaID"].duplicated(keep=False)
        if duplicate_ids.any():
            work.loc[duplicate_ids, "mediaID"] = (
                work.loc[duplicate_ids, "mediaID"]
                + "_"
                + work.loc[duplicate_ids].groupby("mediaID").cumcount().astype(str)
            )
        work["observationID"] = "obs_" + work["mediaID"].str.replace("med_", "", regex=False)
        blank_media = work["media_file"].astype(str).str.strip().eq("")
        work.loc[blank_media, "media_file"] = work.loc[blank_media, "mediaID"] + ".jpg"

        work = work.sort_values(["species", "camera_id", "timestamp_dt", "mediaID"]).reset_index(drop=True)
        gap = work.groupby(["species", "camera_id"])["timestamp_dt"].diff().dt.total_seconds().div(60)
        work["new_event"] = gap.isna() | (gap > float(min_gap_minutes))
        work["event_seq"] = work.groupby(["species", "camera_id"])["new_event"].cumsum().astype(int)
        work["eventID"] = (
            "ev_"
            + work["camera_id"].map(self._slugify_text)
            + "_"
            + work["species"].map(self._slugify_text)
            + "_"
            + work["event_seq"].astype(str).str.zfill(6)
        )
        grouped = work.groupby("eventID", sort=False)
        work["eventStart"] = grouped["timestamp_dt"].transform("min")
        work["eventEnd"] = grouped["timestamp_dt"].transform("max")

        events = grouped.agg(
            deploymentID=("deploymentID", "first"),
            camera_id=("camera_id", "first"),
            species=("species", "first"),
            scientificName=("scientificName", "first"),
            eventStart=("timestamp_dt", "min"),
            eventEnd=("timestamp_dt", "max"),
            event_record_count=("timestamp_dt", "size"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
        ).reset_index()
        events["event_time"] = events["eventStart"] + (events["eventEnd"] - events["eventStart"]) / 2

        source_stem = self._slugify_text(package_name or Path(table_path).stem or "tabular_camera_trap")
        if output_dir is None:
            output_dir = self._make_unique_dir(CAMTRAP_IMPORTS_DIR, f"{source_stem}_camtrapdp")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        deployments = work.groupby("deploymentID", sort=True).agg(
            locationID=("camera_id", "first"),
            locationName=("camera_id", "first"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            deploymentStart=("timestamp_dt", "min"),
            deploymentEnd=("timestamp_dt", "max"),
            cameraID=("camera_id", "first"),
        ).reset_index()
        deployments["coordinateUncertainty"] = ""
        deployments["setupBy"] = ""
        deployments["cameraModel"] = ""
        deployments["cameraDelay"] = ""
        deployments["cameraHeight"] = ""
        deployments["cameraDepth"] = ""
        deployments["cameraTilt"] = ""
        deployments["cameraHeading"] = ""
        deployments["detectionDistance"] = ""
        deployments["timestampIssues"] = "false"
        deployments["baitUse"] = "false"
        deployments["featureType"] = ""
        deployments["habitat"] = ""
        deployments["deploymentGroups"] = source_stem
        deployments["deploymentTags"] = ""
        deployments["deploymentComments"] = "Deployment inferred from a generic tabular camera-trap conversion."

        media = pd.DataFrame({
            "mediaID": work["mediaID"],
            "deploymentID": work["deploymentID"],
            "captureMethod": "activityDetection",
            "timestamp": work["timestamp_dt"],
            "filePath": work["media_file"],
            "filePublic": "false",
            "fileName": work["media_file"].map(lambda value: Path(str(value)).name),
            "fileMediatype": work["media_file"].map(lambda value: self._guess_media_mimetype(value, fallback="application/octet-stream")),
            "exifData": "",
            "favorite": "",
            "mediaComments": "Converted by convert_table_to_camtrap_dp; source_row=" + work["source_row"].astype(str),
        })

        event_observations = pd.DataFrame({
            "observationID": events["eventID"] + "_event",
            "deploymentID": events["deploymentID"],
            "mediaID": "",
            "eventID": events["eventID"],
            "eventStart": events["eventStart"],
            "eventEnd": events["eventEnd"],
            "observationLevel": "event",
            "observationType": "animal",
            "cameraSetupType": "",
            "scientificName": events["scientificName"],
            "count": 1,
            "lifeStage": "",
            "sex": "",
            "behavior": "",
            "individualID": "",
            "individualPositionRadius": "",
            "individualPositionAngle": "",
            "individualSpeed": "",
            "bboxX": "",
            "bboxY": "",
            "bboxWidth": "",
            "bboxHeight": "",
            "classificationMethod": "human",
            "classifiedBy": "convert_table_to_camtrap_dp",
            "classificationTimestamp": "",
            "classificationProbability": "",
            "observationTags": "independent_event",
            "observationComments": "Independent event summary; event_record_count is stored in agent_independent_events.csv.",
        })
        media_observations = pd.DataFrame({
            "observationID": work["observationID"],
            "deploymentID": work["deploymentID"],
            "mediaID": work["mediaID"],
            "eventID": work["eventID"],
            "eventStart": work["eventStart"],
            "eventEnd": work["eventEnd"],
            "observationLevel": "media",
            "observationType": "animal",
            "cameraSetupType": "",
            "scientificName": work["scientificName"],
            "count": work["count"],
            "lifeStage": "",
            "sex": "",
            "behavior": work["behavior"],
            "individualID": "",
            "individualPositionRadius": "",
            "individualPositionAngle": "",
            "individualSpeed": "",
            "bboxX": "",
            "bboxY": "",
            "bboxWidth": "",
            "bboxHeight": "",
            "classificationMethod": "human",
            "classifiedBy": "convert_table_to_camtrap_dp",
            "classificationTimestamp": "",
            "classificationProbability": work["classificationProbability"],
            "observationTags": "",
            "observationComments": "Converted from a generic tabular camera-trap source.",
        })
        observations = pd.concat([event_observations, media_observations], ignore_index=True)

        for frame in [deployments, media, observations, events]:
            for col in frame.columns:
                if pd.api.types.is_datetime64_any_dtype(frame[col]):
                    frame[col] = pd.to_datetime(frame[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        deployment_fields = [
            "deploymentID", "locationID", "locationName", "latitude", "longitude", "coordinateUncertainty",
            "deploymentStart", "deploymentEnd", "setupBy", "cameraID", "cameraModel", "cameraDelay",
            "cameraHeight", "cameraDepth", "cameraTilt", "cameraHeading", "detectionDistance", "timestampIssues",
            "baitUse", "featureType", "habitat", "deploymentGroups", "deploymentTags", "deploymentComments",
        ]
        deployments[deployment_fields].to_csv(output_dir / "deployments.csv", index=False, encoding="utf-8")
        media.to_csv(output_dir / "media.csv", index=False, encoding="utf-8")
        observations.to_csv(output_dir / "observations.csv", index=False, encoding="utf-8")

        event_cache = events.rename(columns={"eventID": "event_id"}).copy()
        event_cache["start_time"] = event_cache["eventStart"]
        event_cache["end_time"] = event_cache["eventEnd"]
        event_cache[[
            "event_id", "deploymentID", "camera_id", "species", "scientificName", "event_time",
            "start_time", "end_time", "event_record_count", "latitude", "longitude",
        ]].to_csv(output_dir / "agent_independent_events.csv", index=False, encoding="utf-8")

        datapackage = {
            "resources": [
                {"name": "deployments", "path": "deployments.csv", "profile": "tabular-data-resource", "format": "csv", "mediatype": "text/csv", "encoding": "utf-8"},
                {"name": "media", "path": "media.csv", "profile": "tabular-data-resource", "format": "csv", "mediatype": "text/csv", "encoding": "utf-8"},
                {"name": "observations", "path": "observations.csv", "profile": "tabular-data-resource", "format": "csv", "mediatype": "text/csv", "encoding": "utf-8"},
            ],
            "profile": "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0.2/camtrap-dp-profile.json",
            "name": source_stem,
            "id": self._stable_hash_id(table_path, raw_rows, len(events), prefix="pkg_"),
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "title": package_title or f"CamtrapDP conversion from {Path(table_path).name}",
            "description": "Generic camera-trap table converted into CamtrapDP for agent ecological analysis.",
            "version": "1.0.2",
            "project": {
                "title": package_title or source_stem,
                "samplingDesign": "generic tabular camera-trap conversion",
                "captureMethod": ["activityDetection"],
                "individualAnimals": False,
                "observationLevel": ["media", "event"],
            },
        }
        (output_dir / "datapackage.json").write_text(json.dumps(datapackage, ensure_ascii=False, indent=2), encoding="utf-8")

        species_summary = {}
        for species, group in work.groupby("species"):
            species_events = events[events["species"] == species]
            species_summary[species] = {
                "media_records": int(len(group)),
                "independent_events": int(len(species_events)),
                "camera_count": int(group["camera_id"].nunique()),
                "date_start": str(group["timestamp_dt"].min()),
                "date_end": str(group["timestamp_dt"].max()),
            }
        report = {
            "source_table": str(table_path),
            "output_dir": str(output_dir),
            "datapackage_path": str(output_dir / "datapackage.json"),
            "source_columns": source_columns,
            "field_mapping": resolved,
            "raw_rows": raw_rows,
            "kept_media_records": int(len(work)),
            "removed_rows": invalid_rows,
            "independent_events": int(len(events)),
            "deployment_count": int(len(deployments)),
            "min_gap_minutes": int(min_gap_minutes),
            "source_timezone": str(source_timezone or "UTC"),
            "species_summary": species_summary,
            "conversion_issues": issues,
            "note": "Generic tabular source converted to CamtrapDP; missing/unparseable required fields were excluded before analysis.",
        }
        (output_dir / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        backend_result = (
            self.set_camtrap_data_dir(
                output_dir,
                activate=activate_backend,
                persist_active=persist_active,
            )
            if activate_backend
            else None
        )
        return {
            "status": "ok",
            "source_table": str(table_path),
            "output_dir": str(output_dir),
            "datapackage_path": str(output_dir / "datapackage.json"),
            "field_mapping": resolved,
            "raw_rows": raw_rows,
            "kept_media_records": int(len(work)),
            "removed_rows": invalid_rows,
            "independent_events": int(len(events)),
            "deployment_count": int(len(deployments)),
            "source_timezone": str(source_timezone or "UTC"),
            "issues": issues,
            "backend_result": backend_result,
            "note": "Converted table is ready as the current CamtrapDP tabular backend for question answering, filtering, SVG figures, and interactive maps.",
        }

    def convert_species_csv_directory_to_camtrap_dp(
        self,
        input_dir,
        output_dir=None,
        species_config=None,
        column_map=None,
        start_date=None,
        end_date=None,
        lon_min=None,
        lon_max=None,
        lat_min=None,
        lat_max=None,
        min_gap_minutes=30,
        package_name="wildlife_camtrapdp",
        package_title="Camera-trap observations converted to CamtrapDP",
        package_description="Camera-trap observations converted from species CSV files into CamtrapDP for ecological analysis.",
        activate_backend=True,
        data_license=None,
        media_license=None,
    ):
        """
        Convert a directory of per-species CSV files into a CamtrapDP package.

        This is the reusable version of the 0525 conversion: each CSV stem is
        matched to species_config, columns are normalized through column_map,
        invalid/out-of-scope rows are filtered, and 30-minute independent events
        are cached for downstream ecology tools.
        """
        report = convert_species_csv_directory_to_camtrap_dp(
            input_dir=input_dir,
            output_dir=output_dir,
            species_config=species_config,
            column_map=column_map,
            start_date=start_date,
            end_date=end_date,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            min_gap_minutes=min_gap_minutes,
            package_name=package_name,
            package_title=package_title,
            package_description=package_description,
            classified_by="convert_species_csv_directory_to_camtrap_dp",
            data_license=data_license,
            media_license=media_license,
        )
        backend_result = None
        if activate_backend:
            backend_result = self.set_camtrap_data_dir(report["output_dir"], activate=True)
        return {
            "status": "ok",
            "output_dir": report["output_dir"],
            "datapackage_path": report["datapackage_path"],
            "kept_media_records": report["kept_media_records"],
            "independent_events": report["independent_events"],
            "deployment_count": report["deployment_count"],
            "source_file_count": len(report.get("source_files", [])),
            "backend_result": backend_result,
            "note": "Species CSV directory converted to CamtrapDP and is ready for tabular backend analysis.",
        }

    def run_ecology_figure_suite(
        self,
        package_dir=None,
        output_dir=None,
        species_order=None,
        ecological_role_mapping=None,
        include_core=True,
        include_exploratory=True,
        write_dashboard=True,
    ):
        """
        Generate the standardized ecology figure suite for the active or supplied CamtrapDP package.
        """
        if package_dir is None:
            if self._camtrap_package_dir is None:
                raise ValueError("No CamtrapDP package is active; pass package_dir or load a dataset first.")
            package_dir = self._camtrap_package_dir
        manifest = run_ecology_figure_suite(
            package_dir=package_dir,
            output_dir=output_dir,
            species_order=species_order,
            ecological_role_mapping=ecological_role_mapping,
            include_core=include_core,
            include_exploratory=include_exploratory,
            write_dashboard=write_dashboard,
        )
        return {
            "status": "ok",
            "package_dir": manifest["package_dir"],
            "output_dir": manifest["output_dir"],
            "figure_count": manifest["figure_count"],
            "manifest_json": manifest["manifest_json"],
            "manifest_md": manifest["manifest_md"],
            "dashboard": manifest.get("dashboard", ""),
            "figures": [
                {
                    "figure_id": item.get("id"),
                    "label": item.get("ecological_question"),
                    "path": item.get("path"),
                    "figure_status": item.get("figure_status"),
                    "interpretation_boundary": item.get("interpretation_boundary"),
                }
                for item in manifest.get("figures", [])
                if item.get("status") == "success" and item.get("path")
            ],
            "note": "Ecology figures use standardized naming and keep exploratory interpretation boundaries in the manifest.",
        }

    def _supported_detection_media_suffixes(self):
        return {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
            ".mp4", ".mov", ".avi", ".mkv", ".webm",
        }

    def _default_detection_record_dir(self, media_dir, detector, classifier):
        detector_slug = self._slugify_text(detector)
        classifier_slug = self._slugify_text(classifier)
        return DETECTION_RECORDS_DIR / f"{media_dir.name}_{detector_slug}_{classifier_slug}"

    def _load_detection_sidecar(self, media_path, inferred_sidecar_path=None):
        sidecar_candidates = [
            Path(inferred_sidecar_path) if inferred_sidecar_path else None,
            media_path.with_suffix(media_path.suffix + ".json"),
            media_path.with_suffix(".json"),
        ]
        for sidecar_path in sidecar_candidates:
            if sidecar_path is None or not sidecar_path.exists():
                continue
            try:
                loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(loaded, list):
                loaded = loaded[0] if loaded and isinstance(loaded[0], dict) else {}
            if isinstance(loaded, dict):
                return loaded
        return {}

    def _load_media_metadata_index(
        self,
        metadata_table=None,
        field_mapping=None,
        media_root=None,
    ):
        """Index optional deployment/media metadata without assuming a dataset layout."""
        if metadata_table is None or not str(metadata_table).strip():
            return {
                "rows": {},
                "source": "",
                "field_mapping": {},
                "row_count": 0,
            }

        metadata_table = Path(metadata_table).expanduser().resolve()
        df = self._read_tabular_input(metadata_table)
        resolved = self._resolve_table_mapping(df, field_mapping)
        media_column = resolved.get("media")
        if not media_column or media_column not in df.columns:
            raise ValueError(
                "metadata_table must include a media/file column or map one through field_mapping."
            )

        semantic_fields = {
            "camera_id": "camera_id",
            "timestamp": "timestamp",
            "latitude": "latitude",
            "longitude": "longitude",
            "species": "species",
            "scientific_name": "scientific_name",
            "count": "count",
            "confidence": "confidence",
            "behavior": "behavior",
        }
        media_root = Path(media_root).resolve() if media_root else None
        indexed = {}
        ambiguous = set()

        def _register(key, payload):
            key = str(key or "").strip()
            if not key or key in ambiguous:
                return
            previous = indexed.get(key)
            if previous is not None and previous != payload:
                indexed.pop(key, None)
                ambiguous.add(key)
                return
            indexed[key] = payload

        for _, row in df.iterrows():
            media_value = str(row.get(media_column, "")).strip()
            if not media_value:
                continue
            payload = {
                target: row.get(column)
                for semantic, target in semantic_fields.items()
                for column in [resolved.get(semantic)]
                if column in df.columns
            }
            payload["metadata_table_row"] = int(row.name) + 2 if isinstance(row.name, int) else str(row.name)
            payload["metadata_table_path"] = str(metadata_table)

            media_candidate = Path(media_value).expanduser()
            absolute_candidate = (
                media_candidate.resolve()
                if media_candidate.is_absolute()
                else (
                    (media_root / media_candidate).resolve()
                    if media_root is not None
                    else (metadata_table.parent / media_candidate).resolve()
                )
            )
            keys = {
                media_value,
                media_candidate.as_posix(),
                media_candidate.name,
                media_candidate.stem,
                str(absolute_candidate),
            }
            if media_root is not None:
                try:
                    keys.add(absolute_candidate.relative_to(media_root).as_posix())
                except ValueError:
                    pass
            for key in keys:
                _register(key, payload)

        return {
            "rows": indexed,
            "source": str(metadata_table),
            "field_mapping": resolved,
            "row_count": int(len(df)),
            "ambiguous_key_count": len(ambiguous),
        }

    def _metadata_for_media_path(self, media_path, media_root, metadata_index):
        rows = (metadata_index or {}).get("rows", {})
        if not rows:
            return {}
        media_path = Path(media_path).resolve()
        candidates = [
            str(media_path),
            media_path.name,
            media_path.stem,
        ]
        if media_root is not None:
            try:
                candidates.insert(1, media_path.relative_to(Path(media_root).resolve()).as_posix())
            except ValueError:
                pass
        for candidate in candidates:
            if candidate in rows:
                return dict(rows[candidate])
        return {}

    def _merge_nonblank_metadata(self, *payloads):
        merged = {}
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                if value is None:
                    continue
                try:
                    if not isinstance(value, (list, tuple, dict)) and pd.isna(value):
                        continue
                except (TypeError, ValueError):
                    pass
                if isinstance(value, str) and not value.strip():
                    continue
                merged[key] = value
        return merged

    def run_detection_pipeline(
        self,
        media_dir=None,
        detector="megadetector",
        classifier="speciesnet",
        output_dir=None,
        default_species="Unknown",
        min_confidence=None,
        activate_json_backend=True,
        run_inference=False,
        country=None,
        admin1_region=None,
        metadata_table=None,
        field_mapping=None,
        video_frame_interval_seconds=1.0,
        max_video_frames=12,
        run_timestamp_ocr=True,
        timestamp_ocr_max_frames=5,
        allow_metadata_only=False,
    ):
        media_source = self._resolve_media_dir(media_dir).expanduser().resolve()
        if not media_source.exists():
            raise FileNotFoundError(f"Media path not found: {media_source}")

        supported_suffixes = self._supported_detection_media_suffixes()
        if media_source.is_file():
            media_files = (
                [media_source]
                if media_source.suffix.lower() in supported_suffixes
                else []
            )
            media_root = media_source.parent
        else:
            media_files = [
                path for path in sorted(media_source.rglob("*"))
                if path.is_file() and path.suffix.lower() in supported_suffixes
            ]
            media_root = media_source
        if not media_files:
            raise ValueError(f"No supported image/video files found under: {media_source}")

        if output_dir is None:
            output_dir = self._default_detection_record_dir(media_source, detector, classifier)
        else:
            output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        provenance_dir = output_dir / "_provenance"
        inference_dir = provenance_dir / "inference"
        provenance_dir.mkdir(parents=True, exist_ok=True)

        metadata_index = self._load_media_metadata_index(
            metadata_table=metadata_table,
            field_mapping=field_mapping,
            media_root=media_root,
        )
        record_rows = []
        placeholder_count = 0
        skipped_without_prediction_count = 0
        confidence_filtered_count = 0
        sidecar_loaded_count = 0
        model_inferred_count = 0
        model_inferred_image_count = 0
        model_inferred_video_frame_count = 0
        records_with_bbox_count = 0
        records_with_confidence_count = 0
        records_with_coordinates_count = 0
        records_with_observed_timestamp_count = 0
        timestamp_ocr_count = 0
        detected_species = Counter()
        observed_model_versions = Counter()
        observed_prediction_sources = Counter()
        inference_failures = []
        timestamp_ocr_failures = []
        video_manifests = []
        timestamp_ocr_manifests = []
        min_conf = float(min_confidence) if min_confidence is not None else None
        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        video_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        supported_model_pair = (
            str(detector).lower() == "megadetector"
            and str(classifier).lower() == "speciesnet"
        )

        for media_path in media_files:
            table_payload = self._metadata_for_media_path(
                media_path,
                media_root,
                metadata_index,
            )
            sidecar_payload = self._load_detection_sidecar(media_path)
            prediction_payloads = []
            timestamp_ocr_payload = {}

            if run_inference:
                if not supported_model_pair:
                    raise ValueError(
                        "Built-in inference currently supports detector='megadetector' "
                        "with classifier='speciesnet'. Other model outputs can be "
                        "ingested through sidecar JSON without natural-language routing."
                    )
                try:
                    if media_path.suffix.lower() in image_suffixes:
                        prediction = run_speciesnet_on_image(
                            media_path,
                            output_json=(
                                inference_dir
                                / f"{self._stable_hash_id(media_path, prefix='img_')}.json"
                            ),
                            country=country,
                            admin1_region=admin1_region,
                        )
                        prediction_payloads = [{
                            **prediction,
                            "_perception_source": "speciesnet_image_inference",
                            "_frame_offset_seconds": 0.0,
                            "_derived_frame_path": "",
                            "_bbox_format": "xywh",
                        }]
                    elif media_path.suffix.lower() in video_suffixes:
                        video_result = run_speciesnet_on_video(
                            media_path,
                            output_dir=inference_dir,
                            frame_interval_seconds=video_frame_interval_seconds,
                            max_frames=max_video_frames,
                            country=country,
                            admin1_region=admin1_region,
                        )
                        video_manifests.append(video_result.get("manifest_json", ""))
                        prediction_payloads = [{
                            **prediction,
                            "_perception_source": "speciesnet_sampled_video_frame_inference",
                            "_frame_offset_seconds": prediction.get("frame_offset_seconds", 0.0),
                            "_derived_frame_path": prediction.get("frame_path", ""),
                            "_bbox_format": "xywh",
                        } for prediction in video_result.get("predictions", [])]
                        inference_failures.extend(video_result.get("failures", []))
                except Exception as exc:
                    inference_failures.append({
                        "media_path": str(media_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    if not sidecar_payload:
                        continue

            if (
                run_timestamp_ocr
                and media_path.suffix.lower() in video_suffixes
                and not self._merge_nonblank_metadata(
                    table_payload,
                    sidecar_payload,
                ).get("timestamp")
            ):
                frame_paths = [
                    item.get("_derived_frame_path")
                    for item in prediction_payloads
                    if item.get("_derived_frame_path")
                ]
                try:
                    ocr_result = extract_video_timestamp_with_paddleocr(
                        media_path,
                        output_dir=inference_dir / "timestamp_ocr",
                        frame_paths=frame_paths or None,
                        max_frames=timestamp_ocr_max_frames,
                    )
                    timestamp_ocr_payload = {
                        "timestamp": ocr_result["timestamp"],
                        "timestamp_source": ocr_result["timestamp_source"],
                        "timestamp_is_synthetic": False,
                        "timestamp_ocr_score": ocr_result.get("ocr_score"),
                        "timestamp_ocr_vote_count": ocr_result.get("vote_count"),
                        "timestamp_ocr_manifest": ocr_result.get("manifest_json"),
                    }
                    timestamp_ocr_count += 1
                    timestamp_ocr_manifests.append(
                        ocr_result.get("manifest_json", "")
                    )
                except Exception as exc:
                    timestamp_ocr_failures.append({
                        "media_path": str(media_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    })

            if not prediction_payloads and sidecar_payload:
                prediction_payloads = [{
                    **sidecar_payload,
                    "_perception_source": "sidecar_prediction",
                    "_frame_offset_seconds": sidecar_payload.get("frame_offset_seconds", 0.0),
                    "_derived_frame_path": sidecar_payload.get("frame_path", ""),
                    "_bbox_format": sidecar_payload.get("bbox_format", "xyxy"),
                }]
            elif not prediction_payloads and allow_metadata_only:
                prediction_payloads = [{
                    "_perception_source": "metadata_only",
                    "_frame_offset_seconds": 0.0,
                    "_derived_frame_path": "",
                    "_bbox_format": "xyxy",
                }]
            elif not prediction_payloads:
                skipped_without_prediction_count += 1
                continue

            for prediction_index, prediction_payload in enumerate(
                prediction_payloads,
                start=1,
            ):
                combined = self._merge_nonblank_metadata(
                    table_payload,
                    timestamp_ocr_payload,
                    sidecar_payload,
                    prediction_payload,
                )
                perception_source = str(
                    prediction_payload.get("_perception_source", "")
                ).strip()
                species_name = (
                    str(combined.get("species", "")).strip()
                    or str(combined.get("scientific_name", "")).strip()
                    or default_species
                )
                scientific_name = (
                    str(combined.get("scientific_name", "")).strip()
                    or species_name
                )
                confidence = self._extract_confidence_value(combined)
                model_version = str(
                    combined.get("model_version", "")
                ).strip()
                prediction_source = str(
                    combined.get("prediction_source", "")
                ).strip()
                if min_conf is not None and (
                    confidence is None or confidence < min_conf
                ):
                    confidence_filtered_count += 1
                    continue

                raw_bbox = combined.get("bbox")
                if (
                    str(prediction_payload.get("_bbox_format", "")).lower() == "xywh"
                    and isinstance(raw_bbox, (list, tuple))
                    and len(raw_bbox) == 4
                ):
                    x, y, width, height = [
                        self._coerce_float(value)
                        for value in raw_bbox
                    ]
                    if None not in {x, y, width, height}:
                        raw_bbox = [x, y, x + width, y + height]
                bbox = self._normalize_bbox(raw_bbox, metadata=combined)

                explicit_timestamp = self.safe_parse_timestamp(
                    combined.get("timestamp")
                )
                parsed_time = explicit_timestamp
                if parsed_time is not None:
                    timestamp_source = (
                        "metadata_table"
                        if table_payload.get("timestamp") is not None
                        else str(
                            combined.get(
                                "timestamp_source",
                                "prediction_sidecar",
                            )
                        )
                    )
                    timestamp_is_synthetic = bool(
                        combined.get("timestamp_is_synthetic", False)
                    )
                else:
                    parsed_time = self._infer_timestamp_from_media_path(media_path)
                    filename_has_timestamp = bool(
                        re.search(r"\d{8}[_-]?\d{6}", media_path.stem)
                    )
                    timestamp_source = (
                        "media_filename"
                        if filename_has_timestamp
                        else "filesystem_modified_time"
                    )
                    timestamp_is_synthetic = not filename_has_timestamp

                frame_offset_seconds = float(
                    prediction_payload.get("_frame_offset_seconds", 0.0) or 0.0
                )
                if parsed_time is not None and frame_offset_seconds:
                    parsed_time = parsed_time + timedelta(
                        seconds=frame_offset_seconds
                    )
                    timestamp_source += "_plus_video_frame_offset"
                timestamp_text = (
                    parsed_time.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")
                    if parsed_time
                    else ""
                )
                camera_id = (
                    str(combined.get("camera_id", "")).strip()
                    or self._infer_camera_id_from_path(media_path)
                )
                try:
                    individuals = max(int(float(combined.get("count", 1) or 1)), 1)
                except Exception:
                    individuals = 1

                metadata = {
                    "filename": media_path.name,
                    "camera_id": camera_id,
                    "latitude": combined.get("latitude"),
                    "longitude": combined.get("longitude"),
                    "timestamp": timestamp_text,
                    "timestamp_original_date": timestamp_text[:10] if timestamp_text else "",
                    "timestamp_source": timestamp_source,
                    "timestamp_is_synthetic": timestamp_is_synthetic,
                    "timestamp_ocr_score": combined.get(
                        "timestamp_ocr_score",
                        "",
                    ),
                    "timestamp_ocr_vote_count": combined.get(
                        "timestamp_ocr_vote_count",
                        "",
                    ),
                    "timestamp_ocr_manifest": combined.get(
                        "timestamp_ocr_manifest",
                        "",
                    ),
                    "species": species_name or "Unknown",
                    "scientific_name": scientific_name or "",
                    "confidence": round(confidence, 4) if confidence is not None else "",
                    "individuals": individuals,
                    "behavior": str(combined.get("behavior", "")).strip() or "Unknown",
                    "special_observations": str(
                        combined.get("notes", "")
                    ).strip(),
                    "ecological_report": "",
                    "media_abspath": str(media_path),
                    "camtrap_file_path": str(media_path),
                    "camtrap_file_mediatype": self._guess_media_mimetype(
                        media_path.name,
                        fallback="application/octet-stream",
                    ),
                    "camtrap_source_frame_offset_seconds": round(
                        frame_offset_seconds,
                        6,
                    ),
                    "derived_frame_path": str(
                        prediction_payload.get("_derived_frame_path", "")
                    ),
                    "visual_frontend_detector": detector,
                    "visual_frontend_classifier": classifier,
                    "visual_frontend_status": perception_source,
                    "perception_model_version": model_version,
                    "prediction_source": prediction_source,
                    "visual_frontend_prediction_json": str(
                        prediction_payload.get("predictions_json", "")
                    ),
                    "bbox": bbox,
                }
                if len(bbox) == 4:
                    metadata["bbox_x"] = bbox[0]
                    metadata["bbox_y"] = bbox[1]
                    metadata["bbox_width"] = round(bbox[2] - bbox[0], 4)
                    metadata["bbox_height"] = round(bbox[3] - bbox[1], 4)

                record_id = (
                    f"{self._slugify_text(camera_id)}_"
                    f"{self._slugify_text(media_path.stem)}_"
                    f"{self._stable_hash_id(media_path, prediction_index, frame_offset_seconds, length=10)}"
                )
                (output_dir / f"{record_id}.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                detected_species[metadata["species"]] += 1
                if model_version:
                    observed_model_versions[model_version] += 1
                if prediction_source:
                    observed_prediction_sources[prediction_source] += 1
                record_rows.append({"id": record_id, "metadata": metadata})

                if perception_source == "metadata_only":
                    placeholder_count += 1
                elif perception_source == "sidecar_prediction":
                    sidecar_loaded_count += 1
                else:
                    model_inferred_count += 1
                    if media_path.suffix.lower() in video_suffixes:
                        model_inferred_video_frame_count += 1
                    else:
                        model_inferred_image_count += 1
                if len(bbox) == 4:
                    records_with_bbox_count += 1
                if confidence is not None:
                    records_with_confidence_count += 1
                if (
                    self._coerce_float(metadata.get("latitude")) is not None
                    and self._coerce_float(metadata.get("longitude")) is not None
                ):
                    records_with_coordinates_count += 1
                if timestamp_text and not timestamp_is_synthetic:
                    records_with_observed_timestamp_count += 1

        if not record_rows:
            raise ValueError(
                "No analysis records were produced. Enable model inference, provide "
                "prediction sidecars, or explicitly set allow_metadata_only=True."
            )

        backend_result = (
            self.set_json_data_dir(output_dir)
            if activate_json_backend
            else None
        )
        if model_inferred_count and sidecar_loaded_count:
            pipeline_mode = "mixed_model_inference_and_sidecar_ingest"
        elif model_inferred_count:
            pipeline_mode = "raw_media_model_inference"
        elif sidecar_loaded_count:
            pipeline_mode = "raw_media_with_prediction_sidecars"
        else:
            pipeline_mode = "metadata_only"
        incomplete_media = (
            skipped_without_prediction_count
            + len(inference_failures)
            + placeholder_count
        )
        pipeline_status = (
            "complete"
            if incomplete_media == 0
            else "complete_with_recorded_gaps"
        )
        record_count = len(record_rows)
        readiness = {
            "taxonomic_analysis_ready": any(
                str(item["metadata"].get("species", "")).strip().lower()
                not in {"", "unknown"}
                for item in record_rows
            ),
            "temporal_analysis_ready": records_with_observed_timestamp_count > 0,
            "spatial_analysis_ready": records_with_coordinates_count > 0,
            "camtrap_export_ready": all(
                str(item["metadata"].get("camera_id", "")).strip()
                and str(item["metadata"].get("timestamp", "")).strip()
                for item in record_rows
            ),
        }
        manifest = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "media_source": str(media_source),
            "metadata_table": metadata_index.get("source", ""),
            "metadata_field_mapping": metadata_index.get("field_mapping", {}),
            "media_file_count": len(media_files),
            "record_count": record_count,
            "detector": detector,
            "classifier": classifier,
            "model_versions": dict(observed_model_versions.most_common()),
            "prediction_sources": dict(
                observed_prediction_sources.most_common()
            ),
            "run_inference": bool(run_inference),
            "video_sampling": {
                "frame_interval_seconds": float(video_frame_interval_seconds),
                "max_frames": int(max_video_frames),
                "video_manifests": [
                    path for path in video_manifests if path
                ],
            },
            "pipeline_mode": pipeline_mode,
            "pipeline_status": pipeline_status,
            "counts": {
                "metadata_only": placeholder_count,
                "prediction_sidecars": sidecar_loaded_count,
                "model_inferred": model_inferred_count,
                "model_inferred_images": model_inferred_image_count,
                "model_inferred_video_frames": model_inferred_video_frame_count,
                "skipped_without_prediction": skipped_without_prediction_count,
                "confidence_filtered": confidence_filtered_count,
                "inference_failures": len(inference_failures),
                "timestamp_ocr_successes": timestamp_ocr_count,
                "timestamp_ocr_failures": len(timestamp_ocr_failures),
                "with_bbox": records_with_bbox_count,
                "with_confidence": records_with_confidence_count,
                "with_coordinates": records_with_coordinates_count,
                "with_observed_timestamp": records_with_observed_timestamp_count,
            },
            "analysis_readiness": readiness,
            "species_breakdown": dict(detected_species.most_common()),
            "inference_failures": inference_failures,
            "timestamp_ocr_manifests": [
                path for path in timestamp_ocr_manifests if path
            ],
            "timestamp_ocr_failures": timestamp_ocr_failures,
        }
        manifest_path = provenance_dir / "detection_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "status": "ok",
            "media_source": str(media_source),
            "output_dir": str(output_dir),
            "record_count": record_count,
            "placeholder_record_count": placeholder_count,
            "skipped_without_prediction_count": skipped_without_prediction_count,
            "sidecar_loaded_record_count": sidecar_loaded_count,
            "model_inferred_record_count": model_inferred_count,
            "model_inferred_image_count": model_inferred_image_count,
            "model_inferred_video_frame_count": model_inferred_video_frame_count,
            "records_with_bbox_count": records_with_bbox_count,
            "records_with_confidence_count": records_with_confidence_count,
            "records_with_coordinates_count": records_with_coordinates_count,
            "records_with_observed_timestamp_count": records_with_observed_timestamp_count,
            "timestamp_ocr_count": timestamp_ocr_count,
            "timestamp_ocr_failure_count": len(timestamp_ocr_failures),
            "detector": detector,
            "classifier": classifier,
            "model_versions": dict(observed_model_versions.most_common()),
            "prediction_sources": dict(
                observed_prediction_sources.most_common()
            ),
            "video_sampling": {
                "frame_interval_seconds": float(
                    video_frame_interval_seconds
                ),
                "max_frames": int(max_video_frames),
            },
            "pipeline_mode": pipeline_mode,
            "pipeline_status": pipeline_status,
            "analysis_readiness": readiness,
            "species_breakdown": dict(detected_species.most_common()),
            "backend_result": backend_result,
            "detection_manifest": str(manifest_path),
            "sample_records": [
                self._build_record_preview(item)
                for item in record_rows[:5]
            ],
            "note": (
                "Raw images are analysed directly by SpeciesNet; raw videos are "
                "sampled into auditable frames and each retained frame is analysed "
                "by the same SpeciesNet ensemble (including its MegaDetector stage). "
                "Existing model predictions can instead be ingested from generic "
                "sidecar JSON. No ecological-question keywords are used for routing."
            ),
        }

    def prepare_camera_trap_dataset(
        self,
        input_path,
        input_type="auto",
        output_dir=None,
        field_mapping=None,
        metadata_table=None,
        detector="megadetector",
        classifier="speciesnet",
        run_inference=True,
        country=None,
        admin1_region=None,
        min_confidence=None,
        video_frame_interval_seconds=1.0,
        max_video_frames=12,
        run_timestamp_ocr=True,
        timestamp_ocr_max_frames=5,
        min_gap_minutes=30,
        package_name=None,
        package_title=None,
        source_timezone="UTC",
        activate_backend=True,
        persist_active=True,
        allow_metadata_only=False,
    ):
        """
        Prepare one of three interoperable inputs for downstream agent tools.

        This tool dispatches only on an explicit/inspectable data format
        (raw media, tabular records, or CamtrapDP). It does not inspect the
        ecological question and does not choose downstream analysis tools.
        """
        source = Path(input_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Input path not found: {source}")

        requested_type = str(input_type or "auto").strip().lower()
        aliases = {
            "raw": "raw_media",
            "media": "raw_media",
            "video": "raw_media",
            "image": "raw_media",
            "csv": "table",
            "tsv": "table",
            "excel": "table",
            "camtrap": "camtrap_dp",
            "camtrapdp": "camtrap_dp",
            "dp": "camtrap_dp",
        }
        requested_type = aliases.get(requested_type, requested_type)
        supported_types = {"auto", "raw_media", "table", "camtrap_dp"}
        if requested_type not in supported_types:
            raise ValueError(
                "input_type must be one of auto, raw_media, table, or camtrap_dp."
            )

        detected_type = requested_type
        if requested_type == "auto":
            if (
                (source.is_file() and source.name == "datapackage.json")
                or (source.is_dir() and (source / "datapackage.json").exists())
            ):
                detected_type = "camtrap_dp"
            elif source.is_file() and source.suffix.lower() in {
                ".csv", ".tsv", ".tab", ".xlsx", ".xls",
            }:
                detected_type = "table"
            elif (
                source.is_file()
                and source.suffix.lower() in self._supported_detection_media_suffixes()
            ) or (
                source.is_dir()
                and any(
                    path.is_file()
                    and path.suffix.lower() in self._supported_detection_media_suffixes()
                    for path in source.rglob("*")
                )
            ):
                detected_type = "raw_media"
            else:
                raise ValueError(
                    "Could not infer the input data format. Set input_type explicitly."
                )

        if detected_type == "camtrap_dp":
            backend_result = self.set_camtrap_data_dir(
                source,
                activate=activate_backend,
                persist_active=persist_active,
                write_validation=True,
            )
            return {
                "status": "ok",
                "input_type": detected_type,
                "input_path": str(source),
                "pipeline_stages": ["CamtrapDP validation", "backend activation"],
                "datapackage_path": backend_result["datapackage_path"],
                "backend_result": backend_result,
                "analysis_readiness": {
                    "camtrap_dp_ready": (
                        backend_result["validation"].get("error_count", 0) == 0
                    ),
                    "active_for_downstream_tools": bool(activate_backend),
                },
                "routing_note": (
                    "The LLM selected this generic preparation tool. Local code "
                    "only validated the supplied file format and did not route an "
                    "ecological question to an analysis."
                ),
            }

        if output_dir is None:
            prepared_root = self._make_unique_dir(
                CAMTRAP_IMPORTS_DIR,
                f"{source.stem or source.name}_prepared",
            )
        else:
            prepared_root = Path(output_dir).expanduser().resolve()
        prepared_root.mkdir(parents=True, exist_ok=True)

        if detected_type == "table":
            conversion = self.convert_table_to_camtrap_dp(
                table_path=source,
                field_mapping=field_mapping,
                output_dir=prepared_root,
                package_name=package_name,
                package_title=package_title,
                min_gap_minutes=min_gap_minutes,
                activate_backend=False,
                source_timezone=source_timezone,
            )
            if conversion.get("status") != "ok":
                return {
                    **conversion,
                    "input_type": detected_type,
                    "pipeline_stages": [
                        "tabular field mapping",
                        "CamtrapDP conversion",
                    ],
                }
            backend_result = self.set_camtrap_data_dir(
                conversion["datapackage_path"],
                activate=activate_backend,
                persist_active=persist_active,
                write_validation=True,
            )
            return {
                "status": "ok",
                "input_type": detected_type,
                "input_path": str(source),
                "pipeline_stages": [
                    "tabular field mapping",
                    "timestamp normalization",
                    "CamtrapDP conversion",
                    "CamtrapDP validation",
                    "backend activation",
                ],
                "datapackage_path": conversion["datapackage_path"],
                "conversion": conversion,
                "backend_result": backend_result,
                "analysis_readiness": {
                    "camtrap_dp_ready": (
                        backend_result["validation"].get("error_count", 0) == 0
                    ),
                    "active_for_downstream_tools": bool(activate_backend),
                },
                "routing_note": (
                    "The LLM selected this generic preparation tool; deterministic "
                    "code mapped the declared table fields to CamtrapDP."
                ),
            }

        record_dir = prepared_root / "structured_records"
        package_dir = prepared_root / "camtrap_dp"
        detection = self.run_detection_pipeline(
            media_dir=source,
            detector=detector,
            classifier=classifier,
            output_dir=record_dir,
            default_species="Unknown",
            min_confidence=min_confidence,
            activate_json_backend=True,
            run_inference=run_inference,
            country=country,
            admin1_region=admin1_region,
            metadata_table=metadata_table,
            field_mapping=field_mapping,
            video_frame_interval_seconds=video_frame_interval_seconds,
            max_video_frames=max_video_frames,
            run_timestamp_ocr=run_timestamp_ocr,
            timestamp_ocr_max_frames=timestamp_ocr_max_frames,
            allow_metadata_only=allow_metadata_only,
        )
        export_result = self.export_camtrap_dp(
            output_dir=package_dir,
            include_event_observations=True,
            min_gap_minutes=min_gap_minutes,
            package_name=package_name,
            package_title=package_title,
            source_timezone=source_timezone,
        )
        backend_result = self.set_camtrap_data_dir(
            export_result["datapackage_path"],
            activate=activate_backend,
            persist_active=persist_active,
            write_validation=True,
        )
        validation = backend_result["validation"]
        perception_executed = detection["model_inferred_record_count"] > 0
        data_pipeline_complete = bool(
            detection.get("record_count", 0)
            and export_result.get("status") == "ok"
            and validation.get("error_count", 0) == 0
            and detection.get("placeholder_record_count", 0) == 0
        )
        full_pipeline_complete = bool(
            data_pipeline_complete
            and perception_executed
            and detection.get("analysis_readiness", {}).get(
                "taxonomic_analysis_ready"
            )
        )
        return {
            "status": "ok" if data_pipeline_complete else "warning",
            "input_type": detected_type,
            "input_path": str(source),
            "pipeline_stages": [
                "raw-media discovery",
                (
                    "sampled-video-frame perception"
                    if detection.get("model_inferred_video_frame_count", 0)
                    else "image perception or prediction-sidecar ingestion"
                ),
                "structured observation records",
                "CamtrapDP export",
                "CamtrapDP validation",
                "backend activation",
            ],
            "perception_executed": perception_executed,
            "data_to_camtrap_pipeline_complete": data_pipeline_complete,
            "full_raw_media_pipeline_complete": full_pipeline_complete,
            "structured_records_dir": detection["output_dir"],
            "datapackage_path": export_result["datapackage_path"],
            "detection": detection,
            "camtrap_export": export_result,
            "backend_result": backend_result,
            "analysis_readiness": {
                **detection.get("analysis_readiness", {}),
                "camtrap_dp_ready": validation.get("error_count", 0) == 0,
                "active_for_downstream_tools": bool(activate_backend),
            },
            "routing_note": (
                "The LLM selected this registered, format-agnostic preparation "
                "tool. Its internal branching uses only the supplied data format; "
                "it contains no ecological-question keyword router."
            ),
        }

    def predict_image_with_speciesnet(
        self,
        image_path,
        country=None,
        admin1_region=None,
        output_json=None,
        write_sidecar=True,
    ):
        result = run_speciesnet_on_image(
            image_path,
            country=country,
            admin1_region=admin1_region,
            output_json=output_json,
        )
        sidecar_path = None
        if write_sidecar:
            sidecar_path = write_speciesnet_sidecar(result)
        return {
            "image_path": result["image_path"],
            "species": result.get("species"),
            "scientific_name": result.get("scientific_name"),
            "confidence": result.get("confidence"),
            "prediction": result.get("prediction"),
            "prediction_score": result.get("prediction_score"),
            "prediction_source": result.get("prediction_source"),
            "model_version": result.get("model_version"),
            "bbox": result.get("bbox") or [],
            "bbox_format": "xywh_normalized",
            "detections": result.get("detections") or [],
            "classifications": result.get("classifications") or [],
            "predictions_json": result.get("predictions_json"),
            "sidecar_json": sidecar_path,
            "note": "Ran the official SpeciesNet ensemble, including MegaDetector object detection and SpeciesNet classification.",
        }

    def _build_export_event_groups(
        self,
        records,
        min_gap_minutes=30,
        source_timezone="UTC",
    ):
        min_gap_minutes = max(int(min_gap_minutes), 0)
        prepared = []
        for record in records:
            metadata = record.get("metadata", {}) or {}
            parsed_time = self._source_datetime_to_utc(
                metadata.get("timestamp"),
                source_timezone=source_timezone,
            )
            if parsed_time is None:
                continue
            deployment_id = str(metadata.get("camtrap_deployment_id", "")).strip() or (
                f"dep_{str(metadata.get('camera_id', '')).strip()}"
            )
            species = str(metadata.get("scientific_name") or metadata.get("species") or "Unknown").strip()
            prepared.append({
                "record": record,
                "deployment_id": deployment_id,
                "species": species,
                "parsed_time": parsed_time,
            })

        prepared.sort(key=lambda item: (item["deployment_id"], item["species"], item["parsed_time"], item["record"]["id"]))
        groups = []
        grouped = defaultdict(list)
        for item in prepared:
            grouped[(item["deployment_id"], item["species"])].append(item)

        for (deployment_id, species), items in grouped.items():
            current = []
            for item in items:
                if not current:
                    current = [item]
                    continue

                prev_time = current[-1]["parsed_time"]
                gap_minutes = (item["parsed_time"] - prev_time).total_seconds() / 60.0
                if gap_minutes <= min_gap_minutes:
                    current.append(item)
                    continue

                groups.append((deployment_id, species, current))
                current = [item]

            if current:
                groups.append((deployment_id, species, current))

        event_rows = []
        record_to_event = {}
        for index, (deployment_id, species, group_items) in enumerate(groups, start=1):
            start_dt = group_items[0]["parsed_time"]
            end_dt = group_items[-1]["parsed_time"]
            event_id = f"{self._slugify_text(deployment_id)}_{self._slugify_text(species)}_{index:05d}"
            counts = []
            behaviors = []
            for item in group_items:
                record_id = item["record"]["id"]
                record_to_event[record_id] = {
                    "event_id": event_id,
                    "event_start": start_dt,
                    "event_end": end_dt,
                }
                metadata = item["record"].get("metadata", {}) or {}
                try:
                    count_value = int(metadata.get("individuals", 1))
                    if count_value > 0:
                        counts.append(count_value)
                except Exception:
                    pass
                behavior = str(metadata.get("behavior", "")).strip()
                if behavior and behavior.lower() != "unknown":
                    behaviors.append(behavior)

            event_rows.append({
                "observationID": f"{event_id}_event",
                "deploymentID": deployment_id,
                "mediaID": "",
                "eventID": event_id,
                "eventStart": self._to_camtrap_iso(start_dt),
                "eventEnd": self._to_camtrap_iso(end_dt),
                "observationLevel": "event",
                "observationType": "animal",
                "cameraSetupType": "",
                "scientificName": species,
                "count": max(counts) if counts else 1,
                "lifeStage": "",
                "sex": "",
                "behavior": Counter(behaviors).most_common(1)[0][0] if behaviors else "",
                "individualID": "",
                "individualPositionRadius": "",
                "individualPositionAngle": "",
                "individualSpeed": "",
                "bboxX": "",
                "bboxY": "",
                "bboxWidth": "",
                "bboxHeight": "",
                "classificationMethod": "machine",
                "classifiedBy": "camagent",
                "classificationTimestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "classificationProbability": "",
                "observationTags": "",
                "observationComments": "Exported event-level summary from camagent",
            })

        return record_to_event, event_rows

    def export_camtrap_dp(
        self,
        output_dir=None,
        species=None,
        include_event_observations=True,
        min_gap_minutes=30,
        package_name=None,
        package_title=None,
        package_version="1.0.2",
        data_license=None,
        media_license=None,
        source_timezone="UTC",
    ):
        """
        Export the current internal records as a CamtrapDP package.
        Internal records may not contain full deployment metadata, so missing deployments are aggregated by camera.
        """
        source_records = self.query_by_species(species) if species else self.get_all_records()
        if not source_records:
            raise ValueError("No records available for CamtrapDP export.")

        exportable = []
        species_counter = Counter()
        for record in source_records:
            metadata = record.get("metadata", {}) or {}
            camera_id = str(metadata.get("camera_id", "")).strip()
            timestamp_text = str(metadata.get("timestamp", "")).strip()
            parsed_time = self._source_datetime_to_utc(
                timestamp_text,
                source_timezone=source_timezone,
            )
            if not camera_id or parsed_time is None:
                continue
            exportable.append({
                "record": record,
                "metadata": metadata,
                "camera_id": camera_id,
                "parsed_time": parsed_time,
            })
            export_species = str(metadata.get("scientific_name") or metadata.get("species") or "Unknown").strip()
            species_counter[export_species] += 1

        if not exportable:
            raise ValueError("No exportable records have both camera_id and timestamp.")

        package_name = package_name or f"{self._slugify_text(species or 'wildlife')}_camtrapdp_export"
        package_title = package_title or f"CamtrapDP export from camagent ({species or 'all species'})"

        if output_dir is None:
            output_dir = self._make_unique_dir(CAMTRAP_EXPORTS_DIR, package_name)
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        deployment_groups = defaultdict(list)
        for item in exportable:
            metadata = item["metadata"]
            deployment_id = str(metadata.get("camtrap_deployment_id", "")).strip() or f"dep_{item['camera_id']}"
            deployment_groups[deployment_id].append(item)

        deployment_rows = []
        deployment_lookup = {}
        for deployment_id, items in sorted(deployment_groups.items()):
            first_meta = items[0]["metadata"]
            timestamps = [item["parsed_time"] for item in items]
            deployment_rows.append({
                "deploymentID": deployment_id,
                "locationID": str(first_meta.get("camtrap_location_id", "")).strip() or items[0]["camera_id"],
                "locationName": str(first_meta.get("camtrap_location_name", "")).strip() or items[0]["camera_id"],
                "latitude": first_meta.get("latitude", ""),
                "longitude": first_meta.get("longitude", ""),
                "coordinateUncertainty": str(first_meta.get("camtrap_coordinate_uncertainty", "")).strip(),
                "deploymentStart": self._to_camtrap_iso(min(timestamps)),
                "deploymentEnd": self._to_camtrap_iso(max(timestamps)),
                "setupBy": str(first_meta.get("camtrap_setup_by", "")).strip(),
                "cameraID": items[0]["camera_id"],
                "cameraModel": str(first_meta.get("camtrap_camera_model", "")).strip(),
                "cameraDelay": str(first_meta.get("camtrap_camera_delay", "")).strip(),
                "cameraHeight": str(first_meta.get("camtrap_camera_height", "")).strip(),
                "cameraDepth": str(first_meta.get("camtrap_camera_depth", "")).strip(),
                "cameraTilt": str(first_meta.get("camtrap_camera_tilt", "")).strip(),
                "cameraHeading": str(first_meta.get("camtrap_camera_heading", "")).strip(),
                "detectionDistance": str(first_meta.get("camtrap_detection_distance", "")).strip(),
                "timestampIssues": self._stringify_bool(first_meta.get("timestamp_is_synthetic")),
                "baitUse": self._stringify_bool(first_meta.get("camtrap_bait_use")),
                "featureType": str(first_meta.get("camtrap_feature_type", "")).strip(),
                "habitat": str(first_meta.get("camtrap_habitat", "")).strip(),
                "deploymentGroups": str(first_meta.get("camtrap_deployment_groups", "")).strip(),
                "deploymentTags": str(first_meta.get("camtrap_deployment_tags", "")).strip(),
                "deploymentComments": str(first_meta.get("camtrap_deployment_comments", "")).strip() or "Deployment interval inferred from first and last exported detection; not camera-operation effort.",
            })
            deployment_lookup[deployment_id] = deployment_rows[-1]

        record_to_event = {}
        event_rows = []
        if include_event_observations:
            record_to_event, event_rows = self._build_export_event_groups(
                [item["record"] for item in exportable],
                min_gap_minutes=min_gap_minutes,
                source_timezone=source_timezone,
            )

        media_rows = []
        observation_rows = list(event_rows)
        for item in exportable:
            record = item["record"]
            metadata = item["metadata"]
            deployment_id = str(metadata.get("camtrap_deployment_id", "")).strip() or f"dep_{item['camera_id']}"
            record_id = str(record.get("id", "")).strip() or uuid.uuid4().hex[:12]
            media_id = str(metadata.get("camtrap_media_id", "")).strip() or record_id
            file_path = str(metadata.get("camtrap_file_path", "")).strip() or str(metadata.get("filename", "")).strip()
            file_name = str(metadata.get("filename", "")).strip() or Path(file_path).name or f"{media_id}.jpg"
            media_rows.append({
                "mediaID": media_id,
                "deploymentID": deployment_id,
                "captureMethod": str(metadata.get("camtrap_capture_method", "")).strip() or "activityDetection",
                "timestamp": self._to_camtrap_iso(item["parsed_time"]),
                "filePath": file_path or file_name,
                "filePublic": self._stringify_bool(metadata.get("camtrap_file_public")),
                "fileName": file_name,
                "fileMediatype": str(metadata.get("camtrap_file_mediatype", "")).strip() or self._guess_media_mimetype(file_name),
                "exifData": "",
                "favorite": "",
                "mediaComments": (
                    str(metadata.get("camtrap_observation_comments", "")).strip()
                    + f"; timestamp_source={metadata.get('timestamp_source', 'unspecified')}"
                    + f"; timestamp_is_synthetic={str(bool(metadata.get('timestamp_is_synthetic', False))).lower()}"
                ).strip("; "),
            })

            export_species = str(metadata.get("scientific_name") or metadata.get("species") or "Unknown").strip()
            event_info = record_to_event.get(record_id)
            if event_info is None:
                fallback_event_id = str(metadata.get("camtrap_event_id", "")).strip() or f"{self._slugify_text(deployment_id)}_{self._slugify_text(export_species)}_{record_id}"
                event_info = {
                    "event_id": fallback_event_id,
                    "event_start": item["parsed_time"],
                    "event_end": item["parsed_time"],
                }

            try:
                exported_count = int(float(metadata.get("individuals", 1) or 1))
                if exported_count <= 0:
                    exported_count = 1
            except Exception:
                exported_count = 1
            bbox = self._normalize_bbox(metadata.get("bbox"), metadata=metadata)
            if len(bbox) == 4:
                bbox_x, bbox_y, bbox_x2, bbox_y2 = bbox
                bbox_width = max(bbox_x2 - bbox_x, 0.0)
                bbox_height = max(bbox_y2 - bbox_y, 0.0)
            else:
                bbox_x = bbox_y = bbox_width = bbox_height = ""

            observation_rows.append({
                "observationID": str(metadata.get("camtrap_observation_id", "")).strip() or f"{media_id}_1",
                "deploymentID": deployment_id,
                "mediaID": media_id,
                "eventID": event_info["event_id"],
                "eventStart": self._to_camtrap_iso(event_info["event_start"]),
                "eventEnd": self._to_camtrap_iso(event_info["event_end"]),
                "observationLevel": "media",
                "observationType": "animal",
                "cameraSetupType": "",
                "scientificName": export_species,
                "count": exported_count,
                "lifeStage": "",
                "sex": "",
                "behavior": str(metadata.get("behavior", "")).strip() if str(metadata.get("behavior", "")).strip().lower() != "unknown" else "",
                "individualID": str(metadata.get("camtrap_individual_id", "")).strip(),
                "individualPositionRadius": "",
                "individualPositionAngle": "",
                "individualSpeed": "",
                "bboxX": bbox_x,
                "bboxY": bbox_y,
                "bboxWidth": bbox_width,
                "bboxHeight": bbox_height,
                "classificationMethod": str(metadata.get("camtrap_classification_method", "")).strip() or "machine",
                "classifiedBy": str(metadata.get("camtrap_classified_by", "")).strip() or "camagent",
                "classificationTimestamp": str(metadata.get("camtrap_classification_timestamp", "")).strip()
                or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "classificationProbability": self._normalize_probability_text(metadata.get("confidence")),
                "observationTags": str(metadata.get("camtrap_observation_tags", "")).strip(),
                "observationComments": str(metadata.get("special_observations", "")).strip(),
            })

        deployments_fieldnames = [
            "deploymentID", "locationID", "locationName", "latitude", "longitude", "coordinateUncertainty",
            "deploymentStart", "deploymentEnd", "setupBy", "cameraID", "cameraModel", "cameraDelay",
            "cameraHeight", "cameraDepth", "cameraTilt", "cameraHeading", "detectionDistance", "timestampIssues",
            "baitUse", "featureType", "habitat", "deploymentGroups", "deploymentTags", "deploymentComments",
        ]
        media_fieldnames = [
            "mediaID", "deploymentID", "captureMethod", "timestamp", "filePath", "filePublic",
            "fileName", "fileMediatype", "exifData", "favorite", "mediaComments",
        ]
        observations_fieldnames = [
            "observationID", "deploymentID", "mediaID", "eventID", "eventStart", "eventEnd", "observationLevel",
            "observationType", "cameraSetupType", "scientificName", "count", "lifeStage", "sex", "behavior",
            "individualID", "individualPositionRadius", "individualPositionAngle", "individualSpeed", "bboxX",
            "bboxY", "bboxWidth", "bboxHeight", "classificationMethod", "classifiedBy", "classificationTimestamp",
            "classificationProbability", "observationTags", "observationComments",
        ]

        deployments_path = output_dir / "deployments.csv"
        media_path = output_dir / "media.csv"
        observations_path = output_dir / "observations.csv"
        self._write_csv_rows(deployments_path, deployments_fieldnames, deployment_rows)
        self._write_csv_rows(media_path, media_fieldnames, media_rows)
        self._write_csv_rows(observations_path, observations_fieldnames, observation_rows)

        all_times = [item["parsed_time"] for item in exportable]
        datapackage = {
            "resources": [
                {
                    "name": "deployments",
                    "path": "deployments.csv",
                    "profile": "tabular-data-resource",
                    "format": "csv",
                    "mediatype": "text/csv",
                    "encoding": "utf-8",
                    "schema": "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0.2/deployments-table-schema.json",
                },
                {
                    "name": "media",
                    "path": "media.csv",
                    "profile": "tabular-data-resource",
                    "format": "csv",
                    "mediatype": "text/csv",
                    "encoding": "utf-8",
                    "schema": "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0.2/media-table-schema.json",
                },
                {
                    "name": "observations",
                    "path": "observations.csv",
                    "profile": "tabular-data-resource",
                    "format": "csv",
                    "mediatype": "text/csv",
                    "encoding": "utf-8",
                    "schema": "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0.2/observations-table-schema.json",
                },
            ],
            "profile": "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0.2/camtrap-dp-profile.json",
            "name": self._slugify_text(package_name),
            "id": str(uuid.uuid4()),
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "title": package_title,
            "description": "Exported from camagent for offline ecological analysis and interoperable biodiversity sharing.",
            "version": package_version,
            "keywords": ["camera traps", "camtrapdp", "camagent"] + [name for name, _ in species_counter.most_common(10)],
            "licenses": [
                {"name": name, "scope": scope}
                for name, scope in ((data_license, "data"), (media_license, "media"))
                if name
            ],
            "project": {
                "title": "camagent camera trap export",
                "observationLevel": ["media", "event"] if include_event_observations else ["media"],
                "captureMethod": sorted({row["captureMethod"] for row in media_rows if row["captureMethod"]}),
                "individualAnimals": False,
            },
            "temporal": {
                "start": min(all_times).strftime("%Y-%m-%d"),
                "end": max(all_times).strftime("%Y-%m-%d"),
            },
            "taxonomic": [
                {"scientificName": name, "taxonRank": "species"}
                for name, _ in species_counter.most_common()
            ],
        }
        (output_dir / "datapackage.json").write_text(
            json.dumps(datapackage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        species_event_counter = Counter(
            str(row.get("scientificName") or "Unknown").strip() or "Unknown"
            for row in event_rows
        )
        species_report = {}
        for species_name in species_counter:
            species_items = [
                item
                for item in exportable
                if str(
                    item["metadata"].get("scientific_name")
                    or item["metadata"].get("species")
                    or "Unknown"
                ).strip() == species_name
            ]
            species_times = [item["parsed_time"] for item in species_items]
            species_report[species_name] = {
                "media_records": int(len(species_items)),
                "independent_events": int(species_event_counter.get(species_name, 0)),
                "camera_count": int(len({
                    item["camera_id"]
                    for item in species_items
                })),
                "date_start": (
                    self._to_camtrap_iso(min(species_times))
                    if species_times
                    else None
                ),
                "date_end": (
                    self._to_camtrap_iso(max(species_times))
                    if species_times
                    else None
                ),
            }
        conversion_report = {
            "schema_version": "1.0",
            "source_backend": (
                "raw_media_structured_records"
                if any(
                    str(item["metadata"].get("media_abspath", "")).strip()
                    for item in exportable
                )
                else "agent_structured_records"
            ),
            "output_dir": str(output_dir),
            "datapackage_path": str(output_dir / "datapackage.json"),
            "kept_media_records": len(exportable),
            "independent_events": len(event_rows),
            "deployment_count": len(deployment_rows),
            "source_timezone": str(source_timezone or "UTC"),
            "min_gap_minutes": int(min_gap_minutes),
            "species_summary": species_report,
            "perception_provenance": dict(Counter(
                str(item["metadata"].get("visual_frontend_status", "unspecified"))
                for item in exportable
            )),
            "note": (
                "Structured image/video observations were exported to CamtrapDP. "
                "Each media-level observation retains model confidence, bounding "
                "box coordinates when available, and its raw-media file path."
            ),
        }
        (output_dir / "conversion_report.json").write_text(
            json.dumps(conversion_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        validation = self._validate_camtrap_package(output_dir)
        self._write_camtrap_conversion_validation(output_dir, validation)

        return {
            "status": "ok",
            "output_dir": str(output_dir),
            "datapackage_path": str(output_dir / "datapackage.json"),
            "record_count": len(exportable),
            "deployment_count": len(deployment_rows),
            "media_count": len(media_rows),
            "observation_count": len(observation_rows),
            "event_observation_count": len(event_rows),
            "species_breakdown": dict(species_counter.most_common()),
            "package_name": datapackage["name"],
            "package_title": package_title,
            "source_timezone": str(source_timezone or "UTC"),
            "validation": validation,
            "conversion_report": str(output_dir / "conversion_report.json"),
            "license_status": "declared" if data_license and media_license else "requires_rights_holder_confirmation",
            "note": (
                "CamtrapDP export completed. When original deployment metadata are missing, deployments are approximated "
                "by grouping records at the camera level and using the earliest and latest timestamps as the deployment window."
            ),
        }
