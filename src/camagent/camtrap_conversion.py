from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .ecology_config import DEFAULT_0525_COLUMN_MAP, DEFAULT_0525_SPECIES_CONFIG
from .paths import CAMTRAP_IMPORTS_DIR, ensure_runtime_dirs


REQUIRED_CANONICAL_COLUMNS = {"longitude", "latitude", "recordtime", "videoname"}
DEFAULT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def stable_id(*parts, prefix="", length=16):
    text = "||".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{digest}" if prefix else digest


def normalize_species_config(species_config=None):
    config = species_config or DEFAULT_0525_SPECIES_CONFIG
    normalized = {}
    for key, value in config.items():
        if isinstance(value, str):
            normalized[key] = {
                "common_name": value,
                "scientific_name": value,
                "group": "",
            }
        else:
            normalized[key] = {
                "common_name": value.get("common_name") or value.get("species") or key,
                "scientific_name": value.get("scientific_name") or value.get("scientificName") or value.get("common_name") or key,
                "group": value.get("group", ""),
            }
    return normalized


def read_species_csv(path, column_map=None, encodings=DEFAULT_ENCODINGS):
    path = Path(path)
    column_map = column_map or DEFAULT_0525_COLUMN_MAP
    canonical_columns = set(REQUIRED_CANONICAL_COLUMNS) | {"camera_id", "species", "scientificName"}
    accepted_columns = set(column_map) | set(column_map.values()) | canonical_columns
    last_error = None
    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                usecols=lambda col: col in accepted_columns,
            )
            return df, encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path}: {last_error}")


def _coerce_timestamp(series, source_timezone="Asia/Shanghai"):
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:  # pandas < 2.0
        parsed = pd.to_datetime(series, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("UTC")
    return parsed.dt.tz_localize(
        source_timezone,
        ambiguous="NaT",
        nonexistent="NaT",
    ).dt.tz_convert("UTC")


def _extract_filename_timestamp(series, source_timezone="Asia/Shanghai"):
    text = series.astype(str)
    extracted = text.str.extract(r"(20\d{6})(\d{6})", expand=True)
    combined = extracted[0].fillna("") + extracted[1].fillna("")
    combined = combined.where(combined.str.len() == 14, "")
    parsed = pd.to_datetime(combined, format="%Y%m%d%H%M%S", errors="coerce")
    return parsed.dt.tz_localize(
        source_timezone,
        ambiguous="NaT",
        nonexistent="NaT",
    ).dt.tz_convert("UTC")


def _has_second_level_timestamp(series):
    return series.astype(str).str.strip().str.contains(r"\d{1,2}:\d{2}:\d{2}", regex=True, na=False)


def _has_clock_timestamp(series):
    """Return whether a source value contains an observed hour and minute."""
    return series.astype(str).str.strip().str.contains(r"\d{1,2}:\d{2}", regex=True, na=False)


def _coerce_best_timestamp(
    recordtime,
    videoname,
    source_timezone="Asia/Shanghai",
):
    record_ts = _coerce_timestamp(recordtime, source_timezone=source_timezone)
    filename_ts = _extract_filename_timestamp(
        videoname,
        source_timezone=source_timezone,
    )
    record_has_clock = _has_clock_timestamp(recordtime)
    use_filename = filename_ts.notna() & (record_ts.isna() | ~record_has_clock)
    best_ts = record_ts.where(~use_filename, filename_ts)
    source = pd.Series("recordtime", index=recordtime.index, dtype="object")
    source = source.where(~use_filename, "videoname_timestamp")
    source = source.where(best_ts.notna(), "unparseable")
    has_observed_clock = (record_ts.notna() & record_has_clock) | use_filename
    timestamp_is_synthetic = best_ts.notna() & ~has_observed_clock
    return best_ts, source, filename_ts, has_observed_clock, timestamp_is_synthetic


def _timestamp_bound(value, end=False, source_timezone="Asia/Shanghai"):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(source_timezone)
    else:
        ts = ts.tz_convert(source_timezone)
    if end and len(str(value).strip()) <= 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts.tz_convert("UTC")


def _camera_id_from_rows(df):
    if "camera_id" in df.columns:
        camera = df["camera_id"].astype(str).str.strip()
    else:
        camera = df["videoname"].astype(str).str.extract(r"^([A-Za-z0-9_-]+)", expand=False)
    camera = camera.astype(str).str.extract(r"^(\d+)", expand=False).fillna(camera)
    return camera.astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def load_species_csv_directory(
    input_dir,
    species_config=None,
    column_map=None,
    start_date=None,
    end_date=None,
    lon_min=None,
    lon_max=None,
    lat_min=None,
    lat_max=None,
    source_timezone="Asia/Shanghai",
):
    input_dir = Path(input_dir)
    config = normalize_species_config(species_config)
    column_map = column_map or DEFAULT_0525_COLUMN_MAP
    frames = []
    file_reports = []

    if not input_dir.exists():
        raise FileNotFoundError(f"Input CSV directory not found: {input_dir}")

    for csv_path in sorted(input_dir.glob("*.csv")):
        species_key = csv_path.stem
        if species_key not in config:
            continue

        df, encoding = read_species_csv(csv_path, column_map=column_map)
        df = df.rename(columns=column_map)
        missing = sorted(REQUIRED_CANONICAL_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        raw_count = int(len(df))
        species = config[species_key]
        work = df[["longitude", "latitude", "recordtime", "videoname"]].copy()
        work["source_file"] = csv_path.name
        work["source_row"] = df.index.astype(int) + 2
        work["species_source"] = species_key
        work["species"] = species["common_name"]
        work["scientificName"] = species["scientific_name"]
        work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
        work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
        (
            work["timestamp_dt"],
            work["timestamp_source"],
            work["filename_timestamp_dt"],
            work["timestamp_has_observed_clock"],
            work["timestamp_is_synthetic"],
        ) = _coerce_best_timestamp(
            work["recordtime"],
            work["videoname"],
            source_timezone=source_timezone,
        )
        work["camera_id"] = _camera_id_from_rows(work)

        valid = work.dropna(subset=["longitude", "latitude", "timestamp_dt", "camera_id"]).copy()
        keep = pd.Series(True, index=valid.index)
        if lon_min is not None:
            keep &= valid["longitude"] >= float(lon_min)
        if lon_max is not None:
            keep &= valid["longitude"] <= float(lon_max)
        if lat_min is not None:
            keep &= valid["latitude"] >= float(lat_min)
        if lat_max is not None:
            keep &= valid["latitude"] <= float(lat_max)
        if start_date is not None:
            keep &= valid["timestamp_dt"] >= _timestamp_bound(
                start_date,
                source_timezone=source_timezone,
            )
        if end_date is not None:
            keep &= valid["timestamp_dt"] <= _timestamp_bound(
                end_date,
                end=True,
                source_timezone=source_timezone,
            )

        cleaned = valid.loc[keep].copy()
        file_reports.append({
            "file": str(csv_path),
            "encoding": encoding,
            "species_source": species_key,
            "species": species["common_name"],
            "scientificName": species["scientific_name"],
            "raw_rows": raw_count,
            "parse_valid_rows": int(len(valid)),
            "kept_rows": int(len(cleaned)),
            "removed_rows": int(raw_count - len(cleaned)),
            "removed_missing_or_parse": int(raw_count - len(valid)),
            "removed_outside_bbox_or_dates": int(len(valid) - len(cleaned)),
            "duplicate_videoname_rows": int(work["videoname"].duplicated().sum()),
            "recordtime_with_seconds_rows": int(_has_second_level_timestamp(work["recordtime"]).sum()),
            "recordtime_with_clock_rows": int(_has_clock_timestamp(work["recordtime"]).sum()),
            "filename_timestamp_rows": int(work["filename_timestamp_dt"].notna().sum()),
            "timestamp_from_filename_rows": int(work["timestamp_source"].eq("videoname_timestamp").sum()),
            "timestamp_from_recordtime_rows": int(work["timestamp_source"].eq("recordtime").sum()),
            "timestamp_without_observed_clock_rows": int(work["timestamp_is_synthetic"].sum()),
            "source_timezone": source_timezone,
        })
        frames.append(cleaned)

    if not frames:
        raise ValueError(f"No configured species CSV files were found in {input_dir}")

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["species", "camera_id", "timestamp_dt", "videoname", "source_file", "source_row"]).reset_index(drop=True)
    data["deploymentID"] = "dep_" + data["camera_id"].astype(str)
    data["locationID"] = data["camera_id"].astype(str)
    hash_source = data[["source_file", "source_row", "videoname", "recordtime", "species"]].astype(str)
    media_hash = pd.util.hash_pandas_object(hash_source, index=False).astype("uint64")
    data["mediaID"] = "med_" + media_hash.map(lambda value: f"{int(value):016x}")
    duplicate_ids = data["mediaID"].duplicated(keep=False)
    if duplicate_ids.any():
        data.loc[duplicate_ids, "mediaID"] = (
            data.loc[duplicate_ids, "mediaID"]
            + "_"
            + data.loc[duplicate_ids].groupby("mediaID").cumcount().astype(str)
        )
    data["observationID"] = "obs_" + data["mediaID"].str.replace("med_", "", regex=False)
    return data, file_reports


def add_independent_events(data, min_gap_minutes=30):
    data = data.sort_values(["species", "camera_id", "timestamp_dt", "mediaID"]).copy()
    gap = data.groupby(["species", "camera_id"])["timestamp_dt"].diff().dt.total_seconds().div(60)
    data["new_event"] = gap.isna() | (gap > float(min_gap_minutes))
    data["event_seq"] = data.groupby(["species", "camera_id"])["new_event"].cumsum().astype(int)
    species_slug = data["species"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    data["eventID"] = (
        "ev_"
        + data["camera_id"].astype(str)
        + "_"
        + species_slug
        + "_"
        + data["event_seq"].astype(str).str.zfill(6)
    )
    grouped = data.groupby("eventID", sort=False)
    data["eventStart"] = grouped["timestamp_dt"].transform("min")
    data["eventEnd"] = grouped["timestamp_dt"].transform("max")
    data["event_record_count"] = grouped["timestamp_dt"].transform("size").astype(int)

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
        observed_clock_time_records=("timestamp_has_observed_clock", "sum"),
        synthetic_time_records=("timestamp_is_synthetic", "sum"),
    ).reset_index()
    events["event_time"] = events["eventStart"] + (events["eventEnd"] - events["eventStart"]) / 2
    events["has_observed_clock_time"] = events["observed_clock_time_records"].gt(0)
    return data, events


def _format_datetimes(*frames):
    for df in frames:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_coordinate_resolution_audit(data, package_name="wildlife_camtrapdp"):
    """Build a privacy-preserving camera-coordinate resolution audit."""
    required = {"camera_id", "latitude", "longitude"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Coordinate audit input is missing required columns: {missing}")

    coordinate_source = data[["camera_id", "latitude", "longitude"]].copy()
    coordinate_source["coordinate_pair"] = (
        coordinate_source["longitude"].round(6).astype(str)
        + ","
        + coordinate_source["latitude"].round(6).astype(str)
    )
    coordinate_audit = coordinate_source.groupby("camera_id", sort=True).agg(
        source_record_count=("coordinate_pair", "size"),
        unique_coordinate_pair_count=("coordinate_pair", "nunique"),
        latitude_min=("latitude", "min"),
        latitude_max=("latitude", "max"),
        longitude_min=("longitude", "min"),
        longitude_max=("longitude", "max"),
        resolved_latitude=("latitude", "median"),
        resolved_longitude=("longitude", "median"),
    ).reset_index()
    coordinate_audit["camera_token"] = coordinate_audit["camera_id"].map(
        lambda value: stable_id(package_name, value, prefix="cam_", length=12)
    )
    latitude_span = (coordinate_audit["latitude_max"] - coordinate_audit["latitude_min"]) * 111.32
    longitude_span = (
        (coordinate_audit["longitude_max"] - coordinate_audit["longitude_min"])
        * 111.32
        * coordinate_audit["resolved_latitude"].map(lambda value: math.cos(math.radians(value)))
    )
    coordinate_audit["maximum_coordinate_span_km"] = (
        latitude_span.pow(2) + longitude_span.pow(2)
    ).pow(0.5).round(6)
    coordinate_audit["coordinate_conflict"] = coordinate_audit["unique_coordinate_pair_count"].gt(1)
    coordinate_audit["resolution_method"] = "median_latitude_and_longitude_per_camera"
    return coordinate_audit


def write_coordinate_resolution_audit(data, output_path, package_name="wildlife_camtrapdp"):
    """Write the public-safe coordinate audit and return its in-memory full summary."""
    coordinate_audit = build_coordinate_resolution_audit(data, package_name=package_name)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coordinate_audit[[
        "camera_token",
        "source_record_count",
        "unique_coordinate_pair_count",
        "maximum_coordinate_span_km",
        "coordinate_conflict",
        "resolution_method",
    ]].to_csv(output_path, index=False, encoding="utf-8")
    return coordinate_audit


def _complete_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def archive_coordinate_resolution_audit_from_sources(
    input_dir,
    package_dir,
    species_config=None,
    column_map=None,
):
    """Reconstruct and archive a coordinate audit without rewriting the CamtrapDP tables."""
    package_dir = Path(package_dir).resolve()
    report_path = package_dir / "conversion_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Conversion report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    filters = report.get("filters", {})
    if species_config is None:
        species_config = {
            Path(item["file"]).stem: {
                "common_name": item["species"],
                "scientific_name": item.get("scientificName") or item["species"],
                "group": "",
            }
            for item in report.get("source_files", [])
            if item.get("file") and item.get("species")
        }
    data, source_reports = load_species_csv_directory(
        input_dir=input_dir,
        species_config=species_config,
        column_map=column_map,
        start_date=filters.get("start_date"),
        end_date=filters.get("end_date"),
        lon_min=filters.get("lon_min"),
        lon_max=filters.get("lon_max"),
        lat_min=filters.get("lat_min"),
        lat_max=filters.get("lat_max"),
    )
    expected_media = report.get("kept_media_records")
    if expected_media is not None and int(expected_media) != int(len(data)):
        raise ValueError(
            "Coordinate audit reconstruction does not match the versioned conversion: "
            f"expected {expected_media} retained rows, reconstructed {len(data)}."
        )

    package_name = package_dir.name
    audit_path = package_dir / "coordinate_resolution_audit.csv"
    coordinate_audit = write_coordinate_resolution_audit(
        data,
        audit_path,
        package_name=package_name,
    )
    conflict_count = int(coordinate_audit["coordinate_conflict"].sum())
    expected_conflicts = report.get("camera_coordinate_conflict_count")
    if expected_conflicts is not None and int(expected_conflicts) != conflict_count:
        audit_path.unlink(missing_ok=True)
        raise ValueError(
            "Coordinate conflict count changed during audit reconstruction: "
            f"expected {expected_conflicts}, reconstructed {conflict_count}."
        )

    source_manifest = []
    for item in source_reports:
        source_path = Path(item["file"])
        source_manifest.append({
            "file": source_path.name,
            "size_bytes": source_path.stat().st_size,
            "sha256": _complete_sha256(source_path),
            "raw_rows": int(item["raw_rows"]),
            "kept_rows": int(item["kept_rows"]),
        })
    audit_manifest = {
        "schema_version": "1.0",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_name": package_name,
        "filters": filters,
        "source_files": source_manifest,
        "retained_record_count": int(len(data)),
        "camera_count": int(len(coordinate_audit)),
        "coordinate_conflict_camera_count": conflict_count,
        "resolution_method": "median_latitude_and_longitude_per_camera",
        "privacy": {
            "raw_camera_ids_in_audit": False,
            "exact_coordinates_in_audit": False,
            "camera_token_method": "salted_stable_sha1_prefix",
        },
        "audit_file": audit_path.name,
        "audit_sha256": _complete_sha256(audit_path),
        "interpretation": (
            "A coordinate conflict means that retained source records for one camera identifier "
            "contained more than one rounded coordinate pair. Package coordinates use the per-camera "
            "median; the audit records conflict magnitude without publishing exact station locations."
        ),
    }
    manifest_path = package_dir / "coordinate_resolution_audit_manifest.json"
    manifest_path.write_text(
        json.dumps(audit_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["coordinate_resolution_method"] = "median_latitude_and_longitude_per_camera"
    report["coordinate_audit_file"] = audit_path.name
    report["coordinate_audit_manifest_file"] = manifest_path.name
    report["coordinate_audit_status"] = "resolved_and_audited"
    report["coordinate_audit_archived_at_utc"] = audit_manifest["created_at_utc"]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit_manifest


def write_camtrap_package(
    data,
    events,
    output_dir,
    reports,
    min_gap_minutes=30,
    filters=None,
    package_name="wildlife_camtrapdp",
    package_title="Camera-trap observations converted to CamtrapDP",
    package_description="Camera-trap observations converted from species CSV files into CamtrapDP for ecological analysis.",
    deployment_group=None,
    classified_by="csv_to_camtrapdp",
    activate_note=None,
    data_license=None,
    media_license=None,
    source_timezone="Asia/Shanghai",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filters = filters or {}
    deployment_group = deployment_group or package_name

    coordinate_audit = write_coordinate_resolution_audit(
        data,
        output_dir / "coordinate_resolution_audit.csv",
        package_name=package_name,
    )

    deployments = data.groupby("deploymentID", sort=True).agg(
        locationID=("locationID", "first"),
        locationName=("camera_id", "first"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        deploymentStart=("timestamp_dt", "min"),
        deploymentEnd=("timestamp_dt", "max"),
        cameraID=("camera_id", "first"),
    ).reset_index()
    for column, value in {
        "coordinateUncertainty": "",
        "setupBy": "",
        "cameraModel": "",
        "cameraDelay": "",
        "cameraHeight": "",
        "cameraDepth": "",
        "cameraTilt": "",
        "cameraHeading": "",
        "detectionDistance": "",
        "timestampIssues": "false",
        "baitUse": "false",
        "featureType": "",
        "habitat": "",
        "deploymentGroups": deployment_group,
        "deploymentTags": "",
        "deploymentComments": "Deployment approximated from camera ID in source CSV.",
    }.items():
        deployments[column] = value

    media = pd.DataFrame({
        "mediaID": data["mediaID"],
        "deploymentID": data["deploymentID"],
        "captureMethod": "activityDetection",
        "timestamp": data["timestamp_dt"],
        "filePath": data["videoname"].astype(str),
        "filePublic": "false",
        "fileName": data["videoname"].astype(str).map(lambda value: Path(value).name),
        "fileMediatype": "video/mp4",
        "exifData": "",
        "favorite": "",
        "mediaComments": (
            "source_file="
            + data["source_file"].astype(str)
            + "; source_row="
            + data["source_row"].astype(str)
            + "; source_species="
            + data["species_source"].astype(str)
            + "; source_recordtime="
            + data["recordtime"].astype(str)
            + "; timestamp_source="
            + data["timestamp_source"].astype(str)
            + "; timestamp_has_observed_clock="
            + data["timestamp_has_observed_clock"].astype(bool).astype(str).str.lower()
            + "; timestamp_is_synthetic="
            + data["timestamp_is_synthetic"].astype(bool).astype(str).str.lower()
        ),
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
        "classifiedBy": classified_by,
        "classificationTimestamp": "",
        "classificationProbability": "",
        "observationTags": "independent_event",
        "observationComments": "Independent event summary; event_record_count is stored in agent_independent_events.csv.",
    })
    media_observations = pd.DataFrame({
        "observationID": data["observationID"],
        "deploymentID": data["deploymentID"],
        "mediaID": data["mediaID"],
        "eventID": data["eventID"],
        "eventStart": data["eventStart"],
        "eventEnd": data["eventEnd"],
        "observationLevel": "media",
        "observationType": "animal",
        "cameraSetupType": "",
        "scientificName": data["scientificName"],
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
        "classifiedBy": classified_by,
        "classificationTimestamp": "",
        "classificationProbability": "",
        "observationTags": data["species_source"],
        "observationComments": "Converted from species CSV source.",
    })
    observations = pd.concat([event_observations, media_observations], ignore_index=True)

    event_cache = events.rename(columns={"eventID": "event_id"}).copy()
    event_cache["start_time"] = event_cache["eventStart"]
    event_cache["end_time"] = event_cache["eventEnd"]
    _format_datetimes(deployments, media, observations, event_cache)

    deployment_fields = [
        "deploymentID", "locationID", "locationName", "latitude", "longitude", "coordinateUncertainty",
        "deploymentStart", "deploymentEnd", "setupBy", "cameraID", "cameraModel", "cameraDelay",
        "cameraHeight", "cameraDepth", "cameraTilt", "cameraHeading", "detectionDistance", "timestampIssues",
        "baitUse", "featureType", "habitat", "deploymentGroups", "deploymentTags", "deploymentComments",
    ]
    deployments[deployment_fields].to_csv(output_dir / "deployments.csv", index=False, encoding="utf-8")
    media.to_csv(output_dir / "media.csv", index=False, encoding="utf-8")
    observations.to_csv(output_dir / "observations.csv", index=False, encoding="utf-8")
    event_cache[[
        "event_id", "deploymentID", "camera_id", "species", "scientificName", "event_time",
        "start_time", "end_time", "event_record_count", "latitude", "longitude",
        "observed_clock_time_records", "synthetic_time_records", "has_observed_clock_time",
    ]].to_csv(output_dir / "agent_independent_events.csv", index=False, encoding="utf-8")

    spatial = None
    if all(key in filters for key in ("lon_min", "lon_max", "lat_min", "lat_max")):
        spatial = {
            "type": "Polygon",
            "coordinates": [[
                [filters["lon_min"], filters["lat_min"]],
                [filters["lon_max"], filters["lat_min"]],
                [filters["lon_max"], filters["lat_max"]],
                [filters["lon_min"], filters["lat_max"]],
                [filters["lon_min"], filters["lat_min"]],
            ]],
        }
    species_taxa = []
    for report in reports:
        item = {
            "scientificName": report["scientificName"],
            "vernacularName": report["species"],
            "taxonRank": "species",
        }
        if item not in species_taxa:
            species_taxa.append(item)

    datapackage = {
        "resources": [
            {"name": "deployments", "path": "deployments.csv", "profile": "tabular-data-resource", "format": "csv", "mediatype": "text/csv", "encoding": "utf-8", "schema": "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0.2/deployments-table-schema.json"},
            {"name": "media", "path": "media.csv", "profile": "tabular-data-resource", "format": "csv", "mediatype": "text/csv", "encoding": "utf-8", "schema": "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0.2/media-table-schema.json"},
            {"name": "observations", "path": "observations.csv", "profile": "tabular-data-resource", "format": "csv", "mediatype": "text/csv", "encoding": "utf-8", "schema": "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0.2/observations-table-schema.json"},
        ],
        "profile": "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0.2/camtrap-dp-profile.json",
        "name": package_name,
        "id": stable_id(package_name, len(data), len(events), prefix="pkg_"),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "title": package_title,
        "description": package_description,
        "version": "1.0.2",
        "keywords": ["camera traps", "camtrapdp", "camagent"],
        "licenses": [
            {"name": name, "scope": scope}
            for name, scope in ((data_license, "data"), (media_license, "media"))
            if name
        ],
        "project": {
            "title": package_title,
            "samplingDesign": "camera-trap observations converted from species CSV files",
            "captureMethod": ["activityDetection"],
            "individualAnimals": False,
            "observationLevel": ["media", "event"],
        },
        "temporal": {"start": filters.get("start_date", ""), "end": filters.get("end_date", "")},
        "agentTimestampProvenance": {
            "sourceTimezone": source_timezone,
            "storageTimezone": "UTC",
            "naiveSourceTimesLocalizedBeforeUTCConversion": True,
        },
        "taxonomic": species_taxa,
    }
    if spatial:
        datapackage["spatial"] = spatial
    (output_dir / "datapackage.json").write_text(json.dumps(datapackage, ensure_ascii=False, indent=2), encoding="utf-8")

    species_summary = {}
    for species, group in data.groupby("species"):
        species_events = events[events["species"] == species]
        species_summary[species] = {
            "media_records": int(len(group)),
            "independent_events": int(len(species_events)),
            "camera_count": int(group["camera_id"].nunique()),
            "date_start": str(group["timestamp_dt"].min()),
            "date_end": str(group["timestamp_dt"].max()),
        }
    report = {
        "output_dir": str(output_dir),
        "datapackage_path": str(output_dir / "datapackage.json"),
        "filters": {**filters, "min_gap_minutes": min_gap_minutes},
        "source_files": reports,
        "total_source_rows": int(sum(item["raw_rows"] for item in reports)),
        "kept_media_records": int(len(data)),
        "independent_events": int(len(events)),
        "deployment_count": int(len(deployments)),
        "species_summary": species_summary,
        "camera_coordinate_conflict_count": int(coordinate_audit["coordinate_conflict"].sum()),
        "coordinate_resolution_method": "median_latitude_and_longitude_per_camera",
        "coordinate_audit_file": "coordinate_resolution_audit.csv",
        "coordinate_audit_manifest_file": "coordinate_resolution_audit_manifest.json",
        "coordinate_audit_status": "resolved_and_audited",
        "conversion_note": activate_note or "Rows outside configured filters or without parseable timestamp, coordinates, or camera ID were excluded.",
        "license_status": "declared" if data_license and media_license else "requires_rights_holder_confirmation",
        "source_timezone": source_timezone,
        "storage_timezone": "UTC",
        "naive_source_times_localized_before_utc_conversion": True,
    }
    audit_path = output_dir / "coordinate_resolution_audit.csv"
    audit_created_at = datetime.now(timezone.utc).isoformat()
    coordinate_source_manifest = []
    for item in reports:
        source_path = Path(item["file"])
        coordinate_source_manifest.append({
            "file": source_path.name,
            "size_bytes": source_path.stat().st_size,
            "sha256": _complete_sha256(source_path),
            "raw_rows": int(item["raw_rows"]),
            "kept_rows": int(item["kept_rows"]),
        })
    coordinate_audit_manifest = {
        "schema_version": "1.0",
        "status": "completed",
        "created_at_utc": audit_created_at,
        "package_name": package_name,
        "filters": filters,
        "source_files": coordinate_source_manifest,
        "retained_record_count": int(len(data)),
        "camera_count": int(len(coordinate_audit)),
        "coordinate_conflict_camera_count": int(coordinate_audit["coordinate_conflict"].sum()),
        "resolution_method": "median_latitude_and_longitude_per_camera",
        "privacy": {
            "raw_camera_ids_in_audit": False,
            "exact_coordinates_in_audit": False,
            "camera_token_method": "salted_stable_sha1_prefix",
        },
        "audit_file": audit_path.name,
        "audit_sha256": _complete_sha256(audit_path),
        "interpretation": (
            "A coordinate conflict means that retained source records for one camera identifier "
            "contained more than one rounded coordinate pair. Package coordinates use the per-camera "
            "median; the audit records conflict magnitude without publishing exact station locations."
        ),
    }
    (output_dir / "coordinate_resolution_audit_manifest.json").write_text(
        json.dumps(coordinate_audit_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["coordinate_audit_archived_at_utc"] = audit_created_at
    (output_dir / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def convert_species_csv_directory_to_camtrap_dp(
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
    classified_by="csv_to_camtrapdp",
    data_license=None,
    media_license=None,
    source_timezone="Asia/Shanghai",
):
    ensure_runtime_dirs()
    if output_dir is None:
        output_dir = CAMTRAP_IMPORTS_DIR / package_name
    filters = {
        "start_date": start_date,
        "end_date": end_date,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
    }
    filters = {key: value for key, value in filters.items() if value is not None}
    data, reports = load_species_csv_directory(
        input_dir=input_dir,
        species_config=species_config,
        column_map=column_map,
        start_date=start_date,
        end_date=end_date,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        source_timezone=source_timezone,
    )
    data, events = add_independent_events(data, min_gap_minutes=min_gap_minutes)
    return write_camtrap_package(
        data=data,
        events=events,
        output_dir=output_dir,
        reports=reports,
        min_gap_minutes=min_gap_minutes,
        filters=filters,
        package_name=package_name,
        package_title=package_title,
        package_description=package_description,
        deployment_group=package_name,
        classified_by=classified_by,
        data_license=data_license,
        media_license=media_license,
        source_timezone=source_timezone,
    )
