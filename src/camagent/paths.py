import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
PACKAGE_DIR = SRC_DIR / "camagent"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
CAMTRAP_PROJECT_DIR = PROJECT_ROOT / "camtrap-dp-main"
CAMTRAP_EXAMPLE_DIR = CAMTRAP_PROJECT_DIR / "example"

WORKSPACE_DIR = PROJECT_ROOT / "workspace"
ASSETS_DIR = WORKSPACE_DIR / "assets"
PARK_BOUNDARY_DIR = ASSETS_DIR / "park_boundary"
PARK_SHAPEFILE_PATH = PARK_BOUNDARY_DIR / "东北虎豹国家公园.shp"

DATASETS_DIR = WORKSPACE_DIR / "datasets"
RAW_DATA_DIR = DATASETS_DIR / "raw"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
TIGER_RAW_DIR = RAW_DATA_DIR / "tiger2024"
TIGER_VIDEO_DIR = TIGER_RAW_DIR / "2024"
CAMERA_LOCATIONS_CSV = TIGER_RAW_DIR / "camera_locations_cleaned.csv"
TIGER_JSON_DIR = PROCESSED_DATA_DIR / "tiger2024_json"
REAL_TIMESTAMP_JSON_DIR = WORKSPACE_DIR / "timestamp_json_paddleocr"
ANIMAL_CAMDATA_JSON_DIR = AGENT_ROOT / "Animal" / "camdata_json"

DATABASES_DIR = WORKSPACE_DIR / "databases"
VECTOR_DB_DIR = DATABASES_DIR / "vector_db"
VECTOR_DB_SQLITE = VECTOR_DB_DIR / "chroma.sqlite3"

OUTPUTS_DIR = WORKSPACE_DIR / "outputs"
OVERLAP_RESULT_BUNDLES_DIR = OUTPUTS_DIR / "overlap_result_bundles"
TIMESTAMP_BACKUP_DIR = OUTPUTS_DIR / "timestamp_backups"
LEGACY_DIR = WORKSPACE_DIR / "legacy"
LEGACY_OUTPUT_DIR = LEGACY_DIR / "output"
LEGACY_TIGER_JSON_DIR = LEGACY_OUTPUT_DIR / "tiger2024_json"
LEGACY_VECTOR_DB_DIR = LEGACY_OUTPUT_DIR / "vector_db"
LEGACY_VECTOR_DB_SQLITE = LEGACY_VECTOR_DB_DIR / "chroma.sqlite3"
CAMTRAP_DP_DIR = WORKSPACE_DIR / "camtrap_dp"
CAMTRAP_IMPORTS_DIR = CAMTRAP_DP_DIR / "imports"
CAMTRAP_EXPORTS_DIR = CAMTRAP_DP_DIR / "exports"
CAMTRAP_ACTIVE_DATASET = CAMTRAP_DP_DIR / "active_dataset.json"
DETECTION_PIPELINE_DIR = WORKSPACE_DIR / "detection_pipeline"
DETECTION_INPUT_DIR = DETECTION_PIPELINE_DIR / "input"
DETECTION_RECORDS_DIR = DETECTION_PIPELINE_DIR / "records"
DETECTION_PREDICTIONS_DIR = DETECTION_PIPELINE_DIR / "predictions"
MEDIA_EXPORTS_DIR = WORKSPACE_DIR / "media_exports"

CACHE_DIR = WORKSPACE_DIR / "cache"
RUNTIME_CACHE_DIR = CACHE_DIR / "runtime"
MPLCONFIG_DIR = CACHE_DIR / "matplotlib"


def ensure_runtime_dirs():
    managed_dirs = [
        WORKSPACE_DIR,
        ASSETS_DIR,
        DATASETS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        TIGER_JSON_DIR,
        DATABASES_DIR,
        VECTOR_DB_DIR,
        OUTPUTS_DIR,
        OVERLAP_RESULT_BUNDLES_DIR,
        LEGACY_DIR,
        LEGACY_OUTPUT_DIR,
        CAMTRAP_DP_DIR,
        CAMTRAP_IMPORTS_DIR,
        CAMTRAP_EXPORTS_DIR,
        DETECTION_PIPELINE_DIR,
        DETECTION_INPUT_DIR,
        DETECTION_RECORDS_DIR,
        DETECTION_PREDICTIONS_DIR,
        MEDIA_EXPORTS_DIR,
        CACHE_DIR,
        RUNTIME_CACHE_DIR,
        MPLCONFIG_DIR,
    ]
    for path in managed_dirs:
        path.mkdir(parents=True, exist_ok=True)


def safe_join(value, sep=", "):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return sep.join(str(item) for item in value)
    except TypeError:
        return str(value)


def display_path(path):
    """Return a user-facing project-relative path when possible."""
    if path is None:
        return ""
    path_text = str(path)
    try:
        value_path = Path(path_text).expanduser()
        if value_path.is_absolute():
            resolved = value_path.resolve()
            for root, label in (
                (OUTPUTS_DIR.resolve(), "outputs"),
                (WORKSPACE_DIR.resolve(), "workspace"),
                (PROJECT_ROOT.resolve(), ""),
                (AGENT_ROOT.resolve(), ""),
            ):
                try:
                    rel = resolved.relative_to(root)
                except ValueError:
                    continue
                return str(Path(label) / rel) if label else str(rel)
    except Exception:
        pass
    return path_text


def sanitize_for_display(value):
    """Recursively replace local absolute paths in user-visible structures."""
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key.endswith("_path") or key.endswith("_dir") or key in {"path", "output_dir", "target_dir"}:
                sanitized[key] = display_path(item)
            else:
                sanitized[key] = sanitize_for_display(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_for_display(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_for_display(item) for item in value)
    if isinstance(value, str):
        text = value
        path_pattern = re.compile(r"(?:/Users/[^/\s]+|/home/[^/\s]+)[^\s`'\"),]+")
        for match in sorted(set(path_pattern.findall(text)), key=len, reverse=True):
            text = text.replace(match, display_path(match))
        if text.startswith(str(AGENT_ROOT)):
            return display_path(text)
        return text
    return value
