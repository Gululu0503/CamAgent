import hashlib
import json
import random
import re
from datetime import datetime, timedelta


DATE_ONLY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FULL_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

SYNTHETIC_TIMESTAMP_SOURCE = "synthetic_random_v1"
OBSERVED_TIMESTAMP_SOURCE = "paddleocr_video_timestamp"
CAMDATA_TIMESTAMP_SOURCE = "camdata_json_observed"
SYNTHETIC_TIMESTAMP_FLAG_FIELD = "timestamp_is_synthetic"
SYNTHETIC_TIMESTAMP_ORIGINAL_FIELD = "timestamp_original_date"


def _stable_seed(date_text: str, filename: str, camera_id: str) -> int:
    seed_text = f"CamAgent|{date_text}|{filename}|{camera_id}|{SYNTHETIC_TIMESTAMP_SOURCE}"
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def generate_synthetic_timestamp(date_text: str, filename: str, camera_id: str) -> str:
    """
    Complete a date-level timestamp with reproducible pseudo-random hour, minute, and second values.
    The same record gets the same generated time on every run, which keeps JSON and DB sync repeatable.
    """
    base_date = datetime.strptime(date_text, "%Y-%m-%d")
    rng = random.Random(_stable_seed(date_text, filename, camera_id))
    seconds_offset = rng.randint(0, 24 * 60 * 60 - 1)
    return (base_date + timedelta(seconds=seconds_offset)).strftime("%Y-%m-%d %H:%M:%S")


def ensure_precise_timestamp(timestamp_text: str, filename: str, camera_id: str) -> tuple[str, bool, str]:
    """
    Return (timestamp, is_synthetic, original_date_text).
    """
    if timestamp_text is None:
        raise ValueError("timestamp cannot be None")

    raw_text = str(timestamp_text).strip()
    if not raw_text:
        raise ValueError("timestamp cannot be an empty string")

    if FULL_TIMESTAMP_PATTERN.fullmatch(raw_text):
        return raw_text, False, raw_text[:10]

    if DATE_ONLY_PATTERN.fullmatch(raw_text):
        return generate_synthetic_timestamp(raw_text, filename=filename, camera_id=camera_id), True, raw_text

    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(raw_text, fmt)
            return parsed.strftime("%Y-%m-%d %H:%M:%S"), False, parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    raise ValueError(f"Unrecognized timestamp format: {raw_text}")


def normalize_observed_timestamp(timestamp_text: str) -> tuple[str, str]:
    """
    Normalize an observed timestamp without inventing clock time.

    Returns (timestamp, original_date). Date-only values are rejected so callers can
    skip records that do not have a real hour/minute/second observation.
    """
    if timestamp_text is None:
        raise ValueError("timestamp cannot be None")

    raw_text = str(timestamp_text).strip()
    if not raw_text:
        raise ValueError("timestamp cannot be an empty string")
    if DATE_ONLY_PATTERN.fullmatch(raw_text):
        raise ValueError(f"timestamp is missing real hour-minute-second values: {raw_text}")

    if FULL_TIMESTAMP_PATTERN.fullmatch(raw_text):
        return raw_text, raw_text[:10]

    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(raw_text, fmt)
            return parsed.strftime("%Y-%m-%d %H:%M:%S"), parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    raise ValueError(f"Unrecognized observed timestamp format: {raw_text}")


def timestamp_has_real_clock_time(timestamp_text: str) -> bool:
    try:
        normalize_observed_timestamp(timestamp_text)
        return True
    except ValueError:
        return False


def _strip_json_code_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[7:]
    elif stripped.startswith("```"):
        stripped = stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def update_ecological_report_timestamp(report_text: str, timestamp_text: str) -> str:
    """
    Update the timestamp field inside an ecological_report JSON string.
    Return the original text on parse failure to avoid damaging existing results.
    """
    if not report_text:
        return report_text

    try:
        payload = json.loads(_strip_json_code_block(report_text))
    except Exception:
        return report_text

    if not isinstance(payload, dict):
        return report_text

    payload["timestamp"] = timestamp_text
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def build_observation_document(metadata: dict) -> str:
    return (
        f"Camera {metadata['camera_id']} on {metadata['timestamp']} "
        f"at Latitude {float(metadata['latitude']):.6f}°, Longitude {float(metadata['longitude']):.6f}° "
        f"recorded {metadata['species']} (confidence: {metadata['confidence']}), "
        f"individuals: {metadata['individuals']}, behavior: {metadata['behavior']}, "
        f"special: {metadata['special_observations']}."
    )
