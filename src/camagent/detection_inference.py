import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .paths import DETECTION_PREDICTIONS_DIR, MPLCONFIG_DIR, RUNTIME_CACHE_DIR


def _slugify_text(text):
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
    return slug or "image"


def _speciesnet_python():
    configured = os.environ.get("AGENTTOOL_SPECIESNET_PYTHON")
    if configured:
        return configured
    return sys.executable


def _as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _parse_taxon_label(label):
    text = str(label or "").strip()
    if not text:
        return {"label": "", "common_name": "", "scientific_name": ""}

    parts = [part.strip() for part in text.split(";")]
    scientific_name = ""
    common_name = ""

    if len(parts) >= 7:
        common_name = parts[6]
        genus = parts[4]
        species = parts[5]
        if genus and species:
            scientific_name = f"{genus.capitalize()} {species.lower()}"
        elif parts[1]:
            scientific_name = parts[1]
    elif len(parts) >= 2:
        common_name = parts[-1] or parts[-2]
        scientific_name = parts[-2] if parts[-2] != common_name else ""
    elif parts:
        common_name = parts[0]

    return {
        "label": text,
        "common_name": common_name,
        "scientific_name": scientific_name,
    }


def _normalize_classifications(classifications):
    if isinstance(classifications, dict):
        labels = classifications.get("classes") or classifications.get("labels") or []
        scores = (
            classifications.get("scores")
            or classifications.get("confidences")
            or classifications.get("probabilities")
            or []
        )
        rows = []
        for idx, label in enumerate(labels):
            score = scores[idx] if idx < len(scores) else None
            parsed = _parse_taxon_label(label)
            rows.append({**parsed, "score": _as_float(score)})
        if rows:
            return rows

    if isinstance(classifications, list):
        rows = []
        for item in classifications:
            if isinstance(item, dict):
                label = (
                    item.get("label")
                    or item.get("class")
                    or item.get("category")
                    or item.get("taxon")
                    or item.get("prediction")
                )
                score = (
                    item.get("score")
                    or item.get("conf")
                    or item.get("confidence")
                    or item.get("probability")
                )
                parsed = _parse_taxon_label(label)
                rows.append({**parsed, "score": _as_float(score)})
            elif isinstance(item, (list, tuple)) and item:
                parsed = _parse_taxon_label(item[0])
                score = item[1] if len(item) > 1 else None
                rows.append({**parsed, "score": _as_float(score)})
        return rows

    return []


def _normalize_prediction(prediction):
    parsed_prediction = _parse_taxon_label(prediction.get("prediction"))
    classifications = _normalize_classifications(prediction.get("classifications"))
    detections = prediction.get("detections") or []
    if not isinstance(detections, list):
        detections = []

    top_classification = classifications[0] if classifications else None
    confidence = _as_float(prediction.get("prediction_score"))
    if confidence is None and top_classification:
        confidence = top_classification.get("score")

    species = parsed_prediction.get("common_name") or parsed_prediction.get("label")
    scientific_name = parsed_prediction.get("scientific_name")
    if not species and top_classification:
        species = top_classification.get("common_name") or top_classification.get("label")
        scientific_name = top_classification.get("scientific_name")

    animal_detections = [
        det for det in detections
        if str(det.get("label", "")).lower() == "animal" or str(det.get("category")) == "1"
    ]
    primary_detection = animal_detections[0] if animal_detections else (detections[0] if detections else {})
    primary_bbox = primary_detection.get("bbox") if isinstance(primary_detection, dict) else None

    return {
        "filepath": prediction.get("filepath"),
        "species": species or "Unknown",
        "scientific_name": scientific_name or species or "Unknown",
        "confidence": confidence,
        "prediction": prediction.get("prediction"),
        "prediction_score": confidence,
        "prediction_source": prediction.get("prediction_source"),
        "model_version": prediction.get("model_version"),
        "failures": prediction.get("failures", []),
        "detections": detections,
        "classifications": classifications,
        "primary_detection": primary_detection,
        "bbox": primary_bbox or [],
        "raw_prediction": prediction,
    }


def run_speciesnet_on_image(
    image_path,
    *,
    output_json=None,
    country=None,
    admin1_region=None,
    model=None,
    python_executable=None,
    ignore_existing=True,
    timeout=None,
):
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")

    if output_json is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DETECTION_PREDICTIONS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json = output_dir / f"{_slugify_text(image_path.stem)}_{stamp}.json"
    else:
        output_json = Path(output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)

    python_executable = python_executable or _speciesnet_python()
    cmd = [
        python_executable,
        "-m",
        "speciesnet.scripts.run_model",
        "--filepaths",
        str(image_path),
        "--predictions_json",
        str(output_json),
        "--bypass_prompts",
        "--noprogress_bars",
    ]
    if ignore_existing:
        cmd.append("--ignore_existing_predictions")
    if country:
        cmd.extend(["--country", str(country).upper()])
    if admin1_region:
        cmd.extend(["--admin1_region", str(admin1_region).upper()])
    if model:
        cmd.extend(["--model", str(model)])

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
    env.setdefault("KAGGLEHUB_CACHE", str(RUNTIME_CACHE_DIR / "kagglehub"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["KAGGLEHUB_CACHE"]).mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SpeciesNet inference failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    predictions = payload.get("predictions") or []
    if not predictions:
        raise RuntimeError(f"SpeciesNet produced no predictions in {output_json}")

    normalized = _normalize_prediction(predictions[0])
    return {
        **normalized,
        "image_path": str(image_path),
        "predictions_json": str(output_json),
        "speciesnet_stdout": completed.stdout,
        "speciesnet_stderr": completed.stderr,
    }


def sample_video_frames(
    video_path,
    *,
    output_dir,
    frame_interval_seconds=1.0,
    max_frames=12,
    jpeg_quality=90,
):
    """Extract an auditable, bounded set of frames from a camera-trap video."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Video perception requires the optional `perception` dependencies "
            "(install camagent[perception])."
        ) from exc

    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for frame extraction: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(
                f"Video metadata do not contain a usable frame rate/count: {video_path}"
            )

        duration_seconds = max((frame_count - 1) / fps, 0.0)
        frame_interval_seconds = max(float(frame_interval_seconds), 0.001)
        max_frames = max(int(max_frames), 1)
        candidate_times = []
        current = 0.0
        while current <= duration_seconds + 1e-9:
            candidate_times.append(min(current, duration_seconds))
            current += frame_interval_seconds
        if not candidate_times:
            candidate_times = [0.0]

        if len(candidate_times) > max_frames:
            # Evenly retain positions across the complete clip so the cap does
            # not silently discard the end of a longer camera-trap video.
            positions = [
                round(index * (len(candidate_times) - 1) / max(max_frames - 1, 1))
                for index in range(max_frames)
            ]
            candidate_times = [candidate_times[index] for index in positions]

        samples = []
        seen_frames = set()
        for sample_index, offset_seconds in enumerate(candidate_times, start=1):
            frame_index = min(int(round(offset_seconds * fps)), frame_count - 1)
            if frame_index in seen_frames:
                continue
            seen_frames.add(frame_index)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_path = output_dir / (
                f"{_slugify_text(video_path.stem)}_"
                f"frame_{sample_index:04d}_{int(round(offset_seconds * 1000)):09d}ms.jpg"
            )
            encoded = cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
            )
            if not encoded:
                continue
            samples.append({
                "frame_path": str(frame_path),
                "frame_index": frame_index,
                "offset_seconds": round(frame_index / fps, 6),
            })
    finally:
        cap.release()

    if not samples:
        raise RuntimeError(f"No frames could be extracted from video: {video_path}")
    return {
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": round(duration_seconds, 6),
        "frame_interval_seconds": frame_interval_seconds,
        "max_frames": max_frames,
        "samples": samples,
    }


def run_speciesnet_on_video(
    video_path,
    *,
    output_dir,
    frame_interval_seconds=1.0,
    max_frames=12,
    country=None,
    admin1_region=None,
    model=None,
    python_executable=None,
    timeout=None,
):
    """
    Run the image-based SpeciesNet ensemble on sampled video frames.

    The returned record keeps the raw-video path, frame offsets, extracted-frame
    paths, per-frame predictions, and failures so downstream CamtrapDP export is
    traceable rather than presenting a clip-level label as direct video inference.
    """
    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    frame_dir = output_dir / "sampled_frames" / _slugify_text(video_path.stem)
    prediction_dir = output_dir / "frame_predictions" / _slugify_text(video_path.stem)
    frame_manifest = sample_video_frames(
        video_path,
        output_dir=frame_dir,
        frame_interval_seconds=frame_interval_seconds,
        max_frames=max_frames,
    )

    predictions = []
    failures = []
    for sample in frame_manifest["samples"]:
        frame_path = Path(sample["frame_path"])
        prediction_json = prediction_dir / f"{frame_path.stem}_speciesnet.json"
        try:
            prediction = run_speciesnet_on_image(
                frame_path,
                output_json=prediction_json,
                country=country,
                admin1_region=admin1_region,
                model=model,
                python_executable=python_executable,
                timeout=timeout,
            )
            predictions.append({
                **prediction,
                "source_video_path": str(video_path),
                "frame_path": str(frame_path),
                "frame_index": sample["frame_index"],
                "frame_offset_seconds": sample["offset_seconds"],
            })
        except Exception as exc:
            failures.append({
                "frame_path": str(frame_path),
                "frame_index": sample["frame_index"],
                "frame_offset_seconds": sample["offset_seconds"],
                "error": f"{type(exc).__name__}: {exc}",
            })

    manifest = {
        **frame_manifest,
        "predictions": predictions,
        "failures": failures,
        "prediction_count": len(predictions),
        "failure_count": len(failures),
        "model": model or "speciesnet_default",
        "perception_method": "sampled_video_frames_with_speciesnet",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{_slugify_text(video_path.stem)}_video_predictions.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["manifest_json"] = str(manifest_path)

    if not predictions:
        raise RuntimeError(
            f"SpeciesNet produced no successful frame predictions for video: {video_path}. "
            f"See {manifest_path} for per-frame failures."
        )
    return manifest


def write_speciesnet_sidecar(result, sidecar_path=None):
    image_path = Path(result["image_path"])
    if sidecar_path is None:
        sidecar_path = image_path.with_suffix(".json")
    else:
        sidecar_path = Path(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    bbox_xywh = result.get("bbox") or []
    bbox_xyxy = []
    if isinstance(bbox_xywh, list) and len(bbox_xywh) == 4:
        x, y, width, height = bbox_xywh
        bbox_xyxy = [round(x, 4), round(y, 4), round(x + width, 4), round(y + height, 4)]

    sidecar = {
        "species": result.get("species") or "Unknown",
        "scientific_name": result.get("scientific_name") or result.get("species") or "Unknown",
        "confidence": result.get("confidence"),
        "bbox": bbox_xyxy,
        "bbox_source_format": "xywh_normalized",
        "bbox_source_xywh": bbox_xywh,
        "detections": result.get("detections") or [],
        "classifications": result.get("classifications") or [],
        "prediction": result.get("prediction"),
        "prediction_score": result.get("prediction_score"),
        "prediction_source": result.get("prediction_source"),
        "model_version": result.get("model_version"),
        "visual_frontend_detector": "megadetector",
        "visual_frontend_classifier": "speciesnet",
        "visual_frontend_status": "model_inferred",
        "speciesnet_predictions_json": result.get("predictions_json"),
    }
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(sidecar_path)
