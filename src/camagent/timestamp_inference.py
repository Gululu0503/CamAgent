import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from .detection_inference import sample_video_frames


_PADDLE_OCR_INSTANCE = None
_TIMESTAMP_PATTERN = re.compile(
    r"(?P<year>20\d{2})\s*[/.\-]\s*"
    r"(?P<month>\d{1,2})\s*[/.\-]\s*"
    r"(?P<day>\d{1,2})\s+"
    r"(?P<hour>\d{1,2})\s*[:.]\s*"
    r"(?P<minute>\d{2})\s*[:.]\s*"
    r"(?P<second>\d{2})"
)


def _paddle_ocr():
    global _PADDLE_OCR_INSTANCE
    if _PADDLE_OCR_INSTANCE is not None:
        return _PADDLE_OCR_INSTANCE
    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Timestamp OCR requires the optional `ocr` dependencies "
            "(install camagent[ocr])."
        ) from exc
    _PADDLE_OCR_INSTANCE = PaddleOCR(
        lang="en",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    return _PADDLE_OCR_INSTANCE


def _flatten_ocr_texts(value, inherited_score=None):
    """Yield text/score pairs from PaddleOCR 2.x and 3.x result structures."""
    if value is None:
        return
    if hasattr(value, "json"):
        value = value.json
    if callable(value):
        value = value()
    if isinstance(value, str):
        yield value, inherited_score
        return
    if isinstance(value, dict):
        if "rec_texts" in value:
            texts = value.get("rec_texts") or []
            scores = value.get("rec_scores") or []
            for index, text in enumerate(texts):
                score = scores[index] if index < len(scores) else inherited_score
                yield str(text), score
        for key, item in value.items():
            if key in {"rec_texts", "rec_scores"}:
                continue
            yield from _flatten_ocr_texts(item, inherited_score=inherited_score)
        return
    if isinstance(value, (list, tuple)):
        if (
            len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], (int, float))
        ):
            yield value[0], float(value[1])
            return
        for item in value:
            yield from _flatten_ocr_texts(item, inherited_score=inherited_score)


def _parse_timestamp(text):
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    match = _TIMESTAMP_PATTERN.search(normalized)
    if not match:
        return None
    try:
        parsed = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
        )
    except ValueError:
        return None
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def extract_video_timestamp_with_paddleocr(
    video_path,
    *,
    output_dir,
    frame_paths=None,
    max_frames=5,
    bottom_crop_fraction=0.42,
):
    """Read an overlaid field timestamp from sampled camera-trap video frames."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Timestamp OCR requires OpenCV (install camagent[perception])."
        ) from exc

    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    crop_dir = output_dir / "timestamp_ocr_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    if not frame_paths:
        sampled = sample_video_frames(
            video_path,
            output_dir=output_dir / "timestamp_ocr_frames",
            frame_interval_seconds=1.0,
            max_frames=max_frames,
        )
        frame_paths = [
            item["frame_path"]
            for item in sampled["samples"]
        ]
    frame_paths = [str(path) for path in frame_paths[:max(int(max_frames), 1)]]
    engine = _paddle_ocr()
    candidates = []
    frame_results = []

    for frame_index, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        crop_start = int(
            image.shape[0] * (1.0 - max(min(float(bottom_crop_fraction), 0.9), 0.1))
        )
        crop = image[crop_start:, :, :]
        crop_path = crop_dir / f"{video_path.stem}_timestamp_crop_{frame_index:03d}.jpg"
        cv2.imwrite(str(crop_path), crop)
        result = engine.predict(crop)
        texts = []
        for text, score in _flatten_ocr_texts(result):
            timestamp = _parse_timestamp(text)
            if timestamp is None:
                continue
            numeric_score = float(score) if score is not None else 0.0
            texts.append({
                "timestamp": timestamp,
                "text": text,
                "ocr_score": numeric_score,
            })
            candidates.append({
                "timestamp": timestamp,
                "text": text,
                "ocr_score": numeric_score,
                "frame_index": frame_index,
                "frame_path": str(frame_path),
                "crop_path": str(crop_path),
            })
        frame_results.append({
            "frame_index": frame_index,
            "frame_path": str(frame_path),
            "crop_path": str(crop_path),
            "candidates": texts,
        })

    if not candidates:
        raise RuntimeError(
            f"PaddleOCR found no complete YYYY/MM/DD HH:MM:SS timestamp in {video_path}."
        )

    votes = Counter(item["timestamp"] for item in candidates)
    selected = sorted(
        candidates,
        key=lambda item: (
            -votes[item["timestamp"]],
            -item["ocr_score"],
            item["frame_index"],
        ),
    )[0]
    manifest = {
        "video_path": str(video_path),
        "timestamp": selected["timestamp"],
        "timestamp_source": "paddleocr_video_timestamp",
        "timestamp_is_synthetic": False,
        "ocr_score": selected["ocr_score"],
        "vote_count": votes[selected["timestamp"]],
        "selected": selected,
        "frame_results": frame_results,
    }
    manifest_path = output_dir / f"{video_path.stem}_timestamp_ocr.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["manifest_json"] = str(manifest_path)
    return manifest
