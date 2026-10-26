import argparse
import hashlib
import io
import json
import math
import os
import sys
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from camagent.camtrap_conversion import convert_species_csv_directory_to_camtrap_dp
from camagent.detection_inference import run_speciesnet_on_image, write_speciesnet_sidecar
from camagent.ecology_config import DEFAULT_0525_COLUMN_MAP, DEFAULT_0525_SPECIES_CONFIG
from camagent.llm_agent import AgentRunPolicy, WildlifeAgent
from camagent.paths import AGENT_ROOT, CAMTRAP_IMPORTS_DIR, MPLCONFIG_DIR, OUTPUTS_DIR, RUNTIME_CACHE_DIR, TIGER_JSON_DIR, VECTOR_DB_DIR, display_path, safe_join, sanitize_for_display, ensure_runtime_dirs
from camagent.timestamp_utils import build_observation_document


DEFAULT_COLLECTION_NAME = "tiger_observations_2024"
EMBEDDING_DIMENSION = 384


def configure_runtime():
    ensure_runtime_dirs()
    os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE_DIR))
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
    warnings.filterwarnings(
        "ignore",
        message=r"urllib3 .*doesn't match a supported version!",
        category=Warning,
    )
    if hasattr(sys.stdin, "buffer"):
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="ignore")
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
    env_paths = [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]
    for env_path in env_paths:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def hash_embedding(text: str, dimension: int = EMBEDDING_DIMENSION):
    values = []
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    counter = 0
    while len(values) < dimension:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for byte in block:
            values.append((byte / 127.5) - 1.0)
            if len(values) >= dimension:
                break
        counter += 1
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def normalize_scalar_metadata(metadata: dict):
    normalized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, bool)):
            normalized[key] = value
        elif isinstance(value, float):
            if not math.isnan(value) and not math.isinf(value):
                normalized[key] = value
        elif value is None:
            continue
        else:
            normalized[key] = json.dumps(value, ensure_ascii=False)
    return normalized


def record_id_for(path: Path, payload: dict):
    camera_id = str(payload.get("camera_id", "")).strip()
    filename = str(payload.get("filename", "")).strip()
    if camera_id and filename:
        return f"{camera_id}_{filename}"
    return path.stem


def document_for(payload: dict):
    try:
        return build_observation_document(payload)
    except Exception:
        return (
            f"Camera {payload.get('camera_id', 'unknown')} recorded "
            f"{payload.get('species', 'Unknown')} at {payload.get('timestamp', '')}. "
            f"Behavior: {payload.get('behavior', 'Unknown')}; "
            f"individuals: {payload.get('individuals', 'Unknown')}."
        )


def load_json_payloads(json_dirs, recursive=False):
    seen_ids = set()
    skipped = []
    records = []
    for json_dir in json_dirs:
        root = Path(json_dir)
        pattern = "**/*.json" if recursive else "*.json"
        if not root.exists():
            skipped.append({"path": str(root), "reason": "json_dir_not_found"})
            continue
        for path in sorted(root.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                skipped.append({"path": str(path), "reason": f"json_read_failed: {exc}"})
                continue
            item_id = record_id_for(path, payload)
            if item_id in seen_ids:
                skipped.append({"path": str(path), "reason": f"duplicate_record_id: {item_id}"})
                continue
            seen_ids.add(item_id)
            records.append((item_id, path, payload))
    return records, skipped


def command_sync_db(args):
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError("chromadb is required for JSON -> DB sync.") from exc

    json_dirs = args.json_dir or [str(TIGER_JSON_DIR)]
    records, skipped = load_json_payloads(json_dirs, recursive=args.recursive)
    ids = []
    documents = []
    metadatas = []
    for item_id, _path, payload in records:
        document = document_for(payload)
        ids.append(item_id)
        documents.append(document)
        metadatas.append(normalize_scalar_metadata(payload))

    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    collection = chromadb.PersistentClient(path=str(db_dir)).get_or_create_collection(args.collection_name)
    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=[hash_embedding(document) for document in documents],
        )

    print(json.dumps(sanitize_for_display({
        "flow": "recognition_json -> chroma_db -> agent",
        "db_dir": str(db_dir),
        "collection_name": args.collection_name,
        "input_json_dirs": json_dirs,
        "upserted_records": len(ids),
        "collection_count": collection.count(),
        "skipped_count": len(skipped),
        "skipped_preview": skipped[:20],
    }), ensure_ascii=False, indent=2))


def command_recognize_image(args):
    result = run_speciesnet_on_image(
        args.image_path,
        country=args.country,
        admin1_region=args.admin1_region,
        output_json=args.output_json,
    )
    sidecar_path = None if args.no_sidecar else write_speciesnet_sidecar(result)
    print(json.dumps(sanitize_for_display({
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
    }), ensure_ascii=False, indent=2))


def command_convert_csv(args):
    report = convert_species_csv_directory_to_camtrap_dp(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        species_config=DEFAULT_0525_SPECIES_CONFIG,
        column_map=DEFAULT_0525_COLUMN_MAP,
        start_date=args.start_date,
        end_date=args.end_date,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        min_gap_minutes=args.min_gap_minutes,
        package_name="0525_wildlife_camtrapdp",
        package_title="0525 Northeast China wildlife camera-trap observations",
        package_description="Five-species CSV source package retained for provenance; the agent runtime reads Chroma DB.",
        classified_by="0525_csv_conversion",
        source_timezone=args.source_timezone,
    )
    print(json.dumps(sanitize_for_display(report), ensure_ascii=False, indent=2))


def build_llm():
    class MissingAPILLM:
        def __init__(self, reason):
            self.reason = reason

        def invoke(self, _messages):
            raise RuntimeError(self.reason)

    azure_api_key = (
        os.getenv("AZURE_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("AZURE_OPENAI_KEY")
    )
    azure_endpoint = os.getenv("AZURE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    requested_provider = os.getenv("LLM_PROVIDER", "").strip().casefold()
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    openai_api_key = (
        deepseek_api_key
        if requested_provider == "deepseek"
        else os.getenv("OPENAI_API_KEY")
    )
    openai_base_url = (
        os.getenv("DEEPSEEK_BASE_URL")
        if requested_provider == "deepseek"
        else None
    ) or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    openai_model = (
        (
            os.getenv("DEEPSEEK_MODEL")
            if requested_provider == "deepseek"
            else None
        )
        or os.getenv("OPENAI_MODEL")
        or os.getenv("MODEL_NAME")
        or ("deepseek-v4-flash" if requested_provider == "deepseek" else "gpt-4.1-mini")
    )
    request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

    try:
        from langchain_openai import AzureChatOpenAI, ChatOpenAI
    except ModuleNotFoundError:
        return MissingAPILLM("langchain_openai is not installed in this Python environment.")

    if requested_provider in {"openai", "openai_compatible", "deepseek"} and not openai_api_key:
        return MissingAPILLM(
            f"LLM_PROVIDER={requested_provider} requires "
            f"{'DEEPSEEK_API_KEY' if requested_provider == 'deepseek' else 'OPENAI_API_KEY'}."
        )

    if azure_api_key and requested_provider not in {
        "openai",
        "openai_compatible",
        "deepseek",
    }:
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint or "https://<your-azure-endpoint>",
            api_key=azure_api_key,
            api_version=os.getenv("AZURE_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview",
            azure_deployment=azure_deployment or "gpt-4.1-mini",
            timeout=request_timeout,
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
            temperature=temperature,
        )

    if openai_api_key:
        kwargs = {
            "api_key": openai_api_key,
            "model": openai_model,
            "timeout": request_timeout,
            "max_retries": int(os.getenv("LLM_MAX_RETRIES", "1")),
            "temperature": temperature,
        }
        if requested_provider == "deepseek":
            kwargs["base_url"] = openai_base_url or "https://api.deepseek.com"
        elif openai_base_url:
            kwargs["base_url"] = openai_base_url
        return ChatOpenAI(**kwargs)

    return MissingAPILLM(
        "No API key is configured. Set AZURE_API_KEY, AZURE_OPENAI_API_KEY, "
        "OPENAI_API_KEY, or DEEPSEEK_API_KEY before running CamAgent."
    )


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _print_json_block(value, indent="  "):
    text = _json_text(sanitize_for_display(value))
    for line in text.splitlines():
        print(f"{indent}{line}")


def _fmt_value(value, default="unavailable"):
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _safe_join(value, sep=", "):
    return safe_join(value, sep=sep)


def _display_path(path):
    return display_path(path)


def _task_names(tasks):
    return [task.get("tool_name") for task in tasks if task.get("tool_name")]


def _first_task_arg(tasks, tool_name, arg_name, default=None):
    for task in tasks:
        if task.get("tool_name") == tool_name:
            return task.get("args", {}).get(arg_name, default)
    return default


def _trace_section(title, bullets):
    lines = [title]
    lines.extend(f"- {bullet}" for bullet in bullets)
    return lines


_INTENT_ORDER = ("quality", "temporal", "spatial", "community", "movement")
_INTENT_LABELS = {
    "quality": "data-quality audit",
    "temporal": "temporal activity and overlap",
    "spatial": "spatial distribution and detection hotspots",
    "community": "community composition and descriptive species relationships",
    "movement": "camera-detection adjacency hypotheses",
}
_QUALITY_TOOLS = {
    "analyze_camera_data_quality",
    "check_missing_fields",
    "detect_invalid_timestamps",
    "detect_label_noise",
    "detect_duplicate_like_records",
    "summarize_camera_record_volume",
    "summarize_duplicate_burden_by_camera",
    "analyze_data_quality",
    "plot_camera_quality_problem_barh",
}
_TEMPORAL_TOOLS = {
    "check_timestamp_quality",
    "build_hourly_activity_distribution",
    "detect_activity_peaks",
    "compute_hourly_relative_frequency",
    "compare_multi_species_hourly_activity",
    "build_species_activity_guilds",
    "circular_time_summary",
    "compare_day_vs_night_activity",
    "estimate_activity_overlap",
    "estimate_pairwise_activity_overlap",
    "estimate_hourly_activity_curve_with_ci",
    "build_hourly_activity_figure_data",
    "build_monthly_activity_distribution",
    "detect_monthly_activity_peaks",
    "compare_seasonal_activity",
    "build_weekday_activity_distribution",
    "analyze_temporal_activity_pattern",
}
_SPATIAL_TOOLS = {
    "build_detection_map",
    "build_hotspot_rank",
    "build_occupancy_summary",
    "build_detection_history_matrix",
    "fit_single_season_occupancy_model",
    "build_local_hotspot_rank",
    "summarize_species_by_camera",
    "analyze_spatial_distribution",
}
_COMMUNITY_TOOLS = {
    "summarize_species_inventory",
    "summarize_community_composition",
    "analyze_community_ecology",
    "build_camera_species_richness",
    "build_species_cooccurrence_matrix",
    "analyze_species_pair_relationship",
    "summarize_species_relationship_network",
}
_MOVEMENT_TOOLS = {
    "analyze_movement_steps",
    "summarize_movement_distance",
    "summarize_movement_direction",
    "infer_transition_corridors",
    "infer_longest_transition_steps",
    "analyze_animal_movement_pattern",
}


def _tool_intents(tool_name):
    name = str(tool_name or "")
    intents = set()
    if name in _QUALITY_TOOLS:
        intents.add("quality")
    if name in _TEMPORAL_TOOLS:
        intents.add("temporal")
    if name in _SPATIAL_TOOLS:
        intents.add("spatial")
    if name in _COMMUNITY_TOOLS:
        intents.add("community")
    if name in _MOVEMENT_TOOLS:
        intents.add("movement")

    if name.startswith("plot_"):
        if any(term in name for term in ("movement", "transition", "corridor")):
            intents.add("movement")
        if any(term in name for term in ("activity", "temporal", "overlap", "phenology")):
            intents.add("temporal")
        if any(
            term in name
            for term in (
                "community",
                "richness",
                "cooccurrence",
                "relationship",
                "inventory",
                "rank_abundance",
                "alpha_diversity",
                "accumulation",
                "rai_heatmap",
            )
        ):
            intents.add("community")
        if any(
            term in name
            for term in (
                "distribution",
                "hotspot",
                "occupancy",
                "detection_history",
                "ecology_map",
                "encounter_index",
            )
        ):
            intents.add("spatial")
        if "camera_quality" in name:
            intents.add("quality")
    return intents


def _intent_profile(tasks):
    detected = set()
    for task in tasks or []:
        detected.update(_tool_intents(task.get("tool_name")))
    return [intent for intent in _INTENT_ORDER if intent in detected]


def _intent_labels(tasks):
    return [_INTENT_LABELS[intent] for intent in _intent_profile(tasks)]


def _task_constraints(tasks):
    constraints = []
    for task in tasks:
        args = task.get("args", {})
        if args.get("min_gap_minutes") is not None:
            constraints.append(
                f"independent-event gap = {args.get('min_gap_minutes')} minutes"
            )
            break
    for task in tasks:
        args = task.get("args", {})
        if args.get("radius_km") is not None:
            constraints.append(f"spatial radius = {args.get('radius_km')} km")
            break
    for task in tasks:
        args = task.get("args", {})
        if args.get("max_time_gap_hours") is not None:
            constraints.append(
                f"maximum adjacency time gap = {args.get('max_time_gap_hours')} hours"
            )
            if args.get("max_distance_km") is not None:
                constraints.append(
                    f"maximum adjacency distance = {args.get('max_distance_km')} km"
                )
            break
    return constraints


def _format_tool_plan_summary(tasks):
    if not tasks:
        return "no executable tools selected"
    pieces = []
    for task in tasks:
        name = task.get("tool_name", "unknown_tool")
        args = task.get("args", {})
        if name == "estimate_activity_overlap":
            pieces.append(f"{name}({args.get('species_b', 'comparison_group')})")
        elif name == "estimate_pairwise_activity_overlap":
            pieces.append(f"{name}({safe_join(args.get('species_list', []))})")
        elif name == "plot_pairwise_activity_overlap_panels":
            pieces.append(f"{name}({safe_join(args.get('comparison_species', []))})")
        elif args.get("species"):
            pieces.append(f"{name}({args.get('species')})")
        elif args.get("species_a"):
            pieces.append(f"{name}({args.get('species_a')})")
        else:
            pieces.append(name)
    return " -> ".join(pieces)


def _visible_reasoning_trace_en(question, planning, tasks):
    selected_tools = _task_names(tasks)
    focal_species = planning.get("target_species") or "unspecified"
    comparison_species = _comparison_species_from_tasks(tasks)
    intents = _intent_profile(tasks)
    intent_labels = _intent_labels(tasks)
    visualization = any(str(task.get("tool_name", "")).startswith("plot_") for task in tasks)
    if not tasks:
        question_bullets = [
            "No executable registry tool remained after plan validation.",
            "The final response should therefore explain the evidence or constraint that prevents execution.",
        ]
    elif len(intents) > 1:
        question_bullets = [
            f"The query spans {safe_join(intent_labels)}.",
            "The requested outputs require a dependency-ordered multi-domain workflow.",
        ]
    else:
        label = intent_labels[0] if intent_labels else "general camera-trap analysis"
        question_bullets = [
            f"The query asks for {label}.",
            "The requested evidence must be produced by registered tools before answer synthesis.",
        ]

    entity_bullets = [f"Detected focal species: {focal_species}."]
    if focal_species == "Siberian tiger":
        entity_bullets.append("Normalized species name: Siberian tiger / Amur tiger.")
    else:
        entity_bullets.append(f"Normalized species name: {focal_species}.")
    if comparison_species:
        entity_bullets.append(
            f"Detected comparison species: {safe_join(comparison_species)}."
        )
    constraints = _task_constraints(tasks)
    if constraints:
        entity_bullets.append(f"Detected analytical constraints: {safe_join(constraints)}.")
    entity_bullets.append(
        f"Detected ecological concepts: {safe_join(intent_labels)}."
        if intent_labels
        else "Detected ecological concepts will be bounded by the validated registry plan."
    )

    assumption_bullets = []
    if "quality" in intents:
        assumption_bullets.append(
            "Data-quality claims require explicit checks of missing fields, timestamps, labels, and duplicate-like records."
        )
    if "temporal" in intents:
        assumption_bullets.extend([
            "Clock-time analyses require reliable timestamp provenance.",
            "Burst detections should be aggregated into independent events before diel summaries.",
        ])
    if "spatial" in intents:
        assumption_bullets.append(
            "Spatial summaries require valid camera locations and must distinguish detection hotspots from occupancy or density."
        )
    if "community" in intents:
        assumption_bullets.append(
            "Community summaries require an explicit focal-species set and must distinguish co-detection from direct interaction."
        )
    if "movement" in intents:
        assumption_bullets.append(
            "Adjacent camera detections lack individual identity and therefore support hypotheses, not tracks or confirmed corridors."
        )
    if not assumption_bullets:
        assumption_bullets.append(
            "Only evidence returned by validated tools may be used in the final answer."
        )
    assumption_bullets.append(
        "Visualization will be generated after analytical evidence is available."
        if visualization
        else "No visualization tool was selected unless explicitly required by the validated plan."
    )

    module_bullets = [
        f"Select the {_INTENT_LABELS[intent]} module."
        for intent in intents
    ]
    if not module_bullets:
        module_bullets.append("Use only the validated general registry tools.")
    if visualization:
        module_bullets.append("Select visualization tools that consume the analytical outputs.")

    lines = []
    lines.extend(_trace_section("Step 1. Understand the ecological question", question_bullets))
    lines.extend(_trace_section("Step 2. Identify entities and ecological concepts", entity_bullets))
    lines.extend(_trace_section("Step 3. Infer required data and assumptions", assumption_bullets))
    lines.extend(_trace_section("Step 4. Select analysis modules", module_bullets))
    lines.extend(_trace_section("Step 5. Translate reasoning into candidate tools", [
        "The LLM selects tools from the current registry schema rather than from local question-routing rules.",
        f"Candidate functions selected for validation: {safe_join(selected_tools)}."
        if selected_tools else "No candidate function remained after validation.",
        "The exact executable tool names must be validated against the current tool registry before execution.",
    ]))
    lines.extend(_trace_section("Step 6. Validate tool chain", [
        "Check that every selected tool exists.",
        "Check that required arguments are present.",
        "Check that no unsupported internal parameter is passed.",
        "Check that the selected tools can be executed in dependency order.",
    ]))
    lines.extend(_trace_section("Step 7. Produce executable plan", [
        f"Validated planner mode: {planning.get('planner_mode', 'llm')}.",
        f"Final validated tools: {safe_join(selected_tools)}." if selected_tools else "No executable tools remained after validation.",
        f"Executable tool chain: {_format_tool_plan_summary(tasks)}.",
    ]))
    return lines


def print_visible_reasoning_trace(question, planning, tasks):
    print("\n[Observable Agent Trace]")
    lines = _visible_reasoning_trace_en(question, planning, tasks)
    for idx, line in enumerate(lines):
        if idx > 0 and line.startswith("Step "):
            print()
        print(line)


def _comparison_species_from_tasks(tasks):
    return (
        _first_task_arg(tasks, "estimate_pairwise_activity_overlap", "species_list")
        or _first_task_arg(tasks, "plot_pairwise_activity_overlap_panels", "comparison_species")
        or []
    )


def _tool_use_decision_lines(planning, tasks):
    lines = []
    if planning.get("planner_raw_response"):
        lines.append("The LLM planner returned a structured JSON tool plan using the registered tool schema.")
    for task in tasks:
        tool_name = task.get("tool_name", "unknown_tool")
        reason = task.get("reason") or "No planner reason was provided."
        lines.append(f"{tool_name}: {reason}")
    if not lines:
        lines.append("No executable tools were returned by the API planner.")
    return lines


def _before_tool_execution_lines(tasks):
    lines = ["The selected tools will be executed in dependency order."]
    for idx, task in enumerate(tasks, start=1):
        name = task.get("tool_name", "unknown_tool")
        args = task.get("args", {})
        if name == "check_timestamp_quality":
            detail = "check timestamp quality before using clock-time analyses"
        elif name == "build_independent_events":
            detail = f"construct independent events with a {args.get('min_gap_minutes', 30)}-minute gap"
        elif name == "build_hourly_activity_distribution":
            detail = f"estimate the hourly activity distribution for {args.get('species', 'the focal species')}"
        elif name == "estimate_activity_overlap":
            detail = "estimate overall activity overlap against the comparison group"
        elif name == "estimate_pairwise_activity_overlap":
            detail = f"estimate pairwise overlap for {safe_join(args.get('species_list', []))}"
        elif name.startswith("plot_"):
            detail = "generate visual outputs after analytical results are available"
        else:
            detail = name
        lines.append(f"Step {idx}: {detail}.")
    lines.append("If a tool fails, successful partial results will be retained and the agent will evaluate whether replanning is needed.")
    return lines


def _result_check_lines(tasks):
    selected_tools = safe_join(_task_names(tasks))
    return [
        f"The LLM reflection step receives compact results for the selected tools: {selected_tools}.",
        "The LLM decides whether the current results are sufficient or whether new tasks are needed.",
    ]


def _synthesis_decision_lines(planning, replanning):
    lines = []
    lines.append("The answer is synthesized by the API model from the compact execution results and reflection JSON.")
    lines.append("The response must stay within successful tool outputs and report limitations from reflection.")
    if replanning.get("figure_outputs"):
        lines.append("Generated figures should be listed so the analysis can be inspected visually.")
    return lines


def _interpretation_boundary_lines(tasks):
    intents = _intent_profile(tasks)
    lines = []
    if "quality" in intents:
        lines.append(
            "Data-quality diagnostics support restrictions and follow-up checks; they do not repair unsupported records automatically."
        )
    if "temporal" in intents:
        lines.append(
            "Temporal overlap measures similarity in recorded-clock activity, not direct interaction or causation."
        )
    if "spatial" in intents:
        lines.append(
            "Detection hotspots are not automatically occupancy estimates, density estimates, or habitat-preference effects."
        )
    if "community" in intents:
        lines.append(
            "Shared-camera occurrence and network edges are descriptive and do not prove ecological interaction."
        )
    if "movement" in intents:
        lines.append(
            "Adjacent camera detections are hypotheses, not individual tracks or confirmed movement corridors."
        )
    return lines


def _print_bullets(lines):
    for line in lines:
        print(f"* {line}")


def _intent_flags_for_display(planning, tasks=None):
    intents = _intent_profile(tasks or [])
    flags = [f"{intent}_analysis" for intent in intents]
    if len(intents) > 1:
        flags.insert(0, "integrated_multi_domain")
    return flags or (["registry_valid_tool_use"] if tasks else ["no_valid_tool_plan"])


def _event_gap_for_display(tasks):
    for task in tasks:
        args = task.get("args", {})
        if args.get("min_gap_minutes") is not None:
            return f"{args.get('min_gap_minutes')} min"
    return "30 min"


def _visualization_for_display(question, tasks):
    if any(str(task.get("tool_name", "")).startswith("plot_") for task in tasks):
        return "requested_and_selected"
    q = question.lower()
    if any(
        term in q
        for term in ["plot", "figure", "visual", "map", "chart", "heatmap", "network"]
    ):
        return "requested_but_not_selected"
    return "not_requested"


def _print_planner_validation(planning):
    validation = planning.get("planner_validation") or {}
    plan_errors = validation.get("plan_errors") or []
    invalid_tools = validation.get("invalid_tools") or []
    missing_args = validation.get("missing_args") or []
    invalid_args = validation.get("invalid_args") or []
    filtered_args = validation.get("filtered_args") or []
    if not any((plan_errors, invalid_tools, missing_args, invalid_args, filtered_args)):
        return
    print("\n[Planner Validation]")
    if plan_errors:
        print("plan_errors:")
        for item in plan_errors:
            print(f"* {item}")
    if invalid_tools:
        print("invalid_tools:")
        for item in invalid_tools:
            print(f"* {item.get('tool_name')}")
            print(f"  reason: {item.get('reason')}")
            print(f"  action: {item.get('action')}")
    if missing_args:
        print("missing_args:")
        for item in missing_args:
            print(f"* {item.get('tool_name')}")
            print(f"  missing: {_safe_join(item.get('missing_args'))}")
            print(f"  action: {item.get('action')}")
    if invalid_args:
        print("invalid_args:")
        for item in invalid_args:
            print(f"* {item.get('tool_name')}.{item.get('argument')}")
            print(f"  reason: {item.get('reason')}")
            print(f"  action: {item.get('action')}")
    if filtered_args:
        print("unsupported_args:")
        for item in filtered_args:
            print(f"* {item.get('tool_name')}: {_safe_join(item.get('filtered_args'))}")
            print(f"  action: {item.get('action')}")


def _print_filtered_args(item):
    filtered = item.get("filtered_args") or []
    if filtered:
        print("filtered_args:")
        for arg in filtered:
            print(f"* {arg}")


def _parse_raw_json(text):
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


def _print_sanitized_json(value):
    print(_json_text(sanitize_for_display(value)))


def _raw_selected_tools(planning):
    raw = _parse_raw_json(planning.get("planner_raw_response", ""))
    return [task.get("tool_name") for task in raw.get("tasks", []) if task.get("tool_name")]


def _comparison_species_for_display(tasks):
    value = _comparison_species_from_tasks(tasks)
    if isinstance(value, str):
        return [value]
    return value or []


def _result_check_summary(execution, replanning):
    successful_tools = [
        item.get("tool_name")
        for item in execution
        if item.get("status") == "success"
    ]
    failed_tools = [
        item.get("tool_name")
        for item in execution
        if item.get("status") in {"failed", "error"}
    ]
    skipped_tools = [
        item.get("tool_name")
        for item in execution
        if item.get("status") == "skipped"
    ]
    returned_result_count = sum(
        item.get("status") == "success" and item.get("result") is not None
        for item in execution
    )
    consistency = replanning.get("consistency_check") or {}
    return {
        "successful_tools": successful_tools,
        "failed_tools": failed_tools,
        "skipped_tools": skipped_tools,
        "successful_tool_count": len(successful_tools),
        "returned_result_count": returned_result_count,
        "visualization result": replanning.get("visualization", "not_applicable"),
        "figure_count": replanning.get("figure_count", 0),
        "reflection_consistency": consistency.get("inconsistency_check", "not_reported"),
        "need_replan": bool(replanning.get("need_replan")),
    }


def print_planner_input(planning):
    summary = planning.get("planner_input_summary") or {}
    print("\n[Planner Input]")
    if not summary:
        print("<planner input summary unavailable>")
        return
    print(f"user_query: {summary.get('user_query')}")
    print(f"available_tool_names: {_safe_join(summary.get('available_tool_names', []))}")
    print(f"species_alias_rules summary: {summary.get('species_alias_rules')}")
    print(f"output_schema summary: {summary.get('output_schema')}")
    print("constraints:")
    for item in summary.get("constraints", []):
        print(f"* {item}")


def print_planner_raw_output(planning):
    print("\n[LLM Planner Raw Output]")
    raw = planning.get("planner_raw_response") or ""
    parsed = _parse_raw_json(raw)
    if parsed:
        _print_sanitized_json(parsed)
    else:
        print(sanitize_for_display(raw))


def print_planner_parse_result(record, planning, tasks):
    validation = planning.get("planner_validation") or {}
    print("\n[Planner Parse Result]")
    print(f"planner_mode: {planning.get('planner_mode', 'llm')}")
    attempt_count = planning.get("planner_attempt_count", 1)
    status = "llm_replanned_after_validation" if attempt_count > 1 else "success"
    print(f"parse_status: {status}")
    print(f"planner_attempt_count: {attempt_count}")
    print(f"target_species: {record.get('species') or record.get('plan', {}).get('species')}")
    print(f"comparison_species: {_safe_join(_comparison_species_for_display(tasks))}")
    print(f"intents: {_safe_join(_intent_flags_for_display(planning, tasks), sep=' + ')}")
    print(f"visualization: {_visualization_for_display(record.get('question', ''), tasks)}")
    print(f"selected_tools_raw: {_safe_join(_raw_selected_tools(planning))}")
    invalid = validation.get("invalid_tools", [])
    print(f"invalid_tools: {_json_text(sanitize_for_display(invalid)) if invalid else '[]'}")
    print(f"validation_action: {validation.get('action', 'validated')}")


def print_tool_call_preparation(tasks, execution):
    print("\n[Tool Call Preparation]")
    by_name = {}
    for item in execution:
        by_name.setdefault(item.get("tool_name"), item)
    for idx, task in enumerate(tasks, start=1):
        item = by_name.get(task.get("tool_name"), {})
        print(f"\ntool_name: {task.get('tool_name')}")
        print(f"raw_args_from_planner: {_json_text(sanitize_for_display(item.get('input_args', task.get('args', {}))))}")
        print(f"validated_args: {_json_text(sanitize_for_display(item.get('args', task.get('args', {}))))}")
        print(f"filtered_args: {_json_text(sanitize_for_display(item.get('filtered_args', [])))}")
        print(f"reason_for_call: {task.get('reason') or item.get('reason')}")
        print(f"dependency_position: {idx}")


def print_result_check(execution, replanning):
    print("\n[LLM Result Check]")
    summary = _result_check_summary(execution, replanning)
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"{key}: {_json_text(value)}")
        else:
            print(f"{key}: {_fmt_value(value)}")


def print_reflection_raw(replanning):
    print("\n[LLM Reflection Raw Output]")
    raw = replanning.get("replanner_raw_response")
    if raw:
        parsed = _parse_raw_json(raw)
        if parsed:
            _print_sanitized_json(parsed)
        else:
            print(sanitize_for_display(raw))
    else:
        print("reflection_mode: deterministic")
        print("reason: computed from tool outputs")


def print_synthesis_input(record):
    print("\n[LLM Synthesis Input]")
    synthesis = record.get("trace", {}).get("synthesis", {})
    evidence = synthesis.get("synthesis_input_evidence") or {}
    _print_sanitized_json(evidence)


def print_final_raw(record):
    print("\n[LLM Final Raw Output]")
    raw = record.get("trace", {}).get("synthesis", {}).get("answer_raw_response", "")
    print(sanitize_for_display(raw) if raw else "<unavailable>")


def _display_peak_period(hourly):
    label = hourly.get("activity_pattern_label")
    if label and label != "insufficient_data":
        return f"{label} activity"
    return hourly.get("peak_period_label", "unavailable")


def _find_execution(execution, tool_name):
    for item in execution:
        if item.get("tool_name") == tool_name:
            return item
    return {}


def _result(item):
    result = item.get("result")
    return result if isinstance(result, dict) else {}


def _relationship(delta):
    if delta is None:
        return "unavailable"
    try:
        delta = float(delta)
    except Exception:
        return "unavailable"
    if delta >= 0.80:
        return "high_overlap"
    if delta >= 0.60:
        return "moderate_overlap"
    return "low_overlap"


def print_agent_record_brief(record):
    print("\n[Selected tools]")
    for task in record["plan"].get("tasks", []):
        print(f"- {task.get('tool_name')}: {task.get('args', {})}")
    print("\n[Final answer]")
    print(record["final_answer"])


def print_agent_record_full_trace(record, brief=False):
    if brief:
        print_agent_record_brief(record)
        return

    trace = record.get("trace", {})
    planning = trace.get("planning", {})
    execution = trace.get("execution", [])
    replanning = trace.get("replanning", {})
    tasks = planning.get("normalized_tasks") or record.get("plan", {}).get("tasks", [])

    print("\n[User Query]")
    print(record.get("question", ""))
    print_planner_input(planning)
    print_planner_raw_output(planning)
    print_planner_parse_result(record, planning, tasks)
    print_visible_reasoning_trace(record.get("question", ""), planning, tasks)

    print("\n[Intent Parsing]")
    print(f"target_species: {record.get('species') or record.get('plan', {}).get('species')}")
    print(f"intent: {_safe_join(_intent_flags_for_display(planning, tasks), sep=' + ')}")
    print(f"event_gap: {_event_gap_for_display(tasks)}")
    print(f"visualization: {_visualization_for_display(record.get('question', ''), tasks)}")
    print(f"planner_mode: {planning.get('planner_mode', 'llm')}")

    print("\n[LLM Tool-Use Decision]")
    _print_bullets(_tool_use_decision_lines(planning, tasks))
    _print_planner_validation(planning)

    print("\n[Tool Planning]")
    for idx, task in enumerate(tasks, start=1):
        tool_name = task.get("tool_name")
        args = task.get("args", {})
        if tool_name == "estimate_activity_overlap":
            print(f"{idx}. {tool_name}: {args.get('species_b')}")
        elif tool_name == "estimate_pairwise_activity_overlap":
            print(f"{idx}. {tool_name}: {_safe_join(args.get('species_list', []))}")
        elif tool_name == "plot_pairwise_activity_overlap_panels":
            print(f"{idx}. {tool_name}: {_safe_join(args.get('comparison_species', []))}")
        elif args.get("species"):
            print(f"{idx}. {tool_name}: {args.get('species')}")
        else:
            print(f"{idx}. {tool_name}")

    print_tool_call_preparation(tasks, execution)

    print("\n[LLM Before Tool Execution]")
    _print_bullets(_before_tool_execution_lines(tasks))

    print_tool_execution(record, execution)
    print_result_check(execution, replanning)
    print_reflection(record, replanning)
    print_reflection_raw(replanning)
    print_synthesis_input(record)

    print("\n[LLM Synthesis Decision]")
    _print_bullets(_synthesis_decision_lines(planning, replanning))
    synthesis_boundaries = [
        "Only use values shown in Tool Execution / Reflection.",
        "Do not introduce numerical values or rankings unless they are present in successful tool outputs.",
        "Use relative paths only.",
    ]
    synthesis_boundaries.extend(_interpretation_boundary_lines(tasks))
    _print_bullets(synthesis_boundaries)
    print_final_raw(record)

    print("\n[Final Answer]")
    print(sanitize_for_display(record["final_answer"]))


def print_tool_execution(record, execution):
    print("\n[Tool Execution]")
    if not execution:
        print("<no tool executions captured>")
        return
    _print_tool_execution_body(execution)


def print_reflection(record, replanning):
    print("\n[Reflection]")
    _print_reflection_body(replanning)


def print_agent_record(record, brief=False, full_trace=False):
    if full_trace:
        print_agent_record_full_trace(record, brief=brief)
        return
    if brief:
        print_agent_record_brief(record)
        return

    trace = record.get("trace", {})
    planning = trace.get("planning", {})
    execution = trace.get("execution", [])
    replanning = trace.get("replanning", {})

    tasks = planning.get("normalized_tasks") or record.get("plan", {}).get("tasks", [])

    print("\n[User Query]")
    print(record.get("question", ""))

    print_visible_reasoning_trace(record.get("question", ""), planning, tasks)

    print("\n[Intent Parsing]")
    print(f"target_species: {record.get('species') or record.get('plan', {}).get('species')}")
    intent_flags = _intent_flags_for_display(planning, tasks)
    print(f"intent: {_safe_join(intent_flags, sep=' + ')}")
    print(f"event_gap: {_event_gap_for_display(tasks)}")
    print(f"visualization: {_visualization_for_display(record.get('question', ''), tasks)}")
    print(f"planner_mode: {planning.get('planner_mode', 'llm')}")

    print("\n[LLM Tool-Use Decision]")
    _print_bullets(_tool_use_decision_lines(planning, tasks))

    _print_planner_validation(planning)

    print("\n[Tool Planning]")
    for idx, task in enumerate(tasks, start=1):
        tool_name = task.get("tool_name")
        args = task.get("args", {})
        if tool_name == "estimate_activity_overlap":
            print(f"{idx}. {tool_name}: {args.get('species_b')}")
        elif tool_name == "estimate_pairwise_activity_overlap":
            species_list = args.get("species_list", [])
            print(f"{idx}. {tool_name}: {_safe_join(species_list)}")
        elif tool_name == "plot_hourly_activity_curve":
            print(f"{idx}. {tool_name}: {args.get('species')}")
        elif tool_name == "plot_pairwise_activity_overlap_panels":
            print(f"{idx}. {tool_name}: {_safe_join(args.get('comparison_species'))}")
        else:
            print(f"{idx}. {tool_name}")

    print("\n[LLM Before Tool Execution]")
    _print_bullets(_before_tool_execution_lines(tasks))

    print("\n[Tool Execution]")
    if not execution:
        print("<no tool executions captured>")
    _print_tool_execution_body(execution)

    print_result_check(execution, replanning)

    print("\n[Reflection]")
    _print_reflection_body(replanning)

    print("\n[LLM Synthesis Decision]")
    _print_bullets(_synthesis_decision_lines(planning, replanning))

    print("\n[Final Answer]")
    print(sanitize_for_display(record["final_answer"]))


def _result_summary_for_display(result):
    if isinstance(result, dict):
        summary = {}
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = value
            elif isinstance(value, (list, tuple)):
                summary[f"{key}_count"] = len(value)
            elif isinstance(value, dict):
                summary[f"{key}_keys"] = list(value.keys())[:12]
        return summary
    if isinstance(result, (list, tuple)):
        return {"item_count": len(result)}
    return {"value": result}


def _print_tool_execution_body(execution):
    if not execution:
        return

    timestamp = _result(_find_execution(execution, "check_timestamp_quality"))
    if timestamp:
        print("✓ check_timestamp_quality")
        print(f"observed_clock_time_ratio: {_fmt_value(timestamp.get('observed_clock_time_ratio'))}")
        print(f"synthetic_clock_time_ratio: {_fmt_value(timestamp.get('synthetic_clock_time_ratio'))}")
        print(f"invalid_timestamp_count: {_fmt_value(timestamp.get('invalid_timestamp_count'))}")
        print(f"timestamp_source: {timestamp.get('timestamp_source') or timestamp.get('timestamp_source_summary')}")
        _print_filtered_args(_find_execution(execution, "check_timestamp_quality"))

    events = _result(_find_execution(execution, "build_independent_events"))
    if events:
        print("\n✓ build_independent_events")
        print(f"raw_records: {_fmt_value(events.get('raw_record_count') or events.get('source_record_count'))}")
        print(f"independent_events: {_fmt_value(events.get('independent_event_count') or events.get('event_count'))}")
        print(f"min_gap_minutes: {_fmt_value(events.get('event_gap_minutes') or events.get('min_gap_minutes'))}")
        _print_filtered_args(_find_execution(execution, "build_independent_events"))

    hourly = _result(_find_execution(execution, "build_hourly_activity_distribution"))
    if hourly:
        print("\n✓ build_hourly_activity_distribution")
        print(f"peak_hours: {hourly.get('peak_hours', [])}")
        print(f"peak_period: {_display_peak_period(hourly)}")
        print(f"major_peak_window: {hourly.get('major_peak_window', 'unavailable')}")
        print(f"secondary_peak_window: {hourly.get('secondary_peak_window', 'none')}")
        print(f"activity_pattern_label: {hourly.get('activity_pattern_label', 'unavailable')}")
        if hourly.get("activity_window_shares") is not None:
            print(f"time_window_shares: {_json_text(hourly.get('activity_window_shares'))}")
        _print_filtered_args(_find_execution(execution, "build_hourly_activity_distribution"))

    overlap = _result(_find_execution(execution, "estimate_activity_overlap"))
    if overlap:
        print("\n✓ estimate_activity_overlap")
        print(f"all_other_species_delta: {_fmt_value(overlap.get('delta_hat'))}")
        print(f"relationship: {_relationship(overlap.get('delta_hat'))}")
        _print_filtered_args(_find_execution(execution, "estimate_activity_overlap"))

    pairwise = _result(_find_execution(execution, "estimate_pairwise_activity_overlap"))
    if pairwise:
        print("\n✓ estimate_pairwise_activity_overlap")
        for row in pairwise.get("pairwise_results", []):
            print(f"{row.get('species_b')}:")
            print(f"  Delta: {_fmt_value(row.get('delta_hat'))}")
            print(f"  overlap_level: {row.get('relationship')}")
            print(f"  ci_low: {_fmt_value(row.get('delta_ci_low'))}")
            print(f"  ci_high: {_fmt_value(row.get('delta_ci_high'))}")
            print(f"  peak_hours_a: {row.get('peak_hours_a', [])}")
            print(f"  peak_hours_b: {row.get('peak_hours_b', [])}")
        for row in pairwise.get("skipped", []):
            print(f"{row.get('species')}: skipped_due_to_insufficient_events")
        _print_filtered_args(_find_execution(execution, "estimate_pairwise_activity_overlap"))

    for item in execution:
        if str(item.get("tool_name", "")).startswith("plot_"):
            if item.get("status") == "success":
                result = item.get("result")
                print(f"\n✓ {item.get('tool_name')}")
                if isinstance(result, str):
                    print(f"output_path: {_display_path(result)}")
                    print("status: generated")
                elif isinstance(result, dict):
                    for figure in result.get("figures", []):
                        species_b = figure.get("species_b") or figure.get("figure_id")
                        if figure.get("png_path"):
                            print(f"{species_b} png_path: {_display_path(figure.get('png_path'))}")
                        if figure.get("svg_path"):
                            print(f"{species_b} svg_path: {_display_path(figure.get('svg_path'))}")
                    for figure in result.get("failed_figures", []):
                        print(f"{figure.get('species_b') or figure.get('figure_id')}: failed")
                        print(f"reason: {figure.get('reason')}")
                    print(f"status: {result.get('status', 'generated')}")
                _print_filtered_args(item)
            elif item.get("status") == "skipped":
                print(f"\n- {item.get('tool_name')}")
                print("visualization_suggestion: true")
                print("visualization_executed: false")
                print(f"reason: {item.get('skip_reason')}")

    community = _result(_find_execution(execution, "summarize_community_composition"))
    if community:
        print("\n✓ summarize_community_composition")
        print("top_sympatric_species:")
        for row in community.get("top_species", []):
            print(f"* species: {row.get('species')}")
            event_count = row.get("independent_event_count") or row.get("event_count") or row.get("record_count")
            if event_count is not None:
                print(f"  independent_events: {event_count}")
            for key in ("rai", "rai_per_100_camera_days", "relative_abundance_index"):
                if row.get(key) is not None:
                    print(f"  {key}: {_fmt_value(row.get(key))}")
        _print_filtered_args(_find_execution(execution, "summarize_community_composition"))

    other_items = [
        item for item in execution
        if item.get("tool_name") not in {
            "check_timestamp_quality",
            "build_independent_events",
            "build_hourly_activity_distribution",
            "estimate_activity_overlap",
            "estimate_pairwise_activity_overlap",
            "summarize_community_composition",
        }
        and not str(item.get("tool_name", "")).startswith("plot_")
    ]
    for item in other_items:
        marker = "✓" if item.get("status") == "success" else "!"
        print(f"\n{marker} {item.get('tool_name')}")
        print(f"status: {item.get('status')}")
        _print_filtered_args(item)
        if item.get("status") == "success":
            print(
                "result_summary: "
                f"{_json_text(_result_summary_for_display(item.get('result')))}"
            )
        if item.get("status") in {"error", "failed"}:
            print(f"reason: {item.get('reason') or item.get('error')}")

def _print_reflection_body(replanning):
    timestamp_ref = replanning.get("timestamp_reliability") or {}
    event_ref = replanning.get("event_count_sufficiency") or {}
    independence_ref = replanning.get("event_independence") or {}
    overlap_ref = replanning.get("overlap_analysis") or {}
    data_quality_ref = replanning.get("data_quality", "not_applicable")
    spatial_ref = replanning.get("spatial_data_sufficiency") or {"status": "not_applicable"}
    print(f"data quality: {data_quality_ref}")
    if timestamp_ref:
        print(f"timestamp reliability: {timestamp_ref.get('status')}")
        print(f"observed_clock_time_ratio: {_fmt_value(timestamp_ref.get('observed_clock_time_ratio'))}")
        print(f"synthetic_clock_time_ratio: {_fmt_value(timestamp_ref.get('synthetic_clock_time_ratio'))}")
    if independence_ref:
        print(f"event independence: {independence_ref.get('status')}")
        print(f"min_gap_minutes: {_fmt_value(independence_ref.get('min_gap_minutes'))}")
    if event_ref:
        print(f"event count sufficiency: {event_ref.get('status')}")
        print(f"tiger_events: {_fmt_value(event_ref.get('tiger_events'))}")
    if overlap_ref:
        print(f"temporal overlap analysis: {overlap_ref.get('status')}")
        print(f"overall_delta: {_fmt_value(overlap_ref.get('overall_delta'))}")
        print(f"pairwise_results: {overlap_ref.get('pairwise_results')}")
    print(f"spatial data sufficiency: {spatial_ref.get('status')}")
    print(f"visualization: {replanning.get('visualization', 'not_suggested')}")
    if replanning.get("figure_count") is not None:
        print(f"figure_count: {replanning.get('figure_count')}")
    if replanning.get("figure_outputs"):
        print("figure_outputs:")
        for figure in replanning.get("figure_outputs", []):
            print(f"- {figure.get('figure_id') or figure.get('label')}:")
            if figure.get("png_path"):
                print(f"    png_path: {_display_path(figure.get('png_path'))}")
            if figure.get("svg_path"):
                print(f"    svg_path: {_display_path(figure.get('svg_path'))}")
            if figure.get("path") and not figure.get("png_path") and not figure.get("svg_path"):
                print(f"    path: {_display_path(figure.get('path'))}")
    if replanning.get("failed_figures"):
        print("failed_figures:")
        for figure in replanning.get("failed_figures", []):
            print(f"- {figure.get('figure_id')}: {figure.get('reason')}")
    if replanning.get("reflection_mode"):
        print(f"reflection_mode: {replanning.get('reflection_mode')}")
    if replanning.get("consistency_check"):
        check = replanning.get("consistency_check") or {}
        print(f"missing_fields: {_json_text(check.get('missing_fields', []))}")
        print(f"inconsistency_check: {check.get('inconsistency_check')}")
    print(f"need_replan: {bool(replanning.get('need_replan'))}")
    if replanning.get("reason"):
        print(f"reason: {replanning.get('reason')}")


def command_chat(args):
    policy = AgentRunPolicy(
        max_replan_rounds=args.max_replan_rounds,
        max_tool_calls=args.max_tool_calls,
        max_tool_retries=args.max_tool_retries,
    )
    agent = WildlifeAgent(build_llm(), default_species=args.default_species, run_policy=policy)

    def save_trace(payload):
        if not args.trace_output:
            return
        trace_path = Path(args.trace_output).expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, list):
            export_payload = [
                agent.exportable_trace(item, detail=args.trace_detail)
                for item in payload
            ]
        else:
            export_payload = agent.exportable_trace(payload, detail=args.trace_detail)
        trace_path.write_text(
            json.dumps(export_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    if args.question:
        print("[Status] Running LLM planning and tool execution...", flush=True)
        record = agent.run(args.question, experiment_id=args.experiment_id)
        save_trace(record)
        print_agent_record(record, brief=args.brief, full_trace=args.full_trace)
        return

    print("CamAgent: An Autonomous LLM-Agent Framework for Camera Trap Data")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except EOFError:
            print("\nEnd of input detected; exiting.")
            break
        except KeyboardInterrupt:
            print("\nInterrupted. Bye.")
            break
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not question:
            continue
        try:
            print("[Status] Running LLM planning and tool execution...", flush=True)
            record = agent.run(question, experiment_id=args.experiment_id)
            save_trace(agent.history)
            print_agent_record(record, brief=args.brief, full_trace=args.full_trace)
        except Exception as exc:
            print(f"[ERROR] {type(exc).__name__}: {exc}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Single journal-facing entrypoint for recognition, JSON->DB sync, and DB-backed agent analysis."
    )
    subparsers = parser.add_subparsers(dest="command")

    sync_db = subparsers.add_parser("sync-db", help="Sync normalized JSON records into the Chroma DB.")
    sync_db.add_argument("--json-dir", action="append", default=None)
    sync_db.add_argument("--db-dir", default=str(VECTOR_DB_DIR))
    sync_db.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    sync_db.add_argument("--recursive", action="store_true")
    sync_db.set_defaults(func=command_sync_db)

    recognize = subparsers.add_parser("recognize-image", help="Run MegaDetector + SpeciesNet on one image.")
    recognize.add_argument("image_path")
    recognize.add_argument("--country")
    recognize.add_argument("--admin1-region")
    recognize.add_argument("--output-json")
    recognize.add_argument("--no-sidecar", action="store_true")
    recognize.set_defaults(func=command_recognize_image)

    convert = subparsers.add_parser("convert-csv", help="Convert retained five-species CSV files for provenance.")
    convert.add_argument("--input-dir", default=str(AGENT_ROOT / "0525newdata"))
    convert.add_argument("--output-dir", default=str(CAMTRAP_IMPORTS_DIR / "0525_wildlife_camtrapdp"))
    convert.add_argument("--start-date", default="2023-01-01")
    convert.add_argument("--end-date", default="2024-12-31 23:59:59")
    convert.add_argument("--lon-min", type=float, default=129.0)
    convert.add_argument("--lon-max", type=float, default=132.0)
    convert.add_argument("--lat-min", type=float, default=42.0)
    convert.add_argument("--lat-max", type=float, default=45.0)
    convert.add_argument("--min-gap-minutes", type=int, default=30)
    convert.add_argument(
        "--source-timezone",
        default="Asia/Shanghai",
        help="IANA timezone for naive field-clock timestamps in source CSV files.",
    )
    convert.set_defaults(func=command_convert_csv)

    chat = subparsers.add_parser("chat", help="Ask questions using the DB-backed agent.")
    chat.add_argument("--question")
    chat.add_argument(
        "--experiment-id",
        choices=(
            "QUAL", "TEMP", "SPAT", "COMM", "MOVE", "INTEG",
            "RAW_MEDIA_PIPELINE",
        ),
        help=(
            "Attach a prespecified ecological-query ID, or the separate "
            "RAW_MEDIA_PIPELINE engineering-validation ID, to the saved trace."
        ),
    )
    chat.add_argument(
        "--default-species",
        default=os.getenv("DEFAULT_SPECIES") or None,
        help="Optional fallback focal species. If omitted, the Agent uses the active inventory or asks for clarification.",
    )
    chat.add_argument("--brief", action="store_true", help="Print only selected tools and the final answer.")
    chat.add_argument(
        "--full-trace",
        action="store_true",
        help="Print the complete observable planning-and-execution trace.",
    )
    chat.add_argument("--trace-output", help="Write the sanitized JSON run trace (or interactive trace history) here.")
    chat.add_argument(
        "--trace-detail",
        choices=("public", "full"),
        default="public",
        help=(
            "Public traces omit raw records/model text and redact camera "
            "locations, local paths, and media names; full traces may contain "
            "sensitive data."
        ),
    )
    chat.add_argument("--max-replan-rounds", type=int, default=2)
    chat.add_argument("--max-tool-calls", type=int, default=32)
    chat.add_argument("--max-tool-retries", type=int, default=1)
    chat.set_defaults(func=command_chat)
    return parser


def main():
    configure_runtime()
    argv = sys.argv[1:] or ["chat"]
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
