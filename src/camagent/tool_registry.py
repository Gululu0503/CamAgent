import inspect

from .paths import OUTPUTS_DIR
from .visualization_tools import VisualizationTools

class ToolRegistry:
    _PLANNER_EXPOSED_TOOLS = {
        # Query and event evidence
        "check_timestamp_quality",
        "build_independent_events",
        "summarize_species_inventory",
        "summarize_community_composition",
        "analyze_community_ecology",
        "infer_complete_community",
        "filter_records",
        "browse_records",
        "evidence_lookup",
        # Spatial and relationship analysis
        "build_detection_map",
        "build_hotspot_rank",
        "build_occupancy_summary",
        "build_detection_history_matrix",
        "fit_single_season_occupancy_model",
        "build_local_hotspot_rank",
        "summarize_species_by_camera",
        "build_camera_species_richness",
        "build_species_cooccurrence_matrix",
        "analyze_species_pair_relationship",
        "summarize_species_relationship_network",
        "analyze_spatial_distribution",
        # Temporal analysis
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
        # Quality and movement analysis
        "analyze_camera_data_quality",
        "check_missing_fields",
        "detect_invalid_timestamps",
        "detect_label_noise",
        "detect_duplicate_like_records",
        "summarize_camera_record_volume",
        "summarize_duplicate_burden_by_camera",
        "analyze_data_quality",
        "analyze_movement_steps",
        "summarize_movement_distance",
        "summarize_movement_direction",
        "infer_transition_corridors",
        "infer_longest_transition_steps",
        "analyze_animal_movement_pattern",
        # Perception, figures, and exports
        "prepare_camera_trap_dataset",
        "run_detection_pipeline",
        "evaluate_perception_benchmark",
        "run_ecology_figure_suite",
        "export_camtrap_dp",
        "export_media_by_species",
        "plot_species_distribution_map",
        "plot_local_hotspot_map",
        "plot_effort_corrected_hotspot_map",
        "plot_predator_prey_encounter_index_map",
        "plot_multi_species_ecology_map",
        "plot_monthly_activity_bar",
        "plot_hourly_activity_curve",
        "plot_activity_overlap_panel",
        "plot_pairwise_activity_overlap_panels",
        "plot_pairwise_temporal_overlap",
        "plot_activity_overlap_triptych",
        "plot_multi_species_occupancy_forest",
        "plot_species_inventory_bar",
        "plot_camera_species_richness_bar",
        "plot_species_cooccurrence_heatmap",
        "plot_species_relationship_network",
        "plot_community_overview_panel",
        "plot_species_pair_relationship_panel",
        "plot_multi_species_hourly_activity",
        "plot_species_rank_abundance_curve",
        "plot_multi_species_monthly_phenology",
        "plot_camera_alpha_diversity_panel",
        "plot_species_accumulation_curve",
        "plot_species_rai_heatmap",
        "plot_temporal_overlap_matrix",
        "plot_community_spatial_association_panel",
        "plot_complete_community_inference",
        "plot_transition_corridor_map",
        "plot_weekday_activity_bar",
        "plot_seasonal_activity_bar",
        "plot_hotspot_rank_bar",
        "plot_hotspot_rank_barh",
        "plot_movement_direction_bar",
        "plot_movement_distance_hist",
        "plot_camera_quality_problem_barh",
        "plot_species_distribution_static",
        "plot_local_hotspot_static",
        "plot_detection_history_heatmap",
        "plot_occupancy_model_forest",
        "plot_occupancy_bootstrap_density",
    }

    _PARAM_DESCRIPTIONS = {
        "species": "Species name, for example Siberian tiger.",
        "species_a": "First species name.",
        "species_b": "Second species name.",
        "focal_species": "Focal species name.",
        "comparison_species": "Comparison species name for pairwise temporal-overlap plotting.",
        "species_list": "Optional list of species names, or a comma-separated string.",
        "camera_id": "Camera identifier, for example 213215.",
        "center_camera_id": "Center camera identifier for radius-based spatial queries.",
        "radius_km": "Radius in kilometers.",
        "top_k": "Number of top results to return.",
        "save_path": "Optional output file path.",
        "output_dir": "Optional output directory.",
        "target_dir": "Optional target directory for generated JSON records.",
        "datapackage_path": "Path to a CamtrapDP datapackage.json file or its containing directory.",
        "table_path": "Path to a CSV, TSV, or Excel camera-trap table to convert to CamtrapDP.",
        "field_mapping": "Dictionary or JSON object mapping semantic fields to input columns, such as species, camera_id, timestamp, latitude, longitude, and media.",
        "sheet_name": "Excel sheet name or index when converting spreadsheet input.",
        "tabular_backend": "Whether to activate CamtrapDP directly as a tabular backend without creating per-record JSON.",
        "persist_active": "Whether to persist this package as the default dataset for future sessions.",
        "write_validation": "Whether to write validation results back into the input package.",
        "prefer_media_level": "Whether to prefer media-level observations. Defaults to True.",
        "include_blank": "Whether to include blank observations. Defaults to False.",
        "include_event_fallback": "Whether to fall back to event-level observations when media-level observations are missing.",
        "activate_json_backend": "Whether to switch the active JSON backend after generation or import.",
        "include_unknown": "Whether to include Unknown records. Defaults to False.",
        "min_gap_minutes": "Minimum gap in minutes used to merge records into independent events.",
        "abundance_metric": "Abundance metric to use, such as rai, independent_events, or raw_records.",
        "min_effort_days_for_rai": "Minimum camera effort days required for RAI calculations.",
        "min_observed_span_days_for_rai": "Minimum observed date span required for RAI calculations.",
        "min_independent_events_for_rai": "Minimum independent events required for RAI ranking.",
        "min_independent_events": "Minimum independent-event count required for a species to enter the requested inference.",
        "min_camera_records": "Minimum record or event count required for a species-camera relationship.",
        "min_records_per_species": "Minimum independent events per species required for richness calculations.",
        "min_rai_per_100_camera_days": "Minimum RAI per 100 camera-days required for inclusion.",
        "include_relationships": "Whether to include relationship-network summaries.",
        "max_pairs": "Maximum number of candidate species pairs to evaluate.",
        "min_detected_cameras": "Minimum number of cameras detecting a species before it enters community inference.",
        "accumulation_permutations": "Number of randomized camera orders used for the species-accumulation interval.",
        "association_fdr_alpha": "Benjamini-Hochberg false-discovery-rate threshold for conditional shared-camera tests.",
        "max_beta_pairs": "Maximum number of camera pairs used for beta-diversity summaries.",
        "fit_detection_corrected_occupancy": "Whether to fit multi-species repeated-survey occupancy models when documented deployment effort is available.",
        "occupancy_species_limit": (
            "Optional computational cap for occupancy fitting; omit it to fit "
            "every sufficiently supported species."
        ),
        "occupancy_bootstrap_iters": "Number of camera-level bootstrap iterations for each occupancy model.",
        "display_top_species": "Number of leading species displayed in a community figure; inference can still use all eligible species.",
        "include_event_observations": "Whether to export additional event-level observations.",
        "package_name": "Optional CamtrapDP package name.",
        "package_title": "Optional CamtrapDP package title.",
        "package_version": "Optional CamtrapDP package version.",
        "data_license": "SPDX-style license identifier confirmed by the data rights holder.",
        "media_license": "SPDX-style license identifier confirmed by the media rights holder.",
        "use_symlinks": "Whether to use symlinks instead of copying media files.",
        "min_confidence": "Optional minimum detection or classification confidence.",
        "limit": "Maximum number of records or samples to return.",
        "start_date": "Optional start date or timestamp.",
        "end_date": "Optional end date or timestamp.",
        "start_hour": "Optional start hour.",
        "end_hour": "Optional end hour.",
        "sort_by": "Sort mode.",
        "media_dir": "Directory containing images or videos.",
        "input_path": "Path to raw image/video media, a tabular observation file, or a CamtrapDP package.",
        "input_type": "Input format: auto, raw_media, table, or camtrap_dp. Auto detection inspects only the file format.",
        "metadata_table": "Optional CSV/TSV/Excel file mapping media files to camera IDs, field timestamps, and coordinates.",
        "source_timezone": "IANA timezone for naive field timestamps, for example Asia/Shanghai.",
        "video_frame_interval_seconds": "Sampling interval in seconds for auditable frame-level video perception.",
        "max_video_frames": "Maximum sampled frames per video.",
        "run_timestamp_ocr": "Whether raw videos lacking an explicit field timestamp should be read with the optional PaddleOCR timestamp stage.",
        "timestamp_ocr_max_frames": "Maximum video frames examined for an overlaid field timestamp.",
        "allow_metadata_only": "Whether files without model predictions may become explicitly labelled metadata-only records.",
        "annotation_path": "CSV, TSV, JSON, or JSONL annotation table containing media paths and independently reviewed ground-truth labels.",
        "media_root": "Optional root directory used to resolve relative media paths in an annotation table.",
        "media_path_column": "Annotation-table column containing an image or video path.",
        "true_label_column": "Annotation-table column containing the reviewed ground-truth species label.",
        "predicted_label_column": "Column containing or receiving the model top-label prediction.",
        "confidence_column": "Column containing or receiving prediction confidence.",
        "group_column": "Camera or independent sampling-unit column used for clustered bootstrap intervals.",
        "sample_id_column": "Unique sample-ID column used to detect duplication and resume batch inference.",
        "max_samples": "Optional deterministic cap on the number of annotated media samples to evaluate.",
        "label_aliases": "Optional dictionary or JSON object mapping synonymous labels to the same evaluated taxon.",
        "confidence_threshold": "Prediction-confidence threshold for thresholded accuracy and coverage.",
        "calibration_bins": "Number of bins used for reliability and calibration summaries.",
        "camera_disjoint_verified": "Whether the held-out camera split has been independently verified; a development-camera file provides machine verification.",
        "human_review_status": "Annotation status: unverified, human_reviewed, adjudicated, or expert_consensus.",
        "split_note": "Optional description of the held-out split protocol.",
        "development_camera_ids_path": "Optional one-column file of development camera IDs used to verify zero camera overlap.",
        "annotation_manifest_path": "Optional signed or archived annotation-provenance JSON with protocol, reviewer roles, review time, and annotation hash.",
        "minimum_class_support": "Minimum reviewed sample count required per reported class.",
        "resume": "Whether to resume from a previously written scored-record checkpoint.",
        "activate_backend": "Whether to activate the prepared CamtrapDP package for subsequent tools in the same Agent run.",
        "persist_active": "Whether to persist the activated CamtrapDP package as the current dataset for later sessions.",
        "detector": "Detector name, such as megadetector or yolo.",
        "classifier": "Species classifier name, such as speciesnet.",
        "default_species": "Default species name to use when classifier output is missing.",
        "run_inference": "Whether to run real detector and classifier inference.",
        "country": "Optional ISO 3166-1 alpha-3 country code for SpeciesNet filtering.",
        "admin1_region": "Optional first-level administrative region code for SpeciesNet filtering.",
        "image_path": "Input image path.",
        "output_json": "Optional output path for raw SpeciesNet predictions JSON.",
        "write_sidecar": "Whether to write a sidecar JSON next to the image.",
        "occasion_days": "Number of days per repeated-survey occasion.",
        "min_active_occasions": "Minimum valid survey occasions required for a camera to enter the model.",
        "bootstrap_iters": "Number of bootstrap iterations.",
        "smooth_sigma": "Smoothing parameter for activity curves.",
        "permutation_iters": "Number of randomization-test iterations.",
        "random_seed": "Integer seed used for bootstrap and permutation reproducibility.",
        "allow_inferred_effort": "Allow an explicitly exploratory occupancy fit when deployment intervals were inferred from detections; defaults to False.",
        "records_a": "First record list.",
        "records_b": "Second record list.",
        "show_ci": "Whether to show bootstrap confidence bands.",
        "annotate_peaks": "Peak annotation mode.",
        "subtitle_level": "Subtitle detail level.",
        "ymax": "Optional manual y-axis maximum.",
        "panel_label": "Optional panel label.",
        "compare_species_list": "Optional comparison species list, or a comma-separated string.",
        "reference_species": "Reference species name.",
        "ecological_role_mapping": "Explicit species-role object supplied by the user or dataset configuration, such as predators and prey lists; roles are never inferred from fixed species names.",
        "compare_occasion_days": "Optional occasion-window list for sensitivity comparison.",
        "normalize": "Whether to normalize values by species.",
        "permutations": "Number of random permutations.",
        "step_count": "Number of sampled curve steps.",
        "top_cameras": "Number of top cameras to show.",
        "metric": "Metric to use, such as jaccard or shared_camera_count.",
        "max_time_gap_hours": "Maximum allowed time gap in hours.",
        "max_distance_km": "Maximum allowed distance in kilometers.",
        "steps": "Optional precomputed movement-step list.",
        "use_ratio": "Whether to plot proportions instead of counts.",
        "bins": "Number of histogram bins.",
        "annotate_top_k": "Number of hotspot cameras to annotate.",
        "privacy_grid_degrees": (
            "Coordinate-grid width in decimal degrees for a privacy-safe public "
            "map; use a positive value such as 0.1 rather than exact camera locations."
        ),
        "max_cameras": "Maximum number of cameras to show.",
        "panels": "Structured panel list for a forest or coefficient plot.",
        "x_label": "X-axis label.",
        "y_label": "Y-axis label.",
        "zero_line": "Reference zero-line position.",
        "panel_labels": "Optional panel-label list.",
        "caption": "Optional figure caption.",
        "panel_items": "Structured panel items for a density grid.",
        "ncols": "Number of columns.",
        "reference_line": "Optional global reference line.",
        "prey_species": "Optional prey-guild species list for predator-prey temporal panels.",
        "min_events": "Minimum independent events required before reporting a pairwise overlap estimate.",
    }

    _TOOL_DESCRIPTIONS = {
        "analyze_spatial_distribution": "Run global or local spatial analysis. Use center_camera_id and radius_km for local hotspot questions.",
        "build_hotspot_rank": "Rank hotspot cameras for a species across the full camera network.",
        "plot_species_distribution_static": (
            "Plot a single-species independent-event distribution map. Public "
            "outputs aggregate coordinates to privacy_grid_degrees and omit camera identifiers."
        ),
        "plot_hotspot_rank_barh": (
            "Plot cameras ranked by independent-event count for one focal species; "
            "this is a detection-hotspot summary, not occupancy or population density."
        ),
        "plot_species_distribution_map": (
            "Plot an access-controlled interactive map with camera locations. "
            "Do not select this exact-location tool when the user requests a privacy-safe public map."
        ),
        "build_local_hotspot_rank": "Rank hotspot cameras within a radius around a center camera.",
        "fit_single_season_occupancy_model": "Fit an intercept-only single-season occupancy model; inferential use requires independently documented camera-operation effort.",
        "build_detection_history_matrix": "Build a camera-by-occasion detection-history matrix for occupancy workflows.",
        "analyze_temporal_activity_pattern": "Summarize temporal activity across months, seasons, weekdays, hours, or day-night periods.",
        "check_timestamp_quality": "Check whether timestamp records are reliable enough for hourly temporal analysis.",
        "estimate_activity_overlap": (
            "Estimate one two-species 24-hour activity-overlap result from "
            "independent events and persist a versioned, tamper-evident result "
            "bundle for later figures and Agent runs."
        ),
        "estimate_pairwise_activity_overlap": (
            "Preferred analysis when one focal species is compared with one or "
            "more species. When the question says 'other species' without naming "
            "taxa, omit species_list so the tool derives eligible non-focal "
            "comparisons from the active dataset inventory rather than inserting "
            "or guessing a fixed species list. Estimate all Delta "
            "values once and persist the same versioned result bundles that "
            "later pairwise figures must reuse. Before this tool, validate "
            "timestamps, construct focal-species independent events, build the "
            "focal hourly distribution, and summarize community composition to "
            "document the comparison-species context."
        ),
        "analyze_camera_data_quality": "Summarize camera-level data-quality issues with a heuristic quality score.",
        "detect_duplicate_like_records": "Find duplicate-like records grouped by camera, timestamp, and species.",
        "analyze_animal_movement_pattern": "Summarize potential local movement steps, distances, directions, and transition corridors.",
        "infer_transition_corridors": "Identify common potential transition edges between temporally adjacent camera observations.",
        "prepare_camera_trap_dataset": (
            "Prepare raw images/videos, a generic observation table, or an existing "
            "CamtrapDP package as the active CamtrapDP backend. For raw videos the "
            "tool performs bounded frame sampling and model inference before "
            "creating structured records and CamtrapDP. Format dispatch is "
            "deterministic and contains no ecological-question keyword routing."
        ),
        "run_detection_pipeline": (
            "Run model inference on raw images or sampled video frames, or ingest "
            "generic prediction sidecars, and create traceable structured records. "
            "Use prepare_camera_trap_dataset when CamtrapDP conversion and same-run "
            "downstream analysis are also required."
        ),
        "evaluate_perception_benchmark": (
            "Run a generic batch perception benchmark from an annotation table. "
            "The tool can execute SpeciesNet on linked images or sampled video "
            "frames, preserve prediction provenance, and report camera-clustered "
            "accuracy, macro precision/recall/F1, per-class errors, confusion, "
            "confidence calibration, and manuscript figures. It never derives "
            "ground truth from folder names or query keywords."
        ),
        "predict_image_with_speciesnet": "Run SpeciesNet image prediction and return structured detections and classifications.",
        "load_camtrap_dp_dataset": "Import a CamtrapDP package and switch the active JSON backend for follow-up analysis.",
        "import_camtrap_dp": "Import a CamtrapDP package into the project JSON record format.",
        "convert_table_to_camtrap_dp": "Convert a generic CSV/TSV/Excel camera-trap table into CamtrapDP and activate it as the current tabular backend.",
        "convert_species_csv_directory_to_camtrap_dp": "Convert a directory of per-species CSV files into CamtrapDP with independent-event caches.",
        "run_ecology_figure_suite": (
            "Generate the complete dataset-wide 11-figure ecology gallery with "
            "caption interpretation boundaries. Species are drawn from the active "
            "CamtrapDP inventory; predator/prey figures are added only when an "
            "explicit ecological_role_mapping is supplied. Use this broad suite only when "
            "the user requests a full figure collection; use focused plot tools "
            "for a concise question-specific report."
        ),
        "export_camtrap_dp": "Export the active internal records as a CamtrapDP package.",
        "export_media_by_species": "Export local image or video media into species-organized folders.",
        "filter_records": "Filter standardized records by species, camera, date, hour, radius, or confidence.",
        "browse_records": "Browse standardized sample records with lightweight media-page style filters.",
        "evidence_lookup": "Return raw-media evidence samples that support an analysis result.",
        "summarize_species_inventory": "Summarize species inventory and record counts for the active data source.",
        "summarize_community_composition": "Summarize multi-species community composition, diversity, and relative abundance.",
        "analyze_community_ecology": "Combine community composition, richness hotspots, co-occurrence, activity guilds, and relationship summaries.",
        "infer_complete_community": (
            "Run dataset-wide community inference across every eligible species: "
            "observed and Chao2 richness, Hill diversity, camera alpha diversity, "
            "Jaccard and Bray-Curtis beta diversity, species accumulation, "
            "conditional shared-camera association tests with FDR correction, "
            "and multi-species detection-corrected occupancy when documented "
            "deployment histories support repeated-survey inference."
        ),
        "build_camera_species_richness": "Rank cameras by multi-species richness and independent-event volume.",
        "build_species_cooccurrence_matrix": "Build a species-by-species camera-level co-occurrence matrix.",
        "analyze_species_pair_relationship": "Analyze descriptive spatial, temporal, and same-camera proximity relationships for two species.",
        "summarize_species_relationship_network": "Summarize a multi-species relationship network from co-occurrence and optional temporal overlap.",
        "plot_effort_corrected_hotspot_map": "Generate an interactive basemap of species hotspots ranked by RAI rather than raw counts.",
        "plot_predator_prey_encounter_index_map": "Generate an interactive predator-prey map from the active inventory and an explicit ecological_role_mapping, using normalized prey RAI times normalized predator RAI.",
        "plot_predator_prey_temporal_panel": "Generate a temporal comparison for an explicit focal species and explicitly configured prey species; biological roles are never inferred from species names.",
        "plot_multi_species_ecology_map": "Generate an interactive multi-layer species RAI map from the active inventory; optional explicit ecological roles add role-level camera metrics.",
        "plot_community_spatial_association_panel": (
            "Generate a privacy-masked composite and return the underlying "
            "independent-event composition, focal-richness hotspot ranks, and "
            "camera-level observed/expected association summaries."
        ),
        "plot_complete_community_inference": (
            "Generate a publication-ready dataset-wide community figure from "
            "the complete community-inference tool, with composition, species "
            "accumulation, camera detection or detection-corrected occupancy, "
            "and conditional association effect sizes."
        ),
        "plot_hourly_activity_curve": (
            "Render the focal-species diel activity figure after the hourly "
            "activity distribution has been analysed."
        ),
        "plot_activity_overlap_panel": (
            "Render a single overlap panel after an overlap analysis tool has "
            "created the numerical result bundle. For several comparison "
            "species, prefer the aggregate pairwise analysis and panel tools."
        ),
        "plot_pairwise_temporal_overlap": (
            "Render a generic pairwise diel activity-overlap figure for two "
            "explicitly supplied species using the versioned numerical result "
            "bundle created by an earlier overlap analysis."
        ),
        "plot_pairwise_activity_overlap_panels": (
            "Preferred figure tool when one focal species is compared with one "
            "or more selected species. For unnamed 'other species', omit "
            "comparison_species and use the same active-dataset selection rule "
            "as the analysis tool. Run "
            "estimate_pairwise_activity_overlap "
            "first with identical species and estimator arguments; this tool "
            "reuses those exact result bundles and reports per-figure status."
        ),
    }

    _TOOL_DEPENDENCIES = {
        "build_hourly_activity_distribution": {
            "all_of": [
                "check_timestamp_quality",
                "build_independent_events",
            ],
        },
        "estimate_activity_overlap": {
            "all_of": [
                "check_timestamp_quality",
                "build_independent_events",
                "build_hourly_activity_distribution",
            ],
        },
        "estimate_pairwise_activity_overlap": {
            "all_of": [
                "check_timestamp_quality",
                "build_independent_events",
                "build_hourly_activity_distribution",
                "summarize_community_composition",
            ],
        },
        "plot_hourly_activity_curve": {
            "all_of": ["build_hourly_activity_distribution"],
        },
        "plot_activity_overlap_panel": {
            "any_of": [
                "estimate_activity_overlap",
                "estimate_pairwise_activity_overlap",
            ],
        },
        "plot_pairwise_temporal_overlap": {
            "any_of": [
                "estimate_activity_overlap",
                "estimate_pairwise_activity_overlap",
            ],
        },
        "plot_pairwise_activity_overlap_panels": {
            "all_of": ["estimate_pairwise_activity_overlap"],
        },
    }

    def _tool_description(self, tool_name: str) -> str:
        if tool_name in self._TOOL_DESCRIPTIONS:
            return self._TOOL_DESCRIPTIONS[tool_name]
        readable = tool_name.replace("_", " ")
        return f"Use this tool to {readable}."

    def _tool_params(self, tool_name: str, names):
        return {name: self._PARAM_DESCRIPTIONS.get(name, f"Parameter `{name}`.") for name in names}

    def get_tool_dependencies(self, tool_name: str):
        dependencies = self._TOOL_DEPENDENCIES.get(tool_name, {})
        return {
            key: list(value)
            for key, value in dependencies.items()
        }

    def __init__(self):
        self.viz = VisualizationTools(output_dir=str(OUTPUTS_DIR))
        # VisualizationTools inherits the complete ecological toolset. A single
        # shared instance is essential: if an LLM plan first prepares/activates a
        # new dataset and then analyses or plots it, every later task must observe
        # that newly active backend within the same Agent run.
        self.tools = self.viz

        self.registry = {
            # =========================================================
            # Base tools
            # =========================================================
            "get_all_records": {
                "category": "base",
                "planner_exposed": False,
                "description": self._tool_description("get_all_records"),
                "params": self._tool_params("get_all_records", []),
                "func": self.tools.get_all_records,
            },
            "query_by_species": {
                "category": "base",
                "planner_exposed": False,
                "description": self._tool_description("query_by_species"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.query_by_species,
            },
            "query_by_camera": {
                "category": "base",
                "planner_exposed": False,
                "description": self._tool_description("query_by_camera"),
                "params": self._tool_params("params", ['camera_id']),
                "func": self.tools.query_by_camera,
            },
            "set_json_data_dir": {
                "category": "base",
                "planner_exposed": False,
                "description": self._tool_description("set_json_data_dir"),
                "params": self._tool_params("params", ['json_data_dir']),
                "func": self.tools.set_json_data_dir,
            },
            "set_camtrap_data_dir": {
                "category": "base",
                "planner_exposed": True,
                "description": "Activate a CamtrapDP package as the tabular backend without creating per-record JSON.",
                "params": self._tool_params("params", ['datapackage_path']),
                "func": self.tools.set_camtrap_data_dir,
            },
            "use_collection_backend": {
                "category": "base",
                "planner_exposed": False,
                "description": self._tool_description("use_collection_backend"),
                "params": self._tool_params("use_collection_backend", []),
                "func": self.tools.use_collection_backend,
            },
            "safe_parse_timestamp": {
                    "category": "base",
                    "planner_exposed": False,
                    "description": self._tool_description("safe_parse_timestamp"),
                    "params": self._tool_params("params", ['timestamp_str']),
                "func": self.tools.safe_parse_timestamp,
            },
            "get_species_records_with_parsed_time": {
                "category": "base",
                "planner_exposed": False,
                "description": self._tool_description("get_species_records_with_parsed_time"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.get_species_records_with_parsed_time,
            },
            "get_valid_species_records": {
                "category": "base",
                "planner_exposed": False,
                "description": self._tool_description("get_valid_species_records"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.get_valid_species_records,
            },
            "build_independent_events": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("build_independent_events"),
                "params": self._tool_params("params", ['species', 'by_camera', 'min_gap_minutes']),
                "func": self.tools.build_independent_events,
            },
            "calculate_distance_km": {
                "category": "base",
                "planner_exposed": False,
                "description": self._tool_description("calculate_distance_km"),
                "params": self._tool_params("params", ['lat1', 'lon1', 'lat2', 'lon2']),
                "func": self.tools.calculate_distance_km,
            },
            "get_camera_locations": {
                "category": "base",
                "planner_exposed": False,
                "description": self._tool_description("get_camera_locations"),
                "params": self._tool_params("get_camera_locations", []),
                "func": self.tools.get_camera_locations,
            },
            "get_cameras_in_radius": {
                "category": "base",
                "planner_exposed": False,
                "description": self._tool_description("get_cameras_in_radius"),
                "params": self._tool_params("params", ['center_camera_id', 'radius_km']),
                "func": self.tools.get_cameras_in_radius,
            },
            "import_camtrap_dp": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("import_camtrap_dp"),
                "params": self._tool_params("params", ['datapackage_path', 'target_dir', 'prefer_media_level', 'include_blank', 'include_event_fallback']),
                "func": self.tools.import_camtrap_dp,
            },
            "load_camtrap_dp_dataset": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("load_camtrap_dp_dataset"),
                "params": self._tool_params("params", ['datapackage_path', 'target_dir', 'prefer_media_level', 'include_blank', 'include_event_fallback', 'activate_json_backend']),
                "func": self.tools.load_camtrap_dp_dataset,
            },
            "prepare_camera_trap_dataset": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("prepare_camera_trap_dataset"),
                "params": self._tool_params("params", [
                    'input_path', 'input_type', 'output_dir', 'field_mapping',
                    'metadata_table', 'detector', 'classifier', 'run_inference',
                    'country', 'admin1_region', 'min_confidence',
                    'video_frame_interval_seconds', 'max_video_frames',
                    'run_timestamp_ocr', 'timestamp_ocr_max_frames',
                    'min_gap_minutes', 'package_name', 'package_title',
                    'source_timezone', 'activate_backend', 'persist_active',
                    'allow_metadata_only',
                ]),
                "func": self.tools.prepare_camera_trap_dataset,
            },
            "convert_table_to_camtrap_dp": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("convert_table_to_camtrap_dp"),
                "params": self._tool_params("params", ['table_path', 'field_mapping', 'output_dir', 'sheet_name', 'package_name', 'package_title', 'min_gap_minutes', 'activate_backend', 'persist_active', 'source_timezone']),
                "func": self.tools.convert_table_to_camtrap_dp,
            },
            "convert_species_csv_directory_to_camtrap_dp": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("convert_species_csv_directory_to_camtrap_dp"),
                "params": self._tool_params("params", ['input_dir', 'output_dir', 'species_config', 'column_map', 'start_date', 'end_date', 'lon_min', 'lon_max', 'lat_min', 'lat_max', 'min_gap_minutes', 'package_name', 'package_title', 'activate_backend']),
                "func": self.tools.convert_species_csv_directory_to_camtrap_dp,
            },
            "run_ecology_figure_suite": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("run_ecology_figure_suite"),
                "params": self._tool_params("params", ['package_dir', 'output_dir', 'species_order', 'ecological_role_mapping', 'include_core', 'include_exploratory', 'write_dashboard']),
                "func": self.tools.run_ecology_figure_suite,
            },
            "summarize_species_inventory": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("summarize_species_inventory"),
                "params": self._tool_params("params", ['top_k']),
                "func": self.tools.summarize_species_inventory,
            },
            "summarize_community_composition": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("summarize_community_composition"),
                "params": self._tool_params("params", ['top_k', 'include_unknown', 'min_gap_minutes', 'abundance_metric', 'min_effort_days_for_rai', 'min_observed_span_days_for_rai', 'min_independent_events_for_rai']),
                "func": self.tools.summarize_community_composition,
            },
            "analyze_community_ecology": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("analyze_community_ecology"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'min_camera_records', 'min_records_per_species', 'min_gap_minutes', 'min_rai_per_100_camera_days', 'include_relationships', 'max_pairs']),
                "func": self.tools.analyze_community_ecology,
            },
            "infer_complete_community": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description(
                    "infer_complete_community"
                ),
                "params": self._tool_params("params", [
                    "species_list", "top_k", "min_independent_events",
                    "min_detected_cameras", "min_gap_minutes",
                    "accumulation_permutations", "bootstrap_iters",
                    "association_fdr_alpha", "max_beta_pairs",
                    "fit_detection_corrected_occupancy",
                    "occupancy_species_limit", "occasion_days",
                    "min_active_occasions", "occupancy_bootstrap_iters",
                    "random_seed",
                ]),
                "func": self.tools.infer_complete_community,
            },
            "export_camtrap_dp": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("export_camtrap_dp"),
                "params": self._tool_params("params", ['output_dir', 'species', 'include_event_observations', 'min_gap_minutes', 'package_name', 'package_title', 'package_version', 'source_timezone']),
                "func": self.tools.export_camtrap_dp,
            },
            "export_media_by_species": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("export_media_by_species"),
                "params": self._tool_params("params", ['species', 'output_dir', 'use_symlinks', 'min_confidence', 'limit']),
                "func": self.tools.export_media_by_species,
            },
            "filter_records": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("filter_records"),
                "params": self._tool_params("params", ['species', 'camera_id', 'start_date', 'end_date', 'start_hour', 'end_hour', 'center_camera_id', 'radius_km', 'min_confidence', 'limit', 'sort_by']),
                "func": self.tools.filter_records,
            },
            "browse_records": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("browse_records"),
                "params": self._tool_params("params", ['species', 'camera_id', 'start_date', 'end_date', 'start_hour', 'end_hour', 'center_camera_id', 'radius_km', 'min_confidence', 'limit', 'sort_by']),
                "func": self.tools.browse_records,
            },
            "evidence_lookup": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("evidence_lookup"),
                "params": self._tool_params("params", ['species', 'camera_id', 'start_date', 'end_date', 'start_hour', 'end_hour', 'center_camera_id', 'radius_km', 'min_confidence', 'limit', 'sort_by']),
                "func": self.tools.evidence_lookup,
            },
            "run_detection_pipeline": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("run_detection_pipeline"),
                "params": self._tool_params("params", [
                    'media_dir', 'detector', 'classifier', 'output_dir',
                    'default_species', 'min_confidence',
                    'activate_json_backend', 'run_inference', 'country',
                    'admin1_region', 'metadata_table', 'field_mapping',
                    'video_frame_interval_seconds', 'max_video_frames',
                    'run_timestamp_ocr', 'timestamp_ocr_max_frames',
                    'allow_metadata_only',
                ]),
                "func": self.tools.run_detection_pipeline,
            },
            "evaluate_perception_benchmark": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description(
                    "evaluate_perception_benchmark"
                ),
                "params": self._tool_params("params", [
                    "annotation_path", "output_dir", "media_root",
                    "media_path_column", "true_label_column",
                    "predicted_label_column", "confidence_column",
                    "group_column", "sample_id_column", "run_inference",
                    "country", "admin1_region", "model",
                    "video_frame_interval_seconds", "max_video_frames",
                    "max_samples", "label_aliases",
                    "confidence_threshold", "calibration_bins",
                    "bootstrap_iters", "random_seed",
                    "camera_disjoint_verified", "human_review_status",
                    "split_note", "development_camera_ids_path",
                    "annotation_manifest_path", "minimum_class_support",
                    "resume",
                ]),
                "func": self.tools.evaluate_perception_benchmark,
            },
            "predict_image_with_speciesnet": {
                "category": "base",
                "planner_exposed": True,
                "description": self._tool_description("predict_image_with_speciesnet"),
                "params": self._tool_params("params", ['image_path', 'country', 'admin1_region', 'output_json', 'write_sidecar']),
                "func": self.tools.predict_image_with_speciesnet,
            },

            # =========================================================
            # Spatial tools
            # =========================================================
            "build_detection_map": {
                "category": "spatial",
                "planner_exposed": True,
                "description": self._tool_description("build_detection_map"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.build_detection_map,
            },
            "build_hotspot_rank": {
                "category": "spatial",
                "planner_exposed": True,
                "description": self._tool_description("build_hotspot_rank"),
                "params": self._tool_params("params", ['species', 'top_k']),
                "func": self.tools.build_hotspot_rank,
            },
            "build_occupancy_summary": {
                "category": "spatial",
                "planner_exposed": True,
                "description": self._tool_description("build_occupancy_summary"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.build_occupancy_summary,
            },
            "build_detection_history_matrix": {
                "category": "spatial",
                "planner_exposed": True,
                "description": self._tool_description("build_detection_history_matrix"),
                "params": self._tool_params("params", ['species', 'occasion_days', 'min_gap_minutes', 'min_active_occasions']),
                "func": self.tools.build_detection_history_matrix,
            },
            "fit_single_season_occupancy_model": {
                "category": "spatial",
                "planner_exposed": True,
                "description": self._tool_description("fit_single_season_occupancy_model"),
                "params": self._tool_params("params", ['species', 'occasion_days', 'min_gap_minutes', 'min_active_occasions', 'bootstrap_iters']),
                "func": self.tools.fit_single_season_occupancy_model,
            },
            "build_local_hotspot_rank": {
                "category": "spatial",
                "planner_exposed": True,
                "description": self._tool_description("build_local_hotspot_rank"),
                "params": self._tool_params("params", ['species', 'center_camera_id', 'radius_km', 'top_k']),
                "func": self.tools.build_local_hotspot_rank,
            },
            "summarize_species_by_camera": {
                "category": "spatial",
                "planner_exposed": True,
                "description": self._tool_description("summarize_species_by_camera"),
                "params": self._tool_params("params", ['camera_id', 'top_k']),
                "func": self.tools.summarize_species_by_camera,
            },
            "build_camera_species_richness": {
                "category": "spatial",
                "planner_exposed": True,
                "description": self._tool_description("build_camera_species_richness"),
                "params": self._tool_params("params", ['top_k', 'min_records_per_species', 'min_gap_minutes', 'min_rai_per_100_camera_days']),
                "func": self.tools.build_camera_species_richness,
            },
            "build_species_cooccurrence_matrix": {
                "category": "spatial",
                "planner_exposed": True,
                "description": self._tool_description("build_species_cooccurrence_matrix"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'min_camera_records']),
                "func": self.tools.build_species_cooccurrence_matrix,
            },
            "analyze_species_pair_relationship": {
                "category": "spatial",
                "planner_exposed": True,
                "description": self._tool_description("analyze_species_pair_relationship"),
                "params": self._tool_params("params", ['species_a', 'species_b', 'min_camera_records', 'min_gap_minutes', 'temporal_window_hours', 'smooth_sigma', 'bootstrap_iters', 'permutation_iters']),
                "func": self.tools.analyze_species_pair_relationship,
            },
            "summarize_species_relationship_network": {
                "category": "spatial",
                "planner_exposed": True,
                "description": self._tool_description("summarize_species_relationship_network"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'min_camera_records', 'min_shared_cameras', 'min_gap_minutes', 'temporal_window_hours', 'include_temporal_overlap', 'max_pairs']),
                "func": self.tools.summarize_species_relationship_network,
            },
            "analyze_spatial_distribution": {
                "category": "spatial",
                "planner_exposed": True,
                "description": self._tool_description("analyze_spatial_distribution"),
                "params": self._tool_params("params", ['species', 'center_camera_id', 'radius_km', 'top_k']),
                "func": self.tools.analyze_spatial_distribution,
            },

            # =========================================================
            # Temporal tools
            # =========================================================
            "check_timestamp_quality": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("check_timestamp_quality"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.check_timestamp_quality,
            },
            "build_hourly_activity_distribution": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("build_hourly_activity_distribution"),
                "params": self._tool_params("params", ['species', 'min_gap_minutes']),
                "func": self.tools.build_hourly_activity_distribution,
            },
            "detect_activity_peaks": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("detect_activity_peaks"),
                "params": self._tool_params("params", ['species', 'top_k', 'min_gap_minutes']),
                "func": self.tools.detect_activity_peaks,
            },
            "compute_hourly_relative_frequency": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("compute_hourly_relative_frequency"),
                "params": self._tool_params("params", ['species', 'min_gap_minutes']),
                "func": self.tools.compute_hourly_relative_frequency,
            },
            "compute_rai_by_hour": {
                "category": "temporal",
                "planner_exposed": False,
                "description": self._tool_description("compute_rai_by_hour"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.compute_rai_by_hour,
            },
            "compare_multi_species_hourly_activity": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("compare_multi_species_hourly_activity"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'min_gap_minutes']),
                "func": self.tools.compare_multi_species_hourly_activity,
            },
            "build_species_activity_guilds": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("build_species_activity_guilds"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'min_gap_minutes']),
                "func": self.tools.build_species_activity_guilds,
            },
            "circular_time_summary": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("circular_time_summary"),
                "params": self._tool_params("params", ['species', 'min_gap_minutes']),
                "func": self.tools.circular_time_summary,
            },
            "calculate_time_overlap_between_two_sets": {
                "category": "temporal",
                "planner_exposed": False,
                "description": self._tool_description("calculate_time_overlap_between_two_sets"),
                "params": self._tool_params("params", ['records_a', 'records_b']),
                "func": self.tools.calculate_time_overlap_between_two_sets,
            },
            "compare_day_vs_night_activity": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("compare_day_vs_night_activity"),
                "params": self._tool_params("params", ['species', 'min_gap_minutes']),
                "func": self.tools.compare_day_vs_night_activity,
            },
            "estimate_activity_overlap": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("estimate_activity_overlap"),
                "params": self._tool_params("params", ['species_a', 'species_b', 'min_gap_minutes', 'smooth_sigma', 'bootstrap_iters', 'permutation_iters']),
                "func": self.tools.estimate_activity_overlap,
            },
            "estimate_pairwise_activity_overlap": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("estimate_pairwise_activity_overlap"),
                "params": self._tool_params("params", ['species_a', 'species_list', 'top_k', 'min_gap_minutes', 'smooth_sigma', 'bootstrap_iters', 'permutation_iters', 'min_events']),
                "func": self.tools.estimate_pairwise_activity_overlap,
            },
            "estimate_hourly_activity_curve_with_ci": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("estimate_hourly_activity_curve_with_ci"),
                "params": self._tool_params("params", ['species', 'min_gap_minutes', 'smooth_sigma', 'bootstrap_iters']),
                "func": self.tools.estimate_hourly_activity_curve_with_ci,
            },
            "build_hourly_activity_figure_data": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("build_hourly_activity_figure_data"),
                "params": self._tool_params("params", ['species', 'min_gap_minutes', 'smooth_sigma', 'bootstrap_iters']),
                "func": self.tools.build_hourly_activity_figure_data,
            },
            "build_monthly_activity_distribution": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("build_monthly_activity_distribution"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.build_monthly_activity_distribution,
            },
            "detect_monthly_activity_peaks": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("detect_monthly_activity_peaks"),
                "params": self._tool_params("params", ['species', 'top_k']),
                "func": self.tools.detect_monthly_activity_peaks,
            },
            "compare_seasonal_activity": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("compare_seasonal_activity"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.compare_seasonal_activity,
            },
            "build_weekday_activity_distribution": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("build_weekday_activity_distribution"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.build_weekday_activity_distribution,
            },
            "analyze_temporal_activity_pattern": {
                "category": "temporal",
                "planner_exposed": True,
                "description": self._tool_description("analyze_temporal_activity_pattern"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.analyze_temporal_activity_pattern,
            },

            # =========================================================
            # Quality tools
            # =========================================================
            "analyze_camera_data_quality": {
                "category": "quality",
                "planner_exposed": True,
                "description": self._tool_description("analyze_camera_data_quality"),
                "params": self._tool_params("params", ['species', 'top_k']),
                "func": self.tools.analyze_camera_data_quality,
            },
            "check_missing_fields": {
                "category": "quality",
                "planner_exposed": True,
                "description": self._tool_description("check_missing_fields"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.check_missing_fields,
            },
            "detect_invalid_timestamps": {
                "category": "quality",
                "planner_exposed": True,
                "description": self._tool_description("detect_invalid_timestamps"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.detect_invalid_timestamps,
            },
            "detect_label_noise": {
                "category": "quality",
                "planner_exposed": True,
                "description": self._tool_description("detect_label_noise"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.detect_label_noise,
            },
            "detect_duplicate_like_records": {
                "category": "quality",
                "planner_exposed": True,
                "description": self._tool_description("detect_duplicate_like_records"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.detect_duplicate_like_records,
            },
            "summarize_camera_record_volume": {
                "category": "quality",
                "planner_exposed": True,
                "description": self._tool_description("summarize_camera_record_volume"),
                "params": self._tool_params("params", ['species', 'top_k']),
                "func": self.tools.summarize_camera_record_volume,
            },
            "summarize_duplicate_burden_by_camera": {
                "category": "quality",
                "planner_exposed": True,
                "description": self._tool_description("summarize_duplicate_burden_by_camera"),
                "params": self._tool_params("params", ['species', 'top_k']),
                "func": self.tools.summarize_duplicate_burden_by_camera,
            },
            "analyze_data_quality": {
                "category": "quality",
                "planner_exposed": True,
                "description": self._tool_description("analyze_data_quality"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.analyze_data_quality,
            },

            # =========================================================
            # Movement tools
            # =========================================================
            "build_time_sorted_camera_sequence": {
                "category": "movement",
                "planner_exposed": False,
                "description": self._tool_description("build_time_sorted_camera_sequence"),
                "params": self._tool_params("params", ['species']),
                "func": self.tools.build_time_sorted_camera_sequence,
            },
            "analyze_movement_steps": {
                "category": "movement",
                "planner_exposed": True,
                "description": self._tool_description("analyze_movement_steps"),
                "params": self._tool_params("params", ['species', 'max_time_gap_hours', 'max_distance_km']),
                "func": self.tools.analyze_movement_steps,
            },
            "summarize_movement_distance": {
                "category": "movement",
                "planner_exposed": True,
                "description": self._tool_description("summarize_movement_distance"),
                "params": self._tool_params("params", ['species', 'max_time_gap_hours', 'max_distance_km', 'steps']),
                "func": self.tools.summarize_movement_distance,
            },
            "summarize_movement_direction": {
                "category": "movement",
                "planner_exposed": True,
                "description": self._tool_description("summarize_movement_direction"),
                "params": self._tool_params("params", ['species', 'max_time_gap_hours', 'max_distance_km', 'steps']),
                "func": self.tools.summarize_movement_direction,
            },
            "infer_transition_corridors": {
                "category": "movement",
                "planner_exposed": True,
                "description": self._tool_description("infer_transition_corridors"),
                "params": self._tool_params("params", ['species', 'max_time_gap_hours', 'max_distance_km', 'top_k', 'steps']),
                "func": self.tools.infer_transition_corridors,
            },
            "infer_longest_transition_steps": {
                "category": "movement",
                "planner_exposed": True,
                "description": self._tool_description("infer_longest_transition_steps"),
                "params": self._tool_params("params", ['species', 'max_time_gap_hours', 'max_distance_km', 'top_k', 'steps']),
                "func": self.tools.infer_longest_transition_steps,
            },
            "analyze_animal_movement_pattern": {
                "category": "movement",
                "planner_exposed": True,
                "description": self._tool_description("analyze_animal_movement_pattern"),
                "params": self._tool_params("params", ['species', 'max_time_gap_hours', 'max_distance_km']),
                "func": self.tools.analyze_animal_movement_pattern,
            },
            # =========================================================
            # Visualization tools
            # =========================================================
            "plot_species_distribution_map": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_species_distribution_map"),
                "params": self._tool_params("params", ['species', 'annotate_top_k', 'save_path']),
                "func": self.viz.plot_species_distribution_map,
            },
            "plot_local_hotspot_map": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_local_hotspot_map"),
                "params": self._tool_params("params", ['species', 'center_camera_id', 'radius_km', 'annotate_top_k', 'save_path']),
                "func": self.viz.plot_local_hotspot_map,
            },
            "plot_effort_corrected_hotspot_map": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_effort_corrected_hotspot_map"),
                "params": self._tool_params("params", ['species', 'top_k', 'save_path']),
                "func": self.viz.plot_effort_corrected_hotspot_map,
            },
            "plot_predator_prey_encounter_index_map": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_predator_prey_encounter_index_map"),
                "params": self._tool_params("params", ['species_list', 'ecological_role_mapping', 'save_path', 'max_cameras']),
                "func": self.viz.plot_predator_prey_encounter_index_map,
            },
            "plot_multi_species_ecology_map": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_multi_species_ecology_map"),
                "params": self._tool_params("params", ['species_list', 'ecological_role_mapping', 'save_path', 'max_cameras_per_species']),
                "func": self.viz.plot_multi_species_ecology_map,
            },
            "plot_monthly_activity_bar": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_monthly_activity_bar"),
                "params": self._tool_params("params", ['species', 'save_path']),
                "func": self.viz.plot_monthly_activity_bar,
            },
            "plot_hourly_activity_curve": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_hourly_activity_curve"),
                "params": self._tool_params("params", ['species', 'min_gap_minutes', 'smooth_sigma', 'show_ci', 'bootstrap_iters', 'annotate_peaks', 'subtitle_level', 'ymax', 'save_path']),
                "func": self.viz.plot_hourly_activity_curve,
            },
            "plot_activity_overlap_panel": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_activity_overlap_panel"),
                "params": self._tool_params("params", ['species_a', 'species_b', 'min_gap_minutes', 'smooth_sigma', 'bootstrap_iters', 'panel_label', 'save_path']),
                "func": self.viz.plot_activity_overlap_panel,
            },
            "plot_pairwise_activity_overlap_panels": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_pairwise_activity_overlap_panels"),
                "params": self._tool_params("params", ['species_a', 'comparison_species', 'top_k', 'min_gap_minutes', 'smooth_sigma', 'bootstrap_iters']),
                "func": self.viz.plot_pairwise_activity_overlap_panels,
            },
            "plot_pairwise_temporal_overlap": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_pairwise_temporal_overlap"),
                "params": self._tool_params("params", ['focal_species', 'comparison_species', 'min_gap_minutes', 'smooth_sigma', 'bootstrap_iters', 'save_path']),
                "func": self.viz.plot_pairwise_temporal_overlap,
            },
            "plot_activity_overlap_triptych": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_activity_overlap_triptych"),
                "params": self._tool_params("params", ['reference_species', 'compare_species_list', 'top_k', 'min_gap_minutes', 'smooth_sigma', 'bootstrap_iters', 'save_path']),
                "func": self.viz.plot_activity_overlap_triptych,
            },
            "plot_multi_species_occupancy_forest": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_multi_species_occupancy_forest"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'occasion_days', 'min_gap_minutes', 'min_active_occasions', 'bootstrap_iters', 'save_path']),
                "func": self.viz.plot_multi_species_occupancy_forest,
            },
            "plot_species_inventory_bar": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_species_inventory_bar"),
                "params": self._tool_params("params", ['top_k', 'include_unknown', 'save_path']),
                "func": self.viz.plot_species_inventory_bar,
            },
            "plot_camera_species_richness_bar": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_camera_species_richness_bar"),
                "params": self._tool_params("params", ['top_k', 'min_records_per_species', 'min_gap_minutes', 'min_rai_per_100_camera_days', 'save_path']),
                "func": self.viz.plot_camera_species_richness_bar,
            },
            "plot_species_cooccurrence_heatmap": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_species_cooccurrence_heatmap"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'min_camera_records', 'min_gap_minutes', 'metric', 'save_path']),
                "func": self.viz.plot_species_cooccurrence_heatmap,
            },
            "plot_species_relationship_network": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_species_relationship_network"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'min_camera_records', 'min_shared_cameras', 'min_gap_minutes', 'temporal_window_hours', 'include_temporal_overlap', 'max_pairs', 'save_path']),
                "func": self.viz.plot_species_relationship_network,
            },
            "plot_community_overview_panel": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_community_overview_panel"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'min_camera_records', 'min_records_per_species', 'min_gap_minutes', 'min_rai_per_100_camera_days', 'save_path']),
                "func": self.viz.plot_community_overview_panel,
            },
            "plot_species_pair_relationship_panel": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_species_pair_relationship_panel"),
                "params": self._tool_params("params", ['species_a', 'species_b', 'min_camera_records', 'min_gap_minutes', 'temporal_window_hours', 'smooth_sigma', 'bootstrap_iters', 'permutation_iters', 'save_path']),
                "func": self.viz.plot_species_pair_relationship_panel,
            },
            "plot_multi_species_hourly_activity": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_multi_species_hourly_activity"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'min_gap_minutes', 'save_path']),
                "func": self.viz.plot_multi_species_hourly_activity,
            },
            "plot_species_rank_abundance_curve": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_species_rank_abundance_curve"),
                "params": self._tool_params("params", ['top_k', 'include_unknown', 'save_path']),
                "func": self.viz.plot_species_rank_abundance_curve,
            },
            "plot_multi_species_monthly_phenology": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_multi_species_monthly_phenology"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'normalize', 'save_path']),
                "func": self.viz.plot_multi_species_monthly_phenology,
            },
            "plot_camera_alpha_diversity_panel": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_camera_alpha_diversity_panel"),
                "params": self._tool_params("params", ['top_k', 'min_records_per_species', 'min_gap_minutes', 'min_rai_per_100_camera_days', 'save_path']),
                "func": self.viz.plot_camera_alpha_diversity_panel,
            },
            "plot_species_accumulation_curve": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_species_accumulation_curve"),
                "params": self._tool_params("params", ['permutations', 'step_count', 'include_unknown', 'save_path']),
                "func": self.viz.plot_species_accumulation_curve,
            },
            "plot_species_rai_heatmap": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_species_rai_heatmap"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'top_cameras', 'min_gap_minutes', 'save_path']),
                "func": self.viz.plot_species_rai_heatmap,
            },
            "plot_temporal_overlap_matrix": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_temporal_overlap_matrix"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'min_gap_minutes', 'save_path']),
                "func": self.viz.plot_temporal_overlap_matrix,
            },
            "plot_community_spatial_association_panel": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_community_spatial_association_panel"),
                "params": self._tool_params("params", ['species_list', 'top_k', 'save_path', 'privacy_grid_degrees']),
                "func": self.viz.plot_community_spatial_association_panel,
            },
            "plot_complete_community_inference": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description(
                    "plot_complete_community_inference"
                ),
                "params": self._tool_params("params", [
                    "species_list", "top_k", "display_top_species",
                    "min_independent_events", "min_detected_cameras",
                    "min_gap_minutes", "accumulation_permutations",
                    "bootstrap_iters", "association_fdr_alpha",
                    "fit_detection_corrected_occupancy",
                    "occupancy_species_limit", "occasion_days",
                    "min_active_occasions", "occupancy_bootstrap_iters",
                    "random_seed", "save_path",
                ]),
                "func": self.viz.plot_complete_community_inference,
            },
            "plot_predator_prey_temporal_panel": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_predator_prey_temporal_panel"),
                "params": self._tool_params("params", ['focal_species', 'species_list', 'prey_species', 'ecological_role_mapping', 'save_path']),
                "func": self.viz.plot_predator_prey_temporal_panel,
            },
            "plot_tiger_spatial_transition_panel": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_tiger_spatial_transition_panel"),
                "params": self._tool_params("params", ['species', 'top_k', 'max_time_gap_hours', 'max_distance_km', 'save_path', 'steps']),
                "func": self.viz.plot_tiger_spatial_transition_panel,
            },
            "plot_transition_corridor_map": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_transition_corridor_map"),
                "params": self._tool_params("params", ['species', 'max_time_gap_hours', 'max_distance_km', 'top_k', 'save_path', 'steps']),
                "func": self.viz.plot_transition_corridor_map,
            },
            "plot_weekday_activity_bar": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_weekday_activity_bar"),
                "params": self._tool_params("params", ['species', 'save_path']),
                "func": self.viz.plot_weekday_activity_bar,
            },
            "plot_seasonal_activity_bar": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_seasonal_activity_bar"),
                "params": self._tool_params("params", ['species', 'use_ratio', 'save_path']),
                "func": self.viz.plot_seasonal_activity_bar,
            },
                        "plot_hotspot_rank_bar": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_hotspot_rank_bar"),
                "params": self._tool_params("params", ['species', 'top_k', 'save_path']),
                "func": self.viz.plot_hotspot_rank_bar,
            },

            "plot_movement_direction_bar": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_movement_direction_bar"),
                "params": self._tool_params("params", ['species', 'max_time_gap_hours', 'max_distance_km', 'steps', 'save_path']),
                "func": self.viz.plot_movement_direction_bar,
            },

            "plot_movement_distance_hist": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_movement_distance_hist"),
                "params": self._tool_params("params", ['species', 'max_time_gap_hours', 'max_distance_km', 'bins', 'steps', 'save_path']),
                "func": self.viz.plot_movement_distance_hist,
            },
            "plot_hotspot_rank_barh": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_hotspot_rank_barh"),
                "params": self._tool_params("params", ['species', 'top_k']),
                "func": self.viz.plot_hotspot_rank_barh,
            },

            "plot_camera_quality_problem_barh": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_camera_quality_problem_barh"),
                "params": self._tool_params("params", ['species', 'top_k']),
                "func": self.viz.plot_camera_quality_problem_barh,
            },

            "plot_species_distribution_static": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_species_distribution_static"),
                "params": self._tool_params(
                    "params",
                    ['species', 'annotate_top_k', 'privacy_grid_degrees'],
                ),
                "func": self.viz.plot_species_distribution_static,
            },

            "plot_local_hotspot_static": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_local_hotspot_static"),
                "params": self._tool_params("params", ['species', 'center_camera_id', 'radius_km', 'annotate_top_k']),
                "func": self.viz.plot_local_hotspot_static,
            },
            "plot_detection_history_heatmap": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_detection_history_heatmap"),
                "params": self._tool_params("params", ['species', 'occasion_days', 'min_gap_minutes', 'min_active_occasions', 'sort_by', 'max_cameras', 'panel_label', 'save_path']),
                "func": self.viz.plot_detection_history_heatmap,
            },
            "plot_occupancy_model_forest": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_occupancy_model_forest"),
                "params": self._tool_params("params", ['species', 'occasion_days', 'compare_occasion_days', 'min_gap_minutes', 'min_active_occasions', 'bootstrap_iters', 'save_path']),
                "func": self.viz.plot_occupancy_model_forest,
            },
            "plot_occupancy_bootstrap_density": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_occupancy_bootstrap_density"),
                "params": self._tool_params("params", ['species', 'occasion_days', 'compare_occasion_days', 'min_gap_minutes', 'min_active_occasions', 'bootstrap_iters', 'ncols', 'save_path']),
                "func": self.viz.plot_occupancy_bootstrap_density,
            },
            "plot_forest_panel": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_forest_panel"),
                "params": self._tool_params("params", ['panels', 'x_label', 'zero_line', 'panel_labels', 'caption', 'save_path']),
                "func": self.viz.plot_forest_panel,
            },
            "plot_density_panel_grid": {
                "category": "visualization",
                "planner_exposed": True,
                "description": self._tool_description("plot_density_panel_grid"),
                "params": self._tool_params("params", ['panel_items', 'ncols', 'x_label', 'y_label', 'reference_line', 'panel_labels', 'caption', 'save_path']),
                "func": self.viz.plot_density_panel_grid,
            },
        }
        self._apply_planner_visibility_policy()

    def _apply_planner_visibility_policy(self):
        """Expose only the paper-facing agent chain to the LLM planner."""
        for name, meta in self.registry.items():
            meta["planner_exposed"] = name in self._PLANNER_EXPOSED_TOOLS

    def _signature_contract(self, func):
        """Return a JSON-serializable callable contract for planner validation."""
        required = []
        defaults = {}
        annotations = {}
        for name, parameter in inspect.signature(func).parameters.items():
            if name == "self" or parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if parameter.default is inspect.Parameter.empty:
                required.append(name)
            else:
                defaults[name] = parameter.default
            if parameter.annotation is not inspect.Parameter.empty:
                annotations[name] = str(parameter.annotation).replace("typing.", "")
        return {
            "required_params": required,
            "parameter_defaults": defaults,
            "parameter_types": annotations,
        }


    def get_tool_descriptions(self, include_internal: bool = False):
        """
        Return the tool metadata intended for the LLM planner by default.
        If include_internal=True, include lower-level internal tools as well.
        """
        result = []
        for name, meta in self.registry.items():
            if (not include_internal) and (not meta.get("planner_exposed", False)):
                continue

            contract = self._signature_contract(meta["func"])
            result.append({
                "tool_name": name,
                "category": meta["category"],
                "description": meta.get("description") or self._tool_description(name),
                "params": meta["params"],
                "planner_exposed": meta.get("planner_exposed", False),
                "requires_prior_tools": self.get_tool_dependencies(name),
                **contract,
            })
        return result

    def has_tool(self, tool_name: str, *, planner_only: bool = False) -> bool:
        if tool_name not in self.registry:
            return False
        return not planner_only or bool(self.registry[tool_name].get("planner_exposed", False))

    def is_planner_exposed(self, tool_name: str) -> bool:
        return self.has_tool(tool_name, planner_only=True)

    def get_func(self, tool_name: str):
        return self.registry[tool_name]["func"]

    def get_registry(self):
        return self.registry

    def get_tools_by_category(self, category: str, include_internal: bool = False):
        result = []
        for name, meta in self.registry.items():
            if meta["category"] != category:
                continue
            if (not include_internal) and (not meta.get("planner_exposed", False)):
                continue

            contract = self._signature_contract(meta["func"])
            result.append({
                "tool_name": name,
                "category": meta["category"],
                "description": meta.get("description") or self._tool_description(name),
                "params": meta["params"],
                "planner_exposed": meta.get("planner_exposed", False),
                **contract,
            })
        return result
