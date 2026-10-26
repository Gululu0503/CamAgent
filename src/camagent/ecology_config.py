from __future__ import annotations


DEFAULT_0525_SPECIES_CONFIG = {
    "Siberian tiger": {
        "common_name": "Siberian tiger",
        "scientific_name": "Panthera tigris altaica",
        "group": "large_carnivore",
    },
    "Amur leopard": {
        "common_name": "Amur leopard",
        "scientific_name": "Panthera pardus orientalis",
        "group": "large_carnivore",
    },
    "Black bear": {
        "common_name": "Black bear",
        "scientific_name": "Ursus thibetanus",
        "group": "large_carnivore",
    },
    "Sika deer": {
        "common_name": "Sika deer",
        "scientific_name": "Cervus nippon",
        "group": "prey",
    },
    "Roe deer": {
        "common_name": "Roe deer",
        "scientific_name": "Capreolus pygargus",
        "group": "prey",
    },
    "东北虎": {
        "common_name": "Siberian tiger",
        "scientific_name": "Panthera tigris altaica",
        "group": "large_carnivore",
    },
    "东北豹": {
        "common_name": "Amur leopard",
        "scientific_name": "Panthera pardus orientalis",
        "group": "large_carnivore",
    },
    "黑熊": {
        "common_name": "Black bear",
        "scientific_name": "Ursus thibetanus",
        "group": "large_carnivore",
    },
    "梅花鹿": {
        "common_name": "Sika deer",
        "scientific_name": "Cervus nippon",
        "group": "prey",
    },
    "袍": {
        "common_name": "Roe deer",
        "scientific_name": "Capreolus pygargus",
        "group": "prey",
    },
}

DEFAULT_0525_COLUMN_MAP = {
    "longitude": "longitude",
    "latitude": "latitude",
    "recordtime": "recordtime",
    "videoname": "videoname",
    "经度": "longitude",
    "纬度": "latitude",
    "拍摄时间": "recordtime",
    "视频名称": "videoname",
    "camera_id": "camera_id",
    "species": "species",
    "scientificName": "scientificName",
}

EXAMPLE_0525_SPECIES = [
    "Siberian tiger",
    "Amur leopard",
    "Black bear",
    "Sika deer",
    "Roe deer",
]

SPECIES_CN = {
    "Siberian tiger": "Siberian tiger",
    "Amur leopard": "Amur leopard",
    "Black bear": "Black bear",
    "Sika deer": "Sika deer",
    "Roe deer": "Roe deer",
}

def normalize_ecological_role_mapping(role_mapping=None, species_inventory=None):
    """Validate optional, data-scoped ecological roles without inferring them from names."""
    role_mapping = role_mapping or {}
    if not isinstance(role_mapping, dict):
        raise ValueError("ecological_role_mapping must be an object with predator and prey lists.")
    available = {
        str(species).strip()
        for species in (species_inventory or [])
        if str(species).strip()
    }

    def _members(*keys):
        values = []
        for key in keys:
            candidate = role_mapping.get(key, [])
            if isinstance(candidate, str):
                candidate = [candidate]
            if candidate is None:
                candidate = []
            if not isinstance(candidate, (list, tuple, set)):
                raise ValueError(f"ecological_role_mapping.{key} must be a species list.")
            values.extend(str(value).strip() for value in candidate if str(value).strip())
        unique = list(dict.fromkeys(values))
        return [value for value in unique if not available or value in available]

    return {
        "predators": _members("predators", "predator", "large_carnivores"),
        "prey": _members("prey"),
        "source": str(role_mapping.get("source") or "user_or_dataset_configuration"),
    }

ECOLOGY_COLORS = {
    "Siberian tiger": "#8C4A3F",
    "Amur leopard": "#A66A4C",
    "Black bear": "#6F5E76",
    "Sika deer": "#4F7C6A",
    "Roe deer": "#8BA888",
    "Predator": "#8C4A3F",
    "Prey": "#4F7C6A",
    "primary": "#3B6E5A",
    "secondary": "#8BA888",
    "accent": "#C08A53",
    "text": "#263238",
    "axis": "#374151",
    "grid": "#E5E7EB",
    "dark": "#263238",
    "mid": "#6B7280",
    "grey": "#6B7280",
    "light": "#F3F4F6",
    "light_green": "#DDE8D5",
    "light_brown": "#EAD8C0",
}

FIGURE_SPECS = {
    "quality_conversion_audit": {
        "filename": "ecology_quality_conversion_audit.svg",
        "kind": "quality",
        "status": "core",
        "question": "Were the source CSV files cleaned, standardized, and converted into an independent-event analysis unit?",
        "caption": "Shows retained records across five source files, 30-minute independent-event compression, and filtered camera coordinates.",
        "metric_basis": "CamtrapDP media rows and 30-minute independent events.",
        "boundary": "This quality figure evaluates analysis readiness and does not directly answer ecological mechanisms.",
    },
    "species_indices": {
        "filename": "ecology_species_indices.svg",
        "kind": "community",
        "status": "core",
        "question": "How do the five target species differ in effort-corrected activity intensity, occupancy breadth, and detection rate?",
        "caption": "Compares species-level metrics using RAI, naive occupancy proportion, daily detection rate, and event volume.",
        "metric_basis": "RAI = independent events / 100 camera-days; occupancy is occupied cameras / all cameras.",
        "boundary": "These are detection and relative-activity metrics, not population density. If deployment spans were inferred from detections, RAI is exploratory until independent camera-operation logs are supplied.",
    },
    "camera_richness_diversity": {
        "filename": "ecology_camera_richness_diversity.svg",
        "kind": "space",
        "status": "core",
        "question": "Which cameras are key sites for multi-species richness, evenness, or large-carnivore detections?",
        "caption": "Shows target-species richness, Shannon/Simpson diversity, and prey-richness gradients at camera scale.",
        "metric_basis": "Camera-level independent-event counts and presence/absence by target species.",
        "boundary": "Detection richness is not complete biodiversity, and camera placement affects hotspot patterns.",
    },
    "temporal_phenology_overlap": {
        "filename": "ecology_temporal_phenology_overlap.svg",
        "kind": "time",
        "status": "core",
        "question": "Are predator and prey activity patterns synchronized or offset across diel activity and monthly phenology?",
        "caption": "Shows hourly activity curves, monthly phenology heatmaps, and a diel activity-overlap matrix.",
        "metric_basis": "Within-species shares of 30-minute independent events by hour and month.",
        "boundary": "Activity rhythms are inferred from detection times and are affected by trigger probability, seasonal sample size, and camera operation.",
    },
    "predator_prey_hypothesis": {
        "filename": "ecology_exploratory_predator_prey_hypothesis.svg",
        "kind": "relationship",
        "status": "exploratory",
        "question": "Do large carnivores appear more often near cameras with higher prey relative activity?",
        "caption": "Places prey RAI, predator occurrence proportion, and spatial co-detection index in one exploratory panel.",
        "metric_basis": "Camera-level prey RAI, predator RAI, and log-scaled co-detection score.",
        "boundary": "The co-detection index is hypothesis-generating and cannot prove predation events, following behavior, or causality.",
    },
    "spatial_cooccurrence_metrics": {
        "filename": "ecology_exploratory_spatial_cooccurrence.svg",
        "kind": "relationship",
        "status": "exploratory",
        "question": "Do species pairs share camera sites more often than expected by chance?",
        "caption": "Shows camera-level observed/expected co-occurrence and Jaccard shared-camera proportions.",
        "metric_basis": "Camera-level species presence from independent events.",
        "boundary": "Shared camera use does not imply simultaneous co-occurrence or direct interaction.",
    },
    "occupancy_detection_breadth": {
        "filename": "ecology_occupancy_detection_breadth.svg",
        "kind": "occupancy",
        "status": "core",
        "question": "Do species differ in detection breadth across the camera network?",
        "caption": "Compares detection coverage using naive occupancy proportions and Wilson intervals.",
        "metric_basis": "Occupied cameras / total cameras, with binomial Wilson intervals.",
        "boundary": "This is a naive occurrence screen, not a detection-probability-corrected occupancy model; formal occupancy also requires independently observed survey effort.",
    },
    "predator_prey_dashboard": {
        "filename": "ecology_exploratory_predator_prey_dashboard.svg",
        "kind": "relationship",
        "status": "exploratory",
        "question": "Are predator and prey signals coupled across space, diel activity, and season?",
        "caption": "Combines the community event pool, predator-prey RAI relationship, hourly activity, and seasonal activity.",
        "metric_basis": "Independent events and camera-effort corrected RAI.",
        "boundary": "This integrated panel screens for patterns and should not be used alone as mechanistic evidence.",
    },
    "temporal_niche_heatmap": {
        "filename": "ecology_exploratory_temporal_niche_heatmap.svg",
        "kind": "time",
        "status": "exploratory",
        "question": "How do the five species partition activity across the 24-hour day?",
        "caption": "Shows diel niche differences using within-species normalized hourly activity proportions.",
        "metric_basis": "Within-species hourly share of independent events.",
        "boundary": "The hourly heatmap is affected by sample size and camera-trigger differences and should be interpreted alongside event counts.",
    },
    "spatial_association_matrix": {
        "filename": "ecology_exploratory_spatial_association_matrix.svg",
        "kind": "relationship",
        "status": "exploratory",
        "question": "Is shared camera use among species pairs higher than random expectation?",
        "caption": "Uses observed/expected shared cameras to inspect the direction of spatial association.",
        "metric_basis": "Observed shared cameras divided by expected shared cameras under marginal occupancy.",
        "boundary": "Observed/expected co-occurrence indicates spatial overlap or shared-habitat clues and cannot prove interaction.",
    },
    "camera_ecology_biplot": {
        "filename": "ecology_exploratory_camera_biplot.svg",
        "kind": "space",
        "status": "exploratory",
        "question": "Which cameras combine high activity volume, high richness, and large-carnivore records?",
        "caption": "Screens priority cameras using total events, richness, and large-carnivore detection flags.",
        "metric_basis": "Camera-level independent events and target-species richness.",
        "boundary": "Priority cameras are information-rich in detection terms and do not necessarily indicate the highest ecological density.",
    },
}
