import os
import json
import math
import re
import textwrap
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from .paths import MPLCONFIG_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.text as mtext
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
import numpy as np
import pandas as pd

try:
    import folium
    from folium.plugins import MousePosition, MeasureControl
except ModuleNotFoundError:
    folium = None
    MousePosition = None
    MeasureControl = None

try:
    import geopandas as gpd
except ModuleNotFoundError:
    gpd = None

from .agent_router import EcologicalToolset
from .ecology_config import normalize_ecological_role_mapping
from .paths import CAMTRAP_ACTIVE_DATASET, OUTPUTS_DIR, PARK_SHAPEFILE_PATH


def _integrate_trapezoid(y, x):
    integrator = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return integrator(y, x)


SCI_COLORS = {
    "text": "#263238",
    "axis": "#374151",
    "grid": "#E5E7EB",
    "primary": "#3D6696",
    "secondary": "#90BBD7",
    "accent": "#F28E2B",
    "predator": "#2F5E93",
    "prey": "#6FAE54",
    "neutral": "#6B7280",
    "light_green": "#D9EED2",
    "light_brown": "#F7F0E5",
    "light_grey": "#F3F4F6",
    "paper": "#FFFFFF",
    # Backward-compatible semantic aliases used throughout the plotting code.
    "blue": "#3D6696",
    "orange": "#F28E2B",
    "green": "#6FAE54",
    "purple": "#8E63B0",
    "sky": "#9ECAE1",
    "yellow": "#F2CF72",
    "salmon": "#D6786D",
    "brick": "#B63A3A",
    "plum": "#5A1A4A",
    "sage": "#B8D9B0",
    "grey": "#6B7280",
    "dark_grey": "#374151",
    "dark": "#263238",
}

SCI_LINE_PALETTE = [
    SCI_COLORS["blue"],
    SCI_COLORS["salmon"],
    SCI_COLORS["green"],
    SCI_COLORS["purple"],
    SCI_COLORS["sky"],
    SCI_COLORS["orange"],
    SCI_COLORS["brick"],
    SCI_COLORS["plum"],
    SCI_COLORS["sage"],
    SCI_COLORS["grey"],
]

PLOT_FONT_SCALE = 1.25

SCI_BAR_PALETTE = [
    "#3B6E5A",
    "#4F7C6A",
    "#8BA888",
    "#C08A53",
    "#A7C4B5",
    "#D6B56D",
    "#6F5E76",
    "#6B7280",
]

FRESH_HEATMAP_CMAP = plt.get_cmap("viridis")
SPATIAL_INTENSITY_CMAP = plt.get_cmap("YlGnBu")
FRESH_DIVERGING_CMAP = plt.get_cmap("BrBG")


class VisualizationTools(EcologicalToolset):
    """
    Visualization tools:
    - Spatial, hotspot, and transition-corridor outputs: folium interactive HTML maps.
    - Temporal and distribution charts: SVG plus an adjacent publication PDF.
    """

    def __init__(
        self,
        db_path=None,
        collection_name="tiger_observations_2024",
        output_dir=None,
        park_shapefile_path=None,
    ):
        super().__init__(db_path=db_path, collection_name=collection_name)
        self.output_dir = Path(OUTPUTS_DIR if output_dir is None else output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.park_shapefile_path = str(PARK_SHAPEFILE_PATH if park_shapefile_path is None else park_shapefile_path)
        self._park_geojson_cache = None

    # =========================
    # General helpers
    # =========================
    def _safe_name(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
        return text[:80] if text else "plot"

    def _build_path(self, filename: str) -> str:
        if str(filename).lower().endswith(".png"):
            filename = f"{filename[:-4]}.svg"
        path = self.output_dir / filename
        return str(path)

    def _as_svg_path(self, save_path: str) -> str:
        path = str(save_path)
        if path.lower().endswith(".png"):
            return f"{path[:-4]}.svg"
        return path

    def _finalize_plot(self, save_path: str, title: str = None):
        save_path = self._as_svg_path(save_path)
        if title:
            plt.title(title)
        self._scale_figure_text(plt.gcf())
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        return save_path

    def _annotate_bar_values(self, xs, ys, rotation=0, fontsize=8):
        for x, y in zip(xs, ys):
            plt.text(x, y, str(y), ha="center", va="bottom", fontsize=fontsize, rotation=rotation)

    def _sorted_pairs_desc(self, labels, values):
        pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
        labels_sorted = [x[0] for x in pairs]
        values_sorted = [x[1] for x in pairs]
        return labels_sorted, values_sorted

    def _apply_clean_sci_axes(self, ax, grid_axis=None):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")
        ax.tick_params(axis="both", colors="#222222", length=3.5, width=0.8)
        if grid_axis:
            ax.grid(axis=grid_axis, color="#DADDE1", linewidth=0.65, alpha=0.65)
        else:
            ax.grid(False)

    def _panel_title(self, ax, label, title):
        ax.set_title(str(title), loc="left", fontsize=11, fontweight="bold", pad=8)

    def _palette_color(self, idx, palette=None):
        palette = palette or SCI_LINE_PALETTE
        return palette[int(idx) % len(palette)]

    def _fresh_heatmap_cmap(self, diverging=False):
        cmap = (FRESH_DIVERGING_CMAP if diverging else FRESH_HEATMAP_CMAP).copy()
        cmap.set_bad("#F1F3F5")
        return cmap

    def _spatial_intensity_cmap(self):
        cmap = SPATIAL_INTENSITY_CMAP.copy()
        cmap.set_bad("#F1F3F5")
        return cmap

    def _heatmap_text_color(self, value, vmin, vmax):
        try:
            numeric = float(value)
        except Exception:
            return "#1F2933"
        if not np.isfinite(numeric):
            return "#1F2933"
        span = max(float(vmax) - float(vmin), 1e-9)
        scaled = (numeric - float(vmin)) / span
        return "white" if scaled >= 0.62 else "#1F2933"

    def plot_species_inventory_bar(self, top_k=20, include_unknown=False, save_path=None):
        composition = self.summarize_community_composition(
            top_k=top_k,
            include_unknown=include_unknown,
            abundance_metric="rai",
        )
        rows = [row for row in composition.get("top_species", []) if float(row.get("metric_value", 0.0)) > 0]
        if not rows:
            raise ValueError("No species inventory data available for plotting.")

        labels = [row["species"] for row in rows][::-1]
        values = np.asarray([row.get("metric_value", row.get("rai_per_100_camera_days", 0.0)) for row in rows][::-1], dtype=float)
        event_counts = np.asarray([row.get("independent_event_count", row["record_count"]) for row in rows][::-1], dtype=float)
        colors = [self._palette_color(idx, SCI_BAR_PALETTE) for idx in range(len(labels))]

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(8.8, max(4.4, 0.34 * len(labels) + 1.4)))
        bars = ax.barh(
            labels,
            values,
            color=colors,
            edgecolor="#3f464d",
            linewidth=0.55,
            alpha=0.78,
        )
        for bar, value, count in zip(bars, values, event_counts):
            ax.text(
                bar.get_width() + max(values.max(), 1.0) * 0.018,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f} | events={int(count)}",
                va="center",
                fontsize=7.4,
                clip_on=False,
            )
        ax.set_xlabel("Adjusted RAI (independent events / 100 camera-days)")
        ax.set_ylabel("Species")
        ax.set_title("Selected focal mammal composition", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(0, max(float(values.max()) * 1.45, 5.0))
        self._apply_clean_sci_axes(ax, grid_axis="x")
        if save_path is None:
            save_path = self._build_path(f"community_species_inventory_top{top_k}.png")
        return self._finalize_paper_plot(fig, save_path)

    # =========================
    # Planner-visible final demo composite figures
    # =========================
    def _final_demo_package_dir(self):
        if not self._using_camtrap_backend():
            raise RuntimeError(
                "This publication figure requires an active CamtrapDP package."
            )
        return Path(self._camtrap_package_dir)

    def _final_demo_output_path(self, filename):
        demo_dir = self.output_dir / "final_demo_figures"
        demo_dir.mkdir(parents=True, exist_ok=True)
        return str(demo_dir / filename)

    def _save_final_demo_figure(self, fig, save_path):
        path = Path(save_path)
        if path.suffix.lower() not in {".png", ".svg"}:
            path = path.with_suffix(".png")
        png_path = path.with_suffix(".png")
        svg_path = path.with_suffix(".svg")
        pdf_path = path.with_suffix(".pdf")
        layout = getattr(fig, "_agenttool_layout", {})
        bottom = float(layout.get("bottom", 0.08))
        top = float(layout.get("top", 0.94))
        hspace = layout.get("hspace")
        wspace = layout.get("wspace")

        self._scale_figure_text(fig, scale=1.12)
        if bool(getattr(fig, "_agenttool_align_labels", True)):
            try:
                fig.align_labels()
            except Exception:
                pass
        if not bool(getattr(fig, "_agenttool_skip_tight_layout", False)):
            try:
                fig.tight_layout(rect=(0.0, bottom, 1.0, top))
            except Exception:
                pass
        adjust_kwargs = {}
        if bottom > 0.08:
            adjust_kwargs["bottom"] = min(bottom, 0.42)
        if top < 0.94:
            adjust_kwargs["top"] = max(top, 0.55)
        if hspace is not None:
            adjust_kwargs["hspace"] = hspace
        if wspace is not None:
            adjust_kwargs["wspace"] = wspace
        if adjust_kwargs:
            try:
                fig.subplots_adjust(**adjust_kwargs)
            except Exception:
                pass
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return str(png_path)

    def _final_demo_events_table(self, species_list):
        self._final_demo_package_dir()
        events = self._load_camtrap_event_df().copy()
        if events.empty:
            raise ValueError("No CamtrapDP independent events are available for final demo figures.")
        if "species" not in events.columns:
            events["species"] = events.get("scientificName", "").map(self._canonical_species_name)
        events["species"] = events["species"].map(self._canonical_species_name)
        camera_col = "camera_id" if "camera_id" in events.columns else "cameraID"
        events["camera_id"] = events[camera_col].astype(str)
        time_col = "event_time_dt" if "event_time_dt" in events.columns else "event_time"
        events["event_time_dt"] = self._parse_camtrap_timestamp_series(
            events[time_col],
            local_clock=True,
        )
        events["hour"] = events["event_time_dt"].dt.hour
        events["month"] = events["event_time_dt"].dt.month
        events["latitude"] = pd.to_numeric(events.get("latitude"), errors="coerce")
        events["longitude"] = pd.to_numeric(events.get("longitude"), errors="coerce")
        events = events.dropna(subset=["camera_id", "species", "event_time_dt", "latitude", "longitude"]).copy()
        events = events[events["species"].isin(species_list)].copy()

        camera_locs = self.get_camera_locations()
        effort_days = self._camera_effort_days_from_records()
        count_table = events.groupby(["camera_id", "species"], observed=True).size().unstack(fill_value=0)
        for species in species_list:
            if species not in count_table.columns:
                count_table[species] = 0
        loc_rows = []
        for camera_id in sorted(set(count_table.index.astype(str)) | set(str(x) for x in camera_locs.keys())):
            loc = camera_locs.get(str(camera_id), {})
            loc_rows.append({
                "camera_id": str(camera_id),
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
                "effort_days": max(float(effort_days.get(str(camera_id), 1.0)), 1.0),
            })
        table = pd.DataFrame(loc_rows).set_index("camera_id")
        table = table.join(count_table[species_list], how="left").fillna({species: 0 for species in species_list})
        for species in species_list:
            table[species] = pd.to_numeric(table[species], errors="coerce").fillna(0).astype(int)
            table[f"{species}_rai"] = table[species] / table["effort_days"].clip(lower=1) * 100.0
        table["target_richness"] = (table[species_list] > 0).sum(axis=1)
        table["total_independent_events"] = table[species_list].sum(axis=1)
        return events, table.reset_index()

    def _annotate_final_demo_points(self, ax, rows, label_func, offsets=None, fontsize=7.0):
        offsets = offsets or [(16, 10), (18, -12), (-34, 14), (-36, -12), (14, 22), (-30, 24)]
        for idx, (_, row) in enumerate(rows.iterrows()):
            dx, dy = offsets[idx % len(offsets)]
            ax.annotate(
                label_func(row),
                xy=(row["longitude"], row["latitude"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                ha="left" if dx >= 0 else "right",
                va="center",
                arrowprops={"arrowstyle": "-", "color": "#7A8793", "lw": 0.65},
                bbox={"boxstyle": "round,pad=0.13", "fc": "white", "ec": "#B5C0C8", "alpha": 0.92},
            )

    def _hourly_share(self, events, species):
        counts = (
            events.loc[events["species"] == species, "hour"]
            .dropna()
            .astype(int)
            .value_counts()
            .reindex(range(24), fill_value=0)
            .sort_index()
        )
        values = counts.to_numpy(dtype=float)
        return values / values.sum() if values.sum() else values

    def _smooth_hourly_share(self, values):
        values = np.asarray(values, dtype=float)
        if not values.size or values.sum() <= 0:
            return values
        padded = np.r_[values[-2:], values, values[:2]]
        smoothed = np.convolve(padded, np.ones(5) / 5, mode="same")[2:-2]
        return smoothed / smoothed.sum() if smoothed.sum() else smoothed

    def _red_white_cmap(self):
        return mcolors.LinearSegmentedColormap.from_list("agent_red_white", ["#FFFFFF", "#F6B0AE", "#B63A3A"])

    def plot_community_spatial_association_panel(
        self,
        species_list=None,
        top_k=5,
        save_path=None,
        privacy_grid_degrees=0.1,
    ):
        """Generic CamtrapDP composition, richness, and shared-camera panel."""
        if isinstance(species_list, str):
            species_list = [
                item.strip()
                for item in species_list.split(",")
                if item.strip()
            ]
        if not species_list:
            available = self._load_camtrap_event_df().copy()
            if available.empty:
                raise ValueError(
                    "No independent events are available for a community panel."
                )
            if "species" not in available.columns:
                available["species"] = available.get(
                    "scientificName",
                    "",
                ).map(self._canonical_species_name)
            available["species"] = available["species"].map(
                self._canonical_species_name
            )
            species_list = (
                available.loc[available["species"] != "Unknown", "species"]
                .value_counts()
                .head(max(int(top_k), 2))
                .index.tolist()
            )
        species_list = [
            self._canonical_species_name(item)
            for item in species_list
        ]
        events, table = self._final_demo_events_table(species_list)
        effort_total = float(table["effort_days"].sum())
        effort_basis = self._camera_effort_basis()
        has_independent_effort = effort_basis == "documented_deployment_intervals"
        metrics = []
        for species in species_list:
            n_events = int((events["species"] == species).sum())
            metrics.append({
                "species": species,
                "events": n_events,
                "rai": (
                    n_events / max(effort_total, 1.0) * 100.0
                    if has_independent_effort
                    else None
                ),
            })
        metrics_df = pd.DataFrame(metrics).sort_values("events", ascending=True)

        camera_presence = table.copy()
        camera_presence["camera_id"] = [
            str(row["camera_id"])
            for _, row in camera_presence.iterrows()
        ]
        presence = (
            camera_presence.set_index("camera_id")[species_list] > 0
        ).astype(int)
        n_camera = max(len(presence), 1)
        oe = np.full((len(species_list), len(species_list)), np.nan)
        for i, a in enumerate(species_list):
            for j, b in enumerate(species_list):
                if i == j:
                    continue
                observed = int(((presence[a] == 1) & (presence[b] == 1)).sum())
                expected = float(presence[a].sum() * presence[b].sum() / n_camera)
                oe[i, j] = observed / expected if expected else np.nan
        shared_camera_pairs = []
        for i, species_a in enumerate(species_list):
            for j in range(i + 1, len(species_list)):
                species_b = species_list[j]
                detected_a = presence[species_a] == 1
                detected_b = presence[species_b] == 1
                shared = int((detected_a & detected_b).sum())
                union = int((detected_a | detected_b).sum())
                expected = float(detected_a.sum() * detected_b.sum() / n_camera)
                shared_camera_pairs.append({
                    "species_a": species_a,
                    "species_b": species_b,
                    "shared_camera_count": shared,
                    "camera_jaccard": round(shared / union, 4) if union else None,
                    "observed_expected_ratio": round(shared / expected, 4) if expected else None,
                })
        shared_camera_pairs.sort(
            key=lambda item: (
                -item["shared_camera_count"],
                item["species_a"],
                item["species_b"],
            )
        )

        self._set_paper_style()
        fig = plt.figure(figsize=(11.8, 7.5))
        outer = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.16, 1.0],
            wspace=0.42,
        )
        left_grid = outer[0].subgridspec(
            1,
            2,
            width_ratios=[1.0, 0.055],
            wspace=0.08,
        )
        right_grid = outer[1].subgridspec(
            2,
            2,
            width_ratios=[1.0, 0.06],
            height_ratios=[1.0, 1.0],
            hspace=0.52,
            wspace=0.10,
        )
        map_ax = fig.add_subplot(left_grid[0, 0])
        map_cax = fig.add_subplot(left_grid[0, 1])
        bar_ax = fig.add_subplot(right_grid[0, 0])
        bar_spacer_ax = fig.add_subplot(right_grid[0, 1])
        matrix_ax = fig.add_subplot(right_grid[1, 0])
        matrix_cax = fig.add_subplot(right_grid[1, 1])
        bar_spacer_ax.axis("off")
        axes = [bar_ax, map_ax, matrix_ax]

        pastel = ["#B7C3CC", "#D99A92", "#EACD82", "#C8E3C2", "#C8B2DF", "#B6DEEA"]
        ax = axes[0]
        bars = ax.barh(
            metrics_df["species"],
            metrics_df["events"],
            color=[pastel[i % len(pastel)] for i in range(len(metrics_df))],
            edgecolor="#3f464d",
            linewidth=0.65,
        )
        for bar, row in zip(bars, metrics_df.itertuples()):
            ax.text(
                bar.get_width() * 1.06,
                bar.get_y() + bar.get_height() / 2,
                f"{row.events:,}",
                va="center",
                fontsize=7.5,
                clip_on=False,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Independent events (log scale)")
        ax.set_title("Selected-species detections", loc="left", fontweight="bold")
        self._apply_clean_sci_axes(ax, "x")

        ax = axes[1]
        spatial = table.dropna(subset=["longitude", "latitude"]).copy()
        park_gdf = self._load_park_gdf()
        if park_gdf is not None and not park_gdf.empty and gpd is not None:
            try:
                point_gdf = gpd.GeoDataFrame(
                    spatial,
                    geometry=gpd.points_from_xy(spatial["longitude"], spatial["latitude"]),
                    crs="EPSG:4326",
                )
                if getattr(park_gdf, "crs", None) is not None:
                    park_gdf = park_gdf.to_crs(epsg=4326)
                within_mask = point_gdf.geometry.within(park_gdf.unary_union) | point_gdf.geometry.touches(park_gdf.unary_union)
                clipped = spatial.loc[within_mask.to_numpy()].copy()
                if not clipped.empty:
                    spatial = clipped
                    park_gdf.plot(ax=ax, facecolor="#2F5E93", edgecolor="none", alpha=0.03, zorder=1)
            except Exception:
                park_gdf = None
        privacy_grid_degrees = max(float(privacy_grid_degrees), 0.01)
        spatial["plot_longitude"] = (
            spatial["longitude"] / privacy_grid_degrees
        ).round() * privacy_grid_degrees
        spatial["plot_latitude"] = (
            spatial["latitude"] / privacy_grid_degrees
        ).round() * privacy_grid_degrees
        grid_aggregations = {
            species: "sum"
            for species in species_list
            if species in spatial.columns
        }
        grid_aggregations["total_independent_events"] = "sum"
        spatial = (
            spatial.groupby(
                ["plot_longitude", "plot_latitude"],
                as_index=False,
            )
            .agg(grid_aggregations)
        )
        spatial["target_richness"] = (
            spatial[[species for species in species_list if species in spatial.columns]]
            > 0
        ).sum(axis=1)
        ax.scatter(
            spatial["plot_longitude"],
            spatial["plot_latitude"],
            s=np.clip(np.sqrt(spatial["total_independent_events"]) * 1.1, 4, 44),
            color="#9ECAE1",
            alpha=0.24,
            edgecolors="white",
            linewidths=0.18,
            label="Privacy-grid cells",
            zorder=2,
        )
        sc = ax.scatter(
            spatial["plot_longitude"],
            spatial["plot_latitude"],
            c=spatial["target_richness"],
            s=np.clip(np.sqrt(spatial["total_independent_events"]) * 1.35, 5, 58),
            cmap=self._spatial_intensity_cmap(),
            vmin=0,
            vmax=max(1, int(spatial["target_richness"].max())),
            alpha=0.66,
            edgecolors="white",
            linewidths=0.20,
            zorder=3,
        )
        cbar = fig.colorbar(sc, cax=map_cax)
        cbar.set_label("Selected species detected")
        hot = spatial.sort_values(["target_richness", "total_independent_events"], ascending=False).head(int(top_k))
        ax.scatter(
            hot["plot_longitude"],
            hot["plot_latitude"],
            s=36,
            color=SCI_COLORS["orange"],
            edgecolors=SCI_COLORS["text"],
            linewidths=1.0,
            label="Highest-richness grid cells",
            zorder=6,
        )
        if park_gdf is not None and not park_gdf.empty:
            try:
                park_gdf.boundary.plot(ax=ax, color="#374151", linewidth=1.25, zorder=9)
            except Exception:
                pass
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Masked longitude ({privacy_grid_degrees:g}° grid)", labelpad=8)
        ax.set_ylabel(f"Masked latitude ({privacy_grid_degrees:g}° grid)")
        ax.set_title("Selected-species richness", loc="left", fontweight="bold")
        ax.legend(
            frameon=False,
            loc="upper right",
            ncol=1,
            fontsize=7.0,
            borderaxespad=0.45,
            markerscale=1.7,
        )
        self._apply_clean_sci_axes(ax, "both")

        ax = axes[2]
        finite = oe[np.isfinite(oe)]
        vmax = max(float(finite.max()) if finite.size else 1.0, 1.01)
        cmap = mcolors.LinearSegmentedColormap.from_list("oe_blue_orange", ["#F2D39B", "#FFFFFF", "#228C87"])
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax)
        im = ax.imshow(oe, cmap=cmap, norm=norm)
        cbar = fig.colorbar(im, cax=matrix_cax)
        cbar.set_label("Observed / marginal expectation")
        ax.set_xticks(range(len(species_list)))
        ax.set_yticks(range(len(species_list)))
        ax.set_xticklabels(species_list, rotation=38, ha="right")
        ax.set_yticklabels(species_list)
        for i in range(len(species_list)):
            for j in range(len(species_list)):
                label = "NA" if np.isnan(oe[i, j]) else f"{oe[i, j]:.2f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=7.4)
        ax.set_title("Shared-camera association (O/E)", loc="left", fontweight="bold")
        fig._agenttool_align_labels = False
        fig._agenttool_skip_tight_layout = True
        fig._agenttool_layout = {
            "bottom": 0.09,
            "top": 0.95,
        }

        if save_path is None:
            save_path = self._final_demo_output_path(
                "selected_species_spatial_association.png"
            )
        png_path = self._save_final_demo_figure(fig, save_path)
        svg_path = str(Path(png_path).with_suffix(".svg"))
        pdf_path = str(Path(png_path).with_suffix(".pdf"))
        richness_hotspots = []
        for rank, (_, row) in enumerate(hot.iterrows(), start=1):
            richness_hotspots.append({
                "rank": rank,
                "aggregation_unit": "privacy_grid_cell",
                "selected_species_richness": int(row["target_richness"]),
                "total_independent_events": int(row["total_independent_events"]),
            })
        return {
            "status": "success",
            "focal_species": species_list,
            "community_composition": sorted(
                metrics,
                key=lambda item: (-item["events"], item["species"]),
            ),
            "composition_metric": (
                "rai_per_100_camera_days"
                if has_independent_effort
                else "independent_event_count"
            ),
            "effort_basis": effort_basis,
            "rai_inference_status": (
                "supported_by_documented_deployment_intervals"
                if has_independent_effort
                else "not_reported_without_independent_effort"
            ),
            "richness_hotspots": richness_hotspots,
            "shared_camera_pairs": shared_camera_pairs,
            "privacy_grid_degrees": privacy_grid_degrees,
            "interpretation_boundary": (
                "Composition and richness use independent-event detections. "
                + (
                    "RAI uses independently documented deployment intervals. "
                    if has_independent_effort
                    else
                    "Deployment intervals are detection-span proxies, so the "
                    "figure reports event counts rather than formal RAI. "
                )
                +
                "Observed/expected shared-camera ratios compare detections with "
                "marginal camera-detection frequencies and do not demonstrate "
                "simultaneous occurrence or direct interaction."
            ),
            "figures": [{
                "figure_id": "community_spatial_association_panel",
                "label": "Community composition, richness, and shared-camera association",
                "png_path": png_path,
                "svg_path": svg_path,
                "pdf_path": pdf_path,
            }],
        }

    def plot_complete_community_inference(
        self,
        species_list=None,
        top_k=None,
        display_top_species=12,
        min_independent_events=5,
        min_detected_cameras=2,
        min_gap_minutes=30,
        accumulation_permutations=250,
        bootstrap_iters=500,
        association_fdr_alpha=0.05,
        fit_detection_corrected_occupancy=True,
        occupancy_species_limit=None,
        occasion_days=14,
        min_active_occasions=2,
        occupancy_bootstrap_iters=300,
        random_seed=42,
        privacy_grid_degrees=0.1,
        save_path=None,
    ):
        """Plot dataset-wide community inference with a privacy-safe richness map."""
        result = self.infer_complete_community(
            species_list=species_list,
            top_k=top_k,
            min_independent_events=min_independent_events,
            min_detected_cameras=min_detected_cameras,
            min_gap_minutes=min_gap_minutes,
            accumulation_permutations=accumulation_permutations,
            bootstrap_iters=bootstrap_iters,
            association_fdr_alpha=association_fdr_alpha,
            fit_detection_corrected_occupancy=(
                fit_detection_corrected_occupancy
            ),
            occupancy_species_limit=occupancy_species_limit,
            occasion_days=occasion_days,
            min_active_occasions=min_active_occasions,
            occupancy_bootstrap_iters=occupancy_bootstrap_iters,
            random_seed=random_seed,
        )
        display_top_species = max(int(display_top_species), 2)
        species_rows = result["species_summary"][:display_top_species]
        display_species = [row["species"] for row in species_rows]

        self._set_paper_style()
        fig = plt.figure(figsize=(7.2, 5.0))
        # Fixed boxes keep both columns and both rows aligned exactly. Colour
        # bars occupy their own boxes and therefore do not shift the panels.
        map_ax = fig.add_axes([0.075, 0.565, 0.360, 0.315])
        map_cax = fig.add_axes([0.448, 0.565, 0.012, 0.315])
        accumulation_ax = fig.add_axes([0.580, 0.565, 0.340, 0.315])
        rank_ax = fig.add_axes([0.075, 0.125, 0.360, 0.315])
        association_ax = fig.add_axes([0.580, 0.125, 0.340, 0.315])
        association_cax = fig.add_axes([0.935, 0.125, 0.012, 0.315])
        setattr(fig, "_agenttool_skip_tight_layout", True)

        privacy_grid_degrees = max(float(privacy_grid_degrees), 0.01)
        events = self._load_camtrap_event_df(
            min_gap_minutes=min_gap_minutes
        ).copy()
        if "species" not in events:
            events["species"] = events.get("scientificName", "Unknown")
        events["species"] = events["species"].map(
            self._canonical_species_name
        )
        spatial_events = events.loc[
            events["species"].isin(result["species"])
        ].copy()
        spatial_events["longitude"] = pd.to_numeric(
            spatial_events["longitude"],
            errors="coerce",
        )
        spatial_events["latitude"] = pd.to_numeric(
            spatial_events["latitude"],
            errors="coerce",
        )
        spatial_events = spatial_events.dropna(
            subset=["longitude", "latitude", "camera_id"]
        )
        spatial_events["plot_longitude"] = (
            spatial_events["longitude"] / privacy_grid_degrees
        ).round() * privacy_grid_degrees
        spatial_events["plot_latitude"] = (
            spatial_events["latitude"] / privacy_grid_degrees
        ).round() * privacy_grid_degrees
        grid_keys = ["plot_longitude", "plot_latitude"]
        grid_summary = (
            spatial_events.groupby(grid_keys, as_index=False)
            .agg(
                eligible_species_richness=("species", "nunique"),
                independent_event_count=("species", "size"),
                camera_count=("camera_id", "nunique"),
            )
        )

        park_gdf = self._load_park_gdf()
        if park_gdf is not None and not park_gdf.empty:
            park_gdf.plot(
                ax=map_ax,
                facecolor="#F7F9FB",
                edgecolor=SCI_COLORS["text"],
                linewidth=0.85,
                zorder=0,
            )
            xmin, ymin, xmax, ymax = park_gdf.total_bounds
            xpad = max((xmax - xmin) * 0.025, 0.02)
            ypad = max((ymax - ymin) * 0.025, 0.02)
            visible = grid_summary.loc[
                grid_summary["plot_longitude"].between(
                    xmin - xpad,
                    xmax + xpad,
                )
                & grid_summary["plot_latitude"].between(
                    ymin - ypad,
                    ymax + ypad,
                )
            ].copy()
            if not visible.empty:
                grid_summary = visible
            map_ax.set_xlim(xmin - xpad, xmax + xpad)
            map_ax.set_ylim(ymin - ypad, ymax + ypad)
            mean_latitude = 0.5 * (ymin + ymax)
            map_ax.set_aspect(
                1.0 / math.cos(math.radians(mean_latitude)),
                adjustable="box",
            )
        elif not grid_summary.empty:
            xpad = max(
                float(grid_summary["plot_longitude"].max()
                      - grid_summary["plot_longitude"].min()) * 0.04,
                privacy_grid_degrees,
            )
            ypad = max(
                float(grid_summary["plot_latitude"].max()
                      - grid_summary["plot_latitude"].min()) * 0.04,
                privacy_grid_degrees,
            )
            map_ax.set_xlim(
                grid_summary["plot_longitude"].min() - xpad,
                grid_summary["plot_longitude"].max() + xpad,
            )
            map_ax.set_ylim(
                grid_summary["plot_latitude"].min() - ypad,
                grid_summary["plot_latitude"].max() + ypad,
            )

        maximum_richness = max(
            int(grid_summary["eligible_species_richness"].max())
            if not grid_summary.empty
            else int(result["analysed_species_count"]),
            1,
        )
        richness_colors = [
            "#E6F3EE",
            "#8FD3C1",
            "#45A8B6",
            "#2C6E9E",
            "#274C77",
        ]
        richness_cmap = mcolors.ListedColormap(
            richness_colors[:maximum_richness]
            if maximum_richness <= len(richness_colors)
            else plt.get_cmap("YlGnBu")(
                np.linspace(0.18, 0.92, maximum_richness)
            )
        )
        richness_norm = mcolors.BoundaryNorm(
            np.arange(0.5, maximum_richness + 1.5, 1),
            richness_cmap.N,
        )
        marker_sizes = (
            np.clip(
                7.0 + 2.0 * np.sqrt(grid_summary["camera_count"]),
                9.0,
                42.0,
            )
            if not grid_summary.empty
            else np.asarray([])
        )
        scatter = map_ax.scatter(
            grid_summary["plot_longitude"],
            grid_summary["plot_latitude"],
            c=grid_summary["eligible_species_richness"],
            s=marker_sizes,
            cmap=richness_cmap,
            norm=richness_norm,
            edgecolor="white",
            linewidth=0.20,
            alpha=0.76,
            zorder=3,
        )
        top_grid_cells = grid_summary.sort_values(
            [
                "eligible_species_richness",
                "camera_count",
                "independent_event_count",
            ],
            ascending=False,
        ).head(5)
        if not top_grid_cells.empty:
            map_ax.scatter(
                top_grid_cells["plot_longitude"],
                top_grid_cells["plot_latitude"],
                s=np.clip(
                    14.0 + 2.0 * np.sqrt(top_grid_cells["camera_count"]),
                    18.0,
                    48.0,
                ),
                facecolor="none",
                edgecolor=SCI_COLORS["accent"],
                linewidth=1.15,
                zorder=5,
                label="Highest-richness grid cells",
            )
            map_ax.legend(
                loc="lower left",
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=0.86,
                fontsize=6.6,
                handletextpad=0.3,
                borderaxespad=0.35,
            )
        map_colorbar = fig.colorbar(
            scatter,
            cax=map_cax,
            ticks=np.arange(1, maximum_richness + 1),
        )
        map_colorbar.set_label(
            "Eligible species detected",
            fontsize=7.0,
        )
        map_colorbar.ax.tick_params(labelsize=6.8, length=2.4)
        map_ax.set_xlabel("Longitude", fontsize=8.0)
        map_ax.set_ylabel("Latitude", fontsize=8.0)
        map_ax.set_title(
            "Privacy-safe community richness",
            loc="left",
            fontsize=9.2,
            fontweight="bold",
            pad=5,
        )
        map_ax.locator_params(axis="both", nbins=4)
        self._apply_clean_sci_axes(map_ax, None)
        map_ax.grid(
            True,
            color=SCI_COLORS["grid"],
            linewidth=0.45,
            alpha=0.70,
            zorder=-2,
        )

        ranked = pd.DataFrame(species_rows).sort_values(
            "independent_event_count",
            ascending=True,
        )
        species_colors = {
            "Siberian tiger": "#0072B2",
            "Sika deer": "#009E73",
            "Roe deer": "#CC79A7",
            "Black bear": "#4D4D4D",
            "Amur leopard": "#E69F00",
        }
        bars = rank_ax.barh(
            ranked["species"],
            ranked["independent_event_count"],
            color=[
                species_colors.get(species, SCI_COLORS["primary"])
                for species in ranked["species"]
            ],
            edgecolor="white",
            linewidth=0.55,
            height=0.68,
        )
        rank_ax.set_xscale("log")
        rank_ax.set_xlim(
            max(float(ranked["independent_event_count"].min()) * 0.55, 1),
            float(ranked["independent_event_count"].max()) * 3.0,
        )
        for bar, value in zip(
            bars,
            ranked["independent_event_count"],
        ):
            rank_ax.text(
                float(value) * 1.10,
                bar.get_y() + bar.get_height() / 2,
                f"{int(value):,}",
                va="center",
                ha="left",
                fontsize=6.8,
            )
        rank_ax.set_xlabel("Independent events (log scale)", fontsize=8.0)
        rank_ax.set_title(
            "Community event composition",
            loc="left",
            fontweight="bold",
            fontsize=9.2,
            pad=5,
        )
        self._apply_clean_sci_axes(rank_ax, "x")

        accumulation = pd.DataFrame(result["species_accumulation"])
        accumulation_ax.fill_between(
            accumulation["camera_count"].to_numpy(dtype=float),
            accumulation["ci_low"].to_numpy(dtype=float),
            accumulation["ci_high"].to_numpy(dtype=float),
            color="#9ECAE1",
            alpha=0.38,
            linewidth=0,
            label="95% random-order interval",
        )
        accumulation_ax.plot(
            accumulation["camera_count"],
            accumulation["mean_observed_species"],
            color="#146C94",
            linewidth=2.2,
            marker="o",
            markersize=3.3,
            label="Mean accumulation",
        )
        gamma_diversity = result["gamma_diversity"]
        accumulation_ax.axhline(
            gamma_diversity["chao2_richness"],
            color="#D55E00",
            linewidth=1.7,
            linestyle="--",
            label=(
                "Chao2 = "
                f"{gamma_diversity['chao2_richness']:.1f}"
            ),
        )
        accumulation_ax.set_xlabel("Cameras sampled")
        accumulation_ax.set_ylabel("")
        accumulation_ax.set_title(
            "Species accumulation and richness",
            loc="left",
            fontweight="bold",
            fontsize=9.2,
            pad=5,
        )
        accumulation_ax.set_xlabel("Cameras sampled", fontsize=8.0)
        accumulation_ax.set_ylabel("")
        accumulation_ax.legend(
            frameon=False,
            fontsize=6.5,
            loc="lower right",
            handlelength=1.6,
        )
        self._apply_clean_sci_axes(accumulation_ax, "both")

        associations = result.get(
            "conditional_camera_associations",
            [],
        )
        matrix = np.full(
            (len(display_species), len(display_species)),
            np.nan,
            dtype=float,
        )
        lookup = {
            (row["species_a"], row["species_b"]): row
            for row in associations
        }
        for i, species_a in enumerate(display_species):
            matrix[i, i] = 0.0
            for j, species_b in enumerate(display_species):
                if i == j:
                    continue
                row = (
                    lookup.get((species_a, species_b))
                    or lookup.get((species_b, species_a))
                )
                if row:
                    matrix[i, j] = row["standardized_effect_size"]
        finite = matrix[np.isfinite(matrix)]
        bound = max(
            float(np.max(np.abs(finite))) if finite.size else 1.0,
            1.0,
        )
        image = association_ax.imshow(
            np.ma.masked_where(
                np.triu(np.ones_like(matrix, dtype=bool), k=0),
                matrix,
            ),
            cmap=mcolors.LinearSegmentedColormap.from_list(
                "community_effect",
                ["#56B4E9", "white", "#D55E00"],
            ),
            norm=mcolors.TwoSlopeNorm(
                vmin=-bound,
                vcenter=0.0,
                vmax=bound,
            ),
            aspect="auto",
        )
        association_ax.set_xticks(
            range(len(display_species)),
            display_species,
            rotation=32,
            ha="right",
        )
        association_ax.set_yticks(
            range(len(display_species)),
            display_species,
        )
        association_ax.set_title(
            "Conditional shared-camera associations",
            loc="left",
            fontweight="bold",
            fontsize=9.2,
            pad=5,
        )
        for row_index in range(len(display_species)):
            for column_index in range(len(display_species)):
                if row_index <= column_index:
                    continue
                value = matrix[row_index, column_index]
                if not np.isfinite(value):
                    continue
                association_ax.text(
                    column_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color=(
                        "white"
                        if abs(float(value)) >= 0.52 * bound
                        else SCI_COLORS["text"]
                    ),
                    fontweight="bold",
                )
        colorbar = fig.colorbar(
            image,
            cax=association_cax,
        )
        colorbar.set_label("Standardized effect size", fontsize=7.0)
        colorbar.ax.tick_params(labelsize=6.8, length=2.4)
        association_ax.tick_params(axis="both", length=0)

        fig.suptitle(
            "Dataset-wide camera-trap community inference "
            f"({result['analysed_species_count']} eligible species)",
            fontsize=11.0,
            fontweight="bold",
            y=0.975,
        )
        if save_path is None:
            save_path = self._build_path(
                "complete_community_inference.svg"
            )
        svg_path = self._finalize_paper_plot(fig, save_path)
        pdf_path = str(Path(svg_path).with_suffix(".pdf"))
        return {
            **result,
            "spatial_richness_map": {
                "aggregation_unit": (
                    f"{privacy_grid_degrees:g}-degree privacy grid cell"
                ),
                "grid_cell_count": int(len(grid_summary)),
                "maximum_eligible_species_richness": maximum_richness,
                "boundary_path": str(self.park_shapefile_path),
                "boundary_display_name": (
                    "Northeast China Tiger and Leopard National Park"
                ),
            },
            "figures": [{
                "figure_id": "complete_community_inference",
                "label": (
                    "Dataset-wide community richness map, composition, "
                    "accumulation, and conditional association"
                ),
                "svg_path": svg_path,
                "pdf_path": pdf_path,
            }],
        }

    def plot_predator_prey_temporal_panel(
        self,
        focal_species=None,
        species_list=None,
        prey_species=None,
        ecological_role_mapping=None,
        save_path=None,
    ):
        """Composite temporal panel based only on explicit focal and ecological roles."""
        species_list = self._resolve_species_list(
            species_list=species_list,
            top_k=8,
            include_unknown=False,
        )
        roles = normalize_ecological_role_mapping(ecological_role_mapping, species_list)
        if prey_species:
            roles["prey"] = normalize_ecological_role_mapping(
                {"prey": prey_species}, species_list
            )["prey"]
        focal_species = self._canonical_species_name(focal_species) if focal_species else None
        if focal_species is None and len(roles["predators"]) == 1:
            focal_species = roles["predators"][0]
        if not focal_species or not roles["prey"]:
            raise ValueError(
                "A focal_species and explicit prey species or ecological_role_mapping are required."
            )
        if focal_species not in species_list:
            species_list = [focal_species, *species_list]
        prey_species = roles["prey"]
        events, _ = self._final_demo_events_table(species_list)
        hours = np.arange(24)
        focal_obs = self._hourly_share(events, focal_species)
        focal_smooth = self._smooth_hourly_share(focal_obs)
        prey = events[events["species"].isin(prey_species)].copy()
        prey_counts = prey["hour"].dropna().astype(int).value_counts().reindex(range(24), fill_value=0).sort_index().to_numpy(dtype=float)
        prey_share = prey_counts / prey_counts.sum() if prey_counts.sum() else prey_counts
        prey_smooth = self._smooth_hourly_share(prey_share)
        delta_guild = float(np.minimum(focal_smooth, prey_smooth).sum()) if prey_smooth.sum() else 0.0

        curve = self.build_hourly_activity_figure_data(
            species=focal_species,
            min_gap_minutes=30,
            smooth_sigma=1.15,
            bootstrap_iters=300,
        )

        self._set_paper_style()
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18.8, 5.35),
            gridspec_kw={"width_ratios": [1.15, 1.32, 1.28], "wspace": 0.42},
        )

        ax = axes[0]
        ax.axvspan(0, 6, color="#DCE8FF", alpha=0.36, zorder=0)
        ax.axvspan(18, 24, color="#DCE8FF", alpha=0.36, zorder=0)
        ax.axvspan(6, 18, color="#F7F0E5", alpha=0.24, zorder=0)
        ax.fill_between(hours, np.minimum(focal_smooth, prey_smooth), color="#BFC3C7", alpha=0.42, label="Overlap area")
        ax.plot(hours, focal_smooth, color=SCI_COLORS["blue"], lw=2.2, label=f"{focal_species} (events={int((events['species'] == focal_species).sum()):,})")
        ax.plot(hours, prey_smooth, color="#4F7C6A", lw=2.2, ls=(0, (4, 2)), label=f"Prey guild (events={len(prey):,})")
        ax.text(0.04, 0.08, f"Delta={delta_guild:.2f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.1,
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "#D1D5DB", "alpha": 0.9})
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xticklabels(["Midnight", "Sunrise", "Noon", "Sunset", "Midnight"])
        ax.set_xlabel("Time of day")
        ax.set_ylabel("Within-group hourly share")
        ax.set_title("Predator-prey diel overlap", loc="left", fontweight="bold")
        ax.legend(
            frameon=False,
            fontsize=7.4,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),
            borderaxespad=0.0,
        )
        self._apply_clean_sci_axes(ax, "y")

        ax = axes[1]
        ax.axvspan(0, 6, color="#DCE8FF", alpha=0.42, zorder=0, label="Night period")
        ax.axvspan(18, 24, color="#DCE8FF", alpha=0.42, zorder=0)
        ax.axvspan(6, 18, color="#F7F0E5", alpha=0.28, zorder=0, label="Day period")
        observed = curve.get("hourly_distribution", [])
        obs_ratio = np.asarray([row.get("ratio", 0.0) for row in observed], dtype=float)
        smooth = curve.get("smoothed_curve", {})
        grid = np.asarray(smooth.get("hour", []), dtype=float)
        mean = np.asarray(smooth.get("smooth_mean", []), dtype=float)
        ci_low = np.asarray(smooth.get("ci_low", []), dtype=float)
        ci_high = np.asarray(smooth.get("ci_high", []), dtype=float)
        if grid.size and mean.size:
            ax.fill_between(grid, ci_low, ci_high, color="#9DB7F5", alpha=0.28, label="95% bootstrap CI")
            ax.plot(grid, mean, color=SCI_COLORS["blue"], lw=2.4, label="Smoothed activity curve")
        ax.scatter(hours, obs_ratio, color=SCI_COLORS["blue"], edgecolors="white", linewidths=0.55, s=34, zorder=5, label="Observed hourly ratio")
        for item, label, dy in [(curve.get("major_peak"), "Major peak", 0.012), (curve.get("trough"), "Trough", -0.012)]:
            if item:
                h = float(item.get("hour", 0.0))
                v = float(item.get("value", 0.0))
                ax.axvline(h, color="#D09975" if label == "Major peak" else "#9AA6B2", ls="--", lw=0.9, alpha=0.75)
                ax.annotate(
                    f"{label}\n{h:.1f} h",
                    xy=(h, v),
                    xytext=(0, -30 if label == "Major peak" else 22),
                    textcoords="offset points",
                    ha="center",
                    va="top" if label == "Major peak" else "bottom",
                    bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#B5C0C8", "alpha": 0.92},
                    fontsize=7.2,
                )
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xlabel("Time of day (hour)")
        ax.set_ylabel("Relative activity")
        ax.set_title(f"Diel activity pattern of {focal_species}", loc="left", fontweight="bold")
        ax.legend(
            frameon=False,
            fontsize=7.0,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),
            borderaxespad=0.0,
            ncol=2,
            columnspacing=1.1,
            handlelength=1.7,
        )
        self._apply_clean_sci_axes(ax, "y")

        selected = [s for s in species_list if int((events["species"] == s).sum()) > 0][:6]
        matrix = np.eye(len(selected), dtype=float)
        shares = {species: self._smooth_hourly_share(self._hourly_share(events, species)) for species in selected}
        for i, a in enumerate(selected):
            for j, b in enumerate(selected):
                if i == j:
                    continue
                matrix[i, j] = float(np.minimum(shares[a], shares[b]).sum())
        ax = axes[2]
        im = ax.imshow(matrix, cmap=self._red_white_cmap(), vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.025)
        cbar.set_label("Diel overlap index Delta")
        ax.set_xticks(range(len(selected)))
        ax.set_yticks(range(len(selected)))
        ax.set_xticklabels(selected, rotation=40, ha="right")
        ax.set_yticklabels(selected)
        for i in range(len(selected)):
            for j in range(len(selected)):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.4, color="white" if val >= 0.72 else SCI_COLORS["text"])
        ax.set_title("Temporal niche overlap matrix", loc="left", fontweight="bold")

        if save_path is None:
            save_path = self._final_demo_output_path("demo2_predator_prey_temporal_reasoning.png")
        return self._save_final_demo_figure(fig, save_path)

    def plot_final_demo_predator_prey_diel_overlap_single(
        self,
        prey_species=None,
        save_path=None,
    ):
        """Single-panel final demo export for tiger-prey diel overlap."""
        species_list = ["Siberian tiger", "Red fox", "Wild boar", "Amur leopard", "Raccoon dog", "Horse", "Leopard cat", "Roe deer"]
        prey_species = prey_species or ["Wild boar", "Roe deer", "Sika deer"]
        events, _ = self._final_demo_events_table(species_list)
        hours = np.arange(24)
        tiger_obs = self._hourly_share(events, "Siberian tiger")
        tiger_smooth = self._smooth_hourly_share(tiger_obs)
        prey = events[events["species"].isin(prey_species)].copy()
        prey_counts = prey["hour"].dropna().astype(int).value_counts().reindex(range(24), fill_value=0).sort_index().to_numpy(dtype=float)
        prey_share = prey_counts / prey_counts.sum() if prey_counts.sum() else prey_counts
        prey_smooth = self._smooth_hourly_share(prey_share)
        delta_guild = float(np.minimum(tiger_smooth, prey_smooth).sum()) if prey_smooth.sum() else 0.0

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(10.4, 6.0))
        ax.axvspan(0, 6, color="#DCE8FF", alpha=0.36, zorder=0)
        ax.axvspan(18, 24, color="#DCE8FF", alpha=0.36, zorder=0)
        ax.axvspan(6, 18, color="#F7F0E5", alpha=0.24, zorder=0)
        ax.fill_between(hours, np.minimum(tiger_smooth, prey_smooth), color="#BFC3C7", alpha=0.42, label="Overlap area")
        ax.plot(hours, tiger_smooth, color=SCI_COLORS["blue"], lw=2.2, label=f"Siberian tiger (events={int((events['species'] == 'Siberian tiger').sum()):,})")
        ax.plot(hours, prey_smooth, color="#4F7C6A", lw=2.2, ls=(0, (4, 2)), label=f"Prey guild (events={len(prey):,})")
        ax.text(
            0.04,
            0.08,
            f"Delta={delta_guild:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "#D1D5DB", "alpha": 0.9},
        )
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xticklabels(["Midnight", "Sunrise", "Noon", "Sunset", "Midnight"])
        ax.set_xlabel("Time of day")
        ax.set_ylabel("Within-group hourly share")
        ax.set_title("Predator-prey diel overlap", loc="left", fontweight="bold")
        ax.legend(
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),
            borderaxespad=0.0,
        )
        self._apply_clean_sci_axes(ax, "y")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

        if save_path is None:
            save_path = self._final_demo_output_path("demo2a_predator_prey_diel_overlap.png")
        return self._save_final_demo_figure(fig, save_path)

    def plot_final_demo_tiger_diel_activity_single(
        self,
        save_path=None,
    ):
        """Single-panel final demo export for Siberian tiger diel activity."""
        curve = self.build_hourly_activity_figure_data(
            species="Siberian tiger",
            min_gap_minutes=30,
            smooth_sigma=1.15,
            bootstrap_iters=300,
        )
        hours = np.arange(24)
        observed = curve.get("hourly_distribution", [])
        obs_ratio = np.asarray([row.get("ratio", 0.0) for row in observed], dtype=float)
        smooth = curve.get("smoothed_curve", {})
        grid = np.asarray(smooth.get("hour", []), dtype=float)
        mean = np.asarray(smooth.get("smooth_mean", []), dtype=float)
        ci_low = np.asarray(smooth.get("ci_low", []), dtype=float)
        ci_high = np.asarray(smooth.get("ci_high", []), dtype=float)

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(10.4, 6.0))
        ax.axvspan(0, 6, color="#DCE8FF", alpha=0.42, zorder=0, label="Night period")
        ax.axvspan(18, 24, color="#DCE8FF", alpha=0.42, zorder=0)
        ax.axvspan(6, 18, color="#F7F0E5", alpha=0.28, zorder=0, label="Day period")
        if grid.size and mean.size:
            ax.fill_between(grid, ci_low, ci_high, color="#9DB7F5", alpha=0.28, label="95% bootstrap CI")
            ax.plot(grid, mean, color=SCI_COLORS["blue"], lw=2.4, label="Smoothed activity curve")
        ax.scatter(hours, obs_ratio, color=SCI_COLORS["blue"], edgecolors="white", linewidths=0.55, s=34, zorder=5, label="Observed hourly ratio")
        for item, label in [(curve.get("major_peak"), "Major peak"), (curve.get("trough"), "Trough")]:
            if item:
                h = float(item.get("hour", 0.0))
                v = float(item.get("value", 0.0))
                is_peak = label == "Major peak"
                ax.axvline(h, color="#D09975" if is_peak else "#9AA6B2", ls="--", lw=0.9, alpha=0.75)
                ax.annotate(
                    f"{label}\n{h:.1f} h",
                    xy=(h, v),
                    xytext=(0, -30 if is_peak else 22),
                    textcoords="offset points",
                    ha="center",
                    va="top" if is_peak else "bottom",
                    bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#B5C0C8", "alpha": 0.92},
                    fontsize=8.5,
                )
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xlabel("Time of day (hour)")
        ax.set_ylabel("Relative activity")
        ax.set_title("Diel activity pattern of Siberian tiger", loc="left", fontweight="bold")
        ax.legend(
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),
            borderaxespad=0.0,
            ncol=2,
            columnspacing=1.1,
            handlelength=1.7,
        )
        self._apply_clean_sci_axes(ax, "y")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))

        if save_path is None:
            save_path = self._final_demo_output_path("demo2b_tiger_diel_activity_pattern.png")
        return self._save_final_demo_figure(fig, save_path)

    def _format_final_demo_distance(self, distance):
        try:
            value = float(distance)
        except Exception:
            return "NA"
        return "<0.1 km" if value < 0.1 else f"{value:.1f} km"

    def plot_tiger_spatial_transition_panel(self, species=None, top_k=8, max_time_gap_hours=72, max_distance_km=100, save_path=None, steps=None):
        """Backward-compatible generic spatial-intensity and transition panel."""
        if not species or not str(species).strip():
            raise ValueError("species is required for a spatial-transition panel.")
        species = self._canonical_species_name(species)
        species_list = self._resolve_species_list(top_k=8, include_unknown=False)
        if species not in species_list:
            species_list = [species, *species_list]
        events, table = self._final_demo_events_table(species_list)
        focal_events = events[events["species"] == species].copy()
        if focal_events.empty:
            raise ValueError(f"No independent events available for {species}.")
        hotspots = self.build_hotspot_rank(species=species, top_k=max(int(top_k), 5))
        hotspot_ids = [str(row.get("camera_id")) for row in hotspots[:5]]
        hot_df = table[table["camera_id"].isin(hotspot_ids)].copy()
        hot_df["_order"] = hot_df["camera_id"].map({cid: idx for idx, cid in enumerate(hotspot_ids)})
        hot_df = hot_df.sort_values("_order")
        corridors = self.infer_transition_corridors(
            species=species,
            max_time_gap_hours=max_time_gap_hours,
            max_distance_km=max_distance_km,
            top_k=max(int(top_k), 8),
            steps=steps,
        )
        edge_rows = corridors.get("top_transition_corridors", []) if isinstance(corridors, dict) else list(corridors or [])

        self._set_paper_style()
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18.8, 5.35),
            gridspec_kw={"width_ratios": [1.22, 1.12, 1.32], "wspace": 0.44},
        )

        ax = axes[0]
        spatial = table.dropna(subset=["longitude", "latitude"]).copy()
        ax.scatter(
            spatial["longitude"],
            spatial["latitude"],
            s=16,
            color="#9ECAE1",
            alpha=0.38,
            edgecolors="white",
            linewidths=0.25,
            label="Monitoring cameras",
        )
        nonzero = spatial[spatial[species] > 0].copy()
        vmax = max(1.0, float(nonzero[f"{species}_rai"].quantile(0.98))) if not nonzero.empty else 1.0
        sc = ax.scatter(
            nonzero["longitude"],
            nonzero["latitude"],
            c=nonzero[f"{species}_rai"],
            cmap=self._spatial_intensity_cmap(),
            vmin=0,
            vmax=vmax,
            s=np.clip(np.sqrt(nonzero[species]) * 7.5, 18, 105),
            alpha=0.78,
            edgecolors="white",
            linewidths=0.35,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.048, pad=0.025)
        cbar.set_label(f"{species} RAI / 100 camera-days")
        ax.scatter(
            hot_df["longitude"],
            hot_df["latitude"],
            color=SCI_COLORS["orange"],
            s=96,
            edgecolors=SCI_COLORS["text"],
            linewidths=0.9,
            label=f"Top {species} hotspots",
            zorder=4,
        )
        count_by_camera = {str(row.get("camera_id")): int(row.get("count", 0)) for row in hotspots}
        self._annotate_final_demo_points(
            ax,
            hot_df,
            lambda row: f"{row['camera_id']} ({count_by_camera.get(str(row['camera_id']), int(row[species]))})",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{species} intensity across cameras", loc="left", fontweight="bold")
        self._apply_clean_sci_axes(ax, "both")

        ax = axes[1]
        hot_plot = pd.DataFrame(hotspots[:5]).sort_values("count", ascending=True)
        colors = [SCI_COLORS["orange"] if idx == len(hot_plot) - 1 else SCI_COLORS["blue"] for idx in range(len(hot_plot))]
        ax.hlines(np.arange(len(hot_plot)), 0, hot_plot["count"], color=colors, lw=2.4, alpha=0.7)
        ax.scatter(hot_plot["count"], np.arange(len(hot_plot)), s=76, color=colors, edgecolor=SCI_COLORS["text"], zorder=3)
        total_focal = max(int((events["species"] == species).sum()), 1)
        for yi, (_, row) in enumerate(hot_plot.iterrows()):
            count_value = float(row["count"])
            pct = count_value / total_focal * 100.0
            ax.text(count_value + max(float(hot_plot["count"].max()), 1.0) * 0.03, yi, f"{int(count_value)} | {pct:.1f}%", va="center", fontsize=8.0, clip_on=False)
        ax.set_yticks(np.arange(len(hot_plot)))
        ax.set_yticklabels(hot_plot["camera_id"])
        ax.set_xlabel(f"Independent {species} events")
        ax.set_ylabel("Camera ID")
        ax.set_title("Hotspot ranking", loc="left", fontweight="bold")
        ax.set_xlim(0, max(float(hot_plot["count"].max()) * 1.18, 1.0))
        self._apply_clean_sci_axes(ax, "x")

        ax = axes[2]
        edge_df = pd.DataFrame(edge_rows).head(int(top_k))
        if not edge_df.empty:
            count_col = "count" if "count" in edge_df.columns else "transition_count"
            if "camera_pair" not in edge_df.columns:
                edge_df["camera_pair"] = edge_df.apply(
                    lambda row: f"{row.get('from_camera', row.get('camera_a'))}-{row.get('to_camera', row.get('camera_b'))}",
                    axis=1,
                )
            if "distance_km" not in edge_df.columns and "median_distance_km" not in edge_df.columns:
                coord = table.set_index("camera_id")[["latitude", "longitude"]].to_dict("index")
                distances = []
                for _, row in edge_df.iterrows():
                    from_camera = str(row.get("from_camera", row.get("camera_a", "")))
                    to_camera = str(row.get("to_camera", row.get("camera_b", "")))
                    a = coord.get(from_camera, {})
                    b = coord.get(to_camera, {})
                    if all(pd.notna(a.get(k)) and pd.notna(b.get(k)) for k in ["latitude", "longitude"]):
                        distances.append(self.calculate_distance_km(a["latitude"], a["longitude"], b["latitude"], b["longitude"]))
                    else:
                        distances.append(np.nan)
                edge_df["distance_km"] = distances
            distance_col = "distance_km" if "distance_km" in edge_df.columns else "median_distance_km"
            edge_df = edge_df.sort_values(count_col, ascending=True)
            bars = ax.barh(edge_df["camera_pair"], edge_df[count_col], color="#6D8FA9", edgecolor=SCI_COLORS["text"], alpha=0.82)
            for bar, row in zip(bars, edge_df.itertuples()):
                count = int(getattr(row, count_col))
                distance = getattr(row, distance_col, np.nan)
                ax.text(
                    bar.get_width() + max(float(edge_df[count_col].max()), 1.0) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{count} transitions | {self._format_final_demo_distance(distance)}",
                    va="center",
                    fontsize=7.2,
                )
            ax.set_xlim(0, max(float(edge_df[count_col].max()) * 1.6, 1.0))
        ax.set_xlabel("Potential transition count")
        ax.set_ylabel("Camera pair")
        ax.set_title("Potential camera-to-camera transition ranking", loc="left", fontweight="bold")
        self._apply_clean_sci_axes(ax, "x")

        if save_path is None:
            save_path = self._final_demo_output_path(
                f"{self._safe_name(species).lower()}_spatial_movement_reasoning.png"
            )
        return self._save_final_demo_figure(fig, save_path)

    def plot_camera_species_richness_bar(
        self,
        top_k=20,
        min_records_per_species=2,
        min_gap_minutes=30,
        min_rai_per_100_camera_days=1.0,
        save_path=None,
    ):
        richness = self.build_camera_species_richness(
            top_k=top_k,
            min_records_per_species=min_records_per_species,
            min_gap_minutes=min_gap_minutes,
            min_rai_per_100_camera_days=min_rai_per_100_camera_days,
        )
        rows = richness.get("richness_by_camera", [])
        if not rows:
            raise ValueError("No camera species richness data available for plotting.")

        labels = [row["camera_id"] for row in rows][::-1]
        values = [row["species_richness"] for row in rows][::-1]
        totals = [row.get("total_independent_events", row["total_records"]) for row in rows][::-1]
        efforts = [row.get("camera_effort_days", 1) for row in rows][::-1]

        fig, ax = plt.subplots(figsize=(8.6, max(4.4, 0.32 * len(labels) + 1.4)))
        y = np.arange(len(labels))
        ax.hlines(y, 0, values, color=SCI_COLORS["green"], linewidth=1.8, alpha=0.58)
        ax.scatter(values, y, s=56, color=SCI_COLORS["green"], edgecolor="#3F5F36", linewidth=0.8, zorder=3)
        label_pad = max(max(values), 1) * 0.035
        for yi, value, total, effort in zip(y, values, totals, efforts):
            ax.text(
                value + label_pad,
                yi,
                f"{value} spp / {total} events / {effort:g} d",
                va="center",
                ha="left",
                fontsize=7.8,
                clip_on=False,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Filtered species richness")
        ax.set_ylabel("Camera")
        ax.set_title("Camera-level species richness", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(0, max(values) + 1.45)
        self._apply_clean_sci_axes(ax, grid_axis="x")
        if save_path is None:
            save_path = self._build_path(f"camera_species_richness_top{top_k}.png")
        return self._finalize_paper_plot(fig, save_path)

    def _collect_ecology_rows(self, include_unknown=False, min_confidence=None):
        if self._using_camtrap_backend():
            event_df = self._load_camtrap_event_df()
            if event_df.empty:
                return pd.DataFrame()
            df = event_df.copy()
            if "species" not in df.columns:
                df["species"] = df.get("scientificName", "").map(self._canonical_species_name)
            df["species"] = df["species"].map(self._canonical_species_name)
            if not include_unknown:
                df = df[df["species"].str.lower() != "unknown"]
            if "event_time" in df.columns:
                ts = pd.to_datetime(df["event_time"], errors="coerce")
            else:
                ts = pd.to_datetime(df.get("eventStart"), errors="coerce")
            camera_col = "camera_id" if "camera_id" in df.columns else "cameraID"
            rows = pd.DataFrame({
                "record_id": df.get("event_id", df.get("eventID", "")),
                "species": df["species"],
                "camera_id": df[camera_col].astype(str),
                "timestamp": ts,
                "individuals": 1,
                "latitude": pd.to_numeric(df.get("latitude"), errors="coerce"),
                "longitude": pd.to_numeric(df.get("longitude"), errors="coerce"),
                "confidence": None,
            })
            rows["date"] = rows["timestamp"].dt.date
            rows["hour"] = rows["timestamp"].dt.hour
            rows["month"] = rows["timestamp"].dt.month
            return rows

        rows = []
        camera_locations = self.get_camera_locations()
        min_conf = float(min_confidence) if min_confidence is not None else None

        for record in self.get_all_records():
            metadata = record.get("metadata", {}) or {}
            species = self._canonical_species_name(metadata.get("species"))
            if not include_unknown and species.lower() == "unknown":
                continue

            confidence = self._extract_confidence_value(metadata)
            if min_conf is not None and (confidence is None or confidence < min_conf):
                continue

            camera_id = str(metadata.get("camera_id", "")).strip()
            parsed_time = self.safe_parse_timestamp(metadata.get("timestamp"))
            if not camera_id:
                continue

            loc = camera_locations.get(camera_id, {})
            latitude = self._coerce_float(metadata.get("latitude"))
            longitude = self._coerce_float(metadata.get("longitude"))
            if latitude is None:
                latitude = loc.get("latitude")
            if longitude is None:
                longitude = loc.get("longitude")

            individuals = metadata.get("individuals")
            try:
                individuals = max(int(float(individuals)), 1)
            except Exception:
                individuals = 1

            rows.append({
                "record_id": record.get("id"),
                "species": species,
                "camera_id": camera_id,
                "timestamp": parsed_time,
                "date": parsed_time.date() if parsed_time is not None else None,
                "hour": parsed_time.hour if parsed_time is not None else None,
                "month": parsed_time.month if parsed_time is not None else None,
                "individuals": individuals,
                "latitude": latitude,
                "longitude": longitude,
                "confidence": confidence,
            })

        return pd.DataFrame(rows)

    def _camera_effort_days(self, df):
        effort = {}
        if df.empty or "timestamp" not in df:
            return effort
        valid = df.dropna(subset=["timestamp", "camera_id"])
        for camera_id, group in valid.groupby("camera_id"):
            start = group["timestamp"].min()
            end = group["timestamp"].max()
            if pd.isna(start) or pd.isna(end):
                continue
            effort[str(camera_id)] = max((end.date() - start.date()).days + 1, 1)
        return effort

    def plot_species_rank_abundance_curve(self, top_k=12, include_unknown=False, save_path=None):
        composition = self.summarize_community_composition(
            top_k=top_k,
            include_unknown=include_unknown,
            abundance_metric="rai",
        )
        rows = [row for row in composition.get("top_species", []) if float(row.get("metric_value", 0.0)) > 0]
        if not rows:
            raise ValueError("No species composition data available for plotting.")

        labels = [row["species"] for row in rows]
        counts = np.asarray([row.get("metric_value", row.get("rai_per_100_camera_days", 0.0)) for row in rows], dtype=float)
        event_counts = np.asarray([row.get("independent_event_count", row["record_count"]) for row in rows], dtype=float)
        shares = counts / max(float(counts.sum()), 1.0)
        ranks = np.arange(1, len(labels) + 1)

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        ax.plot(ranks, counts, color=SCI_COLORS["blue"], linewidth=2.2, marker="o", markersize=6.0, zorder=3)
        ax.fill_between(ranks, counts, color=SCI_COLORS["blue"], alpha=0.10, zorder=1)
        ax.set_yscale("log" if counts.max() / max(counts.min(), 1.0) >= 10 else "linear")
        if ax.get_yscale() == "log":
            positive_counts = counts[counts > 0]
            ax.set_ylim(max(float(positive_counts.min()) * 0.72, 0.001), float(counts.max()) * 2.15)
        else:
            ax.set_ylim(0, float(counts.max()) * 1.34)
        ax.set_xticks(ranks)
        ax.set_xticklabels(
            [f"{i}. {label}" for i, label in zip(ranks, labels)],
            rotation=35,
            ha="right",
            fontsize=7.0,
        )
        ax.set_xlabel("Species rank")
        ax.set_ylabel("Adjusted RAI")
        self._apply_clean_axis(ax, grid_axis="y")

        for idx, (x, y_value, label, share, event_count) in enumerate(zip(ranks, counts, labels, shares, event_counts)):
            offset = 7 + (idx % 3) * 6
            if idx == 0:
                offset = 6
            ax.annotate(
                f"{y_value:.2f}\n{share:.1%}",
                xy=(x, y_value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.7,
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.78},
                clip_on=False,
                zorder=4,
            )

        self._add_caption_title(
            ax,
            "Community rank-abundance curve",
        )
        self._reserve_figure_space(fig, bottom=0.31)
        if save_path is None:
            save_path = self._build_path(f"community_rank_abundance_top{top_k}.png")
        return self._finalize_paper_plot(fig, save_path)

    def plot_multi_species_monthly_phenology(self, species_list=None, top_k=8, normalize=True, save_path=None):
        selected_species = self._resolve_species_list(species_list=species_list, top_k=top_k, include_unknown=False)
        if not selected_species:
            raise ValueError("No species available for monthly phenology plotting.")

        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(10.4, 5.8))
        for idx, species in enumerate(selected_species):
            monthly = self.build_monthly_activity_distribution(species).get("monthly_distribution", [])
            counts = np.asarray([int(item.get("count", 0)) for item in monthly], dtype=float)
            if counts.sum() <= 0:
                continue
            values = counts / counts.sum() if normalize else counts
            ax.plot(
                np.arange(1, 13),
                values,
                marker="o",
                markersize=4.6,
                linewidth=2.0,
                color=SCI_LINE_PALETTE[idx % len(SCI_LINE_PALETTE)],
                label=f"{species} (n={int(counts.sum())})",
            )

        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(month_labels)
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly share" if normalize else "Detection records")
        if normalize:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
            ax.axhline(1.0 / 12.0, color=SCI_COLORS["grey"], linestyle=":", linewidth=1.0, label="Uniform month")
        self._apply_clean_axis(ax, grid_axis="y")
        ax.legend(frameon=False, ncol=2, loc="best")
        self._reserve_figure_space(fig, top=0.86)
        self._add_caption_title(ax, "Multi-species monthly phenology")
        if save_path is None:
            suffix = "ratio" if normalize else "count"
            save_path = self._build_path(f"multi_species_monthly_phenology_top{top_k}_{suffix}.png")
        return self._finalize_paper_plot(fig, save_path)

    def plot_camera_alpha_diversity_panel(
        self,
        top_k=15,
        min_records_per_species=2,
        min_gap_minutes=30,
        min_rai_per_100_camera_days=1.0,
        save_path=None,
    ):
        richness = self.build_camera_species_richness(
            top_k=max(int(top_k), 1),
            min_records_per_species=min_records_per_species,
            min_gap_minutes=min_gap_minutes,
            min_rai_per_100_camera_days=min_rai_per_100_camera_days,
        )
        rows = richness.get("richness_by_camera", [])
        if not rows:
            raise ValueError("No camera-level diversity data available for plotting.")

        metrics = []
        for row in rows:
            rai_values = list((row.get("species_rai_per_100_camera_days") or {}).values())
            counts = np.asarray(rai_values or list(row.get("species_counts", {}).values()), dtype=float)
            total = float(counts.sum())
            if total <= 0:
                continue
            proportions = counts / total
            shannon = -float(np.sum([p * np.log(p) for p in proportions if p > 0]))
            simpson = 1.0 - float(np.sum(proportions ** 2))
            evenness = shannon / np.log(len(counts)) if len(counts) > 1 else 0.0
            metrics.append({
                "camera_id": row["camera_id"],
                "richness": int(row["species_richness"]),
                "total_records": int(row.get("total_independent_events", row["total_records"])),
                "shannon": shannon,
                "simpson": simpson,
                "evenness": evenness,
            })

        plot_df = pd.DataFrame(metrics).sort_values(["richness", "shannon", "total_records"], ascending=False)
        if plot_df.empty:
            raise ValueError("No alpha diversity metrics available for plotting.")
        plot_df = plot_df.head(max(int(top_k), 1)).iloc[::-1].reset_index(drop=True)

        self._set_paper_style()
        fig, axes = plt.subplots(1, 2, figsize=(12.4, max(4.8, 0.32 * len(plot_df) + 1.4)), sharey=True)
        y = np.arange(len(plot_df))

        ax_a, ax_b = axes
        ax_a.hlines(y, 0, plot_df["richness"], color=SCI_COLORS["green"], linewidth=2.0, alpha=0.72)
        ax_a.scatter(plot_df["richness"], y, s=54, color=SCI_COLORS["green"], edgecolor="#1F5A48", zorder=3)
        ax_a.set_yticks(y)
        ax_a.set_yticklabels(plot_df["camera_id"])
        ax_a.set_xlabel("Filtered species richness")
        ax_a.set_ylabel("Camera ID")
        self._panel_title(ax_a, "A", "Richness")
        self._apply_clean_axis(ax_a, grid_axis="x")

        ax_b.plot(plot_df["shannon"], y, color=SCI_COLORS["blue"], marker="o", linewidth=2.0, label="Shannon")
        ax_b.plot(plot_df["simpson"], y, color=SCI_COLORS["orange"], marker="s", linewidth=1.8, label="Simpson")
        ax_b.plot(plot_df["evenness"], y, color=SCI_COLORS["purple"], marker="^", linewidth=1.8, label="Evenness")
        ax_b.set_xlabel("Index value (RAI-weighted)")
        self._panel_title(ax_b, "B", "Alpha diversity indices")
        self._apply_clean_axis(ax_b, grid_axis="x")
        ax_b.legend(frameon=False, loc="best")
        fig.suptitle("Camera-level alpha diversity", fontsize=13.2, fontweight="bold", y=0.99)
        self._reserve_figure_space(fig, top=0.90, wspace=0.10)

        if save_path is None:
            save_path = self._build_path(f"camera_alpha_diversity_top{top_k}.png")
        return self._finalize_paper_plot(fig, save_path)

    def plot_species_accumulation_curve(self, permutations=120, step_count=28, include_unknown=False, save_path=None):
        selected_species, species_camera_counts, event_times = self._independent_event_counts_by_species_camera(
            include_unknown=include_unknown,
            min_gap_minutes=30,
        )
        species_events = []
        for species_name in selected_species:
            event_count = sum(species_camera_counts.get(species_name, {}).values())
            species_events.extend([species_name] * int(event_count))
        if not species_events:
            raise ValueError("No independent species events available for species accumulation plotting.")

        species = np.asarray(species_events, dtype=object)
        n = len(species)
        steps = np.unique(np.linspace(1, n, min(max(int(step_count), 5), n), dtype=int))
        rng = np.random.default_rng(20260517)
        curves = []
        for _ in range(max(int(permutations), 1)):
            shuffled = rng.permutation(species)
            seen = set()
            values = []
            pointer = 0
            for step in steps:
                while pointer < step:
                    seen.add(shuffled[pointer])
                    pointer += 1
                values.append(len(seen))
            curves.append(values)

        curve_array = np.asarray(curves, dtype=float)
        mean_curve = curve_array.mean(axis=0)
        low_curve = np.percentile(curve_array, 2.5, axis=0)
        high_curve = np.percentile(curve_array, 97.5, axis=0)

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        ax.fill_between(steps, low_curve, high_curve, color=SCI_COLORS["sky"], alpha=0.22, label="95% permutation band")
        ax.plot(steps, mean_curve, color=SCI_COLORS["blue"], linewidth=2.3, label="Mean accumulation")
        ax.scatter([steps[-1]], [mean_curve[-1]], color=SCI_COLORS["orange"], s=70, zorder=4, label="Observed pool")
        ax.set_xlabel("Independent events sampled")
        ax.set_ylabel("Accumulated species richness")
        ax.set_ylim(0, max(high_curve.max(), mean_curve.max()) * 1.16)
        self._apply_clean_axis(ax, grid_axis="y")
        ax.legend(frameon=False, loc="best")
        self._add_caption_title(
            ax,
            "Species accumulation curve",
            subtitle=f"{int(mean_curve[-1])} species reached after {n} independent events; permutations = {max(int(permutations), 1)}.",
        )
        if save_path is None:
            save_path = self._build_path("community_species_accumulation_curve.png")
        return self._finalize_paper_plot(fig, save_path)

    def plot_species_rai_heatmap(self, species_list=None, top_k=8, top_cameras=14, min_gap_minutes=30, save_path=None):
        selected_species = self._resolve_species_list(species_list=species_list, top_k=top_k, include_unknown=False)
        df = self._collect_ecology_rows(include_unknown=False)
        effort_days = self._camera_effort_days(df)
        if not selected_species or not effort_days:
            raise ValueError("No camera effort or species data available for RAI plotting.")

        camera_scores = Counter()
        event_counts = defaultdict(lambda: defaultdict(int))
        if self._using_camtrap_backend():
            event_df = self._load_camtrap_event_df()
            if event_df.empty:
                raise ValueError("No independent events available for RAI plotting.")
            event_df = event_df.copy()
            if "species" not in event_df.columns:
                event_df["species"] = event_df.get("scientificName", "").map(self._canonical_species_name)
            event_df["species"] = event_df["species"].map(self._canonical_species_name)
            camera_col = "camera_id" if "camera_id" in event_df.columns else "cameraID"
            event_df = event_df[event_df["species"].isin(selected_species)]
            grouped = event_df.groupby(["species", camera_col]).size()
            for (species, camera_id), count in grouped.items():
                camera_id = str(camera_id)
                if camera_id and camera_id in effort_days:
                    event_counts[species][camera_id] = int(count)
                    camera_scores[camera_id] += int(count)
        else:
            for species in selected_species:
                events = self.build_independent_events(
                    species=species,
                    by_camera=True,
                    min_gap_minutes=min_gap_minutes,
                ).get("events", [])
                for event in events:
                    camera_id = str(event.get("camera_id", "")).strip()
                    if camera_id and camera_id in effort_days:
                        event_counts[species][camera_id] += 1
                        camera_scores[camera_id] += 1

        cameras = [camera for camera, _ in camera_scores.most_common(max(int(top_cameras), 1))]
        if not cameras:
            raise ValueError("No independent events available for RAI plotting.")

        data = []
        for species in selected_species:
            row = []
            for camera_id in cameras:
                effort = max(float(effort_days.get(camera_id, 1)), 1.0)
                row.append(event_counts[species][camera_id] / effort * 100.0)
            data.append(row)
        data = np.asarray(data, dtype=float)

        self._set_paper_style()
        fig_width = max(8.8, 0.48 * len(cameras) + 3.5)
        fig_height = max(4.8, 0.42 * len(selected_species) + 2.2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        vmax = max(float(np.nanmax(data)), 0.01)
        im = ax.imshow(data, cmap=self._spatial_intensity_cmap(), vmin=0, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("RAI: independent events / 100 camera-days")
        ax.set_xticks(np.arange(len(cameras)))
        ax.set_yticks(np.arange(len(selected_species)))
        ax.set_xticklabels(cameras, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticklabels(selected_species, fontsize=8.2)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if value <= 0:
                    continue
                color = self._heatmap_text_color(value, 0, vmax)
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.5, color=color)
        ax.tick_params(length=0)
        self._add_caption_title(ax, "Relative abundance index by camera")
        self._reserve_figure_space(fig, top=0.90, bottom=0.18)
        if save_path is None:
            save_path = self._build_path(f"species_rai_heatmap_top{top_k}_cameras{top_cameras}.png")
        return self._finalize_paper_plot(fig, save_path)

    def plot_temporal_overlap_matrix(self, species_list=None, top_k=6, min_gap_minutes=30, save_path=None):
        selected_species = self._resolve_species_list(species_list=species_list, top_k=top_k, include_unknown=False)
        if len(selected_species) < 2:
            raise ValueError("At least two species are needed for temporal overlap plotting.")

        n = len(selected_species)
        data = np.full((n, n), np.nan, dtype=float)
        np.fill_diagonal(data, 1.0)
        for i in range(n):
            for j in range(i + 1, n):
                overlap = self.estimate_activity_overlap(
                    selected_species[i],
                    selected_species[j],
                    min_gap_minutes=min_gap_minutes,
                    bootstrap_iters=0,
                    permutation_iters=0,
                )
                value = overlap.get("delta_hat")
                if value is None:
                    value = np.nan
                data[i, j] = data[j, i] = value

        self._set_paper_style()
        fig_size = max(6.4, 0.62 * n + 2.6)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        cmap = self._fresh_heatmap_cmap()
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Diel overlap index Delta")
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(selected_species, rotation=40, ha="right", fontsize=8.0)
        ax.set_yticklabels(selected_species, fontsize=8.0)
        for i in range(n):
            for j in range(n):
                value = data[i, j]
                if not np.isfinite(value):
                    continue
                color = self._heatmap_text_color(value, 0, 1)
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.2, color=color)
        ax.tick_params(length=0)
        self._add_caption_title(ax, "Temporal niche overlap matrix")
        if save_path is None:
            save_path = self._build_path(f"temporal_niche_overlap_top{top_k}.png")
        return self._finalize_paper_plot(fig, save_path)

    def plot_activity_overlap_triptych(
        self,
        reference_species=None,
        compare_species_list=None,
        top_k=3,
        min_gap_minutes=30,
        smooth_sigma=1.25,
        bootstrap_iters=500,
        save_path=None,
    ):
        if not reference_species or not str(reference_species).strip():
            raise ValueError("reference_species is required for an activity-overlap triptych.")
        reference_species = self._canonical_species_name(reference_species)
        if compare_species_list is None:
            candidates = self._resolve_species_list(top_k=max(int(top_k) + 3, 4), include_unknown=False)
            compare_species = [species for species in candidates if species != reference_species][:max(int(top_k), 1)]
        else:
            compare_species = [
                self._canonical_species_name(species)
                for species in self._resolve_species_list(compare_species_list, top_k=top_k, include_unknown=False)
            ]
            compare_species = [species for species in compare_species if species != reference_species]

        if not compare_species:
            raise ValueError("No comparison species available for activity overlap triptych.")

        panels = []
        for species in compare_species:
            result = self.estimate_activity_overlap(
                reference_species,
                species,
                min_gap_minutes=min_gap_minutes,
                smooth_sigma=smooth_sigma,
                bootstrap_iters=bootstrap_iters,
                permutation_iters=0,
            )
            if result.get("hour_grid"):
                panels.append((species, result))

        if not panels:
            raise ValueError("No valid activity overlap panels could be generated.")

        self._set_paper_style()
        ncols = len(panels)
        fig, axes = plt.subplots(1, ncols, figsize=(4.05 * ncols, 3.05), sharey=True, squeeze=False)
        axes = axes.ravel()

        for idx, (ax, (species, result)) in enumerate(zip(axes, panels)):
            hour_grid = np.asarray(result["hour_grid"], dtype=float)
            density_ref = np.asarray(result["density_a"], dtype=float)
            density_cmp = np.asarray(result["density_b"], dtype=float)
            overlap_density = np.asarray(result["overlap_density"], dtype=float)

            ax.fill_between(hour_grid, overlap_density, color="#C8C8C8", alpha=0.68, zorder=1)
            ax.plot(hour_grid, density_ref, color="#222222", linewidth=1.18, zorder=3, label=reference_species)
            ax.plot(hour_grid, density_cmp, color="#222222", linewidth=1.18, linestyle=(0, (3, 2)), zorder=3, label=species)
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.set_xticklabels(["Midnight", "Sunrise", "Noon", "Sunset", "Midnight"], fontsize=7.1)
            if idx == 0:
                ax.set_ylabel("Density of activity")
            ax.set_title(f"{reference_species} - {species}", fontsize=9.2, fontweight="bold", pad=7)
            ax.tick_params(axis="both", length=3.0, width=0.7)
            ax.grid(False)

            delta = result.get("delta_hat")
            ci_low = result.get("delta_ci_low")
            ci_high = result.get("delta_ci_high")
            if delta is not None and ci_low is not None and ci_high is not None:
                ax.text(
                    0.03,
                    0.96,
                    f"Delta = {delta:.2f} ({ci_low:.2f}-{ci_high:.2f})",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7.1,
                )

        fig.supxlabel("Time of day", y=0.035)
        self._reserve_figure_space(fig, bottom=0.18, top=0.86, wspace=0.18)

        if save_path is None:
            save_path = self._build_path(
                f"{self._safe_name(reference_species)}_activity_overlap_triptych_top{len(panels)}.png"
            )
        return self._finalize_paper_plot(fig, save_path)

    def plot_multi_species_occupancy_forest(
        self,
        species_list=None,
        top_k=5,
        occasion_days=14,
        min_gap_minutes=30,
        min_active_occasions=2,
        bootstrap_iters=120,
        save_path=None,
    ):
        selected_species = self._resolve_species_list(
            species_list=species_list,
            top_k=top_k,
            include_unknown=False,
        )
        if not selected_species:
            raise ValueError("No species available for multi-species occupancy forest plot.")

        rows = []
        for species in selected_species:
            try:
                samples = self._collect_occupancy_bootstrap_samples(
                    species=species,
                    occasion_days=occasion_days,
                    min_gap_minutes=min_gap_minutes,
                    min_active_occasions=min_active_occasions,
                    bootstrap_iters=bootstrap_iters,
                )
            except Exception:
                continue
            model = samples.get("model_result", {}) or {}
            psi_samples = np.asarray(samples.get("psi_samples", []), dtype=float)
            p_samples = np.asarray(samples.get("p_samples", []), dtype=float)
            if psi_samples.size == 0 or p_samples.size == 0:
                continue
            psi_hat = model.get("occupancy_probability_psi", model.get("psi_hat"))
            p_hat = model.get("detection_probability_p", model.get("p_hat"))
            if psi_hat is None or p_hat is None:
                continue
            rows.append({
                "species": species,
                "psi": float(psi_hat),
                "psi_low": float(np.percentile(psi_samples, 2.5)),
                "psi_high": float(np.percentile(psi_samples, 97.5)),
                "p": float(p_hat),
                "p_low": float(np.percentile(p_samples, 2.5)),
                "p_high": float(np.percentile(p_samples, 97.5)),
                "sites": int(model.get("surveyed_camera_count", model.get("site_count", 0)) or 0),
            })

        plot_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["psi", "p"])
        if plot_df.empty:
            raise ValueError("No occupancy estimates available for plotting.")
        plot_df = plot_df.sort_values("psi", ascending=True).reset_index(drop=True)

        self._set_paper_style()
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(9.8, max(3.8, 0.46 * len(plot_df) + 1.4)),
            sharey=True,
        )
        y = np.arange(len(plot_df))
        configs = [
            (axes[0], "psi", "psi_low", "psi_high", "Occupancy probability (psi)", "a"),
            (axes[1], "p", "p_low", "p_high", "Detection probability (p)", "b"),
        ]

        for ax, value_col, low_col, high_col, title, panel_label in configs:
            for idx, row in plot_df.iterrows():
                estimate = float(row[value_col])
                low = float(row[low_col])
                high = float(row[high_col])
                color = SCI_COLORS["brick"] if idx == len(plot_df) - 1 else "#222222"
                ax.hlines(idx, low, high, color=color, linewidth=1.05, zorder=2)
                ax.scatter(estimate, idx, s=26, color=color, edgecolors="white", linewidths=0.55, zorder=3)
            ax.axvline(0.0, color="#222222", linestyle=":", linewidth=0.8, zorder=1)
            ax.set_xlim(0, 1.02)
            ax.set_xlabel("Estimated probability")
            ax.set_title(title, loc="left", fontsize=11.0, fontweight="bold", pad=8)
            self._apply_clean_axis(ax, grid_axis="x")

        axes[0].set_yticks(y)
        axes[0].set_yticklabels([f"{row.species}  (sites={int(row.sites)})" for row in plot_df.itertuples()], fontsize=8.1)
        axes[1].tick_params(axis="y", labelleft=False)
        fig.suptitle(
            f"Multi-species occupancy model estimates ({occasion_days}-day occasions)",
            fontsize=12.4,
            fontweight="bold",
            y=0.99,
        )
        self._reserve_figure_space(fig, top=0.84, bottom=0.14, wspace=0.12)

        if save_path is None:
            save_path = self._build_path(
                f"multi_species_occupancy_forest_top{len(plot_df)}_{occasion_days}d.png"
            )
        return self._finalize_paper_plot(fig, save_path)

    def plot_species_cooccurrence_heatmap(
        self,
        species_list=None,
        top_k=10,
        min_camera_records=1,
        min_gap_minutes=30,
        metric="jaccard",
        save_path=None,
    ):
        result = self.build_species_cooccurrence_matrix(
            species_list=species_list,
            top_k=top_k,
            min_camera_records=min_camera_records,
            min_gap_minutes=min_gap_minutes,
        )
        species = result.get("species", [])
        if not species:
            raise ValueError("No species co-occurrence data available for plotting.")

        metric = metric if metric in {"jaccard", "shared_camera_count"} else "jaccard"
        matrix = []
        for row in result.get("matrix", []):
            matrix.append([value[metric] for value in row.get("values", [])])
        data = np.asarray(matrix, dtype=float)

        plot_data = data.copy()
        if metric == "jaccard":
            np.fill_diagonal(plot_data, np.nan)
        finite_values = plot_data[np.isfinite(plot_data)]
        vmax = float(np.nanmax(finite_values)) if finite_values.size else 1.0
        if metric == "jaccard":
            vmax = max(vmax, 0.01)
        cmap = self._spatial_intensity_cmap()

        fig_size = max(5.8, 0.55 * len(species) + 2.0)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(plot_data, cmap=cmap, vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard overlap" if metric == "jaccard" else "Shared cameras")
        ax.set_xticks(range(len(species)))
        ax.set_yticks(range(len(species)))
        ax.set_xticklabels(species, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(species, fontsize=8)
        for i in range(len(species)):
            for j in range(len(species)):
                if metric == "jaccard" and i == j:
                    continue
                value = data[i, j]
                label = f"{value:.2f}" if metric == "jaccard" else str(int(value))
                text_color = self._heatmap_text_color(plot_data[i, j], 0, vmax)
                ax.text(j, i, label, ha="center", va="center", fontsize=7, color=text_color)
        ax.set_title("Camera-level species co-occurrence", fontsize=12, fontweight="bold", pad=8)
        ax.tick_params(length=0)
        if save_path is None:
            save_path = self._build_path(f"species_cooccurrence_{metric}_top{top_k}.png")
        return self._finalize_paper_plot(fig, save_path)

    def plot_species_relationship_network(
        self,
        species_list=None,
        top_k=10,
        min_camera_records=1,
        min_shared_cameras=1,
        min_gap_minutes=30,
        temporal_window_hours=24,
        include_temporal_overlap=True,
        max_pairs=12,
        save_path=None,
    ):
        result = self.summarize_species_relationship_network(
            species_list=species_list,
            top_k=top_k,
            min_camera_records=min_camera_records,
            min_shared_cameras=min_shared_cameras,
            min_gap_minutes=min_gap_minutes,
            temporal_window_hours=temporal_window_hours,
            include_temporal_overlap=include_temporal_overlap,
            max_pairs=max_pairs,
        )
        nodes = result.get("nodes", [])
        edges = result.get("edges", [])
        if not nodes:
            raise ValueError("No species relationship network data available for plotting.")

        species_names = [node["species"] for node in nodes]
        count_by_species = {node["species"]: node.get("record_count", 0) for node in nodes}
        camera_by_species = {node["species"]: node.get("camera_count", 0) for node in nodes}
        angle_grid = np.linspace(0, 2 * np.pi, len(species_names), endpoint=False)
        positions = {
            species: (np.cos(angle), np.sin(angle))
            for species, angle in zip(species_names, angle_grid)
        }

        fig, ax = plt.subplots(figsize=(8.8, 7.8))
        ax.set_aspect("equal")
        ax.axis("off")

        if edges:
            max_score = max(edge.get("relationship_score", 0.0) for edge in edges) or 1.0
            for edge in edges:
                species_a = edge["species_a"]
                species_b = edge["species_b"]
                if species_a not in positions or species_b not in positions:
                    continue
                x1, y1 = positions[species_a]
                x2, y2 = positions[species_b]
                score = edge.get("relationship_score", 0.0)
                width = 0.6 + 4.2 * (score / max_score)
                alpha = 0.28 + 0.50 * (score / max_score)
                color = SCI_COLORS["blue"] if edge.get("temporal_overlap_level") in {"high", "moderate"} else SCI_COLORS["grey"]
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=alpha, zorder=1)

        max_records = max(count_by_species.values()) if count_by_species else 1
        for species in species_names:
            x, y = positions[species]
            record_count = count_by_species.get(species, 0)
            camera_count = camera_by_species.get(species, 0)
            size = 420 + 1600 * ((record_count / max_records) ** 0.5 if max_records else 0)
            ax.scatter(
                [x],
                [y],
                s=size,
                color=SCI_COLORS["yellow"],
                edgecolor=SCI_COLORS["dark"],
                linewidth=1.0,
                zorder=3,
            )
            label_radius = 1.16
            ha = "left" if x >= 0 else "right"
            ax.text(
                x * label_radius,
                y * label_radius,
                f"{species}\n{record_count} events / {camera_count} cams",
                ha=ha,
                va="center",
                fontsize=8,
                clip_on=False,
                zorder=4,
            )

        ax.text(
            0,
            1.24,
            "Species relationship network",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

        if save_path is None:
            save_path = self._build_path(f"species_relationship_network_top{top_k}.png")
        return self._finalize_paper_plot(fig, save_path)

    def plot_multi_species_hourly_activity(
        self,
        species_list=None,
        top_k=6,
        min_gap_minutes=30,
        save_path=None,
    ):
        result = self.compare_multi_species_hourly_activity(
            species_list=species_list,
            top_k=top_k,
            min_gap_minutes=min_gap_minutes,
        )
        profiles = [row for row in result.get("profiles", []) if row.get("hourly_distribution")]
        if not profiles:
            raise ValueError("No multi-species hourly activity data available for plotting.")

        fig, ax = plt.subplots(figsize=(10.2, 5.3))
        color_cycle = SCI_LINE_PALETTE
        hours = np.arange(24)
        for idx, profile in enumerate(profiles):
            color = color_cycle[idx % len(color_cycle)]
            ratios = [
                item.get("ratio", 0.0)
                for item in sorted(profile["hourly_distribution"], key=lambda item: item.get("hour", 0))
            ]
            smooth_x, smooth_y = self._smooth_circular_series(ratios, sigma=1.15, resolution=240)
            ax.plot(
                smooth_x,
                smooth_y,
                marker=None,
                linewidth=1.85,
                color=color,
                label=f"{profile['species']} (n={profile.get('valid_time_records', 0)})",
            )
            ax.scatter(hours + 0.5, ratios, s=9, color=color, alpha=0.42, edgecolors="none", zorder=3)

        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(["Midnight", "Sunrise", "Noon", "Sunset", "Midnight"])
        ax.set_xlabel("Time of day")
        ax.set_ylabel("Relative activity")
        ax.set_title("Multi-species diel activity profiles", fontsize=12, fontweight="bold", pad=8)
        self._apply_clean_sci_axes(ax, grid_axis="y")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
        if save_path is None:
            save_path = self._build_path(f"multi_species_hourly_activity_top{top_k}.png")
        return self._finalize_paper_plot(fig, save_path)

    def plot_community_overview_panel(
        self,
        species_list=None,
        top_k=8,
        min_camera_records=1,
        min_records_per_species=2,
        min_gap_minutes=30,
        min_rai_per_100_camera_days=1.0,
        save_path=None,
    ):
        """
        Publication-style multi-panel figure: community composition, richness hotspots, co-occurrence matrix, and multi-species day-night activity.
        """
        top_k = max(int(top_k), 3)
        if self._using_camtrap_backend():
            return self.plot_community_spatial_association_panel(
                species_list=species_list,
                top_k=top_k,
                save_path=save_path,
            )
        composition = self.summarize_community_composition(
            top_k=top_k,
            include_unknown=False,
            abundance_metric="rai",
        )
        composition_rows = [
            row for row in composition.get("top_species", [])
            if float(row.get("metric_value", 0.0)) > 0
        ]
        if species_list is None:
            selected_species = [row["species"] for row in composition_rows[:top_k]]
        else:
            selected_species = self._resolve_species_list(
                species_list=species_list,
                top_k=top_k,
                include_unknown=False,
            )
        richness = self.build_camera_species_richness(
            top_k=max(top_k, 8),
            min_records_per_species=max(int(min_records_per_species), 2),
            min_gap_minutes=min_gap_minutes,
            min_rai_per_100_camera_days=max(float(min_rai_per_100_camera_days), 1.0),
        )
        cooccurrence = self.build_species_cooccurrence_matrix(
            species_list=selected_species,
            top_k=top_k,
            min_camera_records=min_camera_records,
            min_gap_minutes=min_gap_minutes,
        )
        activity = self.compare_multi_species_hourly_activity(
            species_list=selected_species,
            top_k=top_k,
            min_gap_minutes=min_gap_minutes,
        )

        plt.rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 9,
        })
        fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.2))
        ax_a, ax_b, ax_c, ax_d = axes.flatten()

        top_species = composition_rows[:top_k]
        labels = [row["species"] for row in top_species][::-1]
        counts = [row.get("metric_value", row.get("rai_per_100_camera_days", 0.0)) for row in top_species][::-1]
        event_counts = [row.get("independent_event_count", row["record_count"]) for row in top_species][::-1]
        colors = [self._palette_color(idx, SCI_BAR_PALETTE) for idx in range(len(labels))]
        ax_a.barh(labels, counts, color=colors, edgecolor="#3f464d", linewidth=0.55, alpha=0.78)
        ax_a.set_xlabel("Adjusted RAI")
        self._panel_title(ax_a, "A", "Focal mammal composition")
        self._apply_clean_sci_axes(ax_a, grid_axis="x")
        for y, (value, event_count) in enumerate(zip(counts, event_counts)):
            ax_a.text(
                value + max(counts or [1]) * 0.025,
                y,
                f"{value:.2f} | n={int(event_count)}",
                va="center",
                fontsize=8,
                clip_on=False,
            )
        ax_a.set_xlim(0, max(float(max(counts or [1])), 1.0) * 1.46)

        richness_rows = richness.get("richness_by_camera", [])[:max(top_k, 8)]
        cam_labels = [row["camera_id"] for row in richness_rows][::-1]
        richness_values = [row["species_richness"] for row in richness_rows][::-1]
        richness_events = [row.get("total_independent_events", row.get("total_records", 0)) for row in richness_rows][::-1]
        richness_rai = [row.get("total_rai_per_100_camera_days", 0.0) for row in richness_rows][::-1]
        y_cam = np.arange(len(cam_labels))
        ax_b.hlines(y_cam, 0, richness_values, color=SCI_COLORS["green"], linewidth=1.65, alpha=0.56)
        ax_b.scatter(richness_values, y_cam, color=SCI_COLORS["green"], edgecolor="#3F5F36", s=32, zorder=3)
        ax_b.set_yticks(y_cam)
        ax_b.set_yticklabels(cam_labels)
        ax_b.set_xlabel("Focal-mammal richness")
        self._panel_title(ax_b, "B", "Focal-mammal richness hotspots")
        self._apply_clean_sci_axes(ax_b, grid_axis="x")
        ax_b.set_xlim(0, max(richness_values or [1]) + 1.0)
        for y, value, event_count, rai_value in zip(y_cam, richness_values, richness_events, richness_rai):
            ax_b.text(
                value + 0.12,
                y,
                f"{value} spp / n={event_count} / RAI={rai_value:.1f}",
                va="center",
                fontsize=7,
                clip_on=False,
            )

        species = cooccurrence.get("species", [])
        matrix = []
        for row in cooccurrence.get("matrix", []):
            matrix.append([value["jaccard"] for value in row.get("values", [])])
        data = np.asarray(matrix, dtype=float)
        if data.size:
            np.fill_diagonal(data, 0.0)
            vmax = max(float(np.nanmax(data)), 0.01)
            im = ax_c.imshow(data, cmap=self._spatial_intensity_cmap(), vmin=0, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.08)
            cbar.set_label("Jaccard", labelpad=12)
            ax_c.set_xticks(range(len(species)))
            ax_c.set_yticks(range(len(species)))
            ax_c.set_xticklabels(species, rotation=45, ha="right", fontsize=7)
            ax_c.set_yticklabels(species, fontsize=7)
            for i in range(len(species)):
                for j in range(len(species)):
                    if i == j:
                        continue
                    text_color = self._heatmap_text_color(data[i, j], 0, vmax)
                    ax_c.text(j, i, f"{data[i, j]:.02f}", ha="center", va="center", fontsize=6, color=text_color)
        self._panel_title(ax_c, "C", "Camera co-occurrence")
        ax_c.tick_params(length=0)

        profiles = [row for row in activity.get("profiles", []) if row.get("hourly_distribution")]
        for idx, profile in enumerate(profiles):
            color = SCI_LINE_PALETTE[idx % len(SCI_LINE_PALETTE)]
            ratios = [
                item.get("ratio", 0.0)
                for item in sorted(profile["hourly_distribution"], key=lambda item: item.get("hour", 0))
            ]
            smooth_x, smooth_y = self._smooth_circular_series(ratios, sigma=1.15, resolution=180)
            ax_d.plot(smooth_x, smooth_y, linewidth=1.45, color=color, label=profile["species"])
        ax_d.set_xlim(0, 24)
        ax_d.set_xticks([0, 6, 12, 18, 24])
        ax_d.set_xticklabels(["Midnight", "Sunrise", "Noon", "Sunset", "Midnight"], fontsize=7)
        ax_d.set_xlabel("Time of day")
        ax_d.set_ylabel("Relative frequency", labelpad=10)
        self._panel_title(ax_d, "D", "Diel activity profiles")
        self._apply_clean_sci_axes(ax_d, grid_axis="y")
        ax_d.legend(frameon=False, fontsize=7, ncol=2, loc="best")

        fig.suptitle("Multi-species camera-trap ecology overview", fontsize=14, fontweight="bold", y=0.995)
        if save_path is None:
            save_path = self._build_path(f"community_ecology_overview_top{top_k}.png")
        self._reserve_figure_space(fig, top=0.93, hspace=0.48, wspace=0.38)
        return self._finalize_paper_plot(fig, save_path)

    def plot_species_pair_relationship_panel(
        self,
        species_a,
        species_b,
        min_camera_records=1,
        min_gap_minutes=30,
        temporal_window_hours=24,
        smooth_sigma=1.25,
        bootstrap_iters=500,
        permutation_iters=0,
        save_path=None,
    ):
        """
        Publication-style two-species relationship panel: spatial co-occurrence, activity overlap, near-time same-camera events, and metric summary.
        """
        species_a = self._canonical_species_name(species_a)
        species_b = self._canonical_species_name(species_b)
        relationship = self.analyze_species_pair_relationship(
            species_a=species_a,
            species_b=species_b,
            min_camera_records=min_camera_records,
            min_gap_minutes=min_gap_minutes,
            temporal_window_hours=temporal_window_hours,
            smooth_sigma=smooth_sigma,
            bootstrap_iters=bootstrap_iters,
            permutation_iters=permutation_iters,
        )
        overlap = self.estimate_activity_overlap(
            species_a=species_a,
            species_b=species_b,
            min_gap_minutes=min_gap_minutes,
            smooth_sigma=smooth_sigma,
            bootstrap_iters=bootstrap_iters,
            permutation_iters=permutation_iters,
        )

        plt.rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 9,
        })
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8))
        ax_a, ax_b, ax_c, ax_d = axes.flatten()

        spatial_labels = [species_a, species_b, "Shared cameras"]
        spatial_values = [
            relationship.get("camera_count_a", 0),
            relationship.get("camera_count_b", 0),
            relationship.get("shared_camera_count", 0),
        ]
        ax_a.bar(
            spatial_labels,
            spatial_values,
            color=[SCI_COLORS["blue"], SCI_COLORS["orange"], SCI_COLORS["yellow"]],
            edgecolor=SCI_COLORS["dark"],
            linewidth=0.6,
            alpha=0.88,
        )
        ax_a.set_ylabel("Camera count")
        self._panel_title(ax_a, "A", "Spatial camera overlap")
        ax_a.tick_params(axis="x", labelrotation=18)
        self._apply_clean_sci_axes(ax_a, grid_axis="y")
        for i, value in enumerate(spatial_values):
            ax_a.text(i, value + max(spatial_values or [1]) * 0.02, str(value), ha="center", fontsize=8)
        ax_a.text(
            0.03,
            0.90,
            f"Jaccard={relationship.get('camera_jaccard')}",
            transform=ax_a.transAxes,
            fontsize=9,
            color="#333333",
        )

        hours = np.asarray(overlap.get("hour_grid", []), dtype=float)
        density_a = np.asarray(overlap.get("density_a", []), dtype=float)
        density_b = np.asarray(overlap.get("density_b", []), dtype=float)
        overlap_density = np.asarray(overlap.get("overlap_density", []), dtype=float)
        if len(hours) and len(density_a) and len(density_b):
            ax_b.fill_between(hours, overlap_density, color=SCI_COLORS["yellow"], alpha=0.28, label="Overlap")
            ax_b.plot(hours, density_a, color=SCI_COLORS["blue"], linewidth=2.1, label=species_a)
            ax_b.plot(hours, density_b, color=SCI_COLORS["orange"], linewidth=2.1, label=species_b)
        ax_b.set_xlim(0, 24)
        ax_b.set_xticks(range(0, 25, 4))
        ax_b.set_xlabel("Hour of day")
        ax_b.set_ylabel("Density")
        self._panel_title(ax_b, "B", "Diel activity overlap")
        self._apply_clean_sci_axes(ax_b, grid_axis="y")
        ax_b.legend(frameon=False, fontsize=8, loc="best")
        delta = overlap.get("delta_hat")
        ci_low = overlap.get("delta_ci_low")
        ci_high = overlap.get("delta_ci_high")
        ci_text = f"Delta={delta}"
        if ci_low is not None and ci_high is not None:
            ci_text += f" [{ci_low}, {ci_high}]"
        ax_b.text(0.03, 0.90, ci_text, transform=ax_b.transAxes, fontsize=8, color="#333333")

        proximity = relationship.get("same_camera_temporal_proximity", {}) or {}
        samples = proximity.get("nearest_event_samples", []) or []
        gaps = [float(item.get("gap_hours", 0.0)) for item in samples]
        if gaps:
            bins = np.linspace(0, max(float(temporal_window_hours), max(gaps), 1.0), min(10, max(4, len(gaps) + 1)))
            ax_c.hist(gaps, bins=bins, color=SCI_COLORS["purple"], edgecolor="#552A5F", alpha=0.86)
            ax_c.set_xlabel("Nearest same-camera gap (hours)")
            ax_c.set_ylabel("Event pairs")
            self._apply_clean_sci_axes(ax_c, grid_axis="y")
        else:
            ax_c.axis("off")
            ax_c.text(0.5, 0.56, "No near-time same-camera pairs", ha="center", va="center", fontsize=12)
            ax_c.text(
                0.5,
                0.42,
                f"Window: {temporal_window_hours} h; shared cameras with events: "
                f"{proximity.get('shared_camera_count_with_events', 0)}",
                ha="center",
                va="center",
                fontsize=9,
                color="#555555",
            )
        self._panel_title(ax_c, "C", "Near-time same-camera events")
        ax_c.text(
            0.03,
            0.90,
            f"{proximity.get('proximate_event_pair_count', 0)} pairs / {proximity.get('camera_count_with_proximity', 0)} cameras",
            transform=ax_c.transAxes,
            fontsize=8,
            color="#333333",
        )

        ax_d.axis("off")
        summary_lines = [
            f"{species_a} independent events: {relationship.get('record_count_a')}",
            f"{species_b} independent events: {relationship.get('record_count_b')}",
            f"Shared cameras: {relationship.get('shared_camera_count')}",
            f"Camera Jaccard: {relationship.get('camera_jaccard')}",
            f"Activity Delta: {overlap.get('delta_hat')}",
            (
                "Near-time pairs / cameras: "
                f"{proximity.get('proximate_event_pair_count', 0)} / "
                f"{proximity.get('camera_count_with_proximity', 0)}"
            ),
            f"Median nearest gap (h): {proximity.get('median_nearest_gap_hours')}",
            f"Peak hours {species_a}: {overlap.get('peak_hours_a', [])[:3]}",
            f"Peak hours {species_b}: {overlap.get('peak_hours_b', [])[:3]}",
        ]
        ax_d.text(
            0.02,
            0.96,
            "Descriptive relationship metrics",
            transform=ax_d.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        ax_d.text(
            0.02,
            0.84,
            "\n".join(summary_lines),
            transform=ax_d.transAxes,
            fontsize=9,
            va="top",
            linespacing=1.55,
            family="monospace",
        )
        fig.suptitle(f"{species_a} vs {species_b}: spatial-temporal relationship", fontsize=14, fontweight="bold", y=0.995)
        if save_path is None:
            save_path = self._build_path(
                f"{self._safe_name(species_a)}_vs_{self._safe_name(species_b)}_relationship_panel.png"
            )
        self._reserve_figure_space(fig, top=0.94, hspace=0.34, wspace=0.26)
        return self._finalize_paper_plot(fig, save_path)

    def _build_html_path(self, filename: str) -> str:
        path = self.output_dir / filename
        return str(path)

    def _load_park_geojson(self):
        if self._park_geojson_cache is not None:
            return self._park_geojson_cache

        if gpd is None:
            return None

        shp_path = Path(self.park_shapefile_path)
        if not shp_path.exists():
            return None

        try:
            gdf = gpd.read_file(shp_path)
            if getattr(gdf, "crs", None) is not None:
                gdf = gdf.to_crs(epsg=4326)
            self._park_geojson_cache = gdf.to_json()
            return self._park_geojson_cache
        except Exception:
            return None

    def _create_base_map(self, center_lat: float, center_lon: float, zoom_start: int = 10):
        if folium is None:
            raise ImportError("folium is not installed, so an interactive HTML map cannot be generated.")

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles="OpenStreetMap",
            overlay=True,
            control_scale=True,
            detect_retina=True,
            max_zoom=14,
        )
        return m

    def _add_park_boundary(self, m):
        geojson_data = self._load_park_geojson()
        if not geojson_data:
            return

        folium.GeoJson(
            geojson_data,
            name="Northeast China Tiger and Leopard National Park",
            style_function=lambda x: {
                "color": SCI_COLORS["accent"],
                "weight": 2,
                "fillColor": "transparent",
                "fillOpacity": 0,
            }
        ).add_to(m)

    def _add_map_controls(self, m):
        if folium is None or MousePosition is None or MeasureControl is None:
            raise ImportError("folium plugins are not installed, so interactive HTML map controls cannot be generated.")

        MousePosition(
            position="bottomright",
            separator=" | ",
            empty_string="Coordinates unavailable"
        ).add_to(m)

        MeasureControl(
            position="bottomleft",
            primary_length_unit="meters"
        ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        map_style = """
        <style>
            .leaflet-control-attribution {
                background-color: rgba(255,255,255,0.8) !important;
                font-size: 10px !important;
            }
            .leaflet-tooltip {
                background-color: rgba(255,255,255,0.95) !important;
                border: 1px solid #ddd !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            }
        </style>
        """
        m.get_root().html.add_child(folium.Element(map_style))

    def _save_map(self, m, save_path: str):
        m.save(save_path)
        return save_path

    def _fit_map_to_points(self, m, points):
        valid_points = [(lat, lon) for lat, lon in points if lat is not None and lon is not None]
        if not valid_points:
            return
        m.fit_bounds(valid_points)

    def _count_to_radius(self, count: int, base: float = 4.0, scale: float = 1.2, max_radius: float = 18.0):
        return min(max_radius, base + scale * (count ** 0.5))

    def _count_to_line_weight(self, count: int, max_count: int):
        if max_count <= 0:
            return 2
        return 2 + 6 * (count / max_count)

    def _add_default_legend(self, m, title="Legend", items=None):
        items = items or []
        rows = []
        for label, color in items:
            rows.append(f'''
            <div style="display:flex; align-items:center; margin-bottom:4px;">
                <div style="width:16px; height:16px; background-color:{color};
                            border:2px solid white; border-radius:50%; margin-right:8px;"></div>
                <span>{label}</span>
            </div>
            ''')

        legend_html = f'''
        <div style="
            position: fixed;
            bottom: 30px;
            left: 30px;
            z-index: 1000;
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            font-family: Arial, sans-serif;
            font-size: 12px;
        ">
            <h4 style="margin: 0 0 8px 0; color: #2c3e50;">{title}</h4>
            {''.join(rows)}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
    # =========================
    # Publication-style static figure helpers
    # =========================
    def _load_park_gdf(self):
        if gpd is None:
            return None

        shp_path = Path(self.park_shapefile_path)
        if not shp_path.exists():
            return None

        try:
            gdf = gpd.read_file(shp_path)
            if gdf.empty:
                return None
            if getattr(gdf, "crs", None) is not None:
                gdf = gdf.to_crs(epsg=4326)
            return gdf
        except Exception:
            return None

    def _set_paper_style(self):
        """Apply a restrained, publication-oriented matplotlib style."""
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 9.4,
            "axes.titlesize": 11.4,
            "axes.labelsize": 10,
            "axes.labelpad": 7,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "figure.titlesize": 12.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": SCI_COLORS["axis"],
            "axes.labelcolor": SCI_COLORS["text"],
            "axes.linewidth": 0.75,
            "xtick.color": SCI_COLORS["axis"],
            "ytick.color": SCI_COLORS["axis"],
            "text.color": SCI_COLORS["text"],
            "grid.color": SCI_COLORS["grid"],
            "grid.linewidth": 0.52,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.55,
        })

    def _clean_caption_text(self, text):
        if text is None:
            return None
        text = str(text).strip()
        if not text:
            return None
        return re.sub(r"\s+", " ", text)

    def _wrap_caption_text(self, text, width=96):
        text = self._clean_caption_text(text)
        if not text:
            return None
        return textwrap.fill(text, width=max(int(width), 30))

    def _format_plot_value(self, value, digits=3, missing="NA"):
        try:
            numeric = float(value)
        except Exception:
            return missing
        if not np.isfinite(numeric):
            return missing
        return f"{numeric:.{int(digits)}f}"

    def _reserve_figure_space(self, fig, *, bottom=None, top=None, hspace=None, wspace=None):
        """Record layout reservations so final saving does not crush captions."""
        layout = getattr(fig, "_agenttool_layout", {}).copy()
        if bottom is not None:
            layout["bottom"] = max(float(bottom), float(layout.get("bottom", 0.08)))
        if top is not None:
            layout["top"] = min(float(top), float(layout.get("top", 0.94)))
        if hspace is not None:
            layout["hspace"] = max(float(hspace), float(layout.get("hspace", 0.0)))
        if wspace is not None:
            layout["wspace"] = max(float(wspace), float(layout.get("wspace", 0.0)))
        setattr(fig, "_agenttool_layout", layout)

    def _apply_clean_axis(self, ax, grid_axis="y"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if grid_axis:
            ax.grid(axis=grid_axis, linestyle="--", alpha=0.32, zorder=0)
        ax.tick_params(axis="both", length=3.2, width=0.7, color="#4b5563")

    def _annotate_panel_label(self, ax, label="a"):
        return None

    def _finalize_paper_plot(self, fig, save_path: str):
        save_path = self._as_svg_path(save_path)
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        layout = getattr(fig, "_agenttool_layout", {})
        bottom = float(layout.get("bottom", 0.08))
        top = float(layout.get("top", 0.94))
        hspace = layout.get("hspace")
        wspace = layout.get("wspace")

        self._scale_figure_text(fig)
        try:
            fig.align_labels()
        except Exception:
            pass

        if not getattr(fig, "_agenttool_skip_tight_layout", False):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=(
                            "This figure includes Axes that are not compatible "
                            "with tight_layout"
                        ),
                        category=UserWarning,
                    )
                    fig.tight_layout(rect=(0.0, bottom, 1.0, top))
            except Exception:
                pass

        adjust_kwargs = {}
        if bottom > 0.08:
            adjust_kwargs["bottom"] = min(bottom, 0.42)
        if top < 0.94:
            adjust_kwargs["top"] = max(top, 0.55)
        if hspace is not None:
            adjust_kwargs["hspace"] = hspace
        if wspace is not None:
            adjust_kwargs["wspace"] = wspace
        if adjust_kwargs:
            try:
                fig.subplots_adjust(**adjust_kwargs)
            except Exception:
                pass

        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(
            output_path.with_suffix(".pdf"),
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        return str(output_path)

    def _scale_figure_text(self, fig, scale=PLOT_FONT_SCALE):
        if getattr(fig, "_agenttool_font_scaled", False):
            return
        for text_obj in fig.findobj(match=mtext.Text):
            try:
                current = float(text_obj.get_fontsize())
                text_obj.set_fontsize(current * float(scale))
            except Exception:
                continue
        setattr(fig, "_agenttool_font_scaled", True)

    def _add_figure_note(self, fig, note, *, width=118, x=0.01, y=0.018, bottom=0.16, prefix="Note"):
        """Captions/figure notes are intentionally suppressed for cleaner exports."""
        return None

    def _build_dataframe_from_detection_map(self, detection_map):
        rows = []
        for x in detection_map:
            if x.get("latitude") is None or x.get("longitude") is None:
                continue
            rows.append({
                "camera_id": str(x["camera_id"]),
                "count": int(x["count"]),
                "latitude": float(x["latitude"]),
                "longitude": float(x["longitude"]),
            })
        return pd.DataFrame(rows)

    def _count_to_marker_size_static(self, count, min_size=30, scale=18, max_size=260):
        size = min_size + scale * np.sqrt(max(count, 0))
        return min(size, max_size)

    def _filter_spatial_outliers(self, df, park_gdf=None, quantile=0.05):
        if df.empty:
            return df.copy(), {"mode": "empty", "removed": 0}

        cleaned = df.dropna(subset=["longitude", "latitude"]).copy()
        if cleaned.empty:
            return cleaned, {"mode": "empty", "removed": len(df)}

        original_count = len(cleaned)

        if park_gdf is not None and not park_gdf.empty:
            xmin, ymin, xmax, ymax = park_gdf.total_bounds
            xpad = max((xmax - xmin) * 0.03, 0.05)
            ypad = max((ymax - ymin) * 0.03, 0.05)
            park_filtered = cleaned[
                cleaned["longitude"].between(xmin - xpad, xmax + xpad)
                & cleaned["latitude"].between(ymin - ypad, ymax + ypad)
            ].copy()
            if len(park_filtered) >= max(3, int(original_count * 0.6)):
                return park_filtered, {
                    "mode": "park_extent",
                    "removed": original_count - len(park_filtered),
                }

        if original_count < 6:
            return cleaned, {"mode": "none", "removed": 0}

        lon_low, lon_high = cleaned["longitude"].quantile([quantile, 1 - quantile])
        lat_low, lat_high = cleaned["latitude"].quantile([quantile, 1 - quantile])
        xpad = max((lon_high - lon_low) * 0.18, 0.05)
        ypad = max((lat_high - lat_low) * 0.18, 0.05)

        quantile_filtered = cleaned[
            cleaned["longitude"].between(lon_low - xpad, lon_high + xpad)
            & cleaned["latitude"].between(lat_low - ypad, lat_high + ypad)
        ].copy()

        if len(quantile_filtered) >= max(3, int(original_count * 0.7)):
            return quantile_filtered, {
                "mode": "quantile",
                "removed": original_count - len(quantile_filtered),
            }

        return cleaned, {"mode": "none", "removed": 0}

    def _compute_plot_bounds(self, df, center=None, radius_km=None, pad_ratio=0.12):
        if df.empty and center is None:
            return None

        if df.empty:
            xmin = xmax = float(center[0])
            ymin = ymax = float(center[1])
        else:
            xmin = float(df["longitude"].min())
            xmax = float(df["longitude"].max())
            ymin = float(df["latitude"].min())
            ymax = float(df["latitude"].max())

        if center is not None:
            center_lon, center_lat = center
            xmin = min(xmin, center_lon)
            xmax = max(xmax, center_lon)
            ymin = min(ymin, center_lat)
            ymax = max(ymax, center_lat)

        xspan = max(xmax - xmin, 0.02)
        yspan = max(ymax - ymin, 0.02)
        xpad = max(xspan * pad_ratio, 0.05)
        ypad = max(yspan * pad_ratio, 0.05)

        if center is not None and radius_km is not None:
            center_lon, center_lat = center
            lat_pad = radius_km / 111.0 * 1.08
            lon_pad = radius_km / max(111.0 * np.cos(np.radians(center_lat)), 1e-6) * 1.08
            xmin = min(xmin, center_lon - lon_pad)
            xmax = max(xmax, center_lon + lon_pad)
            ymin = min(ymin, center_lat - lat_pad)
            ymax = max(ymax, center_lat + lat_pad)
            xpad = max(xpad, lon_pad * 0.1)
            ypad = max(ypad, lat_pad * 0.1)

        return (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)

    def _clip_park_to_bounds(self, park_gdf, bounds):
        if park_gdf is None or park_gdf.empty or bounds is None:
            return None

        xmin, xmax, ymin, ymax = bounds
        try:
            clipped = park_gdf.cx[xmin:xmax, ymin:ymax]
            if clipped.empty:
                return None
            return clipped
        except Exception:
            return park_gdf

    def _add_caption_title(self, ax, title, subtitle=None, *, subtitle_width=92):
        title = self._clean_caption_text(title) or "Figure"

        ax.set_title(
            title,
            loc="left",
            pad=12,
            fontsize=12.2,
            fontweight="bold",
            color="#17202a",
        )
        self._reserve_figure_space(ax.figure, top=0.92)

    def _format_timestamp_source_summary(self, summary):
        if not isinstance(summary, dict) or not summary:
            return "timestamp sources unspecified"

        ordered = sorted(summary.items(), key=lambda item: (-item[1], item[0]))
        return "; ".join(f"{key}: {value}" for key, value in ordered)

    def _annotate_points_with_offsets(self, ax, df, label_builder, font_size=8, max_labels=None):
        if df.empty:
            return

        rows = list(df.iterrows())
        if max_labels is not None:
            rows = rows[:max(int(max_labels), 0)]
        if not rows:
            return

        fig = ax.figure
        dpi_scale = fig.dpi / 72.0
        candidate_offsets = [
            (16, 12), (16, -12), (-16, 12), (-16, -12),
            (0, 18), (0, -18), (26, 0), (-26, 0),
            (24, 20), (24, -20), (-24, 20), (-24, -20),
        ]
        placed_boxes = []

        def _estimate_label_box(point_px, dx_pt, dy_pt, text, ha, va):
            dx_px = dx_pt * dpi_scale
            dy_px = dy_pt * dpi_scale
            x_anchor = point_px[0] + dx_px
            y_anchor = point_px[1] + dy_px
            line_count = text.count("\n") + 1
            width = max(36.0, len(max(text.split("\n"), key=len)) * font_size * 0.56 * dpi_scale)
            height = max(14.0, line_count * font_size * 1.35 * dpi_scale)
            pad = 4.0 * dpi_scale

            if ha == "right":
                x0, x1 = x_anchor - width - pad, x_anchor + pad
            elif ha == "center":
                x0, x1 = x_anchor - width / 2 - pad, x_anchor + width / 2 + pad
            else:
                x0, x1 = x_anchor - pad, x_anchor + width + pad

            if va == "top":
                y0, y1 = y_anchor - height - pad, y_anchor + pad
            elif va == "center":
                y0, y1 = y_anchor - height / 2 - pad, y_anchor + height / 2 + pad
            else:
                y0, y1 = y_anchor - pad, y_anchor + height + pad
            return (x0, y0, x1, y1)

        def _overlaps(box_a, box_b, margin=4.0):
            return not (
                box_a[2] + margin < box_b[0]
                or box_a[0] - margin > box_b[2]
                or box_a[3] + margin < box_b[1]
                or box_a[1] - margin > box_b[3]
            )

        for idx, (_, row) in enumerate(rows):
            label = str(label_builder(row))
            point_px = ax.transData.transform((row["longitude"], row["latitude"]))
            chosen = None
            fallback = None

            for dx, dy in candidate_offsets:
                ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
                va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
                box = _estimate_label_box(point_px, dx, dy, label, ha, va)
                if fallback is None:
                    fallback = (dx, dy, ha, va, box)
                if not any(_overlaps(box, existing) for existing in placed_boxes):
                    chosen = (dx, dy, ha, va, box)
                    break

            dx, dy, ha, va, box = chosen or fallback
            placed_boxes.append(box)
            ax.annotate(
                label,
                xy=(row["longitude"], row["latitude"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=font_size,
                ha=ha,
                va=va,
                clip_on=False,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": "#9aa6b2",
                    "alpha": 0.92,
                    "lw": 0.65,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#7a8793",
                    "lw": 0.65,
                    "shrinkA": 0,
                    "shrinkB": 4,
                },
            )

    def _smooth_circular_series(self, values, sigma=1.25, resolution=240):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.array([]), np.array([])

        sigma = max(float(sigma), 0.4)
        radius = max(1, int(round(sigma * 3)))
        offsets = np.arange(-radius, radius + 1)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        weights /= weights.sum()

        padded = np.concatenate([values[-radius:], values, values[:radius]])
        smoothed = np.convolve(padded, weights, mode="same")[radius:-radius]

        base_x = np.arange(values.size + 1, dtype=float)
        base_y = np.append(smoothed, smoothed[0])
        dense_x = np.linspace(0.0, float(values.size), int(resolution) + 1)
        dense_y = np.interp(dense_x, base_x, base_y)
        return dense_x, dense_y

    def _estimate_density_curve(self, values, grid_points=240, lower_bound=None):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.array([]), np.array([])

        if values.size == 1 or np.allclose(values, values[0]):
            center = float(values[0])
            bandwidth = max(abs(center) * 0.08, 0.35)
            x_min = center - bandwidth * 4.0
            x_max = center + bandwidth * 4.0
            if lower_bound is not None:
                x_min = max(float(lower_bound), x_min)
            x = np.linspace(x_min, x_max, int(grid_points))
            density = np.exp(-0.5 * ((x - center) / bandwidth) ** 2)
            density /= max(_integrate_trapezoid(density, x), 1e-9)
            return x, density

        std = float(np.std(values, ddof=1))
        q75, q25 = np.percentile(values, [75, 25])
        iqr = float(q75 - q25)
        sigma = min(std, iqr / 1.34) if iqr > 0 else std
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = max((float(values.max()) - float(values.min())) / 6.0, 0.25)
        bandwidth = max(0.9 * sigma * values.size ** (-0.2), 0.15)

        data_min = float(values.min())
        data_max = float(values.max())
        span = max(data_max - data_min, bandwidth * 6.0)
        x_min = data_min - span * 0.08
        x_max = data_max + span * 0.08
        if lower_bound is not None:
            x_min = max(float(lower_bound), x_min)
        if x_max <= x_min:
            x_max = x_min + bandwidth * 4.0

        x = np.linspace(x_min, x_max, int(grid_points))
        scaled = (x[:, None] - values[None, :]) / bandwidth
        density = np.exp(-0.5 * scaled ** 2).sum(axis=1)
        density /= max(values.size * bandwidth * np.sqrt(2.0 * np.pi), 1e-9)
        integral = _integrate_trapezoid(density, x)
        if integral > 0:
            density /= integral
        return x, density

    def build_forest_plot_data(self, panels, zero_line=0.0):
        normalized_panels = []
        x_mins = []
        x_maxs = []

        for idx, panel in enumerate(panels or []):
            rows = []
            for row in panel.get("rows", []):
                if "term" not in row:
                    continue
                estimate = float(row.get("estimate", 0.0))
                ci_low = float(row.get("ci_low", estimate))
                ci_high = float(row.get("ci_high", estimate))
                rows.append({
                    "term": str(row["term"]),
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                })
                x_mins.append(min(ci_low, estimate, zero_line))
                x_maxs.append(max(ci_high, estimate, zero_line))

            if not rows:
                continue

            normalized_panels.append({
                "title": panel.get("title", f"Panel {idx + 1}"),
                "subtitle": panel.get("subtitle"),
                "rows": rows,
            })

        if not normalized_panels:
            raise ValueError("Forest plot panels are empty.")

        x_min = min(x_mins)
        x_max = max(x_maxs)
        span = max(x_max - x_min, 0.1)
        pad = span * 0.12

        return {
            "panels": normalized_panels,
            "xlim": (x_min - pad, x_max + pad),
            "zero_line": float(zero_line),
        }

    def build_density_panel_data(self, panel_items, reference_line=None):
        normalized_panels = []
        x_mins = []
        x_maxs = []
        y_maxs = []

        for idx, item in enumerate(panel_items or []):
            title = item.get("title", f"Panel {idx + 1}")
            subtitle = item.get("subtitle")
            annotation = item.get("annotation")
            lower_bound = item.get("lower_bound")

            if item.get("samples") is not None:
                x, density = self._estimate_density_curve(
                    item.get("samples", []),
                    lower_bound=lower_bound,
                )
            else:
                x = np.asarray(item.get("x", []), dtype=float)
                density = np.asarray(item.get("density", []), dtype=float)
                valid = np.isfinite(x) & np.isfinite(density)
                x = x[valid]
                density = density[valid]
                if x.size:
                    order = np.argsort(x)
                    x = x[order]
                    density = density[order]

            if x.size == 0 or density.size == 0:
                continue

            x_mins.append(float(x.min()))
            x_maxs.append(float(x.max()))
            y_maxs.append(float(density.max()))
            normalized_panels.append({
                "title": title,
                "subtitle": subtitle,
                "annotation": annotation,
                "reference": item.get("reference", reference_line),
                "x": x,
                "density": density,
            })

        if not normalized_panels:
            raise ValueError("Density panel items are empty.")

        return {
            "panels": normalized_panels,
            "xlim": (min(x_mins), max(x_maxs)),
            "ylim": (0.0, max(y_maxs) * 1.14),
        }

    def _plot_categorical_activity_bar(
        self,
        labels,
        values,
        *,
        counts=None,
        ratios=None,
        title,
        subtitle=None,
        xlabel="Category",
        ylabel="Observation count",
        panel_label="a",
        save_path=None,
        filename="categorical_activity.png",
        y_as_percent=False,
        note=None,
    ):
        labels = list(labels)
        values = np.asarray(values, dtype=float)
        counts = None if counts is None else np.asarray(counts, dtype=float)
        ratios = None if ratios is None else np.asarray(ratios, dtype=float)

        if values.size == 0:
            raise ValueError("No categorical values available for plotting.")

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        x = np.arange(len(labels))
        highlight_idx = int(np.argmax(values))

        line_color = SCI_COLORS["blue"]
        highlight_color = SCI_COLORS["orange"]
        ax.plot(
            x,
            values,
            color=line_color,
            linewidth=2.1,
            marker="o",
            markersize=6.2,
            markerfacecolor=line_color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.fill_between(x, values, color=line_color, alpha=0.12, zorder=1)
        ax.scatter(
            [x[highlight_idx]],
            [values[highlight_idx]],
            s=82,
            color=highlight_color,
            edgecolors="white",
            linewidths=0.9,
            zorder=4,
        )

        baseline = None
        if y_as_percent:
            baseline = 1.0 / len(labels)
        elif counts is not None and counts.sum() > 0:
            baseline = float(counts.sum()) / len(labels)

        if baseline is not None:
            ax.axhline(
                baseline,
                color="#6c757d",
                linestyle=":",
                linewidth=1.0,
                alpha=0.85,
                zorder=2,
            )

        for idx, value in enumerate(values):
            lines = []
            if y_as_percent:
                lines.append(f"{values[idx]:.1%}")
                if counts is not None:
                    lines.append(f"n={int(round(counts[idx]))}")
            else:
                lines.append(f"{int(round(values[idx]))}")
                if ratios is not None:
                    lines.append(f"{ratios[idx]:.1%}")
            offset = max(values.max(), 1.0) * (0.035 if idx % 2 == 0 else 0.06)
            ax.text(
                x[idx],
                value + offset,
                "\n".join(lines),
                ha="center",
                va="bottom",
                fontsize=8.2,
            )

        ax.set_xticks(x)
        if max(len(str(label)) for label in labels) > 8:
            ax.set_xticklabels(labels, rotation=18, ha="right")
        else:
            ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self._apply_clean_axis(ax, grid_axis="y")

        if y_as_percent:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        elif counts is not None and counts.sum() > 0:
            total = float(counts.sum())
            secax = ax.secondary_yaxis(
                "right",
                functions=(
                    lambda y: y / total * 100.0,
                    lambda y: y * total / 100.0,
                ),
            )
            secax.set_ylabel("Share of detections (%)")
            secax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

        ax.set_xlim(-0.45, len(labels) - 0.55)
        ax.set_ylim(0, max(values.max(), 0.01) * 1.34)
        self._add_caption_title(ax, title, subtitle=subtitle)
        self._annotate_panel_label(ax, panel_label)

        legend_handles = [
            Line2D([0], [0], color=line_color, marker="o", lw=2.0, label="Activity trend"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=highlight_color, markersize=7, label="Peak"),
        ]
        if baseline is not None:
            legend_handles.append(
                Line2D([0], [0], color=SCI_COLORS["grey"], linestyle=":", lw=1.0, label="Uniform expectation")
            )
        ax.legend(handles=legend_handles, frameon=False, loc="best")
        self._reserve_figure_space(fig, top=0.84)

        if note:
            self._add_figure_note(fig, note, width=108, bottom=0.18)

        if save_path is None:
            save_path = self._build_path(filename)

        return self._finalize_paper_plot(fig, save_path)

    def plot_forest_panel(
        self,
        panels,
        x_label="Estimate",
        zero_line=0.0,
        panel_labels=None,
        caption=None,
        save_path=None,
    ):
        forest_data = self.build_forest_plot_data(panels, zero_line=zero_line)
        normalized_panels = forest_data["panels"]
        panel_labels = panel_labels or [chr(ord("a") + idx) for idx in range(len(normalized_panels))]

        self._set_paper_style()
        fig_width = max(4.2 * len(normalized_panels), 6.8)
        max_rows = max(len(panel["rows"]) for panel in normalized_panels)
        fig_height = min(max(2.6 + 0.48 * max_rows, 3.4), 5.2)
        fig, axes = plt.subplots(1, len(normalized_panels), figsize=(fig_width, fig_height), squeeze=False)
        axes = axes.ravel()

        for idx, (ax, panel) in enumerate(zip(axes, normalized_panels)):
            rows = panel["rows"]
            y_pos = np.arange(len(rows))

            for row_idx, row in enumerate(rows):
                ci_low = row["ci_low"]
                ci_high = row["ci_high"]
                estimate = row["estimate"]
                significant = ci_low > zero_line or ci_high < zero_line
                ax.hlines(
                    row_idx,
                    ci_low,
                    ci_high,
                    color="#222222",
                    linewidth=1.05,
                    zorder=2,
                )
                ax.scatter(
                    estimate,
                    row_idx,
                    s=34,
                    facecolors=SCI_COLORS["brick"] if significant else "white",
                    edgecolors="#222222",
                    linewidths=1.0,
                    zorder=3,
                )

            ax.axvline(zero_line, color=SCI_COLORS["grey"], linestyle=":", linewidth=1.0, zorder=1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([row["term"] for row in rows])
            ax.invert_yaxis()
            ax.set_ylim(len(rows) - 0.45, -0.55)
            ax.set_xlim(*forest_data["xlim"])
            self._apply_clean_axis(ax, grid_axis="x")
            self._add_caption_title(ax, panel["title"], subtitle=panel.get("subtitle"))
            self._annotate_panel_label(ax, panel_labels[idx] if idx < len(panel_labels) else chr(ord("a") + idx))
            if idx == 0:
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel(x_label)

        self._reserve_figure_space(fig, bottom=0.13, top=0.84, wspace=0.28)
        if caption:
            self._add_figure_note(fig, caption, width=125, bottom=0.18)

        if save_path is None:
            save_path = self._build_path("forest_panel.png")

        return self._finalize_paper_plot(fig, save_path)

    def plot_density_panel_grid(
        self,
        panel_items,
        ncols=2,
        x_label="Days",
        y_label="Probability",
        reference_line=None,
        panel_labels=None,
        caption=None,
        save_path=None,
    ):
        density_data = self.build_density_panel_data(panel_items, reference_line=reference_line)
        normalized_panels = density_data["panels"]
        ncols = max(1, int(ncols))
        nrows = int(np.ceil(len(normalized_panels) / ncols))
        panel_labels = panel_labels or [chr(ord("a") + idx) for idx in range(len(normalized_panels))]

        self._set_paper_style()
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.25 * ncols, 3.15 * nrows),
            squeeze=False,
        )
        flat_axes = axes.ravel()

        for idx, ax in enumerate(flat_axes):
            if idx >= len(normalized_panels):
                ax.axis("off")
                continue

            panel = normalized_panels[idx]
            x = panel["x"]
            density = panel["density"]

            ax.fill_between(x, density, color="#C8C8C8", alpha=0.50, zorder=1)
            ax.plot(x, density, color="#222222", linewidth=1.25, zorder=2)

            if panel.get("reference") is not None:
                ax.axvline(
                    float(panel["reference"]),
                    color=SCI_COLORS["brick"],
                    linewidth=0.95,
                    alpha=0.95,
                    zorder=3,
                )

            ax.set_xlim(*density_data["xlim"])
            ax.set_ylim(*density_data["ylim"])
            self._apply_clean_axis(ax, grid_axis="y")
            self._add_caption_title(ax, panel["title"], subtitle=panel.get("subtitle"))
            self._annotate_panel_label(ax, panel_labels[idx] if idx < len(panel_labels) else chr(ord("a") + idx))

            if panel.get("annotation"):
                ax.text(
                    0.97,
                    0.92,
                    str(panel["annotation"]),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.6,
                    linespacing=1.18,
                    bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d0d0d0", "alpha": 0.9, "lw": 0.6},
                )

        xlabel_y = 0.078 if caption else 0.035
        fig.supxlabel(x_label, y=xlabel_y)
        fig.supylabel(y_label, x=0.018)

        self._reserve_figure_space(fig, bottom=0.11, top=0.88, hspace=0.44, wspace=0.30)
        if caption:
            self._add_figure_note(fig, caption, width=118, bottom=0.16)

        if save_path is None:
            save_path = self._build_path("density_panel_grid.png")

        return self._finalize_paper_plot(fig, save_path)

    def _normalize_occasion_day_windows(self, occasion_days=14, compare_occasion_days=None):
        if compare_occasion_days is None:
            raw_values = [occasion_days]
        elif isinstance(compare_occasion_days, (list, tuple, set)):
            raw_values = list(compare_occasion_days)
        else:
            raw_values = [compare_occasion_days]

        normalized = []
        for value in raw_values:
            try:
                days = max(int(value), 1)
            except Exception:
                continue
            if days not in normalized:
                normalized.append(days)

        return normalized or [max(int(occasion_days), 1)]

    def _collect_occupancy_bootstrap_samples(
        self,
        species=None,
        occasion_days=14,
        min_gap_minutes=30,
        min_active_occasions=2,
        bootstrap_iters=200,
    ):
        model_result = self.fit_single_season_occupancy_model(
            species=species,
            occasion_days=occasion_days,
            min_gap_minutes=min_gap_minutes,
            min_active_occasions=min_active_occasions,
            bootstrap_iters=bootstrap_iters,
        )
        history_result = self.build_detection_history_matrix(
            species=species,
            occasion_days=occasion_days,
            min_gap_minutes=min_gap_minutes,
            min_active_occasions=min_active_occasions,
        )

        surveyed_rows = history_result.get("camera_detection_histories", [])
        detection_histories = [row["detection_history"] for row in surveyed_rows]
        psi_samples = []
        p_samples = []

        point_psi = model_result.get("occupancy_probability_psi")
        point_p = model_result.get("detection_probability_p")

        if point_psi is not None:
            psi_samples.append(float(point_psi))
        if point_p is not None:
            p_samples.append(float(point_p))

        bootstrap_iters = max(int(bootstrap_iters), 0)
        if bootstrap_iters > 1 and len(detection_histories) > 1 and point_p is not None:
            rng = np.random.default_rng(42)
            site_count = len(detection_histories)

            for _ in range(bootstrap_iters):
                sample_idx = rng.choice(site_count, size=site_count, replace=True)
                sample_histories = [detection_histories[int(idx)] for idx in sample_idx]
                sample_fit = self._fit_constant_occupancy_em(sample_histories)
                if sample_fit is None:
                    continue
                psi_samples.append(float(sample_fit["psi_hat"]))
                p_samples.append(float(sample_fit["p_hat"]))

        return {
            "model_result": model_result,
            "history_result": history_result,
            "psi_samples": np.asarray(psi_samples, dtype=float),
            "p_samples": np.asarray(p_samples, dtype=float),
        }

    def plot_detection_history_heatmap(
        self,
        species=None,
        occasion_days=14,
        min_gap_minutes=30,
        min_active_occasions=2,
        sort_by="first_detection",
        max_cameras=80,
        panel_label="a",
        save_path=None,
    ):
        """
        Publication style: camera-by-occasion detection-history heatmap.
        Colors distinguish unsurveyed, surveyed-but-not-detected, and detected occasions; this is a standard intermediate figure for occupancy workflows.
        """
        history_result = self.build_detection_history_matrix(
            species=species,
            occasion_days=occasion_days,
            min_gap_minutes=min_gap_minutes,
            min_active_occasions=min_active_occasions,
        )
        rows = list(history_result.get("camera_detection_histories", []))
        total_occasions = int(history_result.get("total_occasions", 0))

        if not rows or total_occasions <= 0:
            raise ValueError(f"{species} has no detection-history data available for plotting.")

        def _first_detection_index(row):
            for occ_idx, detected in zip(row["occasion_indices"], row["detection_history"]):
                if detected:
                    return int(occ_idx)
            return total_occasions + 1

        sort_mode = str(sort_by).strip().lower()
        if sort_mode == "detection_count":
            rows.sort(key=lambda row: (-row["detection_count"], _first_detection_index(row), row["camera_id"]))
        elif sort_mode == "effort":
            rows.sort(key=lambda row: (-row["occasion_count"], -row["detection_count"], row["camera_id"]))
        elif sort_mode == "camera_id":
            rows.sort(key=lambda row: row["camera_id"])
        else:
            rows.sort(key=lambda row: (_first_detection_index(row), -row["detection_count"], row["camera_id"]))

        if max_cameras is not None:
            max_cameras = max(int(max_cameras), 1)
            rows = rows[:max_cameras]

        matrix = np.full((len(rows), total_occasions), -1.0, dtype=float)
        for row_idx, row in enumerate(rows):
            for occ_idx, detected in zip(row["occasion_indices"], row["detection_history"]):
                if 0 <= int(occ_idx) < total_occasions:
                    matrix[row_idx, int(occ_idx)] = 1.0 if int(detected) == 1 else 0.0

        no_survey_color = "#E9EDF2"
        surveyed_color = "#FFF7EC"
        detected_color = "#B63A3A"
        cmap = matplotlib.colors.ListedColormap([no_survey_color, surveyed_color, detected_color])
        norm = matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

        self._set_paper_style()
        fig_height = min(max(4.8, 0.16 * len(rows) + 2.3), 12.0)
        fig, ax = plt.subplots(figsize=(10.2, fig_height))
        ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

        ax.set_xlabel("Survey occasion")
        ax.set_ylabel("Camera")
        ax.set_xlim(-0.5, total_occasions - 0.5)

        tick_step = max(1, total_occasions // 10)
        xticks = list(range(0, total_occasions, tick_step))
        if xticks[-1] != total_occasions - 1:
            xticks.append(total_occasions - 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(idx + 1) for idx in xticks])

        if len(rows) <= 25:
            yticks = np.arange(len(rows))
            ylabels = [row["camera_id"] for row in rows]
        else:
            y_step = max(1, len(rows) // 18)
            yticks = np.arange(0, len(rows), y_step)
            ylabels = [rows[idx]["camera_id"] for idx in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        detected_cameras = sum(1 for row in rows if row.get("detected"))
        total_detections = sum(int(row.get("detection_count", 0)) for row in rows)
        subtitle = (
            f"{len(rows)} cameras displayed across {total_occasions} occasions; {detected_cameras} cameras had "
            f"at least one detection and the matrix contains {total_detections} detected occasions."
        )
        self._add_caption_title(
            ax,
            f"Detection history matrix for {species}",
            subtitle=subtitle,
        )
        self._annotate_panel_label(ax, panel_label)

        legend_handles = [
            Patch(facecolor=no_survey_color, edgecolor="none", label="No survey"),
            Patch(facecolor=surveyed_color, edgecolor="none", label="Surveyed, no detection"),
            Patch(facecolor=detected_color, edgecolor="none", label="Detected"),
        ]
        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )

        note = (
            "Rows are camera stations and columns are repeated survey occasions. Grey cells mark occasions with no "
            "camera activity record, cream cells indicate surveyed occasions without detection, and green cells "
            "indicate target-species detection. This is the standard input structure for single-season occupancy models."
        )
        self._add_figure_note(fig, note, width=125, bottom=0.18)

        if save_path is None:
            save_path = self._build_path(
                f"{self._safe_name(species)}_{occasion_days}d_detection_history_heatmap.png"
            )

        return self._finalize_paper_plot(fig, save_path)

    def plot_occupancy_model_forest(
        self,
        species=None,
        occasion_days=14,
        compare_occasion_days=None,
        min_gap_minutes=30,
        min_active_occasions=2,
        bootstrap_iters=200,
        save_path=None,
    ):
        """
        Publication style: single-season occupancy model parameter forest plot.
        Draws psi and p by default; compare_occasion_days creates multi-panel comparisons across occasion windows.
        """
        panels = []
        windows = self._normalize_occasion_day_windows(
            occasion_days=occasion_days,
            compare_occasion_days=compare_occasion_days,
        )

        for days in windows:
            result = self.fit_single_season_occupancy_model(
                species=species,
                occasion_days=days,
                min_gap_minutes=min_gap_minutes,
                min_active_occasions=min_active_occasions,
                bootstrap_iters=bootstrap_iters,
            )

            rows = []
            if result.get("occupancy_probability_psi") is not None:
                rows.append({
                    "term": "Occupancy probability (psi)",
                    "estimate": float(result["occupancy_probability_psi"]),
                    "ci_low": float(result.get("psi_ci_low", result["occupancy_probability_psi"])),
                    "ci_high": float(result.get("psi_ci_high", result["occupancy_probability_psi"])),
                })
            if result.get("detection_probability_p") is not None:
                rows.append({
                    "term": "Detection probability (p)",
                    "estimate": float(result["detection_probability_p"]),
                    "ci_low": float(result.get("p_ci_low", result["detection_probability_p"])),
                    "ci_high": float(result.get("p_ci_high", result["detection_probability_p"])),
                })

            if not rows:
                continue

            panels.append({
                "title": f"{days}-day survey occasions",
                "subtitle": (
                    f"n = {result.get('surveyed_camera_count', 0)} cameras; naive occupancy = "
                    f"{result.get('naive_occupancy_ratio', 0):.2f}; bootstrap = {int(bootstrap_iters)}."
                ),
                "rows": rows,
            })

        if not panels:
            raise ValueError(f"{species} has insufficient occupancy-model results for a forest plot.")

        caption = (
            "Points show maximum-likelihood estimates from the detection-corrected single-season occupancy model, "
            "and horizontal lines show 95% bootstrap confidence intervals. Comparing panels across occasion windows "
            "is useful for checking sensitivity to the repeated-survey definition."
        )

        if save_path is None:
            suffix = "comparison" if len(panels) > 1 else f"{windows[0]}d"
            save_path = self._build_path(
                f"{self._safe_name(species)}_occupancy_forest_{suffix}.png"
            )

        return self.plot_forest_panel(
            panels=panels,
            x_label="Estimated probability",
            zero_line=0.0,
            panel_labels=[chr(ord("a") + idx) for idx in range(len(panels))],
            caption=caption,
            save_path=save_path,
        )

    def plot_occupancy_bootstrap_density(
        self,
        species=None,
        occasion_days=14,
        compare_occasion_days=None,
        min_gap_minutes=30,
        min_active_occasions=2,
        bootstrap_iters=400,
        ncols=2,
        save_path=None,
    ):
        """
        Publication style: occupancy-model bootstrap density panel.
        Shows bootstrap distributions for psi and p by default, with optional comparisons across occasion windows.
        """
        panel_items = []
        windows = self._normalize_occasion_day_windows(
            occasion_days=occasion_days,
            compare_occasion_days=compare_occasion_days,
        )

        for days in windows:
            bundle = self._collect_occupancy_bootstrap_samples(
                species=species,
                occasion_days=days,
                min_gap_minutes=min_gap_minutes,
                min_active_occasions=min_active_occasions,
                bootstrap_iters=bootstrap_iters,
            )
            result = bundle["model_result"]
            psi_samples = bundle["psi_samples"]
            p_samples = bundle["p_samples"]
            camera_count = int(result.get("surveyed_camera_count", 0))
            subtitle = f"{days}-day occasions; n = {camera_count} cameras"

            if psi_samples.size > 0 and result.get("occupancy_probability_psi") is not None:
                panel_items.append({
                    "title": "Occupancy probability (psi)",
                    "subtitle": subtitle,
                    "samples": psi_samples,
                    "reference": float(result["occupancy_probability_psi"]),
                    "annotation": (
                        f"est. = {float(result['occupancy_probability_psi']):.3f}\n"
                        f"95% CI {float(result['psi_ci_low']):.3f}-{float(result['psi_ci_high']):.3f}"
                    ),
                    "lower_bound": 0.0,
                })

            if p_samples.size > 0 and result.get("detection_probability_p") is not None:
                panel_items.append({
                    "title": "Detection probability (p)",
                    "subtitle": subtitle,
                    "samples": p_samples,
                    "reference": float(result["detection_probability_p"]),
                    "annotation": (
                        f"est. = {float(result['detection_probability_p']):.3f}\n"
                        f"95% CI {float(result['p_ci_low']):.3f}-{float(result['p_ci_high']):.3f}"
                    ),
                    "lower_bound": 0.0,
                })

        if not panel_items:
            raise ValueError(f"{species} has insufficient occupancy bootstrap results for a density panel.")

        caption = (
            "Panels show bootstrap sampling distributions for occupancy probability (psi) and detection probability (p). "
            "Vertical reference lines mark the point estimates, and the annotations report the corresponding 95% "
            "bootstrap confidence intervals."
        )

        if save_path is None:
            suffix = "comparison" if len(windows) > 1 else f"{windows[0]}d"
            save_path = self._build_path(
                f"{self._safe_name(species)}_occupancy_density_{suffix}.png"
            )

        return self.plot_density_panel_grid(
            panel_items=panel_items,
            ncols=max(1, int(ncols)),
            x_label="Estimated probability",
            y_label="Bootstrap density",
            panel_labels=[chr(ord("a") + idx) for idx in range(len(panel_items))],
            caption=caption,
            save_path=save_path,
        )
    # =========================
    # 1. Global spatial distribution map
    # =========================
    def plot_species_distribution_map(self, species=None, annotate_top_k=5, save_path=None):
        """
        Draw a species global spatial distribution on a real map.
        Output an HTML map.
        """
        detection_map = self.build_detection_map(species)

        valid = [
            x for x in detection_map
            if x.get("latitude") is not None and x.get("longitude") is not None
        ]
        if not valid:
            raise ValueError(f"{species} has no spatial data available for plotting.")

        center_lat = sum(x["latitude"] for x in valid) / len(valid)
        center_lon = sum(x["longitude"] for x in valid) / len(valid)

        m = self._create_base_map(center_lat, center_lon, zoom_start=9)
        self._add_park_boundary(m)

        top_camera_ids = {
            x["camera_id"] for x in sorted(valid, key=lambda x: x["count"], reverse=True)[:annotate_top_k]
        }

        for item in valid:
            cam_id = item["camera_id"]
            lat = item["latitude"]
            lon = item["longitude"]
            count = item["count"]

            is_top = cam_id in top_camera_ids
            color = SCI_COLORS["accent"] if is_top else SCI_COLORS["primary"]
            radius = self._count_to_radius(count)

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; padding: 6px;">
                <h4 style="margin-bottom: 6px;">Monitoring Point</h4>
                <p><strong>Species:</strong> {species}</p>
                <p><strong>Camera ID:</strong> {cam_id}</p>
                <p><strong>Observation Count:</strong> {count}</p>
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color="white",
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{cam_id}: {count}"
            ).add_to(m)

            if is_top:
                folium.Marker(
                    [lat, lon],
                    icon=folium.DivIcon(html=f"""
                        <div style="
                            font-size: 11px;
                            color: #222;
                            background: rgba(255,255,255,0.85);
                            border: 1px solid #ccc;
                            border-radius: 4px;
                            padding: 1px 4px;
                            white-space: nowrap;
                        ">{cam_id} ({count})</div>
                    """)
                ).add_to(m)

        self._fit_map_to_points(m, [(x["latitude"], x["longitude"]) for x in valid])
        self._add_map_controls(m)
        self._add_default_legend(
            m,
            title="Species Distribution",
            items=[
                ("Top hotspots", SCI_COLORS["accent"]),
                ("Other cameras", SCI_COLORS["primary"]),
            ]
        )

        if save_path is None:
            save_path = self._build_html_path(f"{self._safe_name(species)}_distribution_map.html")

        return self._save_map(m, save_path)

    # =========================
    # 2. Local hotspot map
    # =========================
    def plot_local_hotspot_map(
            self,
            species=None,
            center_camera_id="213215",
            radius_km=20,
            annotate_top_k=5,
            save_path=None
    ):
        """
        Draw local hotspot distribution around a center camera on a real map.
        Output an HTML map.
        """
        detection_map = self.build_detection_map(species)
        nearby_cameras = self.get_cameras_in_radius(center_camera_id, radius_km)
        nearby_ids = {str(center_camera_id)} | {str(item.get("camera_id")) for item in nearby_cameras}
        local_detection = [item for item in detection_map if str(item.get("camera_id")) in nearby_ids]
        local_detection.sort(key=lambda item: item.get("count", 0), reverse=True)

        camera_locations = self.get_camera_locations()
        center = camera_locations.get(str(center_camera_id))
        if center is None:
            raise ValueError(f"Center camera {center_camera_id} does not exist or has no coordinates.")

        center_lat = center["latitude"]
        center_lon = center["longitude"]

        m = self._create_base_map(center_lat, center_lon, zoom_start=11)
        self._add_park_boundary(m)

        # Search-radius circle
        folium.Circle(
            location=[center_lat, center_lon],
            radius=radius_km * 1000,
            color="#2b8cbe",
            weight=2,
            fill=True,
            fill_opacity=0.08,
            tooltip=f"Search radius: {radius_km} km"
        ).add_to(m)

        # Nearby cameras in light gray
        for item in nearby_cameras:
            lat = item.get("latitude")
            lon = item.get("longitude")
            cam_id = item.get("camera_id")
            if lat is None or lon is None:
                continue

            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="white",
                weight=1,
                fill=True,
                fill_color="#bdbdbd",
                fill_opacity=0.55,
                tooltip=f"Nearby camera: {cam_id}"
            ).add_to(m)

        # Local detection points in blue; larger markers are hotter
        top_items = sorted(local_detection, key=lambda x: x["count"], reverse=True)[:annotate_top_k]
        top_ids = {x["camera_id"] for x in top_items}

        for item in local_detection:
            lat = item.get("latitude")
            lon = item.get("longitude")
            cam_id = item.get("camera_id")
            count = item.get("count", 0)

            if lat is None or lon is None:
                continue

            is_top = cam_id in top_ids
            color = SCI_COLORS["accent"] if is_top else SCI_COLORS["primary"]
            radius = self._count_to_radius(count, base=4.5, scale=1.5)

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; padding: 6px;">
                <h4 style="margin-bottom: 6px;">Local Hotspot</h4>
                <p><strong>Species:</strong> {species}</p>
                <p><strong>Camera ID:</strong> {cam_id}</p>
                <p><strong>Observation Count:</strong> {count}</p>
                <p><strong>Center Camera:</strong> {center_camera_id}</p>
                <p><strong>Radius:</strong> {radius_km} km</p>
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color="white",
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.88,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{cam_id}: {count}"
            ).add_to(m)

            if is_top:
                folium.Marker(
                    [lat, lon],
                    icon=folium.DivIcon(html=f"""
                        <div style="
                            font-size: 11px;
                            color: #222;
                            background: rgba(255,255,255,0.88);
                            border: 1px solid #ccc;
                            border-radius: 4px;
                            padding: 1px 4px;
                            white-space: nowrap;
                        ">{cam_id} ({count})</div>
                    """)
                ).add_to(m)

        # Center camera in green
        folium.CircleMarker(
            location=[center_lat, center_lon],
            radius=8,
            color="white",
            weight=2,
            fill=True,
            fill_color=SCI_COLORS["prey"],
            fill_opacity=1.0,
            tooltip=f"Center camera: {center_camera_id}"
        ).add_to(m)

        folium.Marker(
            [center_lat, center_lon],
            icon=folium.DivIcon(html=f"""
                <div style="
                    font-size: 11px;
                    color: #111;
                    background: rgba(255,255,255,0.92);
                    border: 1px solid #999;
                    border-radius: 4px;
                    padding: 1px 4px;
                    white-space: nowrap;
                ">center {center_camera_id}</div>
            """)
        ).add_to(m)

        all_points = [(center_lat, center_lon)]
        all_points += [
            (x.get("latitude"), x.get("longitude"))
            for x in nearby_cameras if x.get("latitude") is not None and x.get("longitude") is not None
        ]
        all_points += [
            (x.get("latitude"), x.get("longitude"))
            for x in local_detection if x.get("latitude") is not None and x.get("longitude") is not None
        ]
        self._fit_map_to_points(m, all_points)

        self._add_map_controls(m)

        if local_detection:
            legend_title = f"Local Hotspot ({radius_km} km)"
            legend_items = [
                ("Center camera", SCI_COLORS["prey"]),
                ("Top local hotspots", SCI_COLORS["accent"]),
                ("Other local detections", SCI_COLORS["primary"]),
                ("Nearby cameras", "#bdbdbd"),
            ]
        else:
            legend_title = f"Local Hotspot ({radius_km} km)"
            legend_items = [
                ("Center camera", SCI_COLORS["prey"]),
                ("Nearby cameras", "#bdbdbd"),
            ]

        self._add_default_legend(m, title=legend_title, items=legend_items)

        if save_path is None:
            save_path = self._build_html_path(
                f"{self._safe_name(species)}_local_{center_camera_id}_{radius_km}km.html"
            )

        return self._save_map(m, save_path)

    def _camera_ecology_metrics_table(self, species_list=None, ecological_role_mapping=None):
        selected_species = self._resolve_species_list(species_list=species_list, top_k=8, include_unknown=False)
        if self._using_camtrap_backend():
            events = self._load_camtrap_event_df()
            if events.empty:
                raise ValueError("No independent events are available for interactive ecology mapping.")
            events = events.copy()
            if "species" not in events.columns:
                events["species"] = events.get("scientificName", "").map(self._canonical_species_name)
            events["species"] = events["species"].map(self._canonical_species_name)
            camera_col = "camera_id" if "camera_id" in events.columns else "cameraID"
            events["camera_id"] = events[camera_col].astype(str)
            events = events[events["species"].isin(selected_species)]
        else:
            event_rows = self._collect_exact_independent_event_rows(species_list=selected_species, include_unknown=False)
            if not event_rows:
                raise ValueError("No independent events are available for interactive ecology mapping.")
            events = pd.DataFrame(event_rows)
            events["camera_id"] = events["camera_id"].astype(str)
            events["species"] = events["species"].map(self._canonical_species_name)

        event_counts = events.groupby(["camera_id", "species"], observed=True).size().unstack(fill_value=0)
        for species in selected_species:
            if species not in event_counts.columns:
                event_counts[species] = 0

        effort_days = self._camera_effort_days_from_records()
        camera_locs = self.get_camera_locations()
        loc_rows = []
        for camera_id in sorted(set(event_counts.index.astype(str)) | set(str(key) for key in camera_locs.keys())):
            loc = camera_locs.get(str(camera_id), {})
            loc_rows.append({
                "camera_id": str(camera_id),
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
                "camera_effort_days": max(float(effort_days.get(str(camera_id), 1.0)), 1.0),
            })
        table = pd.DataFrame(loc_rows).set_index("camera_id")
        table = table.join(event_counts[selected_species], how="left").fillna({species: 0 for species in selected_species})
        for species in selected_species:
            table[species] = pd.to_numeric(table[species], errors="coerce").fillna(0).astype(int)
            table[f"{species}_rai"] = table[species] / table["camera_effort_days"].clip(lower=1) * 100.0
        table["detected_richness"] = (table[selected_species] > 0).sum(axis=1)
        table["total_independent_events"] = table[selected_species].sum(axis=1)

        roles = normalize_ecological_role_mapping(ecological_role_mapping, selected_species)
        predators = roles["predators"]
        prey = roles["prey"]
        table["large_carnivore_rai"] = table[[f"{s}_rai" for s in predators]].sum(axis=1) if predators else 0.0
        table["prey_rai"] = table[[f"{s}_rai" for s in prey]].sum(axis=1) if prey else 0.0

        def _minmax(series):
            series = pd.to_numeric(series, errors="coerce").fillna(0.0)
            low = float(series.min())
            high = float(series.max())
            if high <= low:
                return series * 0.0
            return (series - low) / (high - low)

        table["normalized_prey_rai"] = _minmax(table["prey_rai"])
        table["normalized_large_carnivore_rai"] = _minmax(table["large_carnivore_rai"])
        table["relative_encounter_risk_index"] = table["normalized_prey_rai"] * table["normalized_large_carnivore_rai"]
        table = table.reset_index()
        table.attrs["ecological_role_mapping"] = roles
        return table

    def _species_marker_color(self, species):
        return {
            "Siberian tiger": SCI_COLORS["predator"],
            "Amur leopard": "#A66A4C",
            "Black bear": SCI_COLORS["purple"],
            "Sika deer": SCI_COLORS["prey"],
            "Roe deer": SCI_COLORS["secondary"],
        }.get(species, SCI_COLORS["primary"])

    def _html_popup(self, title, rows):
        body = "".join(
            f"<p style=\"margin:2px 0;\"><strong>{label}:</strong> {value}</p>"
            for label, value in rows
        )
        return f"""
        <div style="font-family: Arial, sans-serif; padding: 6px; min-width: 220px;">
            <h4 style="margin: 0 0 6px 0;">{title}</h4>
            {body}
        </div>
        """

    def plot_effort_corrected_hotspot_map(self, species=None, top_k=15, save_path=None):
        """
        Plot effort-corrected species hotspots on an interactive basemap.
        Marker size and ranking use RAI: independent events / 100 camera-days.
        """
        table = self._camera_ecology_metrics_table(species_list=[species])
        species = self._canonical_species_name(species)
        rai_col = f"{species}_rai"
        event_col = species
        plot_df = table.dropna(subset=["latitude", "longitude"]).copy()
        plot_df = plot_df[plot_df[event_col] > 0].sort_values(rai_col, ascending=False)
        if plot_df.empty:
            raise ValueError(f"No spatial independent-event data available for {species}.")

        center_lat = float(plot_df["latitude"].mean())
        center_lon = float(plot_df["longitude"].mean())
        m = self._create_base_map(center_lat, center_lon, zoom_start=9)
        self._add_park_boundary(m)

        top_ids = set(plot_df.head(max(int(top_k), 1))["camera_id"].astype(str))
        max_rai = max(float(plot_df[rai_col].max()), 0.01)
        top_layer = folium.FeatureGroup(name=f"Top {top_k} RAI hotspots", show=True)
        other_layer = folium.FeatureGroup(name="Other detections", show=True)

        for _, row in plot_df.iterrows():
            camera_id = str(row["camera_id"])
            rai = float(row.get(rai_col, 0.0))
            events = int(row.get(event_col, 0))
            effort = float(row["camera_effort_days"])
            is_top = camera_id in top_ids
            color = SCI_COLORS["accent"] if is_top else SCI_COLORS["primary"]
            radius = 5.0 + 13.0 * math.sqrt(rai / max_rai)
            popup_html = self._html_popup(
                "Effort-corrected hotspot",
                [
                    ("Species", species),
                    ("Camera ID", camera_id),
                    ("Independent events", f"{events:,}"),
                    ("Camera-days", f"{effort:,.1f}"),
                    ("RAI", f"{rai:.3f} events / 100 camera-days"),
                    ("Interpretation", "Relative activity hotspot, not density."),
                ],
            )
            marker = folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                color="white",
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.86,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{camera_id}: RAI={rai:.3f}",
            )
            marker.add_to(top_layer if is_top else other_layer)

        top_layer.add_to(m)
        other_layer.add_to(m)
        self._fit_map_to_points(m, list(zip(plot_df["latitude"], plot_df["longitude"])))
        self._add_map_controls(m)
        self._add_default_legend(
            m,
            title="Effort-corrected Hotspots",
            items=[
                ("Top RAI hotspots", SCI_COLORS["accent"]),
                ("Other detections", SCI_COLORS["primary"]),
            ],
        )
        if save_path is None:
            save_path = self._build_html_path(f"{self._safe_name(species)}_effort_corrected_hotspot_map.html")
        return self._save_map(m, save_path)

    def plot_predator_prey_encounter_index_map(
        self,
        species_list=None,
        ecological_role_mapping=None,
        save_path=None,
        max_cameras=1600,
    ):
        """
        Plot a relative predator-prey encounter-risk index on an interactive basemap.
        Formula: normalized prey RAI x normalized large-carnivore RAI.
        """
        selected_species = self._resolve_species_list(
            species_list=species_list,
            top_k=8,
            include_unknown=False,
        )
        roles = normalize_ecological_role_mapping(ecological_role_mapping, selected_species)
        if not roles["predators"] or not roles["prey"]:
            raise ValueError(
                "plot_predator_prey_encounter_index_map requires an explicit "
                "ecological_role_mapping with observed predator and prey species."
            )
        table = self._camera_ecology_metrics_table(
            species_list=selected_species,
            ecological_role_mapping=roles,
        )
        plot_df = table.dropna(subset=["latitude", "longitude"]).copy()
        plot_df = plot_df[(plot_df["prey_rai"] > 0) | (plot_df["large_carnivore_rai"] > 0)]
        if plot_df.empty:
            raise ValueError("No predator-prey spatial data are available for encounter index mapping.")
        plot_df = plot_df.sort_values(
            ["relative_encounter_risk_index", "large_carnivore_rai", "prey_rai"],
            ascending=False,
        ).head(max(int(max_cameras), 1))

        center_lat = float(plot_df["latitude"].mean())
        center_lon = float(plot_df["longitude"].mean())
        m = self._create_base_map(center_lat, center_lon, zoom_start=9)
        self._add_park_boundary(m)

        risk_layer = folium.FeatureGroup(name="Relative encounter-risk index", show=True)
        prey_layer = folium.FeatureGroup(name="Prey RAI", show=False)
        predator_layer = folium.FeatureGroup(name="Large-carnivore RAI", show=False)
        max_risk = max(float(plot_df["relative_encounter_risk_index"].max()), 0.01)
        max_prey = max(float(plot_df["prey_rai"].max()), 0.01)
        max_pred = max(float(plot_df["large_carnivore_rai"].max()), 0.01)

        for _, row in plot_df.iterrows():
            risk = float(row["relative_encounter_risk_index"])
            prey_rai = float(row["prey_rai"])
            pred_rai = float(row["large_carnivore_rai"])
            popup_html = self._html_popup(
                "Predator-prey encounter index",
                [
                    ("Camera ID", row["camera_id"]),
                    ("Prey RAI", f"{prey_rai:.3f}"),
                    ("Large-carnivore RAI", f"{pred_rai:.3f}"),
                    ("Normalized prey RAI", f"{float(row['normalized_prey_rai']):.3f}"),
                    ("Normalized carnivore RAI", f"{float(row['normalized_large_carnivore_rai']):.3f}"),
                    ("Relative encounter-risk index", f"{risk:.3f}"),
                    ("Formula", "normalized prey RAI x normalized large-carnivore RAI"),
                    ("Interpretation", "Exploratory co-detection index, not proof of predation."),
                ],
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4.0 + 14.0 * math.sqrt(risk / max_risk),
                color="white",
                weight=1,
                fill=True,
                fill_color="#7B3294",
                fill_opacity=0.83,
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"{row['camera_id']}: encounter index={risk:.3f}",
            ).add_to(risk_layer)
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3.5 + 12.0 * math.sqrt(prey_rai / max_prey),
                color="white",
                weight=1,
                fill=True,
                fill_color="#4F8A4B",
                fill_opacity=0.70,
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"{row['camera_id']}: prey RAI={prey_rai:.3f}",
            ).add_to(prey_layer)
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3.5 + 12.0 * math.sqrt(pred_rai / max_pred),
                color="white",
                weight=1,
                fill=True,
                fill_color=SCI_COLORS["predator"],
                fill_opacity=0.70,
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"{row['camera_id']}: carnivore RAI={pred_rai:.3f}",
            ).add_to(predator_layer)

        risk_layer.add_to(m)
        prey_layer.add_to(m)
        predator_layer.add_to(m)
        self._fit_map_to_points(m, list(zip(plot_df["latitude"], plot_df["longitude"])))
        self._add_map_controls(m)
        self._add_default_legend(
            m,
            title="Encounter Index",
            items=[
                ("Relative encounter-risk index", SCI_COLORS["accent"]),
                ("Prey RAI layer", SCI_COLORS["prey"]),
                ("Large-carnivore RAI layer", SCI_COLORS["predator"]),
            ],
        )
        if save_path is None:
            save_path = self._build_html_path("predator_prey_encounter_index_map.html")
        return self._save_map(m, save_path)

    def plot_multi_species_ecology_map(
        self,
        species_list=None,
        ecological_role_mapping=None,
        save_path=None,
        max_cameras_per_species=900,
    ):
        """
        Plot a multi-layer interactive map with species-specific RAI and camera ecology metrics.
        """
        selected_species = self._resolve_species_list(
            species_list=species_list,
            top_k=8,
            include_unknown=False,
        )
        roles = normalize_ecological_role_mapping(ecological_role_mapping, selected_species)
        table = self._camera_ecology_metrics_table(
            species_list=selected_species,
            ecological_role_mapping=roles,
        )
        plot_df = table.dropna(subset=["latitude", "longitude"]).copy()
        plot_df = plot_df[plot_df["total_independent_events"] > 0]
        if plot_df.empty:
            raise ValueError("No spatial multi-species event data are available for mapping.")

        center_lat = float(plot_df["latitude"].mean())
        center_lon = float(plot_df["longitude"].mean())
        m = self._create_base_map(center_lat, center_lon, zoom_start=9)
        self._add_park_boundary(m)

        for species in selected_species:
            species = self._canonical_species_name(species)
            event_col = species
            rai_col = f"{species}_rai"
            if event_col not in plot_df.columns:
                continue
            layer_df = plot_df[plot_df[event_col] > 0].copy()
            if layer_df.empty:
                continue
            layer_df = layer_df.sort_values(rai_col, ascending=False).head(max(int(max_cameras_per_species), 1))
            layer = folium.FeatureGroup(name=f"{species} RAI", show=(species == selected_species[0]))
            max_rai = max(float(layer_df[rai_col].max()), 0.01)
            color = self._species_marker_color(species)
            for _, row in layer_df.iterrows():
                events = int(row.get(event_col, 0))
                rai = float(row.get(rai_col, 0.0))
                popup_rows = [
                    ("Species", species),
                    ("Camera ID", row["camera_id"]),
                    ("Independent events", f"{events:,}"),
                    ("Camera-days", f"{float(row['camera_effort_days']):,.1f}"),
                    ("RAI", f"{rai:.3f} events / 100 camera-days"),
                    ("Detected species richness", int(row["detected_richness"])),
                ]
                if roles["predators"] and roles["prey"]:
                    popup_rows.extend([
                        ("Configured prey RAI", f"{float(row['prey_rai']):.3f}"),
                        ("Configured predator RAI", f"{float(row['large_carnivore_rai']):.3f}"),
                    ])
                popup_html = self._html_popup("Camera ecology metrics", popup_rows)
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=4.0 + 12.0 * math.sqrt(rai / max_rai),
                    color="white",
                    weight=1,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.82,
                    popup=folium.Popup(popup_html, max_width=330),
                    tooltip=f"{species} at {row['camera_id']}: RAI={rai:.3f}",
                ).add_to(layer)
            layer.add_to(m)

        self._fit_map_to_points(m, list(zip(plot_df["latitude"], plot_df["longitude"])))
        self._add_map_controls(m)
        self._add_default_legend(
            m,
            title="Species RAI Layers",
            items=[(self._canonical_species_name(species), self._species_marker_color(species)) for species in selected_species],
        )
        if save_path is None:
            save_path = self._build_html_path("multi_species_ecology_interactive_map.html")
        return self._save_map(m, save_path)

    # =========================
    # 3. Hourly activity curve
    # =========================
    def plot_hourly_activity_curve(
        self,
        species=None,
        min_gap_minutes=30,
        smooth_sigma=1.15,
        show_ci=False,
        bootstrap_iters=500,
        annotate_peaks="major_only",
        subtitle_level="short",
        ymax=None,
        save_path=None,
    ):
        """
        Publication style: 24-hour activity-rhythm curve.
        Output only when hourly timestamps are sufficiently reliable.
        """
        figure_data = self.build_hourly_activity_figure_data(
            species,
            min_gap_minutes=min_gap_minutes,
            smooth_sigma=smooth_sigma,
            bootstrap_iters=bootstrap_iters if show_ci else 1,
        )
        hourly = figure_data.get("hourly_distribution", [])

        if not hourly:
            raise ValueError(
                f"{species} hourly activity analysis is unreliable:"
                f"Only {figure_data.get('clock_time_ratio', 0):.1%} of timestamps contain explicit hour-minute values."
            )

        curve_df = pd.DataFrame(hourly).sort_values("hour")
        hours = curve_df["hour"].to_numpy(dtype=float)
        ratios = curve_df["ratio"].to_numpy(dtype=float)
        smooth = figure_data.get("smoothed_curve", {})
        smooth_x = np.asarray(smooth.get("hour", []), dtype=float)
        smooth_y = np.asarray(smooth.get("smooth_mean", []), dtype=float)
        ci_low = np.asarray(smooth.get("ci_low", []), dtype=float)
        ci_high = np.asarray(smooth.get("ci_high", []), dtype=float)

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(9.4, 4.9))

        night_color = SCI_COLORS["sky"]
        day_color = SCI_COLORS["yellow"]
        ax.axvspan(0, 6, color=night_color, alpha=0.10, zorder=0)
        ax.axvspan(18, 24, color=night_color, alpha=0.10, zorder=0)
        ax.axvspan(6, 18, color=day_color, alpha=0.07, zorder=0)

        if show_ci and ci_low.size and ci_high.size:
            ax.fill_between(
                smooth_x,
                ci_low,
                ci_high,
                color=SCI_COLORS["sky"],
                alpha=0.18,
                zorder=1,
                label="95% bootstrap CI",
            )

        ax.fill_between(smooth_x, smooth_y, color=SCI_COLORS["sky"], alpha=0.15, zorder=1)
        ax.plot(smooth_x, smooth_y, color=SCI_COLORS["blue"], linewidth=2.3, zorder=3)
        ax.scatter(hours + 0.5, ratios, s=28, color=SCI_COLORS["blue"], edgecolors="white", linewidths=0.7, zorder=4)

        auto_ymax = max(float(smooth_y.max()) if smooth_y.size else 0.0, float(ratios.max()) if ratios.size else 0.0)
        auto_ymax = max(auto_ymax, float(ci_high.max()) if show_ci and ci_high.size else 0.0, 0.01)
        ymax = float(ymax) if ymax is not None else auto_ymax

        annotation_targets = []
        if annotate_peaks == "major_only":
            if figure_data.get("major_peak"):
                annotation_targets.append(("Major peak", figure_data["major_peak"], SCI_COLORS["orange"]))
            if figure_data.get("trough"):
                annotation_targets.append(("Trough", figure_data["trough"], SCI_COLORS["grey"]))
        elif annotate_peaks == "major_secondary":
            if figure_data.get("major_peak"):
                annotation_targets.append(("Major peak", figure_data["major_peak"], SCI_COLORS["orange"]))
            if figure_data.get("secondary_peak"):
                annotation_targets.append(("Secondary peak", figure_data["secondary_peak"], SCI_COLORS["yellow"]))
            if figure_data.get("trough"):
                annotation_targets.append(("Trough", figure_data["trough"], SCI_COLORS["grey"]))
        elif annotate_peaks == "all":
            for idx, peak in enumerate(figure_data.get("peak_hours", [])[:3]):
                annotation_targets.append((f"Peak {idx + 1}", peak, SCI_COLORS["orange"]))

        for label, point, color in annotation_targets:
            point_hour = float(point["hour"])
            curve_value = float(np.interp(point_hour, smooth_x, smooth_y)) if smooth_x.size else float(point.get("value", 0.0))
            ax.axvline(point_hour, color=color, linestyle="--", linewidth=0.9, alpha=0.55, zorder=2)
            ax.annotate(
                f"{label}\n{point_hour:0.1f} h",
                xy=(point_hour, curve_value),
                xytext=(0, 11 if label != "Trough" else -18),
                textcoords="offset points",
                ha="center",
                va="bottom" if label != "Trough" else "top",
                fontsize=7.8,
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "alpha": 0.92},
            )

        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_xlabel("Time of day (hour)")
        ax.set_ylabel("Relative activity")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(0, ymax * 1.14)
        self._apply_clean_axis(ax, grid_axis="y")

        subtitle_parts = [
            f"n = {figure_data.get('valid_time_records', 0)} independent events",
            f"aggregation gap = {min_gap_minutes} min",
        ]
        if subtitle_level == "full":
            subtitle_parts.append(f"clock-time coverage = {figure_data.get('clock_time_ratio', 0):.1%}")
            if figure_data.get("mean_hour") is not None:
                subtitle_parts.append(f"mean activity time = {figure_data['mean_hour']:0.1f} h")
        subtitle = "; ".join(subtitle_parts) + "."
        self._add_caption_title(
            ax,
            f"Diel activity pattern of {species}",
            subtitle=subtitle,
        )

        legend_handles = [
            Patch(facecolor=night_color, edgecolor="none", alpha=0.10, label="Night period"),
            Patch(facecolor=day_color, edgecolor="none", alpha=0.07, label="Day period"),
            Line2D([0], [0], color=SCI_COLORS["blue"], lw=2.2, label="Smoothed activity curve"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=SCI_COLORS["blue"], markeredgecolor="white", markersize=6, label="Observed hourly ratio"),
        ]
        if show_ci:
            legend_handles.append(Patch(facecolor=SCI_COLORS["sky"], edgecolor="none", alpha=0.18, label="95% bootstrap CI"))
        ax.legend(handles=legend_handles, frameon=False, loc="best", ncol=1)
        self._annotate_panel_label(ax, "a")

        source_summary = self._format_timestamp_source_summary(figure_data.get("timestamp_source_summary", {}))
        caption_lines = [
            figure_data.get("timestamp_quality_note", ""),
            f"Timestamp sources: {source_summary}.",
        ]
        if figure_data.get("activity_concentration") is not None:
            caption_lines.append(
                f"Activity concentration (resultant length) = {figure_data['activity_concentration']:.2f}; "
                f"day/night ratio = {figure_data['day_night_summary'].get('day_ratio', 0):.1%} / "
                f"{figure_data['day_night_summary'].get('night_ratio', 0):.1%}."
            )
        self._add_figure_note(
            fig,
            " ".join(line for line in caption_lines if line),
            width=105,
            bottom=0.30,
            prefix="Data note",
        )

        if save_path is None:
            save_path = self._build_path(f"{self._safe_name(species)}_hourly_activity_curve.png")

        figure_path = self._finalize_paper_plot(fig, save_path)
        return {
            "status": "success",
            "species": species,
            "min_gap_minutes": min_gap_minutes,
            "show_ci": bool(show_ci),
            "independent_event_count": figure_data.get("event_count"),
            "valid_time_records": figure_data.get("valid_time_records"),
            "observed_clock_time_ratio": figure_data.get("observed_clock_time_ratio"),
            "synthetic_clock_time_ratio": figure_data.get("synthetic_clock_time_ratio"),
            "peak_hours": [
                int(item["hour"])
                for item in figure_data.get("peak_hours", [])[:3]
                if item.get("hour") is not None
            ],
            "major_peak": figure_data.get("major_peak"),
            "secondary_peak": figure_data.get("secondary_peak"),
            "day_night_summary": figure_data.get("day_night_summary"),
            "figures": [{
                "figure_id": f"{self._safe_name(species)}_hourly_activity_curve",
                "label": f"{species} hourly activity curve",
                "svg_path": figure_path,
            }],
            "interpretation_boundary": (
                "The curve describes the distribution of independent detection "
                "events across clock time; it does not by itself identify a "
                "behavioural mechanism."
            ),
        }

    # =========================
    # 4. Two-species activity-overlap figure
    # =========================
    def plot_activity_overlap_panel(
        self,
        species_a,
        species_b,
        min_gap_minutes=30,
        smooth_sigma=1.25,
        bootstrap_iters=500,
        panel_label="a",
        save_path=None
    ):
        """
        Publication style: two-species 24-hour activity-overlap figure.
        """
        result = self.estimate_activity_overlap(
            species_a=species_a,
            species_b=species_b,
            min_gap_minutes=min_gap_minutes,
            smooth_sigma=smooth_sigma,
            bootstrap_iters=bootstrap_iters,
        )

        if not result.get("hour_grid"):
            raise ValueError(result.get("note", "Activity-overlap analysis is unavailable."))

        hour_grid = np.asarray(result["hour_grid"], dtype=float)
        density_a = np.asarray(result["density_a"], dtype=float)
        density_b = np.asarray(result["density_b"], dtype=float)
        overlap_density = np.asarray(result["overlap_density"], dtype=float)

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(8.8, 4.8))

        ax.axvspan(0, 6, color=SCI_COLORS["grey"], alpha=0.10, zorder=0)
        ax.axvspan(18, 24, color=SCI_COLORS["grey"], alpha=0.10, zorder=0)
        ax.axvspan(6, 18, color=SCI_COLORS["yellow"], alpha=0.06, zorder=0)

        ax.fill_between(hour_grid, overlap_density, color="#C8C8C8", alpha=0.62, zorder=1, label="Overlap")
        ax.plot(hour_grid, density_a, color="#222222", linewidth=1.45, zorder=3, label=species_a)
        ax.plot(hour_grid, density_b, color="#222222", linewidth=1.45, linestyle=(0, (3, 2)), zorder=3, label=species_b)

        ymax = max(float(density_a.max()), float(density_b.max()), float(overlap_density.max()), 0.01)
        ax.set_ylim(0, ymax * 1.18)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(["Midnight", "Sunrise", "Noon", "Sunset", "Midnight"])
        ax.set_xlabel("Time of day")
        ax.set_ylabel("Density of activity")
        self._apply_clean_axis(ax, grid_axis="y")

        for peak_hour in result.get("peak_hours_a", [])[:2]:
            ax.axvline(peak_hour + 0.5, color=SCI_COLORS["blue"], linestyle=":", linewidth=0.8, alpha=0.55, zorder=2)
        for peak_hour in result.get("peak_hours_b", [])[:2]:
            ax.axvline(peak_hour + 0.5, color=SCI_COLORS["salmon"], linestyle=":", linewidth=0.8, alpha=0.45, zorder=2)

        delta_text = (
            f"Delta = {result['delta_hat']:.2f}"
            f" ({result['delta_ci_low']:.2f}-{result['delta_ci_high']:.2f})"
        )
        if result.get("permutation_p_value") is not None:
            delta_text += f"\nperm. p = {result['permutation_p_value']:.3f}"
        ax.text(
            0.015,
            0.97,
            delta_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#d0d0d0", "alpha": 0.92},
        )

        subtitle = (
            f"Independent events with explicit clock time; aggregation gap = {min_gap_minutes} min; "
            f"n = {result['valid_event_count_a']} vs {result['valid_event_count_b']} events."
        )
        self._add_caption_title(
            ax,
            f"Diel activity overlap: {species_a} vs {species_b}",
            subtitle=subtitle,
        )

        legend_handles = [
            Line2D([0], [0], color="#222222", lw=1.45, label=species_a),
            Line2D([0], [0], color="#222222", lw=1.45, linestyle=(0, (3, 2)), label=species_b),
            Patch(facecolor="#C8C8C8", edgecolor="none", alpha=0.62, label="Overlap area"),
        ]
        ax.legend(handles=legend_handles, frameon=False, loc="best")
        self._annotate_panel_label(ax, panel_label)

        synthetic_note = []
        if result.get("synthetic_clock_time_ratio_a", 0) > 0:
            synthetic_note.append(f"{species_a}: {result['synthetic_clock_time_ratio_a']:.1%} synthetic clock time")
        if result.get("synthetic_clock_time_ratio_b", 0) > 0:
            synthetic_note.append(f"{species_b}: {result['synthetic_clock_time_ratio_b']:.1%} synthetic clock time")
        if synthetic_note:
            self._add_figure_note(
                fig,
                "Synthetic-time disclaimer: " + "; ".join(synthetic_note) + ".",
                width=105,
                bottom=0.14,
                prefix=None,
            )

        if save_path is None:
            save_path = self._build_path(
                f"{self._safe_name(species_a)}_vs_{self._safe_name(species_b)}_activity_overlap.png"
            )

        figure_path = self._finalize_paper_plot(fig, save_path)
        return {
            "status": "success",
            "species_a": species_a,
            "species_b": species_b,
            "min_gap_minutes": min_gap_minutes,
            "valid_event_count_a": result.get("valid_event_count_a"),
            "valid_event_count_b": result.get("valid_event_count_b"),
            "delta_hat": result.get("delta_hat"),
            "delta_ci_low": result.get("delta_ci_low"),
            "delta_ci_high": result.get("delta_ci_high"),
            "permutation_p_value": result.get("permutation_p_value"),
            "temporal_relationship": result.get("temporal_relationship"),
            "statistically_distinguishable_activity_distributions": result.get(
                "statistically_distinguishable_activity_distributions"
            ),
            "supports_strong_temporal_separation": result.get(
                "supports_strong_temporal_separation"
            ),
            "estimator_id": result.get("estimator_id"),
            "estimator_version": result.get("estimator_version"),
            "source_events_sha256": result.get("source_events_sha256"),
            "overlap_result_bundle_id": result.get(
                "overlap_result_bundle_id"
            ),
            "overlap_result_bundle_sha256": result.get(
                "overlap_result_bundle_sha256"
            ),
            "overlap_result_sha256": result.get("overlap_result_sha256"),
            "overlap_result_reused": result.get("overlap_result_reused"),
            "peak_hours_a": result.get("peak_hours_a", []),
            "peak_hours_b": result.get("peak_hours_b", []),
            "note": result.get("note"),
            "figures": [{
                "figure_id": (
                    f"{self._safe_name(species_a)}_vs_"
                    f"{self._safe_name(species_b)}_activity_overlap"
                ),
                "label": f"{species_a} vs {species_b} activity overlap",
                "svg_path": figure_path,
                "delta_hat": result.get("delta_hat"),
                "delta_ci_low": result.get("delta_ci_low"),
                "delta_ci_high": result.get("delta_ci_high"),
                "estimator_id": result.get("estimator_id"),
                "estimator_version": result.get("estimator_version"),
                "source_events_sha256": result.get("source_events_sha256"),
                "overlap_result_bundle_id": result.get(
                    "overlap_result_bundle_id"
                ),
                "overlap_result_bundle_sha256": result.get(
                    "overlap_result_bundle_sha256"
                ),
                "overlap_result_sha256": result.get("overlap_result_sha256"),
                "overlap_result_reused": result.get(
                    "overlap_result_reused"
                ),
            }],
        }

    def plot_pairwise_temporal_overlap(
        self,
        focal_species,
        comparison_species,
        min_gap_minutes=30,
        smooth_sigma=1.25,
        bootstrap_iters=500,
        save_path=None,
    ):
        """
        Generic pairwise diel activity-overlap figure with vector exports.
        """
        focal_species = self._canonical_species_name(focal_species or "")
        comparison_species = self._canonical_species_name(comparison_species or "")
        if not focal_species or focal_species == "Unknown":
            raise ValueError(
                "focal_species is required for pairwise temporal-overlap plotting."
            )
        if not comparison_species or comparison_species == "Unknown":
            raise ValueError("comparison_species is required for pairwise temporal-overlap plotting.")
        if focal_species == comparison_species:
            raise ValueError("focal_species and comparison_species must be different.")

        result = self.estimate_activity_overlap(
            species_a=focal_species,
            species_b=comparison_species,
            min_gap_minutes=min_gap_minutes,
            smooth_sigma=smooth_sigma,
            bootstrap_iters=bootstrap_iters,
            permutation_iters=0,
        )

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(6.8, 3.95))
        if result.get("hour_grid"):
            hour_grid = np.asarray(result["hour_grid"], dtype=float)
            density_focal = np.asarray(result["density_a"], dtype=float)
            density_comparison = np.asarray(result["density_b"], dtype=float)
            overlap_density = np.asarray(result["overlap_density"], dtype=float)

            ax.fill_between(
                hour_grid,
                overlap_density,
                color="#C9CDD2",
                alpha=0.58,
                linewidth=0,
                zorder=1,
                label="Overlap area",
            )
            ax.plot(
                hour_grid,
                density_focal,
                color=SCI_COLORS["blue"],
                linewidth=1.9,
                zorder=3,
                label=f"{focal_species} (n={result['valid_event_count_a']:,})",
            )
            ax.plot(
                hour_grid,
                density_comparison,
                color="#4F7C6A",
                linewidth=1.9,
                linestyle=(0, (4, 2.2)),
                zorder=3,
                label=f"{comparison_species} (n={result['valid_event_count_b']:,})",
            )
            ymax = max(float(density_focal.max()), float(density_comparison.max()), float(overlap_density.max()), 0.01)
        else:
            ax.plot([], [], color=SCI_COLORS["blue"], linewidth=1.9, label=f"{focal_species} (n={result.get('valid_event_count_a', 0):,})")
            ax.plot([], [], color="#4F7C6A", linewidth=1.9, linestyle=(0, (4, 2.2)), label=f"{comparison_species} (n={result.get('valid_event_count_b', 0):,})")
            ax.text(
                0.5,
                0.52,
                "Temporal overlap unavailable\none or both species have no independent events",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.0,
                color=SCI_COLORS["text"],
            )
            ymax = 1.0
        y_headroom = 1.17 if result.get("hour_grid") else 1.0
        ax.set_ylim(0, ymax * y_headroom)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(["Midnight", "06:00", "Noon", "18:00", "Midnight"])
        ax.set_xlabel("Time of day")
        ax.set_ylabel("Density of activity")
        self._apply_clean_sci_axes(ax, grid_axis="y")

        delta_text = "Delta = NA" if result.get("delta_hat") is None else f"Delta = {result['delta_hat']:.2f}"
        if result.get("delta_hat") is not None and result.get("delta_ci_low") is not None and result.get("delta_ci_high") is not None:
            delta_text += f" ({result['delta_ci_low']:.2f}-{result['delta_ci_high']:.2f})"
        ax.text(
            0.025,
            0.955,
            delta_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#D1D5DB", "alpha": 0.94},
        )
        ax.set_title(f"{focal_species} vs {comparison_species}", loc="left", fontweight="bold", pad=7)
        ax.legend(
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.095),
            borderaxespad=0.0,
            handlelength=2.3,
            handletextpad=0.55,
            labelspacing=0.32,
            fontsize=7.8,
        )
        self._reserve_figure_space(fig, bottom=0.10, top=0.86)

        if save_path is None:
            focal_slug = self._safe_name(focal_species).lower()
            comparison_slug = self._safe_name(comparison_species).lower()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(self.output_dir / f"{focal_slug}_vs_{comparison_slug}_overlap.png")

        return self._save_final_demo_figure(fig, save_path)

    def plot_pairwise_activity_overlap_panels(
        self,
        species_a=None,
        comparison_species=None,
        top_k=4,
        min_gap_minutes=30,
        smooth_sigma=1.25,
        bootstrap_iters=500,
    ):
        """
        Generate one pairwise diel activity-overlap figure per comparison species.
        Returns figure paths and per-figure failures instead of raising on partial failure.
        """
        if not species_a or not str(species_a).strip():
            raise ValueError("species_a is required for pairwise activity-overlap panels.")
        species_a = self._canonical_species_name(species_a)
        comparison_species = self._resolve_comparison_species(
            focal_species=species_a,
            species_list=comparison_species,
            top_k=top_k,
        )

        figures = []
        failed_figures = []
        for species_b in comparison_species or []:
            species_b = self._canonical_species_name(species_b)
            figure_id = f"{self._safe_name(species_a).lower()}_vs_{self._safe_name(species_b).lower()}_overlap"
            try:
                overlap_result = self.estimate_activity_overlap(
                    species_a=species_a,
                    species_b=species_b,
                    min_gap_minutes=min_gap_minutes,
                    smooth_sigma=smooth_sigma,
                    bootstrap_iters=bootstrap_iters,
                    permutation_iters=0,
                )
                png_path = self.plot_pairwise_temporal_overlap(
                    focal_species=species_a,
                    comparison_species=species_b,
                    min_gap_minutes=min_gap_minutes,
                    smooth_sigma=smooth_sigma,
                    bootstrap_iters=bootstrap_iters,
                )
                svg_path = str(Path(png_path).with_suffix(".svg"))
                figures.append({
                    "figure_id": figure_id,
                    "species_a": species_a,
                    "species_b": species_b,
                    "status": "generated",
                    "png_path": png_path,
                    "svg_path": svg_path,
                    "delta_hat": overlap_result.get("delta_hat"),
                    "delta_ci_low": overlap_result.get("delta_ci_low"),
                    "delta_ci_high": overlap_result.get("delta_ci_high"),
                    "estimator_id": overlap_result.get("estimator_id"),
                    "estimator_version": overlap_result.get(
                        "estimator_version"
                    ),
                    "source_events_sha256": overlap_result.get(
                        "source_events_sha256"
                    ),
                    "overlap_result_bundle_id": overlap_result.get(
                        "overlap_result_bundle_id"
                    ),
                    "overlap_result_bundle_sha256": overlap_result.get(
                        "overlap_result_bundle_sha256"
                    ),
                    "overlap_result_sha256": overlap_result.get(
                        "overlap_result_sha256"
                    ),
                    "overlap_result_reused": overlap_result.get(
                        "overlap_result_reused"
                    ),
                })
            except Exception as exc:
                failed_figures.append({
                    "figure_id": figure_id,
                    "species_a": species_a,
                    "species_b": species_b,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                })

        if figures and failed_figures:
            status = "partially_generated"
        elif figures:
            status = "generated"
        else:
            status = "failed"

        return {
            "tool_name": "plot_pairwise_activity_overlap_panels",
            "species_a": species_a,
            "comparison_species": [self._canonical_species_name(item) for item in comparison_species or []],
            "comparison_selection": "active_dataset_or_explicit_tool_argument",
            "comparison_top_k": int(top_k),
            "min_gap_minutes": int(min_gap_minutes),
            "smooth_sigma": float(smooth_sigma),
            "bootstrap_iters": int(bootstrap_iters),
            "status": status,
            "figure_count": len(figures),
            "figures": figures,
            "overlap_result_bundle_ids": [
                item["overlap_result_bundle_id"]
                for item in figures
                if item.get("overlap_result_bundle_id")
            ],
            "failed_figures": failed_figures,
        }

    # =========================
    # 5. Monthly activity line chart
    # =========================
    def plot_monthly_activity_bar(self, species=None, save_path=None):
        """
        Publication style: monthly activity-distribution figure.
        """
        result = self.build_monthly_activity_distribution(species)
        monthly = result.get("monthly_distribution", [])

        if not monthly:
            raise ValueError(f"{species} has no monthly data available for plotting.")

        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        counts = [int(x["count"]) for x in monthly]
        ratios = [float(x["ratio"]) for x in monthly]
        top_idx = int(np.argmax(counts))
        valid_records = int(result.get("valid_date_records", sum(counts)))

        subtitle = (
            f"n = {valid_records} detections with valid calendar date; peak month = "
            f"{month_labels[top_idx]} ({counts[top_idx]} detections, {ratios[top_idx]:.1%})."
        )
        note = (
            "Lines show month-level detections; the dotted reference line marks a uniform monthly expectation "
            "if detections were evenly distributed across the calendar year."
        )

        if save_path is None:
            save_path = self._build_path(f"{self._safe_name(species)}_monthly_activity.png")

        return self._plot_categorical_activity_bar(
            month_labels,
            counts,
            counts=counts,
            ratios=ratios,
            title=f"Monthly activity profile of {species}",
            subtitle=subtitle,
            xlabel="Month",
            ylabel="Observation count",
            panel_label="a",
            save_path=save_path,
            filename=f"{self._safe_name(species)}_monthly_activity.png",
            note=note,
        )

    # =========================
    # 4. Potential transition-corridor map
    # =========================
    def plot_transition_corridor_map(
            self,
            species=None,
            max_time_gap_hours=72,
            max_distance_km=100,
            top_k=10,
            steps=None,
            save_path=None
    ):
        """
        Draw potential transition corridors on a real map.
        Output an HTML map.
        """
        corridors = self.infer_transition_corridors(
            species=species,
            max_time_gap_hours=max_time_gap_hours,
            max_distance_km=max_distance_km,
            top_k=top_k,
            steps=steps
        )

        if not corridors:
            raise ValueError(f"{species} has no potential transition corridors available for plotting.")

        camera_locations = self.get_camera_locations()

        used_points = []
        for edge in corridors:
            loc_a = camera_locations.get(edge["from_camera"])
            loc_b = camera_locations.get(edge["to_camera"])
            if loc_a and loc_b:
                used_points.append((loc_a["latitude"], loc_a["longitude"]))
                used_points.append((loc_b["latitude"], loc_b["longitude"]))

        if not used_points:
            raise ValueError(f"{species} has potential transition corridors, but plottable coordinates are missing.")

        center_lat = sum(p[0] for p in used_points) / len(used_points)
        center_lon = sum(p[1] for p in used_points) / len(used_points)

        m = self._create_base_map(center_lat, center_lon, zoom_start=9)
        self._add_park_boundary(m)

        max_count = max(x["count"] for x in corridors) if corridors else 1
        used_nodes = set()

        # Draw lines first
        for edge in corridors:
            cam_a = edge["from_camera"]
            cam_b = edge["to_camera"]
            count = edge["count"]

            loc_a = camera_locations.get(cam_a)
            loc_b = camera_locations.get(cam_b)
            if not loc_a or not loc_b:
                continue

            used_nodes.add(cam_a)
            used_nodes.add(cam_b)

            weight = self._count_to_line_weight(count, max_count)

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; padding: 6px;">
                <h4 style="margin-bottom: 6px;">Transition Corridor</h4>
                <p><strong>Species:</strong> {species}</p>
                <p><strong>From:</strong> {cam_a}</p>
                <p><strong>To:</strong> {cam_b}</p>
                <p><strong>Count:</strong> {count}</p>
                <p><strong>Max time gap:</strong> {max_time_gap_hours} h</p>
                <p><strong>Max distance:</strong> {max_distance_km} km</p>
            </div>
            """

            folium.PolyLine(
                locations=[
                    [loc_a["latitude"], loc_a["longitude"]],
                    [loc_b["latitude"], loc_b["longitude"]],
                ],
                color="#6a3d9a",
                weight=weight,
                opacity=0.75,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{cam_a} → {cam_b}: {count}"
            ).add_to(m)

            mid_lat = (loc_a["latitude"] + loc_b["latitude"]) / 2
            mid_lon = (loc_a["longitude"] + loc_b["longitude"]) / 2
            folium.Marker(
                [mid_lat, mid_lon],
                icon=folium.DivIcon(html=f"""
                    <div style="
                        font-size: 11px;
                        color: #222;
                        background: rgba(255,255,255,0.88);
                        border: 1px solid #ccc;
                        border-radius: 4px;
                        padding: 1px 4px;
                        white-space: nowrap;
                    ">{count}</div>
                """)
            ).add_to(m)

        # Draw nodes afterward
        for cam in used_nodes:
            loc = camera_locations.get(cam)
            if not loc:
                continue

            folium.CircleMarker(
                location=[loc["latitude"], loc["longitude"]],
                radius=6,
                color="white",
                weight=1,
                fill=True,
                fill_color="#1f78b4",
                fill_opacity=0.9,
                tooltip=f"Camera: {cam}"
            ).add_to(m)

            folium.Marker(
                [loc["latitude"], loc["longitude"]],
                icon=folium.DivIcon(html=f"""
                    <div style="
                        font-size: 11px;
                        color: #111;
                        background: rgba(255,255,255,0.9);
                        border: 1px solid #ccc;
                        border-radius: 4px;
                        padding: 1px 4px;
                        white-space: nowrap;
                    ">{cam}</div>
                """)
            ).add_to(m)

        self._fit_map_to_points(m, used_points)
        self._add_map_controls(m)
        self._add_default_legend(
            m,
            title="Transition Corridors",
            items=[
                ("Camera nodes", "#1f78b4"),
                ("Transition corridors", "#6a3d9a"),
            ]
        )

        if save_path is None:
            save_path = self._build_html_path(f"{self._safe_name(species)}_transition_corridors.html")

        return self._save_map(m, save_path)

    def plot_migration_corridor_map(
            self,
            species="Siberian tiger",
            max_time_gap_hours=72,
            max_distance_km=100,
            top_k=10,
            steps=None,
            save_path=None
    ):
        """
        Backward-compatible alias.
        New code should use plot_transition_corridor_map.
        """
        return self.plot_transition_corridor_map(
            species=species,
            max_time_gap_hours=max_time_gap_hours,
            max_distance_km=max_distance_km,
            top_k=top_k,
            steps=steps,
            save_path=save_path
        )

    def plot_transition_corridor_static(
        self,
        species="Siberian tiger",
        max_time_gap_hours=72,
        max_distance_km=100,
        top_k=10,
        steps=None,
        save_path=None,
    ):
        """
        Publication style: static potential transition-corridor map (PNG).
        """
        corridors = self.infer_transition_corridors(
            species=species,
            max_time_gap_hours=max_time_gap_hours,
            max_distance_km=max_distance_km,
            top_k=top_k,
            steps=steps,
        )
        if not corridors:
            raise ValueError(f"{species} has no potential transition corridors available for plotting.")

        camera_locations = self.get_camera_locations()
        rows = []
        for edge in corridors:
            loc_a = camera_locations.get(edge["from_camera"])
            loc_b = camera_locations.get(edge["to_camera"])
            if not loc_a or not loc_b:
                continue
            rows.append({
                "from_camera": str(edge["from_camera"]),
                "to_camera": str(edge["to_camera"]),
                "count": int(edge["count"]),
                "from_latitude": float(loc_a["latitude"]),
                "from_longitude": float(loc_a["longitude"]),
                "to_latitude": float(loc_b["latitude"]),
                "to_longitude": float(loc_b["longitude"]),
            })
        if not rows:
            raise ValueError(f"{species} has potential transition corridors, but plottable coordinates are missing.")

        edge_df = pd.DataFrame(rows)
        node_rows = []
        for _, row in edge_df.iterrows():
            node_rows.append({
                "camera_id": row["from_camera"],
                "latitude": row["from_latitude"],
                "longitude": row["from_longitude"],
            })
            node_rows.append({
                "camera_id": row["to_camera"],
                "latitude": row["to_latitude"],
                "longitude": row["to_longitude"],
            })
        node_df = pd.DataFrame(node_rows).drop_duplicates("camera_id")
        bounds = self._compute_plot_bounds(node_df, pad_ratio=0.22)
        park_window = self._clip_park_to_bounds(self._load_park_gdf(), bounds)

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(8.8, 7.0))
        if park_window is not None:
            park_window.plot(
                ax=ax,
                facecolor="#F7F3E8",
                edgecolor="#66737A",
                linewidth=0.9,
                alpha=0.72,
                zorder=0,
            )

        max_count = max(int(edge["count"]) for edge in rows)
        for _, edge in edge_df.sort_values("count").iterrows():
            width = 1.0 + 3.5 * float(edge["count"]) / max(max_count, 1)
            alpha = 0.35 + 0.45 * float(edge["count"]) / max(max_count, 1)
            ax.plot(
                [edge["from_longitude"], edge["to_longitude"]],
                [edge["from_latitude"], edge["to_latitude"]],
                color=SCI_COLORS["blue"],
                linewidth=width,
                alpha=alpha,
                solid_capstyle="round",
                zorder=2,
            )
            mid_lon = (edge["from_longitude"] + edge["to_longitude"]) / 2.0
            mid_lat = (edge["from_latitude"] + edge["to_latitude"]) / 2.0
            if int(edge["count"]) >= 2:
                ax.text(
                    mid_lon,
                    mid_lat,
                    str(int(edge["count"])),
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color=SCI_COLORS["dark"],
                    bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.82},
                    zorder=4,
                )

        node_counts = {}
        for _, edge in edge_df.iterrows():
            node_counts[edge["from_camera"]] = node_counts.get(edge["from_camera"], 0) + int(edge["count"])
            node_counts[edge["to_camera"]] = node_counts.get(edge["to_camera"], 0) + int(edge["count"])
        node_df["traffic"] = node_df["camera_id"].map(node_counts).fillna(1).astype(float)
        ax.scatter(
            node_df["longitude"],
            node_df["latitude"],
            s=[self._count_to_marker_size_static(v, min_size=55, scale=16, max_size=240) for v in node_df["traffic"]],
            color=SCI_COLORS["orange"],
            edgecolors=SCI_COLORS["dark"],
            linewidths=0.7,
            alpha=0.94,
            zorder=3,
        )

        label_df = node_df.nlargest(min(5, len(node_df)), "traffic")
        self._annotate_points_with_offsets(
            ax,
            label_df,
            lambda row: str(row["camera_id"]),
            font_size=8.0,
            max_labels=5,
        )

        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        self._apply_clean_axis(ax, grid_axis="both")
        self._add_caption_title(ax, f"Potential transition corridors for {species}")
        ax.legend(
            handles=[
                Line2D([0], [0], color=SCI_COLORS["blue"], lw=3.0, label="Potential transition"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor=SCI_COLORS["orange"], markeredgecolor=SCI_COLORS["dark"], markersize=7, label="Camera node"),
            ],
            frameon=False,
            loc="lower right",
        )

        if save_path is None:
            save_path = self._build_path(f"{self._safe_name(species)}_transition_corridors.png")
        return self._finalize_paper_plot(fig, save_path)

    # =========================
    # 5. Weekday activity line chart
    # =========================
    def plot_weekday_activity_bar(self, species=None, save_path=None):
        result = self.build_weekday_activity_distribution(species)
        weekday = result.get("weekday_distribution", [])

        if not weekday:
            raise ValueError(f"{species} has no weekday data available for plotting.")

        names = [x["name"] for x in weekday]
        counts = [int(x["count"]) for x in weekday]
        ratios = [float(x["ratio"]) for x in weekday]
        top_idx = int(np.argmax(counts))
        total = int(sum(counts))
        subtitle = (
            f"n = {total} detections with resolvable weekday; highest weekday = "
            f"{names[top_idx]} ({counts[top_idx]} detections, {ratios[top_idx]:.1%})."
        )
        note = (
            "This figure summarizes date-level activity only. The uniform reference line helps separate "
            "true weekday concentration from small count fluctuations."
        )

        if save_path is None:
            save_path = self._build_path(f"{self._safe_name(species)}_weekday_activity.png")

        return self._plot_categorical_activity_bar(
            names,
            counts,
            counts=counts,
            ratios=ratios,
            title=f"Weekday activity profile of {species}",
            subtitle=subtitle,
            xlabel="Weekday",
            ylabel="Observation count",
            panel_label="a",
            save_path=save_path,
            filename=f"{self._safe_name(species)}_weekday_activity.png",
            note=note,
        )

    # =========================
    # 6. Seasonal activity line chart
    # =========================
    def plot_seasonal_activity_bar(self, species=None, use_ratio=False, save_path=None):
        result = self.compare_seasonal_activity(species)
        season_counts = result.get("season_total_records", {})
        season_ratios = result.get("season_ratio", {})
        month_in_season = result.get("month_in_season_counter", {})

        if use_ratio:
            data = season_ratios
            ylabel = "Observation share"
            suffix = "seasonal_activity_ratio"
        else:
            data = season_counts
            ylabel = "Observation count"
            suffix = "seasonal_activity_count"

        if not data:
            raise ValueError(f"{species} has no seasonal data available for plotting.")

        seasons = ["spring", "summer", "autumn", "winter"]
        season_labels = ["Spring", "Summer", "Autumn", "Winter"]
        counts = [int(season_counts.get(s, 0)) for s in seasons]
        ratios = [float(season_ratios.get(s, 0.0)) for s in seasons]
        values = ratios if use_ratio else counts
        top_idx = int(np.argmax(values))
        dominant_months = month_in_season.get(seasons[top_idx], {})
        month_breakdown = ", ".join(f"{month}:{count}" for month, count in dominant_months.items())
        subtitle = (
            f"Dominant season = {season_labels[top_idx]} "
            f"({counts[top_idx]} detections, {ratios[top_idx]:.1%}); monthly composition within season: "
            f"{month_breakdown}."
        )
        note = (
            "Season blocks follow meteorological grouping (spring: Mar-May, summer: Jun-Aug, autumn: Sep-Nov, "
            "winter: Dec-Feb). The dotted line indicates an equal-share expectation across the four seasons."
        )

        if save_path is None:
            save_path = self._build_path(f"{self._safe_name(species)}_{suffix}.png")

        return self._plot_categorical_activity_bar(
            season_labels,
            values,
            counts=counts,
            ratios=ratios,
            title=f"Seasonal activity profile of {species}",
            subtitle=subtitle,
            xlabel="Season",
            ylabel=ylabel,
            panel_label="a",
            save_path=save_path,
            filename=f"{self._safe_name(species)}_{suffix}.png",
            y_as_percent=use_ratio,
            note=note,
        )

    # =========================
    # 7. Hotspot camera Top-K line chart
    # =========================
    def plot_hotspot_rank_bar(self, species=None, top_k=10, save_path=None):
        """
        Backward-compatible wrapper for the publication-style hotspot line chart.
        """
        return self.plot_hotspot_rank_barh(
            species=species,
            top_k=top_k,
            save_path=save_path,
        )

    def plot_hotspot_rank_barh(self, species=None, top_k=10, save_path=None):
        """
        Publication style: hotspot-ranking line chart.
        """
        hotspot_rank = self.build_hotspot_rank(species, top_k=top_k)
        if not hotspot_rank:
            raise ValueError(f"{species} has no hotspot data available for plotting.")

        rank_df = pd.DataFrame(
            {
                "camera_id": [str(x["camera_id"]) for x in hotspot_rank],
                "count": [int(x["count"]) for x in hotspot_rank],
            }
        ).sort_values("count", ascending=False)
        rank_df["public_rank"] = [
            f"Rank {rank}"
            for rank in range(1, len(rank_df) + 1)
        ]
        total_observations = max(
            int(sum(item["count"] for item in self.build_detection_map(species))),
            1,
        )
        shares = rank_df["count"] / total_observations

        plot_df = rank_df.iloc[::-1].reset_index(drop=True)
        plot_shares = shares.iloc[::-1].reset_index(drop=True)

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(8.6, max(4.6, 0.35 * len(plot_df) + 1.5)))

        y = np.arange(len(plot_df))
        counts = plot_df["count"].to_numpy(dtype=float)
        colors = [SCI_COLORS["blue"]] * len(plot_df)
        if colors:
            colors[-1] = SCI_COLORS["orange"]

        ax.hlines(y, 0, counts, color=colors, linewidth=2.2, alpha=0.72, zorder=2)
        ax.scatter(
            counts,
            y,
            s=[58 if idx < len(counts) - 1 else 78 for idx in range(len(counts))],
            color=colors,
            edgecolors="#1F3A4D",
            linewidths=0.75,
            zorder=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["public_rank"])
        ax.set_xlabel("Observation count")
        ax.set_ylabel("Hotspot rank (camera IDs withheld)")
        ax.set_xlim(0, max(counts.max(), 1.0) * 1.22)
        self._apply_clean_axis(ax, grid_axis="x")

        for idx, (value, share) in enumerate(zip(counts, plot_shares)):
            ax.text(
                value + max(counts.max(), 1.0) * 0.015,
                y[idx],
                f"{int(value)} | {share:.1%}",
                ha="left",
                va="center",
                fontsize=8.5,
            )

        self._add_caption_title(
            ax,
            f"Hotspot ranking for {species}",
        )
        self._annotate_panel_label(ax, "c")

        if save_path is None:
            save_path = self._build_path(
                f"{self._safe_name(species)}_hotspot_top{top_k}_barh.png"
            )

        return self._finalize_paper_plot(fig, save_path)

    # =========================
    # 8. Movement-direction line chart
    # =========================
    def plot_movement_direction_bar(self, species=None,
                                    max_time_gap_hours=72, max_distance_km=100,
                                    steps=None, save_path=None):
        direction_stats = self.summarize_movement_direction(
            species=species,
            max_time_gap_hours=max_time_gap_hours,
            max_distance_km=max_distance_km,
            steps=steps
        )

        if not direction_stats:
            raise ValueError(f"{species} has no movement-direction data available for plotting.")

        direction_order = ["north", "east", "south", "west"]
        direction_map = {item["direction"]: item for item in direction_stats}
        plot_rows = [direction_map[d] for d in direction_order if d in direction_map]
        if not plot_rows:
            plot_rows = direction_stats

        labels = [row["direction"].capitalize() for row in plot_rows]
        counts = np.asarray([int(row["count"]) for row in plot_rows], dtype=float)
        ratios = np.asarray([float(row["ratio"]) for row in plot_rows], dtype=float)
        dominant_idx = int(np.argmax(counts))

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        y = np.arange(len(labels))
        colors = [SCI_COLORS["sky"]] * len(labels)
        if len(colors) > dominant_idx:
            colors[dominant_idx] = SCI_COLORS["orange"]

        ax.barh(
            y,
            counts,
            color=colors,
            edgecolor="#1F3A4D",
            linewidth=0.6,
            alpha=0.90,
            zorder=2,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Potential transition count")
        ax.set_ylabel("Direction")
        ax.set_xlim(0, max(counts.max(), 1.0) * 1.18)
        self._apply_clean_axis(ax, grid_axis="x")

        for idx, value in enumerate(counts):
            ax.text(
                value + max(counts.max(), 1.0) * 0.015,
                y[idx],
                f"{int(counts[idx])} | {ratios[idx]:.1%}",
                ha="left",
                va="center",
                fontsize=8.5,
            )

        self._add_caption_title(
            ax,
            f"Directional tendency of inferred {species} movements",
        )
        self._annotate_panel_label(ax, "a")
        self._add_figure_note(
            fig,
            "Direction is summarized from camera-to-camera potential transitions rather than individual "
            "animal trajectories; labels show count and share for each cardinal direction.",
            width=105,
            bottom=0.20,
        )

        if save_path is None:
            save_path = self._build_path(f"{self._safe_name(species)}_movement_direction.png")

        return self._finalize_paper_plot(fig, save_path)

    # =========================
    # 9. Movement-distance distribution figure
    # =========================
    def plot_movement_distance_hist(self, species=None,
                                    max_time_gap_hours=72, max_distance_km=100,
                                    bins=10, steps=None, save_path=None):
        if steps is None:
            steps = self.analyze_movement_steps(
                species=species,
                max_time_gap_hours=max_time_gap_hours,
                max_distance_km=max_distance_km
            )

        if not steps:
            raise ValueError(f"{species} has no movement-distance data available for plotting.")

        distances = np.asarray([float(x["distance_km"]) for x in steps], dtype=float)
        mean_dist = float(distances.mean())
        median_dist = float(np.median(distances))
        max_dist = float(distances.max())
        p75_dist = float(np.percentile(distances, 75))
        density_x, density_y = self._estimate_density_curve(distances, lower_bound=0.0)
        bin_edges = np.histogram_bin_edges(distances, bins=bins)
        bin_width = float(np.mean(np.diff(bin_edges))) if len(bin_edges) > 1 else 1.0
        density_scaled = density_y * len(distances) * bin_width if density_x.size else np.array([])

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(8.9, 5.4))
        ax.hist(
            distances,
            bins=bin_edges,
            color=SCI_COLORS["sky"],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.55,
            zorder=2,
        )
        if density_x.size:
            ax.plot(
                density_x,
                density_scaled,
                color=SCI_COLORS["blue"],
                linewidth=2.0,
                zorder=4,
            )
            ax.fill_between(density_x, density_scaled, color=SCI_COLORS["blue"], alpha=0.10, zorder=3)

        reference_specs = [
            ("Mean", mean_dist, SCI_COLORS["orange"]),
            ("Median", median_dist, SCI_COLORS["purple"]),
            ("75th percentile", p75_dist, SCI_COLORS["grey"]),
        ]
        ymax = ax.get_ylim()[1]
        for idx, (label, value, color) in enumerate(reference_specs):
            ax.axvline(value, color=color, linestyle="--", linewidth=1.0, alpha=0.9, zorder=5)
            ax.text(
                value,
                ymax * (0.96 - idx * 0.07),
                f"{label}: {value:.1f} km",
                rotation=90,
                ha="right",
                va="top",
                fontsize=7.8,
                color=color,
                backgroundcolor="white",
            )

        ax.set_xlabel("Distance between adjacent cameras (km)")
        ax.set_ylabel("Potential transition count")
        self._apply_clean_axis(ax, grid_axis="y")
        self._add_caption_title(
            ax,
            f"Distribution of inferred {species} movement distances",
            subtitle=(
                f"n = {len(distances)} potential transitions; mean = {mean_dist:.2f} km; "
                f"median = {median_dist:.2f} km; max = {max_dist:.2f} km; thresholds: "
                f"≤ {max_time_gap_hours} h and ≤ {max_distance_km} km."
            ),
        )
        self._annotate_panel_label(ax, "a")
        ax.legend(
            handles=[
                Patch(facecolor=SCI_COLORS["sky"], edgecolor="white", alpha=0.55, label="Histogram"),
                Line2D([0], [0], color=SCI_COLORS["blue"], lw=2.0, label="Smoothed density"),
            ],
            frameon=False,
            loc="best",
        )
        self._add_figure_note(
            fig,
            "Distances are summarized from time-adjacent detections that satisfy the user-defined "
            "time and distance filters; they indicate plausible transition scales, not reconstructed paths.",
            width=108,
            bottom=0.20,
        )

        if save_path is None:
            save_path = self._build_path(f"{self._safe_name(species)}_movement_distance_hist.png")

        return self._finalize_paper_plot(fig, save_path)
    def plot_camera_quality_problem_barh(self, species=None, top_k=10, save_path=None):
        """
        Publication style: dual-encoded camera-quality issue figure.
        One line shows issue burden and the other shows quality score.
        """
        quality = self.analyze_camera_data_quality(species, top_k=top_k)
        if not quality:
            raise ValueError(f"{species} has no quality data available for plotting.")

        selected = quality[:top_k]
        quality_df = pd.DataFrame(selected)
        quality_df["camera_id"] = quality_df["camera_id"].astype(str)
        quality_df["other_issue_ratio"] = (
            quality_df["missing_behavior_ratio"]
            + quality_df["invalid_timestamp_ratio"]
            + quality_df["invalid_individuals_ratio"]
        )
        quality_df["total_problem_ratio"] = (
            quality_df["other_issue_ratio"] + quality_df["duplicate_burden_ratio"]
        )
        quality_df = quality_df.sort_values(
            ["quality_score", "duplicate_burden_ratio", "total_records"],
            ascending=[True, False, False],
        )

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(quality_df))
        other_issue = quality_df["other_issue_ratio"].to_numpy(dtype=float)
        duplicate = quality_df["duplicate_burden_ratio"].to_numpy(dtype=float)
        total_problem = quality_df["total_problem_ratio"].to_numpy(dtype=float)
        quality_scores = quality_df["quality_score"].to_numpy(dtype=float)

        ax.plot(
            x,
            total_problem,
            color=SCI_COLORS["predator"],
            linewidth=2.1,
            marker="o",
            markersize=6.0,
            markerfacecolor=SCI_COLORS["predator"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="Total problem ratio",
            zorder=4,
        )
        ax.plot(
            x,
            duplicate,
            color=SCI_COLORS["accent"],
            linewidth=1.8,
            marker="s",
            markersize=5.2,
            markerfacecolor=SCI_COLORS["accent"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            label="Duplicate burden",
            zorder=3,
        )
        if not np.allclose(other_issue, 0):
            ax.plot(
                x,
                other_issue,
                color=SCI_COLORS["neutral"],
                linewidth=1.8,
                marker="^",
                markersize=5.2,
                markerfacecolor=SCI_COLORS["neutral"],
                markeredgecolor="white",
                markeredgewidth=0.7,
                label="Other issue burden",
                zorder=3,
            )
        ax.fill_between(
            x,
            total_problem,
            color=SCI_COLORS["predator"],
            alpha=0.10,
            zorder=1,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(quality_df["camera_id"], rotation=28, ha="right")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        max_problem_ratio = max(float(quality_df["total_problem_ratio"].max()), 0.05)
        ax.set_ylim(0, min(max_problem_ratio * 1.30, 1.05))
        ax.set_xlim(-0.45, len(quality_df) - 0.55)

        for i, total_v in enumerate(total_problem):
            ax.text(
                x[i],
                min(total_v + max_problem_ratio * 0.045, 1.02),
                f"{total_v:.0%}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        ax_score = ax.twinx()
        ax_score.plot(
            x,
            quality_scores,
            color="#1f3b4d",
            marker="D",
            linewidth=1.8,
            markersize=5.2,
            markerfacecolor="#1f3b4d",
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=5,
            label="Quality score",
        )
        ax_score.set_ylim(0, 100)
        ax_score.set_ylabel("Quality score")
        ax_score.grid(False)

        for i, score in enumerate(quality_scores):
            ax_score.text(
                x[i],
                max(score - 2.8, 0),
                f"{score:.1f}",
                ha="center",
                va="top",
                fontsize=8.2,
                color="#1f3b4d",
            )

        ax.set_xlabel("Camera ID")
        ax.set_ylabel("Problem ratio")
        self._apply_clean_axis(ax, grid_axis="y")

        self._add_caption_title(
            ax,
            f"Lowest-scoring cameras for {species}",
        )

        handles, labels = ax.get_legend_handles_labels()
        score_handles, score_labels = ax_score.get_legend_handles_labels()
        ax.legend(
            handles + score_handles,
            labels + score_labels,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        self._annotate_panel_label(ax, "d")

        if save_path is None:
            save_path = self._build_path(
                f"{self._safe_name(species)}_camera_quality_problem_barh.png"
            )

        return self._finalize_paper_plot(fig, save_path)
    def plot_species_distribution_static(
        self,
        species=None,
        annotate_top_k=5,
        privacy_grid_degrees=0.1,
        save_path=None,
    ):
        """
        Publication-style detection map.

        Public output is privacy-safe by default: camera coordinates are
        aggregated to grid-cell centres and camera identifiers are omitted.
        Pass ``privacy_grid_degrees=None`` only for an access-controlled
        internal figure whose exact locations are authorised for display.
        """
        detection_map = self.build_detection_map(species)
        df = self._build_dataframe_from_detection_map(detection_map)

        if df.empty:
            raise ValueError(f"{species} has no spatial data available for plotting.")

        park_gdf = self._load_park_gdf()
        df, filter_meta = self._filter_spatial_outliers(df, park_gdf=park_gdf)
        privacy_safe = privacy_grid_degrees is not None
        if privacy_safe:
            grid_degrees = float(privacy_grid_degrees)
            if not math.isfinite(grid_degrees) or grid_degrees <= 0:
                raise ValueError("privacy_grid_degrees must be a positive finite value or None.")
            df = df.copy()
            df["longitude"] = (
                np.floor(df["longitude"].astype(float) / grid_degrees) + 0.5
            ) * grid_degrees
            df["latitude"] = (
                np.floor(df["latitude"].astype(float) / grid_degrees) + 0.5
            ) * grid_degrees
            aggregation = {"count": "sum"}
            if "camera_id" in df.columns:
                aggregation["camera_id"] = "nunique"
            df = (
                df.groupby(["longitude", "latitude"], as_index=False)
                .agg(aggregation)
                .rename(columns={"camera_id": "camera_count"})
            )
            df["camera_id"] = [
                f"masked_grid_cell_{index}"
                for index in range(1, len(df) + 1)
            ]
        top_df = df.nlargest(min(annotate_top_k, len(df)), "count").copy()
        bounds = self._compute_plot_bounds(df, pad_ratio=0.08)
        park_window = self._clip_park_to_bounds(park_gdf, bounds)

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(8.8, 7.2))

        if park_window is not None:
            park_window.plot(
                ax=ax,
                facecolor="#f4efe1",
                edgecolor="#2f3e46",
                linewidth=0.9,
                alpha=0.65,
                zorder=0,
            )

        sizes = [self._count_to_marker_size_static(c) for c in df["count"]]
        ax.scatter(
            df["longitude"], df["latitude"],
            s=sizes,
            color="#74a9cf",
            alpha=0.78,
            edgecolors="white",
            linewidths=0.5,
            label="Masked grid cells" if privacy_safe else "Monitoring cameras",
            zorder=2,
        )

        ax.scatter(
            top_df["longitude"], top_df["latitude"],
            s=[
                self._count_to_marker_size_static(c, min_size=70, scale=24, max_size=340)
                for c in top_df["count"]
            ],
            color="#f28e2b",
            alpha=0.95,
            edgecolors="#1f1f1f",
            linewidths=0.9,
            label=(
                f"Top {len(top_df)} masked hotspot cells"
                if privacy_safe
                else f"Top {len(top_df)} hotspots"
            ),
            zorder=3,
        )

        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        self._annotate_points_with_offsets(
            ax,
            top_df.sort_values("count", ascending=False),
            (
                (lambda row: f"{int(row['count'])} events")
                if privacy_safe
                else (lambda row: f"{row['camera_id']} ({int(row['count'])})")
            ),
            font_size=8.2,
        )

        ax.set_xlabel("Approximate longitude" if privacy_safe else "Longitude")
        ax.set_ylabel("Approximate latitude" if privacy_safe else "Latitude")
        self._apply_clean_axis(ax, grid_axis="both")
        subtitle = "Marker size scales with independent-event count."
        if privacy_safe:
            subtitle += (
                f" Camera identifiers are omitted and coordinates are aggregated "
                f"to {grid_degrees:g}-degree cells."
            )
        if filter_meta["removed"] > 0:
            subtitle += f" {filter_meta['removed']} outlier camera point(s) were excluded to preserve the study-area scale."
        self._add_caption_title(
            ax,
            f"Spatial distribution of {species}",
            subtitle=subtitle,
        )
        ax.legend(frameon=False, loc="lower right")
        self._annotate_panel_label(ax, "a")

        if save_path is None:
            suffix = "privacy_safe_distribution" if privacy_safe else "distribution_static"
            save_path = self._build_path(
                f"{self._safe_name(species)}_{suffix}.png"
            )

        figure_path = self._finalize_paper_plot(fig, save_path)
        top_display_units = []
        for rank, (_, row) in enumerate(
            top_df.sort_values("count", ascending=False).iterrows(),
            start=1,
        ):
            item = {
                "rank": rank,
                "independent_event_count": int(row["count"]),
            }
            if privacy_safe:
                item["aggregation_unit"] = "masked_grid_cell"
                item["camera_count_in_cell"] = int(row.get("camera_count", 0))
            else:
                item["aggregation_unit"] = "monitoring_camera"
                item["camera_id"] = str(row["camera_id"])
            top_display_units.append(item)

        return {
            "status": "success",
            "species": species,
            "privacy_safe": privacy_safe,
            "privacy_grid_degrees": grid_degrees if privacy_safe else None,
            "aggregation_unit": (
                "masked_grid_cell" if privacy_safe else "monitoring_camera"
            ),
            "annotated_top_k": len(top_display_units),
            "top_display_units": top_display_units,
            "interpretation_boundary": (
                "The public map aggregates independent-event counts to masked "
                f"{grid_degrees:g}-degree grid cells; annotated cells are not "
                "individual cameras, occupancy estimates, or density estimates."
                if privacy_safe
                else "The map shows camera-level independent-event counts, not "
                "occupancy or density estimates."
            ),
            "figures": [{
                "figure_id": f"{self._safe_name(species)}_privacy_safe_distribution",
                "label": f"Privacy-safe spatial distribution of {species}",
                "svg_path": str(figure_path),
            }],
        }
    def plot_local_hotspot_static(
        self,
        species=None,
        center_camera_id="213215",
        radius_km=20,
        annotate_top_k=5,
        save_path=None
    ):
        """
        Publication style: static local-hotspot map (PNG).
        """
        result = self.analyze_spatial_distribution(
            species=species,
            center_camera_id=center_camera_id,
            radius_km=radius_km,
            top_k=annotate_top_k
        )

        local = result.get("local_analysis", {})
        local_detection = local.get("nearby_detection_map", [])
        nearby_cameras = local.get("nearby_cameras", [])

        camera_locations = self.get_camera_locations()
        center = camera_locations.get(str(center_camera_id))
        if center is None:
            raise ValueError(f"Center camera {center_camera_id} does not exist or has no coordinates.")

        center_lat = float(center["latitude"])
        center_lon = float(center["longitude"])

        det_rows = []
        for x in local_detection:
            if x.get("latitude") is None or x.get("longitude") is None:
                continue
            det_rows.append({
                "camera_id": str(x["camera_id"]),
                "count": int(x["count"]),
                "latitude": float(x["latitude"]),
                "longitude": float(x["longitude"]),
            })
        det_df = pd.DataFrame(det_rows)

        near_rows = []
        for x in nearby_cameras:
            if x.get("latitude") is None or x.get("longitude") is None:
                continue
            near_rows.append({
                "camera_id": str(x["camera_id"]),
                "latitude": float(x["latitude"]),
                "longitude": float(x["longitude"]),
            })
        near_df = pd.DataFrame(near_rows)

        park_gdf = self._load_park_gdf()
        label_k = min(max(int(annotate_top_k), 1), 5)
        top_det_df = (
            det_df.nlargest(label_k, "count").copy().sort_values("count", ascending=False)
            if not det_df.empty
            else pd.DataFrame(columns=["camera_id", "count", "latitude", "longitude"])
        )
        bounds_source = det_df if not det_df.empty else near_df
        bounds = self._compute_plot_bounds(
            bounds_source,
            center=(center_lon, center_lat),
            radius_km=radius_km,
            pad_ratio=0.08,
        )
        park_window = self._clip_park_to_bounds(park_gdf, bounds)

        self._set_paper_style()
        fig, ax = plt.subplots(figsize=(8.8, 7.2))

        if park_window is not None:
            park_window.plot(
                ax=ax,
                facecolor="none",
                edgecolor="#2b2b2b",
                linewidth=1.0,
                zorder=0,
            )

        if not near_df.empty:
            ax.scatter(
                near_df["longitude"], near_df["latitude"],
                s=32,
                color="#9ecae1",
                alpha=0.55,
                edgecolors="white",
                linewidths=0.4,
                label="Nearby cameras",
                zorder=1,
            )

        if not det_df.empty:
            ax.scatter(
                det_df["longitude"], det_df["latitude"],
                s=[self._count_to_marker_size_static(c) for c in det_df["count"]],
                color="#f28e2b",
                alpha=0.78,
                edgecolors="white",
                linewidths=0.5,
                label="Local detections",
                zorder=2,
            )

            ax.scatter(
                top_det_df["longitude"], top_det_df["latitude"],
                s=[
                    self._count_to_marker_size_static(c, min_size=68, scale=24, max_size=310)
                    for c in top_det_df["count"]
                ],
                color="#2ca02c",
                alpha=0.95,
                edgecolors="#1f1f1f",
                linewidths=0.85,
                label=f"Top {len(top_det_df)} local hotspots",
                zorder=3,
            )

        search_ellipse = Ellipse(
            xy=(center_lon, center_lat),
            width=(radius_km / max(111.0 * np.cos(np.radians(center_lat)), 1e-6)) * 2.0,
            height=(radius_km / 111.0) * 2.0,
            facecolor="none",
            edgecolor="#9c6644",
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            zorder=1,
        )
        ax.add_patch(search_ellipse)

        ax.scatter(
            [center_lon], [center_lat],
            s=180,
            marker="*",
            color=SCI_COLORS["accent"],
            edgecolors="black",
            linewidths=0.8,
            label=f"Center camera {center_camera_id}",
            zorder=4,
        )
        ax.annotate(
            f"Center {center_camera_id}",
            xy=(center_lon, center_lat),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=8.4,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#9aa6b2", "alpha": 0.9},
        )

        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        self._annotate_points_with_offsets(
            ax,
            top_det_df,
            lambda row: f"{row['camera_id']} ({int(row['count'])})",
            font_size=8.1,
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        self._apply_clean_axis(ax, grid_axis="both")
        self._add_caption_title(
            ax,
            f"Local hotspot pattern around camera {center_camera_id}",
            subtitle=(
                f"Window clipped to the {radius_km} km search neighborhood; labels are shown for the top "
                f"{len(top_det_df)} local hotspots."
            ),
        )
        ax.legend(frameon=False, loc="lower left")
        self._annotate_panel_label(ax, "b")

        if save_path is None:
            save_path = self._build_path(
                f"{self._safe_name(species)}_local_{center_camera_id}_{radius_km}km_static.png"
            )

        return self._finalize_paper_plot(fig, save_path)
