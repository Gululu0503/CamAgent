from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path

from .paths import CAMTRAP_IMPORTS_DIR, MPLCONFIG_DIR, OUTPUTS_DIR, ensure_runtime_dirs

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .ecology_config import (
    ECOLOGY_COLORS,
    FIGURE_SPECS,
    SPECIES_CN,
    normalize_ecological_role_mapping,
)


HEATMAP_CMAP = "viridis"
SPATIAL_INTENSITY_CMAP = "YlGnBu"
ASSOCIATION_CMAP = "BrBG"


def season_name(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def load_ecology_dataset(package_dir):
    package_dir = Path(package_dir)
    events = pd.read_csv(package_dir / "agent_independent_events.csv")
    deployments = pd.read_csv(package_dir / "deployments.csv")
    report = json.loads((package_dir / "conversion_report.json").read_text(encoding="utf-8"))

    events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce", utc=True)
    events = events.dropna(subset=["event_time", "camera_id", "species"])
    events["camera_id"] = events["camera_id"].astype(str)
    events["date"] = events["event_time"].dt.date
    events["hour"] = events["event_time"].dt.hour
    events["month"] = events["event_time"].dt.month
    events["season"] = events["month"].map(season_name)

    deployments["cameraID"] = deployments["cameraID"].astype(str)
    deployments["deploymentStart"] = pd.to_datetime(deployments["deploymentStart"], errors="coerce", utc=True)
    deployments["deploymentEnd"] = pd.to_datetime(deployments["deploymentEnd"], errors="coerce", utc=True)
    effort = (deployments["deploymentEnd"].dt.date - deployments["deploymentStart"].dt.date).map(
        lambda delta: delta.days if pd.notna(delta) else 0
    ) + 1
    deployments["effort_days"] = effort.fillna(1).astype(int).clip(lower=1)
    return events, deployments, report


def _species_order(events, species_order=None):
    observed_species = set(events["species"].dropna().astype(str))
    if species_order:
        configured = [str(species) for species in species_order]
        observed = [species for species in configured if species in observed_species]
        extra = sorted(observed_species - set(observed))
        return observed + extra
    counts = events["species"].dropna().astype(str).value_counts()
    observed = counts.index.tolist()
    extra = sorted(observed_species - set(observed))
    return observed + extra


def camera_table(events, deployments, species_order=None, ecological_role_mapping=None):
    species_order = _species_order(events, species_order)
    counts = events.groupby(["camera_id", "species"]).size().unstack(fill_value=0)
    for species in species_order:
        if species not in counts:
            counts[species] = 0
    dep = deployments.set_index("cameraID")[["latitude", "longitude", "effort_days"]]
    table = dep.join(counts[species_order], how="left").fillna({species: 0 for species in species_order})
    table[species_order] = table[species_order].astype(int)
    for species in species_order:
        table[f"{species}_rai"] = table[species] / table["effort_days"].clip(lower=1) * 100
    roles = normalize_ecological_role_mapping(ecological_role_mapping, species_order)
    predators = roles["predators"]
    prey = roles["prey"]
    table["richness_presence"] = (table[species_order] > 0).sum(axis=1)
    table["prey_richness_presence"] = (table[prey] > 0).sum(axis=1) if prey else 0
    totals = table[species_order].sum(axis=1)
    props = table[species_order].div(totals.replace(0, np.nan), axis=0)
    props_positive = props.where(props > 0)
    table["shannon_events"] = -(props_positive * np.log(props_positive)).sum(axis=1).fillna(0)
    table["simpson_events"] = (1 - (props ** 2).sum(axis=1)).fillna(0)
    table["prey_rai"] = table[[f"{s}_rai" for s in prey]].sum(axis=1) if prey else 0.0
    table["predator_rai"] = table[[f"{s}_rai" for s in predators]].sum(axis=1) if predators else 0.0
    table["predator_present"] = table[predators].sum(axis=1) > 0 if predators else False
    table["prey_present"] = table[prey].sum(axis=1) > 0 if prey else False
    table["total_events"] = table[species_order].sum(axis=1)
    table = table.reset_index().rename(columns={"cameraID": "camera_id"})
    table.attrs["ecological_role_mapping"] = roles
    return table


def species_metrics(events, deployments, species_order=None):
    species_order = _species_order(events, species_order)
    total_effort = float(deployments["effort_days"].sum())
    total_cameras = int(deployments["cameraID"].nunique())
    unique_event_days = events.groupby(["species", "camera_id", "date"]).size().reset_index(name="n")
    rows = []
    for species in species_order:
        subset = events[events["species"] == species]
        days = unique_event_days[unique_event_days["species"] == species]
        event_count = int(len(subset))
        occupied = int(subset["camera_id"].nunique())
        rows.append({
            "species": species,
            "cn": SPECIES_CN.get(species, species),
            "independent_events": event_count,
            "occupied_cameras": occupied,
            "naive_occupancy": occupied / total_cameras if total_cameras else 0.0,
            "rai_per_100_camera_days": event_count / total_effort * 100 if total_effort else 0.0,
            "daily_detection_probability": len(days) / total_effort if total_effort else 0.0,
            "peak_hour": int(subset["hour"].value_counts().idxmax()) if event_count else None,
            "peak_month": int(subset["month"].value_counts().idxmax()) if event_count else None,
        })
    return pd.DataFrame(rows), total_effort, total_cameras


def wilson_interval(successes, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _style(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ECOLOGY_COLORS["axis"])
    ax.spines["bottom"].set_color(ECOLOGY_COLORS["axis"])
    ax.xaxis.label.set_color(ECOLOGY_COLORS["text"])
    ax.yaxis.label.set_color(ECOLOGY_COLORS["text"])
    ax.title.set_color(ECOLOGY_COLORS["text"])
    if grid:
        ax.grid(axis=grid, color=ECOLOGY_COLORS["grid"], linewidth=0.65, alpha=0.8)
    ax.tick_params(length=3, width=0.7, labelsize=9, colors=ECOLOGY_COLORS["axis"])


def _save(fig, output_dir, spec_key):
    path = Path(output_dir) / FIGURE_SPECS[spec_key]["filename"]
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def plot_quality_conversion_audit(report, metrics, deployments, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    source = pd.DataFrame(report.get("source_files", []))
    ax = axes[0]
    if not source.empty:
        x = np.arange(len(source))
        ax.bar(x, source["raw_rows"], color=ECOLOGY_COLORS["light_green"], label="source rows")
        ax.bar(x, source["kept_rows"], color=ECOLOGY_COLORS["primary"], label="kept rows")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(source["species"], rotation=20, ha="right")
        ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("Rows, log scale")
    ax.set_title("Source rows retained", loc="left", fontweight="bold")
    _style(ax)

    ax = axes[1]
    order = metrics.sort_values("independent_events", ascending=True)
    ax.barh(order["species"], order["independent_events"], color=[ECOLOGY_COLORS.get(s, "#AFA8A6") for s in order["species"]])
    ax.set_xscale("log")
    ax.set_xlabel("30-min independent events")
    ax.set_title("Independent events", loc="left", fontweight="bold")
    _style(ax, "x")

    ax = axes[2]
    ax.scatter(deployments["longitude"], deployments["latitude"], s=4, alpha=0.28, color=ECOLOGY_COLORS["mid"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Camera coordinates after cleaning", loc="left", fontweight="bold")
    _style(ax, "both")
    fig.suptitle("Data source, event, and coordinate audit", fontweight="bold", fontsize=15)
    return _save(fig, output_dir, "quality_conversion_audit")


def plot_species_indices(metrics, total_cameras, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    order = metrics.sort_values("rai_per_100_camera_days", ascending=True)
    colors = [ECOLOGY_COLORS.get(s, "#AFA8A6") for s in order["species"]]

    ax = axes[0, 0]
    ax.barh(order["species"], order["rai_per_100_camera_days"], color=colors)
    ax.set_xlabel("RAI: independent events / 100 camera-days")
    ax.set_title("Effort-corrected activity index", loc="left", fontweight="bold")
    _style(ax, "x")

    ax = axes[0, 1]
    occupancy = metrics.sort_values("naive_occupancy", ascending=True)
    ci = np.array([wilson_interval(row.occupied_cameras, total_cameras) for row in occupancy.itertuples()])
    if len(ci):
        ax.errorbar(
            occupancy["naive_occupancy"],
            occupancy["species"],
            xerr=[occupancy["naive_occupancy"].to_numpy() - ci[:, 0], ci[:, 1] - occupancy["naive_occupancy"].to_numpy()],
            fmt="o",
            color=ECOLOGY_COLORS["dark"],
            ecolor="#9AA6B2",
            capsize=3,
        )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1))
    ax.set_xlabel("Occupied cameras / all cameras")
    ax.set_title("Naive occupancy breadth", loc="left", fontweight="bold")
    _style(ax, "x")

    ax = axes[1, 0]
    daily = metrics.sort_values("daily_detection_probability", ascending=True)
    ax.barh(daily["species"], daily["daily_detection_probability"], color=[ECOLOGY_COLORS.get(s, "#AFA8A6") for s in daily["species"]])
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1))
    ax.set_xlabel("Camera-days with detection / camera effort days")
    ax.set_title("Naive daily detection rate", loc="left", fontweight="bold")
    _style(ax, "x")

    ax = axes[1, 1]
    ax.scatter(
        metrics["occupied_cameras"],
        metrics["rai_per_100_camera_days"],
        s=np.sqrt(metrics["independent_events"].clip(lower=1)) * 3,
        c=[ECOLOGY_COLORS.get(s, "#AFA8A6") for s in metrics["species"]],
        alpha=0.85,
        edgecolors="white",
    )
    for row in metrics.itertuples():
        ax.text(row.occupied_cameras, row.rai_per_100_camera_days, row.species, fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("Occupied cameras")
    ax.set_ylabel("RAI per 100 camera-days")
    ax.set_title("Widespread vs locally active species", loc="left", fontweight="bold")
    _style(ax, "both")
    fig.suptitle("Species-level ecological indices", fontweight="bold", fontsize=15)
    return _save(fig, output_dir, "species_indices")


def plot_camera_richness_diversity(table, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    ax = axes[0, 0]
    rich = table["richness_presence"].value_counts().reindex(range(0, int(table["richness_presence"].max()) + 2), fill_value=0)
    ax.bar(rich.index, rich.values, color=ECOLOGY_COLORS["Prey"])
    ax.set_xlabel("Detected target-species richness per camera")
    ax.set_ylabel("Number of cameras")
    ax.set_title("Camera species richness", loc="left", fontweight="bold")
    _style(ax)

    ax = axes[0, 1]
    top = table.sort_values(["richness_presence", "shannon_events", "total_events"], ascending=False).head(20).sort_values("shannon_events")
    ax.barh(top["camera_id"].astype(str), top["shannon_events"], color=np.where(top["predator_present"], ECOLOGY_COLORS["Predator"], ECOLOGY_COLORS["Prey"]))
    ax.set_xlabel("Shannon diversity from independent-event proportions")
    ax.set_ylabel("Camera ID")
    ax.set_title("Top diverse cameras", loc="left", fontweight="bold")
    _style(ax, "x")

    ax = axes[1, 0]
    sample = table[table["total_events"] > 0].copy()
    ax.hexbin(np.log1p(sample["total_events"]), sample["richness_presence"], gridsize=32, cmap=SPATIAL_INTENSITY_CMAP, mincnt=1)
    ax.set_xlabel("log(1 + independent events)")
    ax.set_ylabel("Presence-based richness")
    ax.set_title("Activity and richness", loc="left", fontweight="bold")
    _style(ax, "both")

    ax = axes[1, 1]
    max_prey = int(table["prey_richness_presence"].max()) if len(table) else 0
    predator_rate = table.groupby("prey_richness_presence")["predator_present"].mean().reindex(range(0, max_prey + 1), fill_value=0)
    ax.plot(predator_rate.index, predator_rate.values, marker="o", color=ECOLOGY_COLORS["Predator"], linewidth=2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))
    ax.set_xlabel("Detected prey/herbivore richness")
    ax.set_ylabel("Share of cameras with large carnivore")
    ax.set_title("Large-carnivore detections by prey richness", loc="left", fontweight="bold")
    _style(ax)
    fig.suptitle("Camera-level richness and diversity", fontweight="bold", fontsize=15)
    return _save(fig, output_dir, "camera_richness_diversity")


def temporal_overlap_matrix(events, species_order):
    hourly = []
    for species in species_order:
        counts = events.loc[events["species"] == species, "hour"].value_counts().reindex(range(24), fill_value=0).sort_index()
        values = counts.to_numpy(dtype=float)
        hourly.append(values / values.sum() if values.sum() else values)
    hourly = np.asarray(hourly)
    overlap = np.minimum(hourly[:, None, :], hourly[None, :, :]).sum(axis=2)
    return hourly, overlap


def plot_temporal_phenology(events, output_dir, species_order=None):
    species_order = _species_order(events, species_order)
    _, overlap = temporal_overlap_matrix(events, species_order)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))

    ax = axes[0]
    for species in species_order:
        counts = events.loc[events["species"] == species, "hour"].value_counts().reindex(range(24), fill_value=0).sort_index()
        values = counts / counts.sum() if counts.sum() else counts
        smooth = np.convolve(np.r_[values.iloc[-2:], values, values.iloc[:2]], np.ones(5) / 5, mode="same")[2:-2]
        ax.plot(range(24), smooth, label=species, color=ECOLOGY_COLORS.get(species, "#AFA8A6"), linewidth=2)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Hour")
    ax.set_ylabel("Within-species share")
    ax.set_title("Hourly activity profile", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[1]
    monthly = events.groupby(["species", "month"]).size().unstack(fill_value=0).reindex(species_order).reindex(columns=range(1, 13), fill_value=0)
    monthly_share = monthly.div(monthly.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    im = ax.imshow(monthly_share, cmap=HEATMAP_CMAP, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="Within-species monthly share")
    ax.set_yticks(range(len(species_order)))
    ax.set_yticklabels(species_order)
    ax.set_xticks(range(12))
    ax.set_xticklabels(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_title("Annual phenology", loc="left", fontweight="bold")

    ax = axes[2]
    im = ax.imshow(overlap, vmin=0, vmax=1, cmap=HEATMAP_CMAP)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="Discrete hourly-profile overlap")
    ax.set_xticks(range(len(species_order)))
    ax.set_yticks(range(len(species_order)))
    ax.set_xticklabels(species_order, rotation=35, ha="right")
    ax.set_yticklabels(species_order)
    for i in range(len(species_order)):
        for j in range(len(species_order)):
            ax.text(j, i, f"{overlap[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if overlap[i, j] > 0.6 else "#222")
    ax.set_title("Descriptive hourly-profile overlap", loc="left", fontweight="bold")
    fig.suptitle("Temporal ecology: activity timing, phenology, and hourly-profile overlap", fontweight="bold", fontsize=15)
    return _save(fig, output_dir, "temporal_phenology_overlap")


def plot_predator_prey_hypothesis(table, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    work = table.copy()
    bins = min(10, max(2, int(work["prey_rai"].rank(method="first").nunique())))
    work["prey_rai_bin"] = pd.qcut(work["prey_rai"].rank(method="first"), bins, labels=False, duplicates="drop")
    dec = work.groupby("prey_rai_bin").agg(
        mean_prey_rai=("prey_rai", "mean"),
        predator_presence=("predator_present", "mean"),
        mean_predator_rai=("predator_rai", "mean"),
    )

    ax = axes[0]
    ax.plot(dec["mean_prey_rai"], dec["predator_presence"], marker="o", color=ECOLOGY_COLORS["Predator"], linewidth=2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))
    ax.set_xlabel("Mean prey RAI by camera bin")
    ax.set_ylabel("Share of cameras with large carnivore")
    ax.set_title("Predator occurrence along prey activity", loc="left", fontweight="bold")
    _style(ax)

    ax = axes[1]
    ax.hexbin(np.log1p(work["prey_rai"]), np.log1p(work["predator_rai"]), gridsize=35, cmap=SPATIAL_INTENSITY_CMAP, mincnt=1)
    corr = np.corrcoef(np.log1p(work["prey_rai"]), np.log1p(work["predator_rai"]))[0, 1] if len(work) > 2 else np.nan
    ax.set_xlabel("log(1 + prey RAI)")
    ax.set_ylabel("log(1 + large-carnivore RAI)")
    ax.set_title(f"Effort-corrected RAI relationship, r={corr:.2f}", loc="left", fontweight="bold")
    _style(ax, "both")

    ax = axes[2]
    map_data = work.dropna(subset=["longitude", "latitude"]).copy()
    score = np.log1p(map_data["prey_rai"]) * (1 + np.log1p(map_data["predator_rai"]))
    sc = ax.scatter(map_data["longitude"], map_data["latitude"], c=score, s=10, cmap=SPATIAL_INTENSITY_CMAP, alpha=0.65, linewidths=0)
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label="prey RAI x predator RAI co-detection score")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Predator-prey co-detection hotspots", loc="left", fontweight="bold")
    _style(ax, "both")
    fig.suptitle("Exploratory predator-prey hypothesis panel", fontweight="bold", fontsize=15)
    return _save(fig, output_dir, "predator_prey_hypothesis")


def plot_spatial_cooccurrence(events, output_dir, species_order=None):
    species_order = _species_order(events, species_order)
    presence = events.groupby(["camera_id", "species"]).size().unstack(fill_value=0)
    for species in species_order:
        if species not in presence:
            presence[species] = 0
    binary = (presence[species_order] > 0).astype(int)
    n = len(binary)
    oe = np.zeros((len(species_order), len(species_order)))
    jaccard = np.zeros_like(oe)
    for i, a in enumerate(species_order):
        for j, b in enumerate(species_order):
            both = int((binary[a].eq(1) & binary[b].eq(1)).sum())
            union = int((binary[a].eq(1) | binary[b].eq(1)).sum())
            exp = binary[a].sum() * binary[b].sum() / max(n, 1)
            oe[i, j] = both / exp if exp else np.nan
            jaccard[i, j] = both / union if union else np.nan
    for i in range(len(species_order)):
        oe[i, i] = np.nan
        jaccard[i, i] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    finite_oe = oe[np.isfinite(oe)]
    vmax = max(float(finite_oe.max()) if len(finite_oe) else 1.0, 1.01)
    norm = matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax)
    panels = [
        (axes[0], oe, "Observed / expected camera-level co-occurrence", "O/E", ASSOCIATION_CMAP, norm),
        (axes[1], jaccard, "Jaccard shared-camera overlap", "Jaccard", SPATIAL_INTENSITY_CMAP, None),
    ]
    for ax, data, title, label, cmap, color_norm in panels:
        im = ax.imshow(data, cmap=cmap, norm=color_norm) if color_norm is not None else ax.imshow(data, cmap=cmap, vmin=0)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label=label)
        ax.set_xticks(range(len(species_order)))
        ax.set_yticks(range(len(species_order)))
        ax.set_xticklabels(species_order, rotation=35, ha="right")
        ax.set_yticklabels(species_order)
        for i in range(len(species_order)):
            for j in range(len(species_order)):
                ax.text(j, i, "NA" if np.isnan(data[i, j]) else f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(title, loc="left", fontweight="bold")
    fig.suptitle("Exploratory spatial relationship metrics from camera presence/absence", fontweight="bold", fontsize=15)
    return _save(fig, output_dir, "spatial_cooccurrence_metrics")


def plot_detection_occupancy(metrics, total_cameras, output_dir):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    order = metrics.sort_values("naive_occupancy", ascending=False)
    x = np.arange(len(order))
    ci = np.array([wilson_interval(row.occupied_cameras, total_cameras) for row in order.itertuples()])
    ax.bar(x, order["naive_occupancy"], color=[ECOLOGY_COLORS.get(s, "#AFA8A6") for s in order["species"]], alpha=0.88)
    if len(ci):
        ax.errorbar(
            x,
            order["naive_occupancy"],
            yerr=[order["naive_occupancy"].to_numpy() - ci[:, 0], ci[:, 1] - order["naive_occupancy"].to_numpy()],
            fmt="none",
            ecolor="#263238",
            capsize=4,
            linewidth=1,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(order["species"], rotation=15, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))
    ax.set_ylabel("Naive occupancy with Wilson 95% CI")
    ax.set_title("Naive occupancy: detection breadth across standardized cameras", loc="left", fontweight="bold")
    _style(ax)
    return _save(fig, output_dir, "occupancy_detection_breadth")


def plot_predator_prey_dashboard(
    events,
    table,
    output_dir,
    species_order=None,
    ecological_role_mapping=None,
):
    species_order = _species_order(events, species_order)
    roles = normalize_ecological_role_mapping(ecological_role_mapping, species_order)
    predators = roles["predators"]
    prey = roles["prey"]
    if not predators or not prey:
        raise ValueError(
            "Predator-prey figures require an explicit ecological_role_mapping "
            "with observed predator and prey species."
        )
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2))
    ax1, ax2, ax3, ax4 = axes.ravel()

    comp = events["species"].value_counts().reindex(species_order).fillna(0).sort_values()
    ax1.barh(comp.index, comp.values, color=[ECOLOGY_COLORS.get(s, "#AFA8A6") for s in comp.index])
    ax1.set_xlabel("Independent events")
    ax1.set_title("Community event pool", loc="left", fontweight="bold")
    _style(ax1, "x")

    x = np.log1p(table["prey_rai"])
    y = np.log1p(table["predator_rai"])
    corr = float(np.corrcoef(x, y)[0, 1]) if len(table) > 2 else np.nan
    ax2.scatter(x, y, s=np.clip(table["richness_presence"] * 18, 18, 110), c=np.where(table["predator_present"], ECOLOGY_COLORS["Predator"], ECOLOGY_COLORS["grey"]), alpha=0.55, edgecolors="white", linewidths=0.4)
    ax2.set_xlabel("log(1 + prey RAI)")
    ax2.set_ylabel("log(1 + predator RAI)")
    ax2.set_title(f"Predator and prey RAI, r={corr:.2f}", loc="left", fontweight="bold")
    _style(ax2, "both")

    activity_species = events["species"].value_counts().reindex(species_order).dropna().head(4).index
    for species in activity_species:
        counts = events.loc[events["species"] == species, "hour"].value_counts().reindex(range(24), fill_value=0).sort_index()
        values = counts / counts.sum() if counts.sum() else counts
        smooth = np.convolve(np.r_[values.iloc[-2:], values, values.iloc[:2]], np.ones(5) / 5, mode="same")[2:-2]
        ax3.plot(range(24), smooth, color=ECOLOGY_COLORS.get(species, "#AFA8A6"), linewidth=2.0, label=f"{species} (n={int(counts.sum())})")
    ax3.set_xticks([0, 6, 12, 18, 23])
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("Share of independent events")
    ax3.set_title("Hourly activity profile", loc="left", fontweight="bold")
    ax3.legend(frameon=False, fontsize=8)
    _style(ax3, "y")

    months = np.arange(1, 13)
    predator_month = events[events["species"].isin(predators)].groupby("month").size().reindex(months, fill_value=0)
    prey_month = events[events["species"].isin(prey)].groupby("month").size().reindex(months, fill_value=0)
    predator_share = predator_month / predator_month.sum() if predator_month.sum() else predator_month
    prey_share = prey_month / prey_month.sum() if prey_month.sum() else prey_month
    ax4.plot(months, prey_share, marker="o", color=ECOLOGY_COLORS["Prey"], linewidth=2.0, label="Prey")
    ax4.plot(months, predator_share, marker="s", color=ECOLOGY_COLORS["Predator"], linewidth=2.0, label="Large carnivores")
    ax4.set_xticks(months)
    ax4.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Monthly share")
    ax4.set_title("Seasonal activity", loc="left", fontweight="bold")
    ax4.legend(frameon=False)
    _style(ax4, "y")
    fig.suptitle("Exploratory predator-prey ecology from CamtrapDP independent events", fontweight="bold", fontsize=15)
    return _save(fig, output_dir, "predator_prey_dashboard")


def plot_temporal_niche_heatmap(events, output_dir, species_order=None):
    species_order = _species_order(events, species_order)
    hourly, _ = temporal_overlap_matrix(events, species_order)
    fig, ax = plt.subplots(figsize=(12.2, 4.8))
    im = ax.imshow(hourly, aspect="auto", cmap=HEATMAP_CMAP)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Within-species hourly share")
    ax.set_yticks(np.arange(len(species_order)))
    ax.set_yticklabels(species_order)
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_title("Exploratory hourly activity-profile heatmap", loc="left", fontweight="bold")
    return _save(fig, output_dir, "temporal_niche_heatmap")


def plot_spatial_association_matrix(events, output_dir, species_order=None):
    species_order = _species_order(events, species_order)
    presence = events.groupby(["camera_id", "species"]).size().unstack(fill_value=0)
    for species in species_order:
        if species not in presence:
            presence[species] = 0
    binary = (presence[species_order] > 0).astype(int)
    n = len(binary)
    data = np.zeros((len(species_order), len(species_order)))
    for i, a in enumerate(species_order):
        for j, b in enumerate(species_order):
            obs = int(((binary[a] == 1) & (binary[b] == 1)).sum())
            exp = float(binary[a].sum() * binary[b].sum() / max(n, 1))
            data[i, j] = obs / exp if exp > 0 else np.nan
    for i in range(len(species_order)):
        data[i, i] = np.nan
    finite = data[np.isfinite(data)]
    vmax = max(float(finite.max()) if len(finite) else 1.0, 1.01)
    norm = matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(data, cmap=ASSOCIATION_CMAP, norm=norm)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Observed / expected shared cameras")
    ax.set_xticks(range(len(species_order)))
    ax.set_yticks(range(len(species_order)))
    ax.set_xticklabels(species_order, rotation=35, ha="right")
    ax.set_yticklabels(species_order)
    for i in range(len(species_order)):
        for j in range(len(species_order)):
            ax.text(j, i, "NA" if np.isnan(data[i, j]) else f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Exploratory observed / expected camera-level co-occurrence", loc="left", fontweight="bold")
    return _save(fig, output_dir, "spatial_association_matrix")


def plot_camera_ecology_biplot(table, output_dir):
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    x = np.log1p(table["total_events"])
    y = table["richness_presence"]
    colors = np.where(table["predator_present"], ECOLOGY_COLORS["Predator"], ECOLOGY_COLORS["grey"])
    ax.scatter(x, y, c=colors, s=np.clip(table["total_events"] ** 0.5 * 8, 12, 180), alpha=0.5, edgecolors="white", linewidths=0.35)
    top = table.sort_values(["richness_presence", "predator_present", "total_events"], ascending=False).head(15)
    for _, row in top.iterrows():
        ax.text(np.log1p(row["total_events"]), row["richness_presence"] + 0.04, str(row["camera_id"]), fontsize=7, ha="center")
    ax.set_xlabel("log(1 + total independent events)")
    ax.set_ylabel("Detected target-species richness")
    ax.set_title("Exploratory camera ecology biplot", loc="left", fontweight="bold")
    _style(ax, "both")
    return _save(fig, output_dir, "camera_ecology_biplot")


def _figure_manifest_item(spec_key, path, context):
    spec = FIGURE_SPECS[spec_key]
    p = Path(path)
    return {
        "id": spec_key,
        "figure": spec["filename"],
        "path": str(path),
        "status": "success" if p.exists() and p.stat().st_size > 0 else "empty",
        "figure_status": spec["status"],
        "kind": spec["kind"],
        "ecological_question": spec["question"],
        "caption": spec["caption"],
        "metric_basis": spec["metric_basis"],
        "interpretation_boundary": spec["boundary"],
        "data_context": context,
        "sanity_check": {
            "file_nonempty": p.exists() and p.stat().st_size > 0,
            "uses_independent_events": True,
            "raw_video_count_not_primary_metric": True,
        },
    }


def write_figure_manifest(manifest, output_dir):
    output_dir = Path(output_dir)
    json_path = output_dir / "ecology_figure_manifest.json"
    md_path = output_dir / "ecology_figure_manifest.md"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Ecology Figure Manifest",
        "",
        f"- Package: `{manifest['package_dir']}`",
        f"- Figures: {manifest['figure_count']}",
        f"- Media rows: {manifest['data_context']['media_records']:,}",
        f"- Independent events: {manifest['data_context']['independent_events_30min']:,}",
        f"- Cameras: {manifest['data_context']['camera_count']:,}",
        "",
        "## Naming Rule",
        "",
        "`ecology_<topic>.svg` is a core descriptive figure; `ecology_exploratory_<topic>.svg` is hypothesis-generating and must keep its interpretation boundary in the caption.",
        "",
        "## Figures",
        "",
    ]
    for item in manifest["figures"]:
        lines.extend([
            f"### {item['figure']}",
            f"- Status: {item['figure_status']}",
            f"- Question: {item['ecological_question']}",
            f"- Caption: {item['caption']}",
            f"- Metric basis: {item['metric_basis']}",
            f"- Boundary: {item['interpretation_boundary']}",
            f"- Check: non-empty={item['sanity_check']['file_nonempty']}",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return str(json_path), str(md_path)


def write_review_dashboard(manifest, output_dir):
    output_dir = Path(output_dir)
    path = output_dir / "ecology_review_dashboard.html"
    cards = []
    for item in manifest["figures"]:
        file_name = item["figure"]
        media = f'<img src="{html.escape(file_name)}" alt="{html.escape(file_name)}">' if Path(item["path"]).exists() else "<div class='missing'>File missing</div>"
        cards.append(f"""
        <article class="card">
          <div class="card-head">
            <h2>{html.escape(item['figure'])}</h2>
            <span class="badge {html.escape(item['figure_status'])}">{html.escape(item['figure_status'])}</span>
          </div>
          <div class="media">{media}</div>
          <dl>
            <dt>Question</dt><dd>{html.escape(item['ecological_question'])}</dd>
            <dt>Caption</dt><dd>{html.escape(item['caption'])}</dd>
            <dt>Metric basis</dt><dd>{html.escape(item['metric_basis'])}</dd>
            <dt>Boundary</dt><dd>{html.escape(item['interpretation_boundary'])}</dd>
          </dl>
        </article>
        """)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ecology Figure Review Dashboard</title>
  <style>
    body {{ margin:0; font-family: Arial, Helvetica, sans-serif; color:#222; background:#f7f8fa; }}
    header {{ padding:28px 36px; background:#263238; color:white; }}
    header h1 {{ margin:0 0 10px; font-size:26px; }}
    header p {{ margin:6px 0; max-width:1050px; line-height:1.5; }}
    main {{ padding:24px 36px 48px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap:18px; align-items:start; }}
    .card {{ background:white; border:1px solid #dfe4ea; border-radius:8px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,.06); }}
    .card-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:14px 16px; border-bottom:1px solid #eef1f4; }}
    .card h2 {{ margin:0; font-size:16px; }}
    .badge {{ font-size:12px; border-radius:999px; padding:3px 8px; background:#eceff1; }}
    .badge.core {{ color:#1b5e20; background:#e8f5e9; }}
    .badge.exploratory {{ color:#8C4A3F; background:#EAD8C0; }}
    img {{ display:block; width:100%; max-height:520px; object-fit:contain; background:white; border-bottom:1px solid #eef1f4; }}
    dl {{ display:grid; grid-template-columns:120px 1fr; gap:8px 12px; padding:14px 16px 18px; margin:0; font-size:13px; line-height:1.4; }}
    dt {{ font-weight:bold; color:#455a64; }}
    dd {{ margin:0; }}
    .missing {{ padding:42px; color:#8C4A3F; text-align:center; }}
  </style>
</head>
<body>
  <header>
    <h1>Ecology Figure Review Dashboard</h1>
    <p>Core figures are descriptive and effort-aware. Exploratory figures are named with <code>ecology_exploratory_</code> and keep their interpretation boundary next to the caption.</p>
    <p>Records: {manifest['data_context']['media_records']:,} media rows -> {manifest['data_context']['independent_events_30min']:,} independent events; cameras: {manifest['data_context']['camera_count']:,}.</p>
  </header>
  <main><div class="grid">{''.join(cards)}</div></main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")
    return str(path)


def run_ecology_figure_suite(
    package_dir=None,
    output_dir=None,
    species_order=None,
    ecological_role_mapping=None,
    include_core=True,
    include_exploratory=True,
    write_dashboard=True,
):
    ensure_runtime_dirs()
    package_dir = Path(package_dir or (CAMTRAP_IMPORTS_DIR / "0525_wildlife_camtrapdp"))
    output_dir = Path(output_dir or OUTPUTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    events, deployments, report = load_ecology_dataset(package_dir)
    species_order = _species_order(events, species_order)
    roles = normalize_ecological_role_mapping(ecological_role_mapping, species_order)
    table = camera_table(events, deployments, species_order, roles)
    metrics, total_effort, total_cameras = species_metrics(events, deployments, species_order)
    context = {
        "package_dir": str(package_dir),
        "media_records": int(report.get("kept_media_records", len(events))),
        "independent_events_30min": int(report.get("independent_events", len(events))),
        "camera_count": int(report.get("deployment_count", deployments["cameraID"].nunique())),
        "total_effort_camera_days": float(total_effort),
        "species_order": species_order,
        "ecological_role_mapping": roles,
        "filters": report.get("filters", {}),
    }

    figure_paths = []
    if include_core:
        figure_paths.extend([
            ("quality_conversion_audit", plot_quality_conversion_audit(report, metrics, deployments, output_dir)),
            ("species_indices", plot_species_indices(metrics, total_cameras, output_dir)),
            ("camera_richness_diversity", plot_camera_richness_diversity(table, output_dir)),
            ("temporal_phenology_overlap", plot_temporal_phenology(events, output_dir, species_order)),
            ("occupancy_detection_breadth", plot_detection_occupancy(metrics, total_cameras, output_dir)),
        ])
    if include_exploratory:
        figure_paths.extend([
            ("spatial_cooccurrence_metrics", plot_spatial_cooccurrence(events, output_dir, species_order)),
            ("temporal_niche_heatmap", plot_temporal_niche_heatmap(events, output_dir, species_order)),
            ("spatial_association_matrix", plot_spatial_association_matrix(events, output_dir, species_order)),
        ])
        if roles["predators"] and roles["prey"]:
            figure_paths.extend([
                ("predator_prey_hypothesis", plot_predator_prey_hypothesis(table, output_dir)),
                (
                    "predator_prey_dashboard",
                    plot_predator_prey_dashboard(
                        events,
                        table,
                        output_dir,
                        species_order,
                        roles,
                    ),
                ),
                ("camera_ecology_biplot", plot_camera_ecology_biplot(table, output_dir)),
            ])

    figures = [_figure_manifest_item(key, path, context) for key, path in figure_paths]
    manifest = {
        "package_dir": str(package_dir),
        "output_dir": str(output_dir),
        "figure_count": len(figures),
        "data_context": context,
        "species_metrics": metrics.to_dict(orient="records"),
        "figures": figures,
        "role_specific_figures": {
            "enabled": bool(roles["predators"] and roles["prey"]),
            "reason": (
                "explicit ecological roles supplied"
                if roles["predators"] and roles["prey"]
                else "no explicit predator/prey role mapping supplied"
            ),
        },
    }
    manifest_json, manifest_md = write_figure_manifest(manifest, output_dir)
    manifest["manifest_json"] = manifest_json
    manifest["manifest_md"] = manifest_md
    if write_dashboard:
        manifest["dashboard"] = write_review_dashboard(manifest, output_dir)
        (Path(manifest_json)).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
