"""Longitudinal Visualization Module for Project 2.

Generates:
1. Metric vs Rolling Baseline Ribbon (with +/- SWC or SD shaded bands).
2. Dual-Axis Composite Readiness vs Wearable Recovery/Strain Overlay.

All synthetic data plots are explicitly labelled 'synthetic' per AGENTS.md Rule 7.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_metric_baseline_ribbon(
    baselines_df: pd.DataFrame,
    athlete_id: str,
    metric: str = "cmj_jump_height_imp",
    side: str = "both",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot longitudinal metric values against rolling personal baseline ribbon."""
    df = baselines_df[
        (baselines_df["athlete_id"] == athlete_id)
        & (baselines_df["metric"] == metric)
        & (baselines_df["side"] == side)
    ].sort_values("date").copy()

    if df.empty:
        raise ValueError(f"No data found for athlete={athlete_id}, metric={metric}, side={side}")

    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Plot rolling baseline ribbon where established
    est_mask = df["baseline_established"] == True
    if est_mask.any():
        est_df = df[est_mask]
        upper_band = est_df["baseline_mean"] + est_df["swc"]
        lower_band = est_df["baseline_mean"] - est_df["swc"]

        ax.fill_between(
            est_df["date"],
            lower_band,
            upper_band,
            color="#2b5c8f",
            alpha=0.18,
            label="Rolling Baseline (±SWC)",
        )
        ax.plot(
            est_df["date"],
            est_df["baseline_mean"],
            color="#2b5c8f",
            linestyle="--",
            linewidth=1.5,
            label="Rolling Baseline Mean (28d)",
        )

    # Plot session observed values
    ax.plot(
        df["date"],
        df["value"],
        color="#1a202c",
        marker="o",
        linewidth=2.0,
        markersize=6,
        label="Observed Session Value",
    )

    # Highlight acute drops
    drop_mask = df["z_score"] < -1.0
    if drop_mask.any():
        ax.scatter(
            df.loc[drop_mask, "date"],
            df.loc[drop_mask, "value"],
            color="#e53e3e",
            s=90,
            zorder=5,
            label="Acute Drop Flag (z < -1.0)",
        )

    unit = df["unit"].iloc[0] if "unit" in df.columns else ""
    ax.set_title(
        f"Athlete {athlete_id}: Longitudinal {metric} vs Rolling Baseline [Synthetic Data]",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel(f"Metric Value ({unit})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved baseline ribbon chart to %s", save_path)

    return fig


def plot_readiness_wearable_overlay(
    merged_df: pd.DataFrame,
    athlete_id: str,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot dual-axis longitudinal readiness index and wearable recovery/strain overlay."""
    df = merged_df[merged_df["athlete_id"] == athlete_id].sort_values("date").copy()

    if df.empty:
        raise ValueError(f"No longitudinal data found for athlete={athlete_id}")

    df["date"] = pd.to_datetime(df["date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, dpi=300)

    # -------------------------------------------------------------
    # Panel 1: Neuromuscular Composite Readiness Index (z-score)
    # -------------------------------------------------------------
    # Shaded normal readiness band [-1.0, +1.0]
    ax1.axhspan(-1.0, 1.0, color="#edf2f7", alpha=0.9, label="Normal Readiness Band (±1.0 SD)")
    ax1.axhline(0.0, color="#718096", linestyle="--", linewidth=1.0)

    # Plot composite readiness z-score
    valid_z = df.dropna(subset=["composite_readiness_z"])
    ax1.plot(
        valid_z["date"],
        valid_z["composite_readiness_z"],
        color="#2b6cb0",
        marker="s",
        linewidth=2.0,
        markersize=6,
        label="Composite Readiness (z-score)",
    )

    # Highlight acute drops
    drops = valid_z[valid_z["acute_drop_flag"] == True]
    if not drops.empty:
        ax1.scatter(
            drops["date"],
            drops["composite_readiness_z"],
            color="#e53e3e",
            s=90,
            zorder=5,
            label="Acute Readiness Drop Flag",
        )

    ax1.set_ylabel("Readiness Index (z-score)", fontsize=10, fontweight="bold")
    ax1.set_title(
        f"Athlete {athlete_id}: Neuromuscular Readiness & Wearable Recovery Overlay [Synthetic Data]",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 2: Daily Wearable Recovery & Strain Overlay
    # -------------------------------------------------------------
    # Recovery zones shading
    min_date, max_date = df["date"].min(), df["date"].max()
    ax2.axhspan(0, 33, color="#fed7d7", alpha=0.5, label="Red Zone (<34%)")
    ax2.axhspan(34, 66, color="#fefcbf", alpha=0.5, label="Yellow Zone (34-66%)")
    ax2.axhspan(67, 100, color="#c6f6d5", alpha=0.5, label="Green Zone (≥67%)")

    # Recovery % line
    if "recovery_pct" in df.columns:
        ax2.plot(
            df["date"],
            df["recovery_pct"],
            color="#2f855a",
            marker="o",
            linewidth=1.8,
            markersize=4,
            label="Recovery %",
        )

    # Twin axis for Day Strain
    if "day_strain" in df.columns and df["day_strain"].notna().any():
        ax2_strain = ax2.twinx()
        ax2_strain.plot(
            df["date"],
            df["day_strain"],
            color="#dd6b20",
            linestyle=":",
            linewidth=1.5,
            label="Day Strain",
        )
        ax2_strain.set_ylabel("Day Strain (AU)", fontsize=9, color="#dd6b20", fontweight="bold")
        ax2_strain.set_ylim(0, 21)

    ax2.set_ylabel("Recovery %", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved readiness overlay chart to %s", save_path)

    return fig
