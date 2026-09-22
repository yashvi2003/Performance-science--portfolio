"""Configuration loader and schema dataclasses for Project 2."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any
import yaml


@dataclass
class BaselineConfig:
    rolling_window_days: int = 28
    min_sessions_for_baseline: int = 3
    swc_method: str = "individual_0.5sd"
    swc_multiplier: float = 0.5
    typical_error_type: str = "cv_te"


@dataclass
class ReadinessConfig:
    composite_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "cmj_jump_height_imp": 0.40,
            "cmj_rsimod": 0.35,
            "cmj_peak_power": 0.25,
        }
    )
    fallback_single_metric: str = "cmj_jump_height_imp"
    drop_threshold_pct: float = 10.0
    use_swc_threshold: bool = True
    heuristic_label: str = "Experimental Heuristic Index (Unvalidated Score)"


@dataclass
class RecoveryZones:
    red_upper: float = 33.0
    yellow_lower: float = 34.0
    yellow_upper: float = 66.0
    green_lower: float = 67.0


@dataclass
class WearableConfig:
    recovery_zones: RecoveryZones = field(default_factory=RecoveryZones)
    consecutive_red_days_alert: int = 2


@dataclass
class ReportEngineConfig:
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    readiness: ReadinessConfig = field(default_factory=ReadinessConfig)
    wearable: WearableConfig = field(default_factory=WearableConfig)


def load_config(config_path: str | Path | None = None) -> ReportEngineConfig:
    """Load configuration from YAML file or return defaults if None."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return ReportEngineConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    b_raw = raw.get("baselines", {})
    baselines = BaselineConfig(
        rolling_window_days=int(b_raw.get("rolling_window_days", 28)),
        min_sessions_for_baseline=int(b_raw.get("min_sessions_for_baseline", 3)),
        swc_method=str(b_raw.get("swc_method", "individual_0.5sd")),
        swc_multiplier=float(b_raw.get("swc_multiplier", 0.5)),
        typical_error_type=str(b_raw.get("typical_error_type", "cv_te")),
    )

    r_raw = raw.get("readiness", {})
    readiness = ReadinessConfig(
        composite_weights=r_raw.get(
            "composite_weights",
            {
                "cmj_jump_height_imp": 0.40,
                "cmj_rsimod": 0.35,
                "cmj_peak_power": 0.25,
            },
        ),
        fallback_single_metric=str(r_raw.get("fallback_single_metric", "cmj_jump_height_imp")),
        drop_threshold_pct=float(r_raw.get("drop_threshold_pct", 10.0)),
        use_swc_threshold=bool(r_raw.get("use_swc_threshold", True)),
        heuristic_label=str(r_raw.get("heuristic_label", "Experimental Heuristic Index (Unvalidated Score)")),
    )

    w_raw = raw.get("wearable", {})
    rz_raw = w_raw.get("recovery_zones", {})
    recovery_zones = RecoveryZones(
        red_upper=float(rz_raw.get("red_upper", 33.0)),
        yellow_lower=float(rz_raw.get("yellow_lower", 34.0)),
        yellow_upper=float(rz_raw.get("yellow_upper", 66.0)),
        green_lower=float(rz_raw.get("green_lower", 67.0)),
    )
    wearable = WearableConfig(
        recovery_zones=recovery_zones,
        consecutive_red_days_alert=int(w_raw.get("consecutive_red_days_alert", 2)),
    )

    return ReportEngineConfig(
        baselines=baselines,
        readiness=readiness,
        wearable=wearable,
    )
