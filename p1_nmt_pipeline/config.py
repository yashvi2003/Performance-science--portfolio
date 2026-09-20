"""Configuration loader and dataclasses for p1_nmt_pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FilterConfig:
    order: int = 4
    cutoffs_hz: dict[str, float] = field(default_factory=lambda: {
        "CMJ": 30.0,
        "SJ": 30.0,
        "DJ": 50.0,
        "IMTP": 20.0,
        "default": 30.0,
    })

    def get_cutoff(self, test_name: str | None) -> float:
        if test_name and test_name.upper() in self.cutoffs_hz:
            return self.cutoffs_hz[test_name.upper()]
        return self.cutoffs_hz.get("default", 30.0)


@dataclass(frozen=True)
class ThresholdConfig:
    quiet_window_s: float = 0.5
    quiet_start_s: float = 0.05
    onset_sd_multiplier: float = 5.0
    integration_lead_s: float = 0.030
    takeoff_threshold_n: float = 10.0
    landing_threshold_n: float = 10.0
    takeoff_consecutive_ms: float = 2.5
    landing_consecutive_ms: float = 2.5
    takeoff_min_samples: int = 5
    correct_flight_phase_offset: bool = False
    flight_offset_model: str = "whole_trace"  # 'whole_trace' or 'flight_only'
    max_quiet_sd_n: float = 15.0
    max_quiet_trend_n_per_s: float = 20.0
    sj_countermovement_threshold_pct: float = 5.0
    imtp_pretension_threshold_n: float = 50.0


@dataclass(frozen=True)
class ClassifierConfig:
    confidence_threshold: float = 0.70
    cmj_min_dip_pct: float = 10.0
    sj_max_dip_pct: float = 5.0
    flight_min_duration_s: float = 0.15
    imtp_min_plateau_pct: float = 150.0


@dataclass(frozen=True)
class PipelineConfig:
    filters: FilterConfig = field(default_factory=FilterConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> PipelineConfig:
        """Load configuration from a YAML file, falling back to defaults."""
        if config_path is None:
            default_path = Path(__file__).resolve().parent / "config.yaml"
            if default_path.exists():
                config_path = default_path
            else:
                return cls()

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        filters_data = raw.get("filters", {})
        thresholds_data = raw.get("thresholds", {})
        classifier_data = raw.get("classifier", {})

        filters = FilterConfig(
            order=filters_data.get("order", 4),
            cutoffs_hz=filters_data.get("cutoffs_hz", {
                "CMJ": 30.0,
                "SJ": 30.0,
                "DJ": 50.0,
                "IMTP": 20.0,
                "default": 30.0,
            }),
        )

        thresholds = ThresholdConfig(
            quiet_window_s=thresholds_data.get("quiet_window_s", 0.5),
            quiet_start_s=thresholds_data.get("quiet_start_s", 0.05),
            onset_sd_multiplier=thresholds_data.get("onset_sd_multiplier", 5.0),
            integration_lead_s=thresholds_data.get("integration_lead_s", 0.030),
            takeoff_threshold_n=thresholds_data.get("takeoff_threshold_n", 10.0),
            landing_threshold_n=thresholds_data.get("landing_threshold_n", 10.0),
            takeoff_consecutive_ms=thresholds_data.get("takeoff_consecutive_ms", 2.5),
            landing_consecutive_ms=thresholds_data.get("landing_consecutive_ms", 2.5),
            takeoff_min_samples=thresholds_data.get("takeoff_min_samples", 5),
            correct_flight_phase_offset=thresholds_data.get("correct_flight_phase_offset", False),
            flight_offset_model=thresholds_data.get("flight_offset_model", "whole_trace"),
            max_quiet_sd_n=thresholds_data.get("max_quiet_sd_n", 15.0),
            max_quiet_trend_n_per_s=thresholds_data.get("max_quiet_trend_n_per_s", 20.0),
            sj_countermovement_threshold_pct=thresholds_data.get("sj_countermovement_threshold_pct", 5.0),
            imtp_pretension_threshold_n=thresholds_data.get("imtp_pretension_threshold_n", 50.0),
        )

        classifier = ClassifierConfig(
            confidence_threshold=classifier_data.get("confidence_threshold", 0.70),
            cmj_min_dip_pct=classifier_data.get("cmj_min_dip_pct", 10.0),
            sj_max_dip_pct=classifier_data.get("sj_max_dip_pct", 5.0),
            flight_min_duration_s=classifier_data.get("flight_min_duration_s", 0.15),
            imtp_min_plateau_pct=classifier_data.get("imtp_min_plateau_pct", 150.0),
        )

        return cls(filters=filters, thresholds=thresholds, classifier=classifier)
