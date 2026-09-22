"""Configuration dataclasses and YAML loader for Project 3."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class ProxiesConfig:
    strain_method: str = "sum_srpe"
    strain_rest_fill: float = 0.0
    recovery_primary_field: str = "readiness"
    recovery_scale_factor: float = 10.0


@dataclass(frozen=True)
class CleaningConfig:
    min_sleep_hours: float = 2.0
    max_sleep_hours: float = 16.0
    min_rhr_bpm: float = 35.0
    max_rhr_bpm: float = 110.0
    min_readiness: float = 1.0
    max_readiness: float = 10.0
    min_fatigue: float = 1.0
    max_fatigue: float = 5.0
    min_soreness: float = 1.0
    max_soreness: float = 5.0
    max_daily_strain: float = 3000.0


@dataclass(frozen=True)
class ModelingConfig:
    alpha: float = 0.05
    person_mean_center: bool = True
    standardize_features: bool = True
    random_intercept: bool = True
    random_slope: bool = False


@dataclass(frozen=True)
class ProfilerConfig:
    raw_dir: Path = Path("data/raw/pmdata")
    output_csv: Path = Path("data/processed/wearable_daily.csv")
    proxies: ProxiesConfig = field(default_factory=ProxiesConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> ProfilerConfig:
        if path is None:
            path = Path(__file__).parent / "config.yaml"
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        ds = data.get("dataset", {})
        px = data.get("proxies", {})
        cl = data.get("cleaning", {})
        md = data.get("modeling", {})

        return cls(
            raw_dir=Path(ds.get("raw_dir", "data/raw/pmdata")),
            output_csv=Path(ds.get("output_csv", "data/processed/wearable_daily.csv")),
            proxies=ProxiesConfig(
                strain_method=px.get("day_strain", {}).get("method", "sum_srpe"),
                strain_rest_fill=px.get("day_strain", {}).get("fill_no_training_days", 0.0),
                recovery_primary_field=px.get("recovery_pct", {}).get("primary_field", "readiness"),
                recovery_scale_factor=px.get("recovery_pct", {}).get("scale_factor", 10.0),
            ),
            cleaning=CleaningConfig(
                min_sleep_hours=cl.get("min_sleep_hours", 2.0),
                max_sleep_hours=cl.get("max_sleep_hours", 16.0),
                min_rhr_bpm=cl.get("min_rhr_bpm", 35.0),
                max_rhr_bpm=cl.get("max_rhr_bpm", 110.0),
                min_readiness=cl.get("min_readiness", 1.0),
                max_readiness=cl.get("max_readiness", 10.0),
                min_fatigue=cl.get("min_fatigue", 1.0),
                max_fatigue=cl.get("max_fatigue", 5.0),
                min_soreness=cl.get("min_soreness", 1.0),
                max_soreness=cl.get("max_soreness", 5.0),
                max_daily_strain=cl.get("max_daily_strain", 3000.0),
            ),
            modeling=ModelingConfig(
                alpha=md.get("alpha", 0.05),
                person_mean_center=md.get("person_mean_center", True),
                standardize_features=md.get("standardize_features", True),
                random_intercept=md.get("random_effects", {}).get("intercept", True),
                random_slope=md.get("random_effects", {}).get("slope", False),
            ),
        )
