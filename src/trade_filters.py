import os
from dataclasses import dataclass
import pandas as pd


def _env_bool(env_val: str, default: bool) -> bool:
    if env_val is None:
        return default
    return str(env_val).strip().lower() in {"1", "true", "yes", "y"}


@dataclass
class FilterConfig:
    """Configuration for trading-time filters."""
    min_dir_conf: float = 0.7
    min_ev_dir: float = 0.0
    ev_quantile: float = 0.95
    mag_quantile: float = 0.65
    mag_min_abs: float = 0.0
    use_abs_ev: bool = True
    min_sweep_cost: float = 0.0  # minimum sweep cost magnitude required to trade
    sweep_cost_quantile: float = 0.0  # if >0, derive min_sweep_cost from this quantile of sweep cost magnitude

    @classmethod
    def from_env(cls, env=os.environ) -> "FilterConfig":
        return cls(
            min_dir_conf=float(env.get("FILTER_MIN_DIR_CONF", cls.min_dir_conf)),
            min_ev_dir=float(env.get("FILTER_MIN_EV_DIR", cls.min_ev_dir)),
            ev_quantile=float(env.get("FILTER_EV_QUANTILE", cls.ev_quantile)),
            mag_quantile=float(env.get("FILTER_MAG_QUANTILE", cls.mag_quantile)),
            mag_min_abs=float(env.get("FILTER_MAG_MIN_ABS", cls.mag_min_abs)),
            use_abs_ev=_env_bool(env.get("FILTER_USE_ABS_EV"), cls.use_abs_ev),
            min_sweep_cost=float(env.get("FILTER_MIN_SWEEP_COST", cls.min_sweep_cost)),
            sweep_cost_quantile=float(env.get("FILTER_SWEEP_COST_QUANTILE", cls.sweep_cost_quantile)),
        )


def compute_thresholds(ev_series: pd.Series, mag_p75: pd.Series, cfg: FilterConfig) -> tuple[float, float]:
    """Derive EV and magnitude cutoffs from quantiles and absolutes."""
    ev_clean = ev_series.dropna()
    ev_source = ev_clean.abs() if cfg.use_abs_ev else ev_clean
    ev_cutoff = float(ev_source.quantile(cfg.ev_quantile)) if not ev_source.empty else 0.0

    mag_clean = mag_p75.dropna()
    mag_floor_q = float(mag_clean.quantile(cfg.mag_quantile)) if not mag_clean.empty else 0.0
    mag_floor = max(mag_floor_q, cfg.mag_min_abs)
    return ev_cutoff, mag_floor


def apply_filters(
    dir_conf: pd.Series,
    ev_dir: pd.Series,
    ev_combined: pd.Series,
    mag_p75: pd.Series,
    ev_cutoff: float,
    mag_floor: float,
    cfg: FilterConfig,
    sweep_cost_mag: pd.Series | None = None,
) -> pd.Series:
    """Return a boolean mask indicating which rows pass all trading gates."""
    ev_metric = ev_combined.abs() if cfg.use_abs_ev else ev_combined
    passes = (
        (dir_conf >= cfg.min_dir_conf)
        & (ev_dir.abs() >= cfg.min_ev_dir)
        & (ev_metric >= ev_cutoff)
        & (mag_p75 >= mag_floor)
    )
    if sweep_cost_mag is not None:
        cutoff = cfg.min_sweep_cost
        if cutoff == 0 and cfg.sweep_cost_quantile > 0:
            cutoff = float(sweep_cost_mag.quantile(cfg.sweep_cost_quantile))
        if cutoff > 0:
            passes &= (sweep_cost_mag >= cutoff)
    return passes
