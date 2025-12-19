import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Allow running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_all_raw_data
from src.feature_engineering import (
    add_basic_features,
    add_orderflow_features,
    ensure_multiindex,
    make_feature_matrix,
)
from src.labels import make_labels, _future_price_time_based
from src.trade_filters import FilterConfig, apply_filters, compute_thresholds


def forward_returns(mid: pd.Series, horizon: int = 1, time_horizon: Optional[pd.Timedelta] = None) -> pd.Series:
    if time_horizon is not None:
        future = _future_price_time_based(mid, delta=time_horizon)
    else:
        future = mid.groupby(level=0).shift(-horizon)
    return (future - mid) / mid


def make_dir_label(fwd: pd.Series, neutral_band: float = 0.0) -> pd.Series:
    band = abs(neutral_band)
    def _lab(x: float) -> float:
        if x > band:
            return 1.0
        if x < -band:
            return -1.0
        return 0.0
    return fwd.apply(_lab).rename("label")


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    total = len(y)
    weights = y.map(lambda cls: total / (len(counts) * counts.get(cls, 1)))
    return weights.to_numpy()


def compute_realized_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Realized short-horizon volatility of log returns.
    """
    log_price = np.log(df["mid"].astype(float))
    log_ret = log_price.groupby(level=0).diff()
    vol = log_ret.groupby(level=0).transform(lambda s: s.rolling(window, min_periods=max(5, window // 2)).std())
    return vol.rename("realized_vol")


def compute_expected_range(mid: pd.Series, horizon: int = 2) -> pd.Series:
    """
    Expected price range over the next `horizon` steps: (future_max - future_min) / current_mid.
    Uses a forward-looking rolling window per symbol. Horizon should be >=2 to capture a true range.
    """
    if horizon < 2:
        horizon = 2

    def _fwd_range(s: pd.Series) -> pd.Series:
        fwd = s.shift(-1)
        fwd_max = fwd.rolling(horizon, min_periods=horizon).max()
        fwd_min = fwd.rolling(horizon, min_periods=horizon).min()
        return (fwd_max - fwd_min) / s

    return mid.groupby(level=0).transform(_fwd_range).rename("expected_range")


def compute_directional_excursion(mid: pd.Series, direction: pd.Series, horizon: int, cost_proxy: pd.Series) -> pd.Series:
    """
    Directional excursion net of costs:
      max_{k<=H} (sign(direction) * delta_price(t->t+k)/mid) - cost_proxy
    """
    if horizon < 2:
        horizon = 2
    dir_sign = direction.astype(float).clip(-1, 1)

    def _excursion(s: pd.Series, sign: pd.Series) -> pd.Series:
        # cumulative forward returns up to horizon
        fwd = s.groupby(level=0).apply(lambda x: x.shift(-1)).droplevel(0)
        # rolling max in direction of sign
        rel = (fwd - s) / s
        rel_signed = rel * sign
        roll_max = rel_signed.groupby(level=0).transform(lambda x: x.rolling(horizon, min_periods=horizon).max())
        return roll_max

    exc = _excursion(mid, dir_sign)
    net = exc - cost_proxy
    return net.rename("dir_excursion_net")


def build_candidate_mask(df: pd.DataFrame) -> pd.Series:
    """
    Candidate gate. Modes:
      - baseline: tight spread + low sweep (cheap regime)
      - commitment: baseline + commitment signals (refill, absorption, vol suppression)
      - imbalance: legacy imbalance-driven gate
    Target rate 5-15%; raises if outside [1%,30%].
    """
    gate_type = os.environ.get("CAND_GATE_TYPE", "commitment").strip().lower()

    spread_bps = (df["spread"] / df["mid"]) * 1e4 if {"spread", "mid"}.issubset(df.columns) else pd.Series(np.inf, index=df.index)
    sweep_bps = None
    if {"sweep_cost_buy1", "sweep_cost_sell1"}.issubset(df.columns):
        sweep_bps = df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1) * 1e4

    # Defaults by quantile
    spread_cap_bps = float(os.environ.get("CAND_SPREAD_MAX_BPS") or spread_bps.quantile(0.3))
    sweep_cap_bps = float(os.environ.get("CAND_SWEEP_MAX_BPS") or (sweep_bps.quantile(0.3) if sweep_bps is not None else np.inf))

    mask = spread_bps <= spread_cap_bps
    if sweep_bps is not None and np.isfinite(sweep_cap_bps):
        mask &= sweep_bps <= sweep_cap_bps

    if gate_type == "commitment":
        # Commitment signals: refill, absorption, vol suppression
        refill = df.get("refill_count", pd.Series(0, index=df.index))
        refill_q = float(os.environ.get("CAND_REFILL_Q", "0.8"))
        refill_thresh = refill.quantile(refill_q)

        absorp = df.get("absorption_ratio", pd.Series(0, index=df.index)).replace([np.inf, -np.inf], np.nan).fillna(0)
        absorp_q = float(os.environ.get("CAND_ABSORB_Q", "0.8"))
        absorp_thresh = absorp.quantile(absorp_q)

        rv_short = df.get("rv_short", pd.Series(np.inf, index=df.index))
        rv_q = float(os.environ.get("CAND_RV_Q", "0.3"))
        rv_thresh = rv_short.quantile(rv_q)

        mask &= (refill >= refill_thresh) & (absorp >= absorp_thresh) & (rv_short <= rv_thresh)
        print(f"Candidate gate (commitment): spread_cap_bps={spread_cap_bps:.2f}, sweep_cap_bps={sweep_cap_bps:.2f}, refill>={refill_thresh:.6g}, absorption>={absorp_thresh:.6g}, rv_short<={rv_thresh:.6g}")

    elif gate_type == "imbalance":
        persist_window = int(os.environ.get("CAND_IMB_WINDOW", "50"))
        imb = df["order_book_imbalance"] if "order_book_imbalance" in df else df.get("depth_imbalance_top5", pd.Series(0, index=df.index))
        imb_roll = imb.groupby(level=0).transform(lambda x: x.rolling(persist_window, min_periods=max(5, persist_window // 2)).mean())
        sign_persist = imb.groupby(level=0).transform(
            lambda x: (np.sign(x).rolling(persist_window, min_periods=max(5, persist_window // 2)).mean()).abs()
        )
        activity = df["volatility_20"] if "volatility_20" in df else df.get("ret_5", pd.Series(0, index=df.index)).abs()
        imb_abs_thresh = float(os.environ.get("CAND_IMB_ABS_THRESH") or imb_roll.abs().quantile(0.8))
        persist_min = float(os.environ.get("CAND_IMB_PERSIST_MIN") or 0.7)
        activity_min = float(os.environ.get("CAND_ACTIVITY_MIN") or activity.quantile(0.5))
        mask &= (imb_roll.abs() >= imb_abs_thresh) & (sign_persist >= persist_min) & (activity >= activity_min)
        print(f"Candidate gate (imbalance): spread_cap_bps={spread_cap_bps:.2f}, sweep_cap_bps={sweep_cap_bps:.2f}, imb_abs_thresh={imb_abs_thresh:.4f}, persist_min={persist_min:.2f}, activity_min={activity_min:.6f}")

    else:  # baseline
        print(f"Candidate gate (baseline): spread_cap_bps={spread_cap_bps:.2f}, sweep_cap_bps={sweep_cap_bps:.2f}")

    rate = mask.mean()
    print(f"Candidate rate: {rate:.2%}")
    if rate < 0.01 or rate > 0.30:
        raise ValueError(f"Candidate rate {rate:.2%} outside [1%,30%]; adjust CAND_* or quantiles.")
    return mask


def compute_regime_flags(df: pd.DataFrame, cfg: FilterConfig) -> tuple[pd.Series, float | None]:
    """
    Classify regimes (fragile/stable) based on sweep cost and book/depth features.
    Returns (regime_series, sweep_reg_cutoff).
    """
    sweep_mag = None
    if {"sweep_cost_buy1", "sweep_cost_sell1"}.issubset(df.columns):
        sweep_mag = df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)
    sweep_reg_cut = None
    if sweep_mag is not None and cfg.regime_sweep_quantile > 0:
        sweep_reg_cut = float(sweep_mag.quantile(cfg.regime_sweep_quantile))

    regime_fragile = pd.Series(False, index=df.index)
    if sweep_mag is not None and sweep_reg_cut is not None:
        regime_fragile |= sweep_mag >= sweep_reg_cut
    if "depth_imbalance_top5" in df:
        regime_fragile |= df["depth_imbalance_top5"].abs() >= cfg.regime_depth_imbalance_abs
    if "book_slope_top5" in df:
        regime_fragile |= df["book_slope_top5"].abs() >= cfg.regime_book_slope_abs

    regime = pd.Series("stable", index=df.index, dtype="object")
    regime.loc[regime_fragile] = "fragile"
    return regime, sweep_reg_cut


def _contiguous_sample_per_symbol(df: pd.DataFrame, rows: int | None = None, frac: float | None = None) -> pd.DataFrame:
    """
    Take a contiguous head slice per symbol to preserve adjacency.
    Allocation is proportional to per-symbol counts.
    """
    if rows is None and (frac is None or frac <= 0):
        return df
    df_sorted = df.sort_values(["Symbol", "Time"])
    counts = df_sorted["Symbol"].value_counts()
    total = len(df_sorted)
    pieces = []
    for sym, cnt in counts.items():
        if rows is not None:
            take = min(cnt, max(1, int(np.ceil(rows * cnt / total))))
        else:
            take = min(cnt, max(1, int(np.ceil(cnt * frac))))
        pieces.append(df_sorted[df_sorted["Symbol"] == sym].head(take))
    return pd.concat(pieces, ignore_index=True)


def _build_direction_estimator(model_type: str, params: dict, n_jobs: Optional[int] = None):
    """
    Construct the requested direction classifier.
    """
    model_type = model_type.lower()
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for TRAIN_DIR_MODEL=xgboost. Install with `pip install xgboost` "
                "or set TRAIN_DIR_MODEL=catboost."
            ) from exc
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 400),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 6),
            min_child_weight=params.get("min_child_weight", 1.0),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            reg_lambda=params.get("reg_lambda", 1.0),
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method=os.environ.get("XGB_TREE_METHOD", "hist"),
            random_state=42,
            n_jobs=n_jobs,
        )
    if model_type == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "CatBoost is required for TRAIN_DIR_MODEL=catboost. Install with `pip install catboost` "
                "or set TRAIN_DIR_MODEL=xgboost."
            ) from exc
        return CatBoostClassifier(
            iterations=params.get("iterations", params.get("n_estimators", 400)),
            learning_rate=params.get("learning_rate", 0.05),
            depth=params.get("depth", params.get("max_depth", 6)),
            l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,  # keep training side-effect free
        )
    raise ValueError(f"Unsupported direction model type: {model_type}")


def _load_multiple_dirs(paths: list[Path], max_rows_per_file: int | None = None) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.exists():
            print(f"Loading data from {p} ...")
            frames.append(load_all_raw_data(p, max_rows_per_file=max_rows_per_file))
        else:
            print(f"Skipping missing data dir: {p}")
    if not frames:
        raise FileNotFoundError("No training data directories contained usable files.")
    # Preserve MultiIndex; load_all_raw_data already sets (Symbol, Time) index.
    combined = pd.concat(frames)
    return ensure_multiindex(combined)


def main():
    default_dirs = os.environ.get(
        "TRAIN_DATA_DIRS",
        ",".join(
            [
                str(PROJECT_ROOT / "data" / "raw"),
                str(PROJECT_ROOT / "data" / "processed"),
            ]
        ),
    )
    user_input = input(f"Training data directories (comma-separated) [{default_dirs}]: ").strip()
    dir_list = (
        [Path(d.strip()) for d in user_input.split(",") if d.strip()]
        if user_input
        else [Path(d.strip()) for d in default_dirs.split(",") if d.strip()]
    )

    max_rows_per_file_env = int(os.environ.get("TRAIN_MAX_READ_ROWS_PER_FILE", "0"))
    max_rows_per_file = max_rows_per_file_env if max_rows_per_file_env > 0 else None

    df = _load_multiple_dirs(dir_list, max_rows_per_file=max_rows_per_file)
    # Drop duplicate Symbol/Time to avoid leaking duplicate samples into training (jsonl files can overlap time ranges).
    if df.index.duplicated().any():
        before = len(df)
        df = df.loc[~df.index.duplicated(keep="last")]
        print(f"Deduped Symbol/Time rows: {before - len(df)} removed, {len(df)} remaining.")

    #print("Computing features ...")
    df_feat = add_basic_features(df)
    df_feat = add_orderflow_features(df_feat)
    # Keep rows even if they contain NaNs; tree-based models handle missing values.
    X_df, _ = make_feature_matrix(df_feat, drop_na=False)

    # Direction label from forward returns with neutral band
    neutral_band = float(os.environ.get("TRAIN_NEUTRAL_BAND", "0.0001"))
    time_horizon_env = os.environ.get("TRAIN_TIME_HORIZON_MS")
    time_horizon = None
    if time_horizon_env:
        try:
            ms = int(time_horizon_env)
            if ms > 0:
                time_horizon = pd.to_timedelta(ms, unit="ms")
        except ValueError as exc:
            raise ValueError(f"Invalid TRAIN_TIME_HORIZON_MS={time_horizon_env}") from exc
    dir_horizon = int(os.environ.get("TRAIN_DIR_HORIZON", "1"))
    fwd_ret = forward_returns(df_feat["mid"], horizon=dir_horizon, time_horizon=time_horizon).rename("fwd_ret")
    dir_label = make_dir_label(fwd_ret, neutral_band=neutral_band)

    # Cost proxy
    sweep_cost_mag_feat = None
    if {"sweep_cost_buy1", "sweep_cost_sell1"}.issubset(df_feat.columns):
        sweep_cost_mag_feat = df_feat[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)
    if "mid" not in df_feat or "spread" not in df_feat:
        raise ValueError("Required columns mid/spread missing after feature construction.")
    spread_rel = df_feat["spread"] / df_feat["mid"]
    cost_proxy_feat = spread_rel / 2
    if sweep_cost_mag_feat is not None:
        cost_proxy_feat = cost_proxy_feat + sweep_cost_mag_feat

    # Magnitude target: directional excursion net of costs over a longer horizon
    range_horizon_env = int(os.environ.get("TRAIN_RANGE_HORIZON", "20"))
    range_horizon = max(range_horizon_env, dir_horizon + 1, 2)
    dir_excursion_net = compute_directional_excursion(df_feat["mid"], dir_label, range_horizon, cost_proxy_feat)
    mag_target_series = dir_excursion_net
    mag_target_name = "dir_excursion_net"

    # Align features, direction label, and magnitude
    X_reset = X_df.reset_index()
    fwd_reset = fwd_ret.reset_index()
    dir_reset = dir_label.reset_index()
    mag_reset = mag_target_series.reset_index()

    merged = X_reset.merge(fwd_reset, on=["Symbol", "Time"], how="inner")
    merged = merged.merge(dir_reset, on=["Symbol", "Time"], how="inner")
    merged = merged.merge(mag_reset, on=["Symbol", "Time"], how="inner")
    merged = merged.dropna(subset=["fwd_ret", "label", mag_target_name])

    # Candidate mask
    cand_mask = build_candidate_mask(merged)
    merged["candidate"] = cand_mask
    merged = merged[merged["candidate"]]
    if merged.empty:
        raise ValueError("No candidate rows after applying candidate mask. Loosen CAND_* gates.")

    # Regime flags (fragile/stable) for magnitude per-regime models
    filter_cfg = FilterConfig.from_env()
    regime_series, sweep_reg_cut = compute_regime_flags(merged, filter_cfg)
    merged["regime"] = regime_series

    # Sweep-high flag for magnitude modeling and gating (robust with floor/cap)
    sweep_cutoff_train = None
    if {"sweep_cost_buy1", "sweep_cost_sell1"}.issubset(merged.columns):
        sweep_mag_train = merged[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)
        if filter_cfg.sweep_cost_quantile > 0:
            sweep_cutoff_train = float(sweep_mag_train.quantile(filter_cfg.sweep_cost_quantile))
        elif filter_cfg.min_sweep_cost > 0:
            sweep_cutoff_train = filter_cfg.min_sweep_cost
        # Clamp cutoff to avoid collapse
        sweep_min_floor = float(os.environ.get("FILTER_SWEEP_MIN_FLOOR", "0.0"))
        sweep_max_cap = float(os.environ.get("FILTER_SWEEP_MAX_CAP", "0.0"))
        if sweep_cutoff_train is not None:
            if sweep_min_floor > 0:
                sweep_cutoff_train = max(sweep_cutoff_train, sweep_min_floor)
            if sweep_max_cap > 0:
                sweep_cutoff_train = min(sweep_cutoff_train, sweep_max_cap)
            merged["sweep_high"] = sweep_mag_train >= sweep_cutoff_train
        else:
            merged["sweep_high"] = False
    else:
        merged["sweep_high"] = False

    # Optional post-label sampling to preserve label adjacency during fast iterations
    fast_sample_rows = int(os.environ.get("TRAIN_FAST_SAMPLE_ROWS", "0"))
    fast_sample_frac = float(os.environ.get("TRAIN_FAST_SAMPLE_FRAC", "0"))
    fast_contig = os.environ.get("TRAIN_FAST_CONTIGUOUS", "0").strip().lower() in {"1", "true", "yes", "y"}
    if fast_sample_rows > 0 or (0 < fast_sample_frac < 1):
        before = len(merged)
        if fast_contig:
            merged = _contiguous_sample_per_symbol(
                merged,
                rows=fast_sample_rows if fast_sample_rows > 0 else None,
                frac=fast_sample_frac if fast_sample_frac > 0 else None,
            )
            print(f"Fast contiguous sample: rows {before} -> {len(merged)} (TRAIN_FAST_CONTIGUOUS=1)")
        else:
            # Sweep-aware sampling: keep all sweep_high, sample remainder
            if "sweep_high" in merged and merged["sweep_high"].any():
                sweep_high_df = merged[merged["sweep_high"]]
                rest_df = merged[~merged["sweep_high"]]
                target = fast_sample_rows if fast_sample_rows > 0 else int(len(merged) * fast_sample_frac)
                take_rest = max(0, target - len(sweep_high_df))
                rest_sampled = rest_df.sample(n=min(take_rest, len(rest_df)), random_state=42) if take_rest > 0 else rest_df.iloc[0:0]
                merged = pd.concat([sweep_high_df, rest_sampled], ignore_index=True)
                print(f"Fast sample (sweep-aware): kept {len(sweep_high_df)} sweep_high, sampled {len(rest_sampled)} rest; rows {before} -> {len(merged)}")
            else:
                if fast_sample_rows > 0:
                    merged = merged.sample(n=min(fast_sample_rows, before), random_state=42)
                    print(f"Fast sample: rows {before} -> {len(merged)} via TRAIN_FAST_SAMPLE_ROWS={fast_sample_rows}")
                else:
                    merged = merged.sample(frac=fast_sample_frac, random_state=42)
                    print(f"Fast sample: rows {before} -> {len(merged)} via TRAIN_FAST_SAMPLE_FRAC={fast_sample_frac}")

    if merged.empty:
        raise ValueError("No overlapping rows between features and forward returns after merge.")

    feature_cols = [c for c in merged.columns if c not in {"Symbol", "Time", "label", "fwd_ret", "regime", "sweep_high", mag_target_name, "candidate"}]
    x_small = merged[feature_cols]
    y_dir = merged["label"]
    y_mag = merged[mag_target_name].abs()
    regimes_all = merged["regime"]
    sweep_high_all = merged["sweep_high"]

    # Default cap to keep runs lightweight; override with TRAIN_MAX_ROWS=0 to disable.
    max_rows = int(os.environ.get("TRAIN_MAX_ROWS", "0"))
    if max_rows > 0 and len(x_small) > max_rows:
        sampled_idx = x_small.index.to_series().sample(
            n=max_rows, random_state=42, replace=False
        ).index
        x_small = x_small.loc[sampled_idx]
        y_dir = y_dir.loc[sampled_idx]
        y_mag = y_mag.loc[sampled_idx]
        print(f"Sampled down to {len(x_small)} rows for training via TRAIN_MAX_ROWS={max_rows}")

    if len(y_dir) == 0 or len(x_small) == 0:
        raise ValueError(
            "No training samples after feature/label alignment. "
            "Likely the raw data lacked usable fields or all rows were dropped. "
            "Check data/raw contents and feature construction."
        )

    #print(f"Total samples: {len(y_small)}, features: {x_small.shape[1]}")

    dir_model_type = os.environ.get("TRAIN_DIR_MODEL", "xgboost").strip().lower()
    n_jobs_env = int(os.environ.get("TRAIN_N_JOBS", "0"))
    dir_n_jobs = None if n_jobs_env <= 0 else n_jobs_env
    print(f"Training direction model type: {dir_model_type} (TRAIN_DIR_MODEL)")

    use_label_encoding = dir_model_type == "xgboost"
    label_encoder = None
    if use_label_encoding:
        label_encoder = LabelEncoder()
        y_dir_enc = pd.Series(label_encoder.fit_transform(y_dir.astype(int)), index=y_dir.index)
    else:
        y_dir_enc = y_dir

    X_train, X_test, y_train, y_test, ymag_train, ymag_test = train_test_split(
        x_small, y_dir_enc, y_mag, test_size=0.2, random_state=42, stratify=y_dir_enc
    )
    regime_train = regimes_all.loc[X_train.index]
    regime_test = regimes_all.loc[X_test.index]
    sweep_high_train = sweep_high_all.loc[X_train.index] if "sweep_high" in merged else pd.Series(False, index=X_train.index)
    sweep_high_test = sweep_high_all.loc[X_test.index] if "sweep_high" in merged else pd.Series(False, index=X_test.index)

    # Sweep-high counts for awareness
    sweep_train_counts = sweep_high_train.value_counts(dropna=False).to_dict()
    sweep_test_counts = sweep_high_test.value_counts(dropna=False).to_dict()
    print(f"Sweep_high counts -> train: {sweep_train_counts}, test: {sweep_test_counts}")
    sample_weight_train = compute_sample_weights(y_train)
    sample_weight_test = compute_sample_weights(y_test)

    # Simple hyperparameter sweep on a validation split (using a subset for speed)
    tune_frac = 0.3
    X_tune = X_train.sample(frac=tune_frac, random_state=42)
    y_tune = y_train.loc[X_tune.index]
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_tune, y_tune, test_size=0.2, random_state=42, stratify=y_tune
    )

    if dir_model_type == "xgboost":
        candidates = [
            {"learning_rate": 0.05, "max_depth": 4, "min_child_weight": 1.0, "subsample": 0.9, "colsample_bytree": 0.9, "n_estimators": 300},
            {"learning_rate": 0.1, "max_depth": 4, "min_child_weight": 1.0, "subsample": 0.9, "colsample_bytree": 0.9, "n_estimators": 300},
            {"learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1.5, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 400},
        ]
    elif dir_model_type == "catboost":
        candidates = [
            {"learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0, "iterations": 400},
            {"learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 5.0, "iterations": 300},
            {"learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 3.0, "iterations": 500},
        ]
    else:
        raise ValueError(f"TRAIN_DIR_MODEL must be 'xgboost' or 'catboost', got {dir_model_type}")

    best = None
    best_f1 = -1.0
    for params in candidates:
        base = _build_direction_estimator(dir_model_type, params, n_jobs=dir_n_jobs)
        model = CalibratedClassifierCV(estimator=base, cv=3, method="isotonic", n_jobs=dir_n_jobs)
        model.fit(X_train_sub, y_train_sub, sample_weight=compute_sample_weights(y_train_sub))
        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        print(f"{dir_model_type} params {params} -> val macro F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best = params

    if best is None:
        raise RuntimeError("No direction model hyperparameters were evaluated; check candidates list.")

    print(f"Best {dir_model_type} params: {best} (val macro F1={best_f1:.4f})")

    dir_base = _build_direction_estimator(dir_model_type, best, n_jobs=dir_n_jobs)
    dir_model = CalibratedClassifierCV(estimator=dir_base, cv=3, method="isotonic", n_jobs=dir_n_jobs)
    dir_model.fit(X_train, y_train, sample_weight=sample_weight_train)

    # Magnitude quantile regressors on realized volatility, trained per regime
    def _fit_mag_models(X: pd.DataFrame, y: pd.Series) -> tuple:
        mag_median = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.5,
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=30,
            random_state=42,
        )
        mag_p75 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.75,
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=30,
            random_state=42,
        )
        mag_median.fit(X, y)
        mag_p75.fit(X, y)
        return mag_median, mag_p75

    mag_models: dict[str, tuple] = {}
    # Train only on sweep_high samples; skip training if no data
    for regime_name in regime_train.unique():
        mask = (regime_train == regime_name) & (sweep_high_train if isinstance(sweep_high_train, pd.Series) else False)
        if mask.any():
            mag_models[regime_name] = _fit_mag_models(X_train[mask], ymag_train[mask])

    # Predict magnitudes only when sweep_high; else zero (don't trade)
    mag_med_pred = pd.Series(0.0, index=X_test.index, dtype=float)
    mag_p75_pred = pd.Series(0.0, index=X_test.index, dtype=float)
    for regime_name, (mm, mp) in mag_models.items():
        mask = (regime_test == regime_name) & (sweep_high_test if isinstance(sweep_high_test, pd.Series) else False)
        if mask.any():
            mag_med_pred.loc[mask] = mm.predict(X_test[mask])
            mag_p75_pred.loc[mask] = mp.predict(X_test[mask])

    #print("Evaluating ...")
    y_pred_enc = dir_model.predict(X_test)
    y_proba = dir_model.predict_proba(X_test)
    classes_enc = dir_model.classes_
    if use_label_encoding and label_encoder is not None:
        # Map encoded classes/preds back to original labels for reporting and saving.
        classes = label_encoder.inverse_transform(classes_enc.astype(int)).astype(float)
        y_pred = label_encoder.inverse_transform(y_pred_enc.astype(int))
    else:
        classes = classes_enc.astype(float)
        y_pred = y_pred_enc
    expected = (y_proba * classes.reshape(1, -1)).sum(axis=1)
    mag_med = np.clip(mag_med_pred.to_numpy(), 0, None)
    mag_hi = np.clip(mag_p75_pred.to_numpy(), 0, None)
    mag_used = np.minimum(mag_hi, mag_med * 2)  # cap median by upper quantile (or 2x median to avoid zero hi)

    if use_label_encoding and label_encoder is not None:
        y_test_report = label_encoder.inverse_transform(y_test.astype(int))
    else:
        y_test_report = y_test

    print("Classification report:\n", classification_report(y_test_report, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test_report, y_pred))
    print("\nPredicted class order:", classes)
    print("Probability preview (first 5 rows):")
    print(y_proba[:5])
    print("Expected value of prediction (mean over test set):", expected.mean())
    print("Expected value quantiles (5/50/95):", pd.Series(expected).quantile([0.05, 0.5, 0.95]).to_dict())
    print("Magnitude median preview (first 5 rows):", mag_med[:5])
    print("Magnitude p75 preview (first 5 rows):", mag_hi[:5])

    preds_df = pd.DataFrame(index=X_test.index)
    preds_df["Symbol"] = merged.loc[X_test.index, "Symbol"].values
    preds_df["Time"] = merged.loc[X_test.index, "Time"].values
    preds_df["y_true"] = y_test
    preds_df["y_pred"] = y_pred
    preds_df["expected_value_dir"] = expected
    preds_df["mag_pred_med"] = mag_med
    preds_df["mag_pred_p75"] = mag_hi
    preds_df["ev_combined"] = mag_used * (y_proba[:, list(classes).index(1.0)] - y_proba[:, list(classes).index(-1.0)])
    # Cost proxy: half-spread (relative) + sweep cost magnitude (relative) if available
    cost_proxy = pd.Series(0.0, index=preds_df.index)
    sweep_cost_mag = None
    if {"sweep_cost_buy1", "sweep_cost_sell1"}.issubset(preds_df.columns):
        sweep_cost_mag = preds_df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)
    if "mid" in preds_df and "spread" in preds_df:
        spread_rel = preds_df["spread"] / preds_df["mid"]
        cost_proxy += spread_rel / 2
    if sweep_cost_mag is not None:
        cost_proxy += sweep_cost_mag
    preds_df["ev_net"] = preds_df["ev_combined"] - cost_proxy
    preds_df["sample_weight"] = sample_weight_test
    for i, cls in enumerate(classes):
        preds_df[f"proba_{int(cls)}"] = y_proba[:, i]
    # Carry sweep_high flag from the split for analysis/gating
    preds_df["sweep_high"] = sweep_high_test.reindex(X_test.index).fillna(False) if "sweep_high" in merged else False

    # Add a few regime-related features for downstream filtering/analysis
    for col in ["spread", "sweep_cost_buy1", "sweep_cost_sell1", "order_book_imbalance", "depth_imbalance_top5", "book_slope_top5"]:
        if col in X_test.columns:
            preds_df[col] = X_test[col]

    save_artifacts = os.environ.get("TRAIN_SAVE_ARTIFACTS", "1").strip().lower() in {"1", "true", "yes", "y"}
    save_preds_csv = os.environ.get("TRAIN_SAVE_PRED_CSV", "1").strip().lower() in {"1", "true", "yes", "y"}
    if save_artifacts:
        artifacts_dir = PROJECT_ROOT / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        model_path = artifacts_dir / "hgb_combo_model.joblib"
        label_encoder_classes = label_encoder.classes_.tolist() if label_encoder is not None else None
        joblib.dump(
            {
                "direction_model": dir_model,
                "direction_model_type": dir_model_type,
                "direction_params": best,
                "direction_label_encoder_classes": label_encoder_classes,
                "magnitude_models": mag_models,
                "magnitude_target": mag_target_name,
                "classes": classes,
                "feature_cols": feature_cols,
            },
            model_path,
        )

        preds_path = artifacts_dir / "hgb_test_predictions.csv"
        if save_preds_csv:
            preds_out = preds_df.copy()
            preds_out.to_csv(preds_path, index=False)
            print(f"\nSaved combo model to {model_path}")
            print(f"Saved test predictions to {preds_path}")
        else:
            print(f"\nSaved combo model to {model_path}")
            print("Skipping test predictions CSV (TRAIN_SAVE_PRED_CSV is falsy).")
    else:
        print("\nSkipping artifact save (TRAIN_SAVE_ARTIFACTS is falsy).")

    # Post-prediction trading gates preview
    filter_cfg = FilterConfig.from_env()
    # Compute thresholds only on rows with non-zero magnitude to avoid dilution by zero-magnitude rows
    ev_metric = preds_df["ev_combined"].abs() if filter_cfg.use_abs_ev else preds_df["ev_combined"]
    mask_thresh = preds_df["mag_pred_p75"] > 0
    ev_metric_subset = ev_metric[mask_thresh] if mask_thresh.any() else ev_metric
    mag_p75_subset = preds_df.loc[mask_thresh, "mag_pred_p75"] if mask_thresh.any() else preds_df["mag_pred_p75"]
    ev_cutoff, mag_floor = compute_thresholds(ev_metric_subset, mag_p75_subset, filter_cfg)

    dir_conf = preds_df[[f"proba_{int(1.0)}", f"proba_{int(-1.0)}"]].max(axis=1)
    sweep_cutoff = sweep_cutoff_train  # reuse training cutoff if available
    if sweep_cost_mag is None:
        if {"sweep_cost_buy1", "sweep_cost_sell1"}.issubset(preds_df.columns):
            sweep_cost_mag = preds_df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)
    if sweep_cost_mag is not None and sweep_cutoff is None:
        if filter_cfg.sweep_cost_quantile > 0:
            sweep_cutoff = float(sweep_cost_mag.quantile(filter_cfg.sweep_cost_quantile))
        elif filter_cfg.min_sweep_cost > 0:
            sweep_cutoff = filter_cfg.min_sweep_cost
    if sweep_cost_mag is not None:
        preds_df["sweep_high"] = sweep_cost_mag >= (sweep_cutoff if sweep_cutoff is not None else 0)
    elif "sweep_high" not in preds_df.columns:
        preds_df["sweep_high"] = False

    # Regime classifier: fragile if any of the conditions hold
    regime = pd.Series("stable", index=preds_df.index, dtype="object")
    regime_fragile_mask = pd.Series(False, index=preds_df.index)
    if sweep_cost_mag is not None and filter_cfg.regime_sweep_quantile > 0:
        sweep_reg_cut = float(sweep_cost_mag.quantile(filter_cfg.regime_sweep_quantile))
        regime_fragile_mask |= sweep_cost_mag >= sweep_reg_cut
    if "depth_imbalance_top5" in preds_df:
        regime_fragile_mask |= preds_df["depth_imbalance_top5"].abs() >= filter_cfg.regime_depth_imbalance_abs
    if "book_slope_top5" in preds_df:
        regime_fragile_mask |= preds_df["book_slope_top5"].abs() >= filter_cfg.regime_book_slope_abs
    regime.loc[regime_fragile_mask] = "fragile"
    preds_df["regime"] = regime

    preds_df["dir_conf"] = dir_conf
    # Stage A: economics gate (candidate gate already applied upstream), optional spread cap
    spread_cap = float(os.environ.get("FILTER_SPREAD_MAX", "0"))
    passes_stage_a = pd.Series(True, index=preds_df.index)
    if spread_cap > 0 and "spread" in preds_df:
        passes_stage_a &= preds_df["spread"] <= spread_cap

    # Stage B: economic gate on net excursion (ev_net > 0)
    passes_stage_b = preds_df["ev_net"] > 0

    preds_df["passes_filters"] = passes_stage_a & passes_stage_b
    # Optional regime filter
    if "regime" in preds_df:
        preds_df["passes_filters"] &= preds_df["regime"] == "fragile"
    pass_rate = preds_df["passes_filters"].mean()
    sweep_high_total = preds_df["sweep_high"].sum() if "sweep_high" in preds_df else 0
    pass_rate_high = (
        preds_df.loc[preds_df["sweep_high"], "passes_filters"].mean()
        if "sweep_high" in preds_df and sweep_high_total > 0
        else 0.0
    )
    sweep_cutoff_str = "none"
    if sweep_cutoff is not None:
        sweep_cutoff_str = f"{sweep_cutoff:.6g}"

    print(
        "\nTrade filter preview on held-out set:\n"
        f"  min_dir_conf={filter_cfg.min_dir_conf}, min_ev_dir={filter_cfg.min_ev_dir}, "
        f"ev_quantile={filter_cfg.ev_quantile}, mag_quantile={filter_cfg.mag_quantile}, "
        f"mag_min_abs={filter_cfg.mag_min_abs}, use_abs_ev={filter_cfg.use_abs_ev}, "
        f"min_sweep_cost={filter_cfg.min_sweep_cost}, sweep_cost_quantile={filter_cfg.sweep_cost_quantile}\n"
        f"  derived ev_cutoff={ev_cutoff:.6g}, mag_floor={mag_floor:.6g}, "
        f"sweep_cutoff={sweep_cutoff_str}\n"
        f"  pass rate (all): {pass_rate:.2%} ({preds_df['passes_filters'].sum()}/{len(preds_df)}), "
        f"pass rate (sweep_high): {pass_rate_high:.2%} ({preds_df.loc[preds_df['sweep_high'], 'passes_filters'].sum() if 'sweep_high' in preds_df else 0}/{sweep_high_total})"
    )

    if len(preds_df["passes_filters"]) > 0:
        filt_df = preds_df[preds_df["passes_filters"]]
        def _ev_summary(df: pd.DataFrame) -> dict:
            if df.empty:
                return {"count": 0, "ev_mean": None, "ev_median": None, "ev_p05": None, "ev_p95": None}
            ev = df["ev_combined"]
            return {
                "count": len(df),
                "ev_mean": ev.mean(),
                "ev_median": ev.median(),
                "ev_p05": ev.quantile(0.05),
                "ev_p95": ev.quantile(0.95),
                "long_mean": ev[ev > 0].mean() if (ev > 0).any() else None,
                "short_mean": ev[ev < 0].mean() if (ev < 0).any() else None,
            }
        def _fmt(val: float | None) -> str:
            return "nan" if val is None else f"{val:.6g}"
        overall_stats = _ev_summary(filt_df)
        print(
            "\nFiltered EV stats (post filters only):\n"
            f"  trades: {overall_stats['count']} of {len(preds_df)} ({pass_rate:.2%})\n"
            f"  ev_mean={_fmt(overall_stats['ev_mean'])} ev_median={_fmt(overall_stats['ev_median'])} "
            f"ev_p05={_fmt(overall_stats['ev_p05'])} ev_p95={_fmt(overall_stats['ev_p95'])}\n"
            f"  long_mean={_fmt(overall_stats['long_mean'])} short_mean={_fmt(overall_stats['short_mean'])}"
        )
        if "ev_net" in filt_df:
            ev_net = filt_df["ev_net"]
            print(
                "  ev_net mean={mn} median={md} p05={p05} p95={p95}".format(
                    mn=_fmt(ev_net.mean()),
                    md=_fmt(ev_net.median()),
                    p05=_fmt(ev_net.quantile(0.05)),
                    p95=_fmt(ev_net.quantile(0.95)),
                )
            )
        if "sweep_high" in filt_df:
            filt_high = filt_df[filt_df["sweep_high"]]
            if not filt_high.empty:
                stats_high = _ev_summary(filt_high)
                print(
                    "  (sweep_high) trades: {cnt} of {total}, ev_mean={evm}, ev_median={evmed}, ev_p05={p05}, ev_p95={p95}".format(
                        cnt=stats_high["count"],
                        total=sweep_high_total,
                        evm=_fmt(stats_high["ev_mean"]),
                        evmed=_fmt(stats_high["ev_median"]),
                        p05=_fmt(stats_high["ev_p05"]),
                        p95=_fmt(stats_high["ev_p95"]),
                    )
                )
        # Magnitude stats on sweep_high rows for visibility
        if "sweep_high" in preds_df and preds_df["sweep_high"].any():
            mag_high = preds_df.loc[preds_df["sweep_high"], ["mag_pred_med", "mag_pred_p75"]]
            print("\nMagnitude stats (sweep_high rows only):")
            print(mag_high.describe())

        # Regime splits: tight vs wide spread, low vs high sweep cost magnitude
        if not filt_df.empty:
            regimes: dict[str, pd.DataFrame] = {}
            if "spread" in filt_df:
                spread_thr = filt_df["spread"].quantile(0.75)
                regimes["spread_tight"] = filt_df[filt_df["spread"] < spread_thr]
                regimes["spread_wide"] = filt_df[filt_df["spread"] >= spread_thr]
            if {"sweep_cost_buy1", "sweep_cost_sell1"}.issubset(filt_df.columns):
                sweep_mag = filt_df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)
                sweep_thr = sweep_mag.quantile(0.75)
                regimes["sweep_low"] = filt_df[sweep_mag < sweep_thr]
                regimes["sweep_high"] = filt_df[sweep_mag >= sweep_thr]
            if "regime" in filt_df:
                regimes["regime_fragile"] = filt_df[filt_df["regime"] == "fragile"]
                regimes["regime_stable"] = filt_df[filt_df["regime"] == "stable"]

            if regimes:
                print("\nFiltered EV by regime buckets:")
                for name, df_reg in regimes.items():
                    stats = _ev_summary(df_reg)
                    if stats["count"] == 0:
                        continue
                    print(
                        f"  {name}: count={stats['count']} "
                        f"ev_mean={_fmt(stats['ev_mean'])} ev_median={_fmt(stats['ev_median'])} "
                        f"ev_p05={_fmt(stats['ev_p05'])} ev_p95={_fmt(stats['ev_p95'])}"
                    )

if __name__ == "__main__":
    main()
