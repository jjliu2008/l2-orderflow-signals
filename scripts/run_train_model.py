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
    dir_horizon = 1
    fwd_ret = forward_returns(df_feat["mid"], horizon=dir_horizon, time_horizon=time_horizon).rename("fwd_ret")
    dir_label = make_dir_label(fwd_ret, neutral_band=neutral_band)

    # Magnitude targets
    vol_window = int(os.environ.get("TRAIN_VOL_WINDOW", "20"))
    realized_vol = compute_realized_vol(df_feat, window=vol_window)
    # Expected range over the same tick horizon as direction label
    range_horizon_env = int(os.environ.get("TRAIN_RANGE_HORIZON", "0"))
    range_horizon = max(range_horizon_env, dir_horizon + 1, 2)
    expected_range = compute_expected_range(df_feat["mid"], horizon=range_horizon)
    mag_target_choice = os.environ.get("TRAIN_MAG_TARGET", "range").strip().lower()

    sweep_cost_mag_feat = None
    if {"sweep_cost_buy1", "sweep_cost_sell1"}.issubset(df_feat.columns):
        sweep_cost_mag_feat = df_feat[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)

    if mag_target_choice == "range":
        mag_target_series = expected_range.abs()
        mag_target_name = "expected_range"
    elif mag_target_choice in {"impact", "range_x_sweep"}:
        if sweep_cost_mag_feat is None:
            raise ValueError("TRAIN_MAG_TARGET=impact requires sweep_cost_* features present.")
        mag_target_series = (expected_range.abs() * sweep_cost_mag_feat).rename("expected_range_x_sweep")
        mag_target_name = "expected_range_x_sweep"
    else:
        mag_target_series = realized_vol.abs()
        mag_target_name = "realized_vol"

    # Align features, direction label, and magnitude
    X_reset = X_df.reset_index()
    fwd_reset = fwd_ret.reset_index()
    dir_reset = dir_label.reset_index()
    mag_reset = mag_target_series.reset_index()

    merged = X_reset.merge(fwd_reset, on=["Symbol", "Time"], how="inner")
    merged = merged.merge(dir_reset, on=["Symbol", "Time"], how="inner")
    merged = merged.merge(mag_reset, on=["Symbol", "Time"], how="inner")
    merged = merged.dropna(subset=["fwd_ret", "label", mag_target_name])

    # Regime flags (fragile/stable) for magnitude per-regime models
    filter_cfg = FilterConfig.from_env()
    regime_series, sweep_reg_cut = compute_regime_flags(merged, filter_cfg)
    merged["regime"] = regime_series

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
            if fast_sample_rows > 0:
                merged = merged.sample(n=min(fast_sample_rows, before), random_state=42)
                print(f"Fast sample: rows {before} -> {len(merged)} via TRAIN_FAST_SAMPLE_ROWS={fast_sample_rows}")
            else:
                merged = merged.sample(frac=fast_sample_frac, random_state=42)
                print(f"Fast sample: rows {before} -> {len(merged)} via TRAIN_FAST_SAMPLE_FRAC={fast_sample_frac}")

    if merged.empty:
        raise ValueError("No overlapping rows between features and forward returns after merge.")

    feature_cols = [c for c in merged.columns if c not in {"Symbol", "Time", "label", "fwd_ret", "regime", mag_target_name}]
    x_small = merged[feature_cols]
    y_dir = merged["label"]
    y_mag = merged[mag_target_name].abs()
    regimes_all = merged["regime"]

    # Default cap to keep runs lightweight; override with TRAIN_MAX_ROWS=0 to disable.
    max_rows = int(os.environ.get("TRAIN_MAX_ROWS", "50000"))
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
    for regime_name in regime_train.unique():
        mask = regime_train == regime_name
        mag_models[regime_name] = _fit_mag_models(X_train[mask], ymag_train[mask])

    # Predict magnitudes per regime
    mag_med_pred = pd.Series(index=X_test.index, dtype=float)
    mag_p75_pred = pd.Series(index=X_test.index, dtype=float)
    for regime_name, (mm, mp) in mag_models.items():
        mask = regime_test == regime_name
        if mask.any():
            mag_med_pred.loc[mask] = mm.predict(X_test[mask])
            mag_p75_pred.loc[mask] = mp.predict(X_test[mask])
    # Fallback if any missing
    if mag_med_pred.isna().any():
        default_regime = "fragile" if "fragile" in mag_models else list(mag_models.keys())[0]
        mm_def, mp_def = mag_models[default_regime]
        missing = mag_med_pred.isna()
        mag_med_pred.loc[missing] = mm_def.predict(X_test[missing])
        mag_p75_pred.loc[missing] = mp_def.predict(X_test[missing])

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
    preds_df["y_true"] = y_test
    preds_df["y_pred"] = y_pred
    preds_df["expected_value_dir"] = expected
    preds_df["mag_pred_med"] = mag_med
    preds_df["mag_pred_p75"] = mag_hi
    preds_df["ev_combined"] = mag_used * (y_proba[:, list(classes).index(1.0)] - y_proba[:, list(classes).index(-1.0)])
    preds_df["sample_weight"] = sample_weight_test
    for i, cls in enumerate(classes):
        preds_df[f"proba_{int(cls)}"] = y_proba[:, i]

    # Add a few regime-related features for downstream filtering/analysis
    for col in ["spread", "sweep_cost_buy1", "sweep_cost_sell1", "order_book_imbalance", "depth_imbalance_top5", "book_slope_top5"]:
        if col in X_test.columns:
            preds_df[col] = X_test[col]

    save_artifacts = os.environ.get("TRAIN_SAVE_ARTIFACTS", "1").strip().lower() in {"1", "true", "yes", "y"}
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
        preds_df.to_csv(preds_path)

        print(f"\nSaved combo model to {model_path}")
        print(f"Saved test predictions to {preds_path}")
    else:
        print("\nSkipping artifact save (TRAIN_SAVE_ARTIFACTS is falsy).")

    # Post-prediction trading gates preview
    filter_cfg = FilterConfig.from_env()
    ev_metric = preds_df["ev_combined"].abs() if filter_cfg.use_abs_ev else preds_df["ev_combined"]
    ev_cutoff, mag_floor = compute_thresholds(ev_metric, preds_df["mag_pred_p75"], filter_cfg)

    dir_conf = preds_df[[f"proba_{int(1.0)}", f"proba_{int(-1.0)}"]].max(axis=1)
    sweep_cost_mag = None
    if {"sweep_cost_buy1", "sweep_cost_sell1"}.issubset(preds_df.columns):
        sweep_cost_mag = preds_df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)
    sweep_cutoff = None
    if sweep_cost_mag is not None:
        if filter_cfg.sweep_cost_quantile > 0:
            sweep_cutoff = float(sweep_cost_mag.quantile(filter_cfg.sweep_cost_quantile))
        elif filter_cfg.min_sweep_cost > 0:
            sweep_cutoff = filter_cfg.min_sweep_cost

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
    preds_df["passes_filters"] = apply_filters(
        dir_conf=dir_conf,
        ev_dir=preds_df["expected_value_dir"],
        ev_combined=preds_df["ev_combined"],
        mag_p75=preds_df["mag_pred_p75"],
        ev_cutoff=ev_cutoff,
        mag_floor=mag_floor,
        cfg=filter_cfg,
        sweep_cost_mag=sweep_cost_mag if sweep_cutoff is not None else None,
    )
    # Only allow trades in fragile regime
    preds_df["passes_filters"] &= preds_df["regime"] == "fragile"
    pass_rate = preds_df["passes_filters"].mean()
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
        f"  pass rate: {pass_rate:.2%} ({preds_df['passes_filters'].sum()}/{len(preds_df)})"
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
