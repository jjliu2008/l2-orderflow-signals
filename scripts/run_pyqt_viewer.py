"""
PyQtGraph live viewer for price and backtest metrics (no Monte Carlo chart).
- Streams mid-price from data/live (or PYQT_DATA_DIR).
- Refreshes price every REFRESH_MS (default 1000 ms).
- Re-runs backtest and updates equity/metrics every METRIC_REFRESH_MS (default 15000 ms).
- Optional charts (prompted at startup): drawdown, EV buckets vs realized return,
  trade PnL distribution, rolling mean return.

Requires: pyqtgraph, PyQt5

Run:
  python scripts/run_pyqt_viewer.py
  # or overrides:
  PYQT_DATA_DIR=data/live PYQT_LOOKBACK_MIN=30 REFRESH_MS=1000 METRIC_REFRESH_MS=15000 python scripts/run_pyqt_viewer.py
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_all_raw_data
from src.feature_engineering import add_basic_features, ensure_multiindex
from scripts import run_backtest  # type: ignore
from contextlib import contextmanager


DATA_DIR = Path(os.environ.get("PYQT_DATA_DIR", PROJECT_ROOT / "data" / "live")).expanduser().resolve()
LOOKBACK_MIN = int(os.environ.get("PYQT_LOOKBACK_MIN", "30"))
REFRESH_MS = int(os.environ.get("REFRESH_MS", "1000"))           # price updates
METRIC_REFRESH_MS = int(os.environ.get("METRIC_REFRESH_MS", "15000"))  # equity/metrics refresh
CLEAR_MS = int(os.environ.get("CLEAR_MS", "120000"))             # periodic reset


@contextmanager
def _patched_input(default_response: str = ""):
    """Disable interactive prompts when reusing backtest."""
    from unittest.mock import patch
    with patch("builtins.input", lambda prompt="": default_response):
        yield


def _env_flag(name: str) -> Optional[bool]:
    val = os.environ.get(name)
    if val is None:
        return None
    val = val.strip().lower()
    return val in {"1", "true", "yes", "y"}


def _has_data_files(path: Path) -> bool:
    return any(path.glob("*.jsonl")) or any(path.glob("*.csv"))


def _pick_data_dir(env_dir: Optional[str] = None) -> Path:
    target = Path(env_dir).expanduser().resolve() if env_dir else DATA_DIR
    if not target.exists() or not _has_data_files(target):
        fallback = PROJECT_ROOT / "data" / "raw"
        if fallback.exists() and _has_data_files(fallback):
            return fallback
        raise FileNotFoundError(f"No data found in {target}")
    return target


def load_mid_series(data_dir: Path, lookback_min: int) -> pd.Series:
    df = ensure_multiindex(load_all_raw_data(data_dir))
    df_feat = add_basic_features(df)
    mid = df_feat["mid"].dropna().sort_index()
    if mid.empty:
        return mid
    cutoff = mid.index.get_level_values("Time").max() - pd.Timedelta(minutes=lookback_min)
    mid = mid[mid.index.get_level_values("Time") >= cutoff]
    return mid


def load_equity_curve() -> Optional[pd.DataFrame]:
    path = PROJECT_ROOT / "artifacts" / "backtest_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Time"])
    if df.empty:
        return None
    return df


class LiveWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph Live Viewer")
        self.resize(1200, 800)

        # Prompt for optional charts
        self.show_price = self._prompt_flag("Show live mid-price? (Y/n): ", env_name="SHOW_PRICE", default=True)
        self.show_eq = self._prompt_flag("Show equity curve? (Y/n): ", env_name="SHOW_EQ", default=True)
        self.show_dd = self._prompt_flag("Show drawdown curve? (y/N): ", env_name="SHOW_DD")
        self.show_ev = self._prompt_flag("Show EV buckets vs realized return? (y/N): ", env_name="SHOW_EV")
        self.show_pnl = self._prompt_flag("Show trade PnL distribution? (y/N): ", env_name="SHOW_PNL")
        self.show_roll = self._prompt_flag("Show rolling mean return? (y/N): ", env_name="SHOW_ROLL")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.price_plot = pg.PlotWidget(title="Live mid-price") if self.show_price else None
        self.eq_plot = pg.PlotWidget(title="Backtest equity (from artifacts)") if self.show_eq else None
        self.dd_plot = None
        self.ev_plot = None
        self.pnl_plot = None
        self.roll_plot = None

        if self.price_plot:
            layout.addWidget(self.price_plot)
        if self.eq_plot:
            layout.addWidget(self.eq_plot)
        if self.show_dd:
            self.dd_plot = pg.PlotWidget(title="Drawdown curve")
            layout.addWidget(self.dd_plot)
        if self.show_ev:
            self.ev_plot = pg.PlotWidget(title="EV buckets vs realized return")
            layout.addWidget(self.ev_plot)
        if self.show_pnl:
            self.pnl_plot = pg.PlotWidget(title="Trade PnL distribution")
            layout.addWidget(self.pnl_plot)
        if self.show_roll:
            self.roll_plot = pg.PlotWidget(title="Rolling mean return")
            layout.addWidget(self.roll_plot)

        self.price_curve = self.price_plot.plot(pen=pg.mkPen(color="#4c78a8", width=2)) if self.price_plot else None
        self.eq_curve = self.eq_plot.plot(pen=pg.mkPen(color="#f58518", width=2)) if self.eq_plot else None
        self.dd_curve = self.dd_plot.plot(pen=pg.mkPen(color="#b279a2", width=2)) if self.dd_plot else None
        self.ev_curve = self.ev_plot.plot(pen=None, symbol="o", symbolSize=6, symbolBrush="#4c78a8") if self.ev_plot else None
        self.pnl_bar = self.pnl_plot.plot(stepMode=True, fillLevel=0, brush=(76, 120, 168, 120)) if self.pnl_plot else None
        self.roll_curve = self.roll_plot.plot(pen=pg.mkPen(color="#e45756", width=2)) if self.roll_plot else None

        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.status_label)

        self.data_dir = _pick_data_dir()

        self.price_timer = QtCore.QTimer()
        self.price_timer.timeout.connect(self.update_price)
        self.price_timer.start(REFRESH_MS)

        self.metric_timer = QtCore.QTimer()
        self.metric_timer.timeout.connect(self.update_equity_and_metrics)
        self.metric_timer.start(METRIC_REFRESH_MS)

        # Periodic clear to avoid stale cache in plots
        self.clear_timer = QtCore.QTimer()
        self.clear_timer.timeout.connect(self.reset_plots)
        self.clear_timer.start(CLEAR_MS)

        # initial draw
        self.update_price()
        self.update_equity_and_metrics()

    def reset_plots(self):
        """Periodically clear plots to ensure fresh redraws."""
        if self.price_plot:
            self.price_plot.clear()
            self.price_curve = self.price_plot.plot(pen=pg.mkPen(color="#4c78a8", width=2))
        if self.eq_plot:
            self.eq_plot.clear()
            self.eq_curve = self.eq_plot.plot(pen=pg.mkPen(color="#f58518", width=2))
        if self.dd_plot:
            self.dd_plot.clear()
            self.dd_curve = self.dd_plot.plot(pen=pg.mkPen(color="#b279a2", width=2))
        if self.ev_plot:
            self.ev_plot.clear()
            self.ev_curve = self.ev_plot.plot(pen=None, symbol="o", symbolSize=6, symbolBrush="#4c78a8")
        if self.pnl_plot:
            self.pnl_plot.clear()
            self.pnl_bar = self.pnl_plot.plot(stepMode=True, fillLevel=0, brush=(76, 120, 168, 120))
        if self.roll_plot:
            self.roll_plot.clear()
            self.roll_curve = self.roll_plot.plot(pen=pg.mkPen(color="#e45756", width=2))
        # Kick an immediate refresh after clearing
        self.update_price()
        self.update_equity_and_metrics()

    def update_price(self):
        try:
            mid = load_mid_series(self.data_dir, LOOKBACK_MIN)
            if mid.empty:
                if self.price_curve:
                    self.price_curve.setData([], [])
                return
            if self.price_curve:
                idx = mid.index.get_level_values("Time")
                times = pd.to_datetime(idx).astype(np.int64) / 1e9
                self.price_curve.setData(times, mid.values)
                self.price_plot.setLabel("bottom", "Time (s, epoch)")
                self.price_plot.setLabel("left", "Price")
        except Exception:
            if self.price_curve:
                self.price_curve.setData([], [])

    def _prompt_flag(self, prompt: str, env_name: str, default: bool = False) -> bool:
        env_val = _env_flag(env_name)
        if env_val is not None:
            return env_val
        # Allow disabling prompts entirely
        if _env_flag("PYQT_NO_PROMPT"):
            return default
        ans = input(prompt).strip().lower()
        return ans in {"y", "yes", "1", "true"}

    def update_equity_and_metrics(self):
        try:
            # Re-run backtest to refresh equity file
            os.environ["BT_DATA_DIR"] = str(self.data_dir)
            os.environ["BACKTEST_SAVE"] = "1"
            with _patched_input(""):
                run_backtest.main()

            df = load_equity_curve()
            if df is None:
                if self.eq_curve:
                    self.eq_curve.setData([], [])
                if self.dd_curve:
                    self.dd_curve.setData([], [])
                if self.ev_curve:
                    self.ev_curve.setData([], [])
                if self.pnl_bar:
                    self.pnl_bar.setData([], [])
                if self.roll_curve:
                    self.roll_curve.setData([], [])
                self.status_label.setText("No equity data available.")
                return

            times = pd.to_datetime(df["Time"]).astype(np.int64) / 1e9
            equity = df["equity"].values
            returns = df["return"].values if "return" in df.columns else None
            expected = df["expected"].values if "expected" in df.columns else None

            if self.eq_curve:
                self.eq_curve.setData(times, equity)
                self.eq_plot.setLabel("bottom", "Time (s, epoch)")
                self.eq_plot.setLabel("left", "Equity")

            if self.dd_curve is not None:
                peak = np.maximum.accumulate(equity)
                dd = equity / peak - 1.0
                self.dd_curve.setData(times, dd)

            if self.ev_curve is not None and expected is not None and returns is not None:
                try:
                    buckets = pd.qcut(expected, q=5, duplicates="drop")
                    bucket_ret = (
                        pd.DataFrame({"bucket": buckets, "ret": returns})
                        .groupby("bucket", observed=False)["ret"]
                        .mean()
                    )
                    xs = np.arange(len(bucket_ret))
                    self.ev_curve.setData(xs, bucket_ret.values)
                    self.ev_plot.getAxis("bottom").setTicks([[(i, str(i)) for i in xs]])
                except Exception:
                    self.ev_curve.setData([], [])

            if self.pnl_bar is not None and returns is not None:
                hist, edges = np.histogram(returns, bins=30)
                self.pnl_bar.setData(x=edges, y=np.append(hist, hist[-1]))

            if self.roll_curve is not None and returns is not None:
                roll = pd.Series(returns).rolling(200, min_periods=20).mean()
                self.roll_curve.setData(np.arange(len(roll)), roll.values)
            self.status_label.setText("Metrics updated successfully.")
        except Exception as exc:
            if self.eq_curve:
                self.eq_curve.setData([], [])
            if self.dd_curve:
                self.dd_curve.setData([], [])
            if self.ev_curve:
                self.ev_curve.setData([], [])
            if self.pnl_bar:
                self.pnl_bar.setData([], [])
            if self.roll_curve:
                self.roll_curve.setData([], [])
            self.status_label.setText(f"Error updating metrics: {exc}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = LiveWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
