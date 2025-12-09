"""
TradingView Lightweight Charts live viewer for price/backtest metrics.

What it does
------------
- Serves a small local HTTP server with JSON endpoints for price and backtest metrics.
- Renders TradingView Lightweight Charts in the browser (loaded from CDN, no extra Python deps).
- Periodically re-runs the backtest (cooldown configurable) so equity artifacts stay fresh.

Run
---
  python scripts/run_lightweight_viewer.py

Environment knobs
-----------------
  TV_DATA_DIR: path to raw/live data (defaults to data/live, falls back to data/raw)
  TV_LOOKBACK_MIN: minutes of mid-price history to show (default 30)
  TV_PRICE_REFRESH_MS: price poll interval for the browser (default 1000)
  TV_METRIC_REFRESH_MS: metrics poll interval for the browser (default 15000)
  TV_BACKTEST_COOLDOWN_SEC: minimum seconds between backtest refreshes (default derived from TV_METRIC_REFRESH_MS)
  TV_PORT: HTTP port (default 8765)
"""

import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_all_raw_data  # noqa: E402
from src.feature_engineering import add_basic_features, ensure_multiindex  # noqa: E402
from scripts import run_backtest  # type: ignore  # noqa: E402


DATA_DIR = Path(os.environ.get("TV_DATA_DIR", PROJECT_ROOT / "data" / "live")).expanduser().resolve()
LOOKBACK_MIN = int(os.environ.get("TV_LOOKBACK_MIN", "30"))
PRICE_REFRESH_MS = int(os.environ.get("TV_PRICE_REFRESH_MS", "1000"))
METRIC_REFRESH_MS = int(os.environ.get("TV_METRIC_REFRESH_MS", "15000"))
BACKTEST_COOLDOWN_SEC = float(
    os.environ.get("TV_BACKTEST_COOLDOWN_SEC", max(5, METRIC_REFRESH_MS / 1000))
)
PORT = int(os.environ.get("TV_PORT", "8765"))

# Allowed candle timeframes for mid-price aggregation.
CANDLE_FREQS = {
    "1m": "60s",
    "3m": "180s",
    "5m": "300s",
    "10m": "600s",
    "15m": "900s",
    "30m": "1800s",
    "45m": "2700s",
    "1h": "3600s",
    "2h": "7200s",
    "3h": "10800s",
    "4h": "14400s",
    "1d": "1d",
}

# In-process cache for price data to avoid rereading full files each tick.
# Structure: {path: {"df": DataFrame, "mtime": float, "size": int}}
_PRICE_CACHE: dict[Path, dict] = {}
USE_PRICE_CACHE = os.environ.get("TV_USE_PRICE_CACHE", "1").lower() not in {"0", "false", "no"}

INDEX_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TradingView Lightweight Charts</title>
  <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --card: #111827;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-2: #f59e0b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "SF Pro Display", system-ui, -apple-system, sans-serif;
      background: radial-gradient(circle at 15% 20%, #0b1222, #0f172a 55%),
                  radial-gradient(circle at 80% 0%, #13203b, #0f172a 50%);
      color: var(--text);
      min-height: 100vh;
      padding: 16px;
    }
    h1 { margin: 0 0 12px 0; font-weight: 600; letter-spacing: 0.4px; }
    p.lead { margin: 0 0 16px 0; color: var(--muted); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }
    .card {
      background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 15px 40px rgba(0,0,0,0.35);
      backdrop-filter: blur(4px);
    }
    .card h2 {
      margin: 0 0 8px 0;
      font-size: 15px;
      font-weight: 600;
      color: var(--text);
      letter-spacing: 0.2px;
    }
    .chart {
      height: 280px;
    }
    .small { height: 200px; }
    .status {
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.1);
      color: var(--accent);
      font-size: 12px;
      margin-left: 8px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      background: rgba(245, 158, 11, 0.12);
      color: var(--accent-2);
      border-radius: 999px;
      font-size: 12px;
    }
  </style>
</head>
<body>
  <h1>TradingView Lightweight Charts <span class="badge">live</span></h1>
  <p class="lead">Mid-price plus backtest metrics pulled from local files. Updates are auto-polled by the browser.</p>
  <div class="grid">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
        <h2 style="margin:0;">Mid-price (lookback: TV_LOOKBACK_MIN_PLACEHOLDER min)</h2>
        <div style="display:flex;align-items:center;gap:6px;">
          <select id="price-tf" class="pill" style="cursor:pointer;border:none;padding:6px 10px;">
            <option value="1m">1m</option>
            <option value="3m">3m</option>
            <option value="5m">5m</option>
            <option value="10m">10m</option>
            <option value="15m">15m</option>
            <option value="30m">30m</option>
            <option value="45m">45m</option>
            <option value="1h">1h</option>
            <option value="2h">2h</option>
            <option value="3h">3h</option>
            <option value="4h">4h</option>
            <option value="1d">1d</option>
          </select>
          <button id="price-toggle" class="pill" style="cursor:pointer;border:none;">Candles</button>
        </div>
      </div>
      <div id="price-chart" class="chart"></div>
      <div class="status" id="price-status"></div>
    </div>
    <div class="card">
      <h2>Equity curve</h2>
      <div id="equity-chart" class="chart"></div>
      <div class="status" id="equity-status"></div>
    </div>
    <div class="card">
      <h2>Drawdown</h2>
      <div id="drawdown-chart" class="small"></div>
    </div>
    <div class="card">
      <h2>Rolling mean return</h2>
      <div id="rolling-chart" class="small"></div>
    </div>
    <div class="card">
      <h2>EV buckets vs realized return</h2>
      <div id="ev-chart" class="small"></div>
    </div>
    <div class="card">
      <h2>Trade PnL distribution</h2>
      <div id="pnl-chart" class="small"></div>
    </div>
  </div>
  <script>
    const priceRefreshMs = PRICE_REFRESH_MS_PLACEHOLDER;
    const metricRefreshMs = METRIC_REFRESH_MS_PLACEHOLDER;

    const palette = {
      accent: "#38bdf8",
      accent2: "#f59e0b",
      accent3: "#a855f7",
      accent4: "#22c55e",
      grid: "rgba(148, 163, 184, 0.2)",
      text: "#e2e8f0",
      muted: "#94a3b8"
    };

    const baseOptions = {
      layout: { background: { color: "transparent" }, textColor: palette.text },
      grid: {
        vertLines: { color: palette.grid },
        horzLines: { color: palette.grid },
      },
      timeScale: { timeVisible: true, secondsVisible: true },
      rightPriceScale: { borderVisible: false },
    };

    const priceChart = LightweightCharts.createChart(document.getElementById("price-chart"), { ...baseOptions, height: 280 });
    const priceAreaSeries = priceChart.addAreaSeries({
      lineColor: palette.accent,
      topColor: "rgba(56, 189, 248, 0.35)",
      bottomColor: "rgba(56, 189, 248, 0.05)",
      lineWidth: 2,
    });
    const priceCandleSeries = priceChart.addCandlestickSeries({
      upColor: "#38bdf8",
      borderUpColor: "#38bdf8",
      wickUpColor: "#38bdf8",
      downColor: "#0ea5e9",
      borderDownColor: "#0ea5e9",
      wickDownColor: "#0ea5e9",
    });
    let priceMode = "area"; // "area" | "candles"
    let priceTf = "1m";
    let lastPriceArea = [];
    let lastPriceCandles = [];
    let evLabels = [];
    let pnlEdges = [];

    const equityChart = LightweightCharts.createChart(document.getElementById("equity-chart"), { ...baseOptions, height: 280 });
    const equitySeries = equityChart.addLineSeries({ color: palette.accent2, lineWidth: 2 });

    const drawdownChart = LightweightCharts.createChart(document.getElementById("drawdown-chart"), { ...baseOptions, height: 200 });
    drawdownChart.timeScale().applyOptions({ timeVisible: true, secondsVisible: false });
    const drawdownSeries = drawdownChart.addLineSeries({
      color: palette.accent3,
      lineWidth: 2,
      priceFormat: { type: "percent", precision: 3 },
    });

    const rollingChart = LightweightCharts.createChart(document.getElementById("rolling-chart"), { ...baseOptions, height: 200 });
    rollingChart.timeScale().applyOptions({ timeVisible: false });
    const rollingSeries = rollingChart.addLineSeries({
      color: palette.accent4,
      lineWidth: 2,
      priceFormat: { type: "price", precision: 2, minMove: 0.01 },
    });

    const evChart = LightweightCharts.createChart(document.getElementById("ev-chart"), {
      ...baseOptions,
      height: 200,
      timeScale: {
        visible: true,
        timeVisible: false,
        secondsVisible: false,
        tickMarkFormatter: (time) => evLabels[time - 1] || "",
      },
      rightPriceScale: { borderVisible: false, visible: false },
    });
    const evSeries = evChart.addHistogramSeries({ color: palette.accent, priceFormat: { type: "price", precision: 6 } });

    const pnlChart = LightweightCharts.createChart(document.getElementById("pnl-chart"), {
      ...baseOptions,
      height: 200,
      timeScale: {
        visible: true,
        timeVisible: false,
        secondsVisible: false,
        tickMarkFormatter: (time) => {
          const idx = Math.max(0, Math.min(pnlEdges.length - 1, Math.round(time) - 1));
          const val = pnlEdges[idx];
          return val != null ? Number(val).toFixed(4) : "";
        },
      },
      rightPriceScale: { borderVisible: false, visible: false },
    });
    const pnlSeries = pnlChart.addHistogramSeries({ color: palette.accent2, base: 0 });

    function sanitize(points = [], label = "") {
      const seen = new Map();
      let dropped = 0;
      const bad = [];
      for (const p of points || []) {
        const t = p?.time;
        const v = p?.value;
        if (
          t == null ||
          v == null ||
          !Number.isFinite(t) ||
          !Number.isFinite(v) ||
          Number.isNaN(t) ||
          Number.isNaN(v)
        ) {
          dropped += 1;
          bad.push(p);
          continue;
        }
        // Keep the last value per timestamp to avoid duplicate times.
        seen.set(Number(t), { time: Number(t), value: Number(v), color: p.color });
      }
      const clean = Array.from(seen.values()).sort((a, b) => a.time - b.time);
      if (label) {
        if (dropped > 0) {
          console.warn(`sanitize(${label}) dropped ${dropped} bad points`);
        }
        if (clean.length === 0 && (points || []).length) {
          console.warn(`sanitize(${label}) produced 0 points`, { sample: (points || []).slice(0, 5) });
        }
      }
      return clean;
    }

    function setStatus(id, text) {
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    }

    function setPriceMode(mode) {
      priceMode = mode;
      if (mode === "area") {
        safeSet(priceAreaSeries, lastPriceArea, "price-area");
        safeSet(priceCandleSeries, [], "price-candles");
      } else {
        safeSet(priceCandleSeries, lastPriceCandles, "price-candles");
        safeSet(priceAreaSeries, [], "price-area");
      }
      const btn = document.getElementById("price-toggle");
      if (btn) btn.textContent = mode === "area" ? "Candles" : "Area";
    }

    async function fetchJson(url) {
      const res = await fetch(url, { cache: "no-store" });
      return res.json();
    }

    function safeSet(series, data, label) {
      try {
        series.setData(data);
      } catch (err) {
        console.error(`setData failed for ${label}: len=${data?.length}`, err, data?.slice ? data.slice(0, 5) : data);
        series.setData([]);
      }
    }

    async function loadPrice() {
      try {
        const data = await fetchJson(`/api/price?freq=${priceTf}`);
        if (data.error) {
          setStatus("price-status", data.error);
          priceAreaSeries.setData([]);
          priceCandleSeries.setData([]);
          return;
        }
        const merged = [];
        (data.series || []).forEach(block => {
          (block.points || []).forEach(pt => merged.push({ time: pt.time, value: pt.value }));
        });
        const clean = sanitize(merged, "price").sort((a, b) => a.time - b.time);
        console.log("price payload", { merged: merged.length, clean: clean.length, sampleClean: clean.slice(0, 3), sampleMerged: merged.slice(0, 3) });
        if ((data.series || []).length && clean.length === 0) {
          console.warn("sanitize(price) dropped all points", { mergedLength: merged.length, raw: merged.slice(0, 5) });
        }
        lastPriceArea = clean;

        const candles = [];
        (data.candles || []).forEach(block => {
          (block.candles || []).forEach(c => candles.push(c));
        });
        const cleanCandles = candles
          .filter(c => c.time != null && c.open != null && c.high != null && c.low != null && c.close != null)
          .map(c => ({
            time: Number(c.time),
            open: Number(c.open),
            high: Number(c.high),
            low: Number(c.low),
            close: Number(c.close),
          }))
          .sort((a, b) => a.time - b.time);
        lastPriceCandles = cleanCandles;

        if (priceMode === "area") {
          safeSet(priceAreaSeries, lastPriceArea, "price-area");
          safeSet(priceCandleSeries, [], "price-candles");
        } else {
          safeSet(priceCandleSeries, lastPriceCandles, "price-candles");
          safeSet(priceAreaSeries, [], "price-area");
        }
        setStatus("price-status", `Updated ${new Date().toLocaleTimeString()} (${priceMode === "area" ? "Area" : "Candles"}, ${priceTf})`);
      } catch (err) {
        setStatus("price-status", `Error: ${err}`);
        priceAreaSeries.setData([]);
        priceCandleSeries.setData([]);
      }
    }

    function histogramFromEdges(edges = [], counts = []) {
      // Lightweight charts expects one value per bar; map histogram edges to centers.
      const pts = [];
      for (let i = 0; i < counts.length; i++) {
        const center = (edges[i] + edges[i + 1]) / 2;
        pts.push({ time: i + 1, value: counts[i], color: palette.accent2 });
      }
      return pts;
    }

    async function loadMetrics() {
      try {
        const data = await fetchJson("/api/equity");
        if (data.error) {
          equitySeries.setData([]);
          drawdownSeries.setData([]);
          rollingSeries.setData([]);
          evSeries.setData([]);
          pnlSeries.setData([]);
          setStatus("equity-status", data.error);
          return;
        }

        const times = data.times || [];
        const equity = sanitize((data.equity || []).map((v, idx) => ({ time: times[idx], value: v })), "equity");
        const dd = sanitize((data.drawdown || []).map((v, idx) => ({ time: times[idx], value: v })), "drawdown");
        const rollingTimes = data.rollingTimes || times;
        const rolling = sanitize((data.rollingMeanBps || []).map((v, idx) => ({ time: rollingTimes[idx], value: v })), "rolling");

        console.log("metrics payload", {
          equity: equity.length,
          drawdown: dd.length,
          rolling: rolling.length,
          ev: (data.evBuckets || []).length,
          pnl: (data.pnlHistogram?.counts || []).length,
          times: times.length,
          rollingTimes: rollingTimes.length,
          sampleRolling: rolling.slice(0, 3),
        });

        safeSet(equitySeries, equity, "equity");
        safeSet(drawdownSeries, dd, "drawdown");
        safeSet(rollingSeries, rolling, "rolling");

        evLabels = (data.evBuckets || []).map(b => b.label || "");
        const evBuckets = sanitize((data.evBuckets || []).map((b, idx) => ({ time: idx + 1, value: b.value, color: palette.accent })), "ev");
        safeSet(evSeries, evBuckets, "ev");

        pnlEdges = data.pnlHistogram?.edges || [];
        const pnlHist = sanitize(histogramFromEdges(pnlEdges, data.pnlHistogram?.counts), "pnl");
        safeSet(pnlSeries, pnlHist, "pnl");

        setStatus("equity-status", `Metrics refreshed ${new Date().toLocaleTimeString()}`);
      } catch (err) {
        setStatus("equity-status", `Error: ${err}`);
      }
    }

    loadPrice();
    loadMetrics();
    setInterval(loadPrice, priceRefreshMs);
    setInterval(loadMetrics, metricRefreshMs);

    document.getElementById("price-toggle")?.addEventListener("click", () => {
      setPriceMode(priceMode === "area" ? "candles" : "area");
    });

    document.getElementById("price-tf")?.addEventListener("change", (e) => {
      priceTf = e.target.value || "1m";
      loadPrice();
    });

    // initialize with area data when first payload arrives
  </script>
</body>
</html>
"""


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


def _parse_jsonl_lines(lines: list[str], symbol_fallback: str) -> pd.DataFrame:
    rows: List[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mtype = msg.get("type")
        if mtype not in {"match", "last_match", "ticker"}:
            continue
        try:
            price = float(msg.get("price", "nan"))
            size = float(msg.get("size", msg.get("last_size", "nan")))
        except (TypeError, ValueError):
            continue
        t = pd.to_datetime(msg.get("time"))
        symbol = msg.get("product_id", symbol_fallback)
        rows.append(
            {
                "Time": t,
                "Symbol": symbol,
                "BidPrice1": price,
                "AskPrice1": price,
                "BidVolume1": 0.0,
                "AskVolume1": 0.0,
                "Volume": size,
            }
        )
    return pd.DataFrame(rows)


def _load_file_cached(path: Path) -> pd.DataFrame:
    """
    Load a file with caching to avoid full rereads.
    JSONL: append only new lines since last size.
    CSV: reload only if mtime/size changed.
    """
    global _PRICE_CACHE
    try:
        stat = path.stat()
    except FileNotFoundError:
        return pd.DataFrame()

    cache = _PRICE_CACHE.get(path)

    if path.suffix.lower() == ".jsonl":
        offset = cache.get("size", 0) if cache else 0
        new_rows: List[dict] = []
        with path.open("r", encoding="utf-8") as f:
            f.seek(offset)
            new_lines = f.readlines()
        new_df = _parse_jsonl_lines(new_lines, path.stem)
        if cache and not cache["df"].empty:
            base_df = cache["df"]
            df = pd.concat([base_df, new_df], ignore_index=True)
        else:
            df = new_df
        _PRICE_CACHE[path] = {"df": df, "mtime": stat.st_mtime, "size": stat.st_size}
        return df

    # CSV: simple mtime check
    if cache and cache.get("mtime") == stat.st_mtime and cache.get("size") == stat.st_size:
        return cache["df"]

    try:
        df = pd.read_csv(path, parse_dates=["Time"])
        if "Symbol" not in df.columns:
            df["Symbol"] = path.stem
        _PRICE_CACHE[path] = {"df": df, "mtime": stat.st_mtime, "size": stat.st_size}
        return df
    except Exception:
        return pd.DataFrame()


def load_mid_series(data_dir: Path, lookback_min: int) -> pd.Series:
    data_dir = data_dir.resolve()
    files = sorted(list(data_dir.glob("*.jsonl")) + list(data_dir.glob("*.csv")))
    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    if USE_PRICE_CACHE:
        frames = []
        for fpath in files:
            df = _load_file_cached(fpath)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.Series(dtype=float)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = load_all_raw_data(data_dir).reset_index()

    df = ensure_multiindex(df)
    df_feat = add_basic_features(df)
    mid = df_feat["mid"].dropna().sort_index()
    if mid.empty:
        return mid
    cutoff = mid.index.get_level_values("Time").max() - pd.Timedelta(minutes=lookback_min)
    return mid[mid.index.get_level_values("Time") >= cutoff]


def build_candles(mid: pd.Series, freq: str = "60s") -> list[dict]:
    """
    Aggregate mid-price into simple OHLC candles per symbol.
    """
    if mid.empty:
        return []
    df = mid.reset_index().rename(columns={0: "mid"})
    df = df.rename(columns={"mid": "price"})
    df["Time"] = pd.to_datetime(df["Time"])
    candles = []
    for symbol, chunk in df.groupby("Symbol"):
        c = (
            chunk.set_index("Time")["price"]
            .resample(freq)
            .agg(["first", "max", "min", "last"])
            .dropna()
        )
        if c.empty:
            continue
        candles.append(
            {
                "symbol": symbol,
                "candles": [
                    {
                        "time": int(ts.value // 1_000_000_000),
                        "open": float(row["first"]),
                        "high": float(row["max"]),
                        "low": float(row["min"]),
                        "close": float(row["last"]),
                    }
                    for ts, row in c.iterrows()
                ],
            }
        )
    return candles


def load_equity_curve() -> Optional[pd.DataFrame]:
    path = PROJECT_ROOT / "artifacts" / "backtest_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Time"])
    if df.empty:
        return None
    return df


def _drawdown(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    peak = np.maximum.accumulate(arr)
    dd = arr / peak - 1.0
    return dd.tolist()


def _ev_buckets(expected: pd.Series, returns: pd.Series, buckets: int = 5) -> List[Dict[str, float]]:
    if expected.empty or returns.empty:
        return []
    try:
        bins = pd.qcut(expected, q=buckets, duplicates="drop")
        bucket_ret = (
            pd.DataFrame({"bucket": bins, "ret": returns})
            .groupby("bucket", observed=False)["ret"]
            .mean()
        )
        labels = [str(idx) for idx in bucket_ret.index]
        return [{"label": lbl, "value": float(val)} for lbl, val in zip(labels, bucket_ret.values)]
    except Exception:
        return []


def _pnl_histogram(returns: pd.Series, bins: int = 30) -> Dict[str, List[float]]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(clean.values, bins=bins)
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def _rolling_mean(returns: pd.Series, window: int = 200, min_periods: int = 20) -> List[Optional[float]]:
    if returns.empty:
        return []
    roll = pd.to_numeric(returns, errors="coerce").rolling(window, min_periods=min_periods).mean()
    return [None if pd.isna(v) else float(v) for v in roll]


def _filter_finite_pairs(times: List[int], values: List[Optional[float]]) -> tuple[List[int], List[float]]:
    """Drop any entries where value is None/NaN or non-finite to keep chart happy."""
    clean_times: List[int] = []
    clean_vals: List[float] = []
    for t, v in zip(times, values):
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(fv)):
            continue
        clean_times.append(t)
        clean_vals.append(fv)
    return clean_times, clean_vals


_last_backtest_run = 0.0
_backtest_lock = threading.Lock()


def refresh_backtest_if_needed():
    """
    Keep artifacts/backtest_results.csv fresh while throttling executions.
    """
    global _last_backtest_run
    now = time.time()
    if now - _last_backtest_run < BACKTEST_COOLDOWN_SEC:
        return
    with _backtest_lock:
        if now - _last_backtest_run < BACKTEST_COOLDOWN_SEC:
            return
        os.environ["BT_DATA_DIR"] = str(_pick_data_dir())
        os.environ["BACKTEST_SAVE"] = "1"
        run_backtest.main()
        _last_backtest_run = time.time()


def _series_to_points(series: pd.Series) -> List[Dict[str, float]]:
    """
    Convert a MultiIndex Series with Time level into lightweight-charts points.
    """
    points: List[Dict[str, float]] = []
    times = pd.to_datetime(series.index.get_level_values("Time"))
    for t, v in zip(times, series.values):
        if pd.isna(v):
            continue
        points.append({"time": int(t.value // 1_000_000_000), "value": float(v)})
    return points


class TVRequestHandler(BaseHTTPRequestHandler):
    server_version = "TVLightweight/0.1"

    def _send_json(self, payload, status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self):
        html = (
            INDEX_HTML_TEMPLATE
            .replace("PRICE_REFRESH_MS_PLACEHOLDER", str(PRICE_REFRESH_MS))
            .replace("METRIC_REFRESH_MS_PLACEHOLDER", str(METRIC_REFRESH_MS))
            .replace("TV_LOOKBACK_MIN_PLACEHOLDER", str(LOOKBACK_MIN))
        )
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        # Quieter logging; keep concise console output.
        sys.stdout.write(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}\n")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_html()
        if parsed.path == "/api/price":
            return self.handle_price()
        if parsed.path == "/api/equity":
            return self.handle_equity()
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def handle_price(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        freq_key = params.get("freq", ["1m"])[0]
        freq = CANDLE_FREQS.get(freq_key, CANDLE_FREQS["1m"])
        try:
            mid = load_mid_series(_pick_data_dir(), LOOKBACK_MIN)
        except Exception as exc:
            return self._send_json({"error": str(exc), "series": []}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        series: List[Dict[str, object]] = []
        for symbol, chunk in mid.groupby(level=0):
            series.append({"symbol": symbol, "points": _series_to_points(chunk)})
        candles = build_candles(mid, freq=freq)
        self._send_json({"series": series, "candles": candles, "freq": freq_key})

    def handle_equity(self):
        try:
            refresh_backtest_if_needed()
            df = load_equity_curve()
        except Exception as exc:
            return self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        if df is None:
            return self._send_json({"error": "No backtest results found. Run a backtest to create artifacts/backtest_results.csv."})

        df = df.sort_values("Time").drop_duplicates(subset=["Time"], keep="last")

        times = pd.to_datetime(df["Time"])
        epoch_times = [int(t.value // 1_000_000_000) for t in times]
        equity = pd.to_numeric(df.get("equity"), errors="coerce").ffill()

        payload = {
            "times": epoch_times,
            "equity": [float(v) if not pd.isna(v) else None for v in equity],
            # Convert to percent for better visibility in the chart.
            "drawdown": [float(v) * 100 for v in _drawdown(equity.tolist())],
        }

        if "return" in df.columns:
            returns = pd.to_numeric(df["return"], errors="coerce")
            payload["pnlHistogram"] = _pnl_histogram(returns)
            # Show rolling mean in basis points for readability.
            rolling_bps = [
                None if v is None else float(v) * 10000 for v in _rolling_mean(returns)
            ]
            rt_times, rt_vals = _filter_finite_pairs(epoch_times, rolling_bps)
            payload["rollingMeanBps"] = rt_vals
            payload["rollingTimes"] = rt_times
        else:
            payload["pnlHistogram"] = {"edges": [], "counts": []}
            payload["rollingMeanBps"] = []
            payload["rollingTimes"] = []

        if {"expected", "return"}.issubset(df.columns):
            expected = pd.to_numeric(df["expected"], errors="coerce")
            returns = pd.to_numeric(df["return"], errors="coerce")
            payload["evBuckets"] = _ev_buckets(expected, returns)
        else:
            payload["evBuckets"] = []

        self._send_json(payload)


def main():
    addr = ("0.0.0.0", PORT)
    httpd = ThreadingHTTPServer(addr, TVRequestHandler)
    print(f"Serving TradingView Lightweight Charts at http://localhost:{PORT}")
    print(f"Data dir: {_pick_data_dir()}")
    print(f"Price refresh: {PRICE_REFRESH_MS} ms | Metric refresh: {METRIC_REFRESH_MS} ms | Backtest cooldown: {BACKTEST_COOLDOWN_SEC} s")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down viewer.")
        httpd.server_close()


if __name__ == "__main__":
    main()
