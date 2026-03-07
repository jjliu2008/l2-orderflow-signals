"""
Two-agent OpenClaw orchestrator for strategy experimentation.

Flow:
  1) Run experiment_agent.py for the current config.
  2) Summarize results and send to the ideas agent.
  3) Validate proposal against guardrails.
  4) Ask coding agent to produce patches.
  5) Apply patches locally.
  6) Loop until stop conditions.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_AGENT = PROJECT_ROOT / "scripts" / "experiment_agent.py"


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        proc = subprocess.run(
            ["powershell", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return "ProcessName" in proc.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_lock(outdir: Path) -> Path:
    lock_path = PROJECT_ROOT / ".experiment_orchestrator.lock"
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0))
        except Exception:
            pid = 0
        if _pid_running(pid):
            raise RuntimeError(
                f"experiment_orchestrator already running (pid={pid}). Remove {lock_path} if stale."
            )
    outdir.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "started_at": time.time()}, indent=2),
        encoding="utf-8",
    )
    return lock_path


@dataclass(frozen=True)
class ScreeningConfig:
    enabled: bool
    train_days: int
    lock_days: int
    top_k: int
    min_valid_runs: int
    full_run_on_positive_only: bool
    positive_lock_ev_threshold: float


@dataclass(frozen=True)
class OrchestratorConfig:
    experiment_config: Path
    outdir: Path
    ideas_agent: str
    coding_agent: str
    openclaw_timeout_seconds: int
    idea_mode: str
    max_iterations: int
    max_no_improve: int
    pivot_min_lock_ev: float
    pivot_min_lock_ev_iterations: int
    pivot_family_fail_limit: int
    max_structural_variants_per_family: int
    pivot_min_right_tail_mfe3: float
    pivot_min_right_tail_mfe4: float
    scoring_enabled: bool
    score_weight_max_dd: float
    score_weight_worst_day: float
    score_weight_breach: float
    score_weight_trade_shortfall: float
    min_promote_score_delta: float
    min_promote_lock_ev_delta: float
    min_promote_trades_lock: int
    revert_on_reject: bool
    allowed_files: List[str]
    allowed_knobs: Dict[str, List[str]]
    allow_any_knob: bool
    allow_fixed_env_changes: bool
    allowed_fixed_env_keys: List[str]
    ideas_require_strict_json: bool
    ideas_allow_embedded_json: bool
    research_require_web_tools: bool
    research_require_source_url: bool
    research_require_tool_evidence: bool
    research_min_web_search_count: int
    research_min_web_fetch_count: int
    research_min_sources: int
    research_max_sources: int
    research_allowed_domains: List[str]
    research_blocked_domains: List[str]
    research_preferred_domains: List[str]
    auto_mode_start: str
    auto_grid_to_structural_no_improve: int
    auto_grid_to_structural_best_lock_below: float
    auto_grid_to_structural_require_tail_failure: bool
    auto_structural_revert_on_improve: bool
    screening: ScreeningConfig | None
    max_proposal_attempts: int
    max_patch_attempts: int
    rotation_request_limit: int


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, object]) -> None:
    path.write_text(json.dumps(_json_safe(data), indent=2), encoding="utf-8")


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value

    # Convert numpy/pandas scalar types (e.g. int64/float64) to native Python.
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    # Serialize pandas timestamps / datetimes consistently.
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


def _extract_code_blocks(text: str, fence: str | None = None) -> List[str]:
    if fence:
        pattern = rf"```{re.escape(fence)}\\s*(.*?)```"
    else:
        pattern = r"```\\s*(.*?)```"
    return [m.strip() for m in re.findall(pattern, text, flags=re.DOTALL)]


def _extract_json_block(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in agent response.")
    return json.loads(match.group(0))


def _extract_json_from_text(text: str) -> Dict[str, object]:
    for fence in ("json", None):
        for block in _extract_code_blocks(text, fence=fence):
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    return _extract_json_block(text)


def _looks_like_input_request(text: str) -> bool:
    lower = text.lower()
    request_markers = [
        "needed_from_you",
        "optimization_target",
        "top_rows",
        "paste the top 3 rows",
        "tell me which direction",
        "which direction you want to optimize",
    ]
    if any(marker in lower for marker in request_markers):
        # If it also includes a grid payload, treat it as potentially valid.
        if "\"grid\"" in lower or "'grid'" in lower:
            return False
        return True
    return False


def _is_strict_json_only(text: str) -> bool:
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False
    try:
        parsed = json.loads(stripped)
    except Exception:
        return False
    return isinstance(parsed, dict)


def _research_source_urls(research: Dict[str, object]) -> List[str]:
    urls: List[str] = []
    sources = research.get("sources", [])
    if not isinstance(sources, list):
        return urls
    for item in sources:
        url = ""
        if isinstance(item, str):
            url = item.strip()
        elif isinstance(item, dict):
            raw = item.get("url", "")
            url = str(raw).strip() if raw is not None else ""
        if re.match(r"^https?://", url, flags=re.IGNORECASE):
            urls.append(url)
    return urls


def _normalize_host(host: str) -> str:
    host = host.strip().lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _url_host(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    return _normalize_host(host)


def _domain_match(host: str, pattern: str) -> bool:
    host = _normalize_host(host)
    pat = _normalize_host(pattern)
    if not host or not pat:
        return False
    return host == pat or host.endswith("." + pat)


def _parse_iso_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ideas_sessions_dir(agent_id: str) -> Path:
    return Path.home() / ".openclaw" / "agents" / agent_id / "sessions"


def _snapshot_agent_sessions(agent_id: str) -> Dict[str, float]:
    sessions_dir = _ideas_sessions_dir(agent_id)
    snap: Dict[str, float] = {}
    if not sessions_dir.exists():
        return snap
    for path in sessions_dir.glob("*.jsonl"):
        try:
            snap[str(path.resolve())] = path.stat().st_mtime
        except Exception:
            continue
    return snap


def _collect_openclaw_tool_evidence(
    agent_id: str,
    start_ts: float,
    end_ts: float,
    before_snapshot: Dict[str, float] | None = None,
) -> Dict[str, object]:
    sessions_dir = _ideas_sessions_dir(agent_id)
    evidence: Dict[str, object] = {
        "agent_id": agent_id,
        "window_start": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
        "window_end": datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
        "session_files": [],
        "tool_calls": [],
        "web_tools_used": False,
        "web_search_count": 0,
        "web_fetch_count": 0,
        "observed_urls": [],
        "observed_domains": [],
        "errors": [],
    }
    if not sessions_dir.exists():
        evidence["errors"] = ["sessions_dir_missing"]
        return evidence

    start_dt = datetime.fromtimestamp(start_ts - 5, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts + 10, tz=timezone.utc)
    before_snapshot = before_snapshot or {}

    url_seen: set[str] = set()
    domain_seen: set[str] = set()
    tool_calls: List[Dict[str, object]] = []
    files_used: List[str] = []

    candidates = sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for path in candidates[:8]:
        try:
            mtime = path.stat().st_mtime
        except Exception:
            continue
        prev_mtime = before_snapshot.get(str(path.resolve()))
        if prev_mtime is not None and mtime <= prev_mtime and mtime < start_ts - 1:
            continue
        if mtime < start_ts - 300:
            continue

        file_used = False
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    event_dt = _parse_iso_ts(rec.get("timestamp"))
                    if event_dt is not None and (event_dt < start_dt or event_dt > end_dt):
                        continue
                    if rec.get("type") != "message":
                        continue
                    msg = rec.get("message", {})
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role")
                    if role == "assistant":
                        content = msg.get("content", [])
                        if not isinstance(content, list):
                            continue
                        for item in content:
                            if not isinstance(item, dict):
                                continue
                            if item.get("type") != "toolCall":
                                continue
                            name = str(item.get("name", ""))
                            if name not in {"web_search", "web_fetch"}:
                                continue
                            file_used = True
                            call_args = item.get("arguments", {})
                            call_entry: Dict[str, object] = {
                                "tool": name,
                                "timestamp": event_dt.isoformat() if event_dt else None,
                                "query": call_args.get("query") if isinstance(call_args, dict) else None,
                                "url": call_args.get("url") if isinstance(call_args, dict) else None,
                            }
                            tool_calls.append(_json_safe(call_entry))  # type: ignore[arg-type]
                            evidence["web_tools_used"] = True
                            if name == "web_search":
                                evidence["web_search_count"] = int(evidence["web_search_count"]) + 1
                            if name == "web_fetch":
                                evidence["web_fetch_count"] = int(evidence["web_fetch_count"]) + 1
                            if isinstance(call_args, dict):
                                raw_url = call_args.get("url")
                                if isinstance(raw_url, str) and re.match(r"^https?://", raw_url, flags=re.IGNORECASE):
                                    url_seen.add(raw_url.strip())
                    elif role == "toolResult":
                        tool_name = str(msg.get("toolName", ""))
                        if tool_name not in {"web_search", "web_fetch"}:
                            continue
                        file_used = True
                        details = msg.get("details", {})
                        if isinstance(details, dict):
                            if tool_name == "web_search":
                                results = details.get("results", [])
                                if isinstance(results, list):
                                    for r in results:
                                        if not isinstance(r, dict):
                                            continue
                                        u = r.get("url")
                                        if isinstance(u, str) and re.match(r"^https?://", u, flags=re.IGNORECASE):
                                            url_seen.add(u.strip())
                            if tool_name == "web_fetch":
                                u = details.get("url")
                                if isinstance(u, str) and re.match(r"^https?://", u, flags=re.IGNORECASE):
                                    url_seen.add(u.strip())
        except Exception as exc:
            errs = evidence.get("errors", [])
            if isinstance(errs, list):
                errs.append(f"{path.name}:{type(exc).__name__}")
            continue
        if file_used:
            files_used.append(str(path))

    for u in sorted(url_seen):
        h = _url_host(u)
        if h:
            domain_seen.add(h)
    evidence["tool_calls"] = tool_calls
    evidence["session_files"] = files_used
    evidence["observed_urls"] = sorted(url_seen)
    evidence["observed_domains"] = sorted(domain_seen)
    return evidence


def _validate_ideas_research_rule(
    proposal: Dict[str, object],
    cfg: OrchestratorConfig,
    tool_evidence: Dict[str, object] | None = None,
) -> Tuple[bool, str]:
    research = proposal.get("research")
    if not isinstance(research, dict):
        return False, "Missing required research object."

    used_web_tools = bool(research.get("used_web_tools", False))
    if cfg.research_require_web_tools and not used_web_tools:
        return False, "research.used_web_tools must be true."

    urls = _research_source_urls(research)
    if cfg.research_require_source_url and len(urls) < max(1, cfg.research_min_sources):
        return False, "research.sources must include at least one http(s) URL."

    if cfg.research_max_sources > 0 and len(urls) > cfg.research_max_sources:
        return False, f"research.sources has too many URLs ({len(urls)} > {cfg.research_max_sources})."

    unique_urls: List[str] = []
    seen_urls: set[str] = set()
    for u in urls:
        if u not in seen_urls:
            unique_urls.append(u)
            seen_urls.add(u)

    for u in unique_urls:
        host = _url_host(u)
        if not host:
            return False, f"Invalid source URL host: {u}"
        if any(_domain_match(host, d) for d in cfg.research_blocked_domains):
            return False, f"Blocked source domain: {host}"
        if cfg.research_allowed_domains and not any(_domain_match(host, d) for d in cfg.research_allowed_domains):
            return False, f"Source domain not allowlisted: {host}"

    if cfg.research_require_tool_evidence:
        if not isinstance(tool_evidence, dict):
            return False, "Missing OpenClaw tool evidence for ideas turn."
        if not bool(tool_evidence.get("web_tools_used", False)):
            return False, "No observed web_search/web_fetch tool usage in OpenClaw session."
        observed_search_count = int(tool_evidence.get("web_search_count", 0) or 0)
        observed_fetch_count = int(tool_evidence.get("web_fetch_count", 0) or 0)
        if observed_search_count < cfg.research_min_web_search_count:
            return (
                False,
                f"Observed web_search count too low ({observed_search_count} < {cfg.research_min_web_search_count}).",
            )
        if observed_fetch_count < cfg.research_min_web_fetch_count:
            return (
                False,
                f"Observed web_fetch count too low ({observed_fetch_count} < {cfg.research_min_web_fetch_count}).",
            )
        observed_urls = {
            str(u).strip()
            for u in tool_evidence.get("observed_urls", [])
            if isinstance(u, str) and u.strip()
        }
        observed_domains = {
            _normalize_host(str(d))
            for d in tool_evidence.get("observed_domains", [])
            if isinstance(d, str) and str(d).strip()
        }
        if cfg.research_require_source_url and unique_urls:
            overlaps = False
            for u in unique_urls:
                if u in observed_urls:
                    overlaps = True
                    break
                host = _url_host(u)
                if host and host in observed_domains:
                    overlaps = True
                    break
            if not overlaps:
                return False, "Reported research.sources do not match observed web tool results."

    return True, ""


def _extract_patch_from_text(text: str) -> str | None:
    for marker in ("*** Begin Patch", "diff --git"):
        idx = text.find(marker)
        if idx != -1:
            return text[idx:].strip()
    for fence in ("diff", "patch"):
        for block in _extract_code_blocks(text, fence=fence):
            if "diff --git" in block or block.startswith("*** Begin Patch"):
                return block.strip()
    return None


def _make_unified_diff(path: Path, before: str, after: str) -> str:
    try:
        rel_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_path = path
    rel = rel_path.as_posix()
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
    )


def _dict_delta(before: Dict[str, object], after: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    delta: Dict[str, Dict[str, object]] = {}
    for key in sorted(set(before.keys()) | set(after.keys())):
        b = before.get(key)
        a = after.get(key)
        if b != a:
            delta[str(key)] = {"before": _json_safe(b), "after": _json_safe(a)}  # type: ignore[arg-type]
    return delta


def _effective_idea_mode(cfg: OrchestratorConfig, family_modes: Dict[str, str], family_id: str) -> str:
    if cfg.idea_mode != "auto":
        return cfg.idea_mode
    return family_modes.get(family_id, cfg.auto_mode_start if cfg.auto_mode_start in {"grid", "structural"} else "grid")


def _should_switch_grid_to_structural(
    cfg: OrchestratorConfig,
    best_expectancy: float,
    no_improve: int,
    tails: Dict[str, object] | None,
) -> Tuple[bool, str]:
    if no_improve < cfg.auto_grid_to_structural_no_improve:
        return False, ""
    if best_expectancy >= cfg.auto_grid_to_structural_best_lock_below:
        return False, ""
    if cfg.auto_grid_to_structural_require_tail_failure:
        tails = tails if isinstance(tails, dict) else {}
        p_mfe_ge_3 = float(tails.get("p_mfe_ge_3", 0.0) or 0.0)
        p_mfe_ge_4 = float(tails.get("p_mfe_ge_4", 0.0) or 0.0)
        if not (
            p_mfe_ge_3 <= cfg.pivot_min_right_tail_mfe3
            and p_mfe_ge_4 <= cfg.pivot_min_right_tail_mfe4
        ):
            return False, ""
    return True, "grid_to_structural_auto"


def _strip_locked_env(
    grid: List[Dict[str, object]],
    locked_keys: Iterable[str],
) -> List[Dict[str, object]]:
    locked = set(locked_keys)
    cleaned: List[Dict[str, object]] = []
    for entry in grid:
        env = entry.get("env", {})
        if not isinstance(env, dict):
            continue
        new_env = {k: v for k, v in env.items() if k not in locked}
        if not new_env:
            continue
        cleaned.append({"name": entry.get("name", "variant"), "env": new_env})
    return cleaned


def _grid_from_idea_text(
    idea_text: str,
    allowed_knobs: Dict[str, List[str]],
    allow_any_knob: bool,
) -> Tuple[Dict[str, object] | None, List[Dict[str, object]]]:
    try:
        idea = _extract_json_from_text(idea_text)
    except Exception:
        return None, []
    grid = idea.get("grid", [])
    if not isinstance(grid, list) or not grid:
        return idea, []
    filtered: List[Dict[str, object]] = []
    for entry in grid:
        if not isinstance(entry, dict):
            continue
        env = entry.get("env", {})
        if not isinstance(env, dict):
            continue
        clean_env: Dict[str, str] = {}
        invalid = False
        for k, v in env.items():
            if not allow_any_knob:
                if k not in allowed_knobs:
                    invalid = True
                    break
                allowed_vals = [str(x) for x in allowed_knobs[k]]
                if str(v) not in allowed_vals:
                    invalid = True
                    break
            clean_env[k] = str(v)
        if invalid or not clean_env:
            continue
        filtered.append(
            {
                "name": entry.get("name", "variant"),
                "env": clean_env,
            }
        )
    return idea, filtered


def _date_from_ymd(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _truncate_date_range(start: str, end: str, max_days: int) -> Tuple[str, str]:
    start_dt = _date_from_ymd(start)
    end_dt = _date_from_ymd(end)
    if start_dt is None or end_dt is None or max_days <= 0:
        return start, end
    cap_end = start_dt + timedelta(days=max_days - 1)
    if cap_end > end_dt:
        cap_end = end_dt
    return start_dt.strftime("%Y-%m-%d"), cap_end.strftime("%Y-%m-%d")


def _build_screening_config(
    experiment_config: Path,
    screen_dir: Path,
    screening: ScreeningConfig,
) -> Tuple[Path, Dict[str, object]]:
    cfg_data = _load_json(experiment_config)
    guardrails = cfg_data.get("guardrails", {})
    if not isinstance(guardrails, dict):
        guardrails = {}
    cfg_data["guardrails"] = {**guardrails, "allow_screening_split_override": True}
    data_days = cfg_data.get("data_days", {})
    if isinstance(data_days, dict):
        ts = str(data_days.get("train_start", ""))
        te = str(data_days.get("train_end", ""))
        ls = str(data_days.get("lock_start", ""))
        le = str(data_days.get("lock_end", ""))
        nts, nte = _truncate_date_range(ts, te, screening.train_days)
        nls, nle = _truncate_date_range(ls, le, screening.lock_days)
        cfg_data["data_days"] = {
            **data_days,
            "train_start": nts,
            "train_end": nte,
            "lock_start": nls,
            "lock_end": nle,
        }
    screen_dir.mkdir(parents=True, exist_ok=True)
    screen_cfg = screen_dir / "screening_config.json"
    _write_json(screen_cfg, cfg_data)
    return screen_cfg, {"data_days": cfg_data.get("data_days", {})}


def _top_grid_from_comparison(
    comparison_path: Path,
    original_grid: List[Dict[str, object]],
    top_k: int,
    min_valid_runs: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if top_k <= 0:
        return original_grid, {"selected_names": [g.get("name") for g in original_grid]}
    try:
        df = pd.read_csv(comparison_path)
    except Exception as exc:
        return original_grid, {"screening_error": f"{type(exc).__name__}: {exc}"}
    if df.empty or "name" not in df.columns:
        return original_grid, {"screening_error": "empty_or_invalid_comparison"}

    ranked = df.copy()
    if "status" in ranked.columns:
        ranked["__ok"] = (ranked["status"].astype(str) == "OK").astype(int)
    else:
        ranked["__ok"] = 0
    if "lock_expectancy_ticks" not in ranked.columns:
        ranked["lock_expectancy_ticks"] = float("-inf")
    ranked["lock_expectancy_ticks"] = pd.to_numeric(ranked["lock_expectancy_ticks"], errors="coerce").fillna(float("-inf"))
    ranked = ranked.sort_values(["__ok", "lock_expectancy_ticks"], ascending=[False, False])

    ok_count = int(ranked["__ok"].sum())
    selected_names = [str(x) for x in ranked["name"].head(top_k).tolist()]
    if ok_count < min_valid_runs:
        # Fall back to the original full grid if screening was too sparse/noisy.
        return original_grid, {
            "selected_names": [g.get("name") for g in original_grid],
            "fallback": "min_valid_runs",
            "ok_count": ok_count,
        }

    grid_by_name = {
        str(item.get("name")): item
        for item in original_grid
        if isinstance(item, dict) and item.get("name") is not None
    }
    selected_grid = [grid_by_name[name] for name in selected_names if name in grid_by_name]
    if not selected_grid:
        return original_grid, {"screening_error": "selected_grid_empty"}
    return selected_grid, {"selected_names": selected_names, "ok_count": ok_count}


def _screening_positive_summary(
    comparison_path: Path,
    threshold: float,
) -> Dict[str, object]:
    try:
        df = pd.read_csv(comparison_path)
    except Exception as exc:
        return {
            "comparison_read_error": f"{type(exc).__name__}: {exc}",
            "best_lock_expectancy": float("-inf"),
            "positive_run_names": [],
            "positive_count": 0,
            "threshold": threshold,
            "has_positive": False,
        }
    if df.empty or "lock_expectancy_ticks" not in df.columns:
        return {
            "best_lock_expectancy": float("-inf"),
            "positive_run_names": [],
            "positive_count": 0,
            "threshold": threshold,
            "has_positive": False,
            "missing_lock_expectancy_ticks": "lock_expectancy_ticks" not in df.columns,
        }
    series = pd.to_numeric(df["lock_expectancy_ticks"], errors="coerce")
    best = float(series.max()) if not series.dropna().empty else float("-inf")
    positive_mask = series > threshold
    positive_count = int(positive_mask.fillna(False).sum())
    names: List[str] = []
    if "name" in df.columns:
        names = [str(x) for x in df.loc[positive_mask, "name"].tolist()]
    return {
        "best_lock_expectancy": best,
        "positive_run_names": names,
        "positive_count": positive_count,
        "threshold": threshold,
        "has_positive": positive_count > 0,
    }


def _run_openclaw_agent(
    agent_id: str,
    message: str,
    timeout: int = 600,
    json_only: bool = False,
    session_id: str | None = None,
) -> str:
    cmd = ["openclaw.cmd", "agent", "--agent", agent_id, "--timeout", str(timeout)]
    if session_id:
        cmd.extend(["--session-id", session_id])
    if json_only:
        cmd.append("--json")
    cmd.extend(["--message", message])
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout + 30,
        check=False,
    )
    return proc.stdout


def _strip_openclaw_preamble(text: str) -> str:
    marker = "|"
    if marker in text:
        idx = text.find(marker)
        return text[idx + 1 :].strip()
    return text.strip()


def _apply_patch(patch_text: str) -> None:
    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=str(PROJECT_ROOT),
        input=patch_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Patch apply failed:\n{proc.stdout}")


def _diff_files_from_patch(patch_text: str) -> List[str]:
    files = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            files.append(line.replace("+++ b/", "").strip())
        elif line.startswith("*** Update File: "):
            files.append(line.replace("*** Update File: ", "").strip())
        elif line.startswith("*** Add File: "):
            files.append(line.replace("*** Add File: ", "").strip())
    return sorted(set(files))


def _validate_config_guardrails(
    baseline: Dict[str, object],
    candidate: Dict[str, object],
    allowed_knobs: Dict[str, List[str]],
    allow_any_knob: bool,
    allow_fixed_env_changes: bool,
    allowed_fixed_env_keys: List[str],
) -> Tuple[bool, str]:
    locked_sections = ["data_days", "min_trades", "risk_limits"]
    if not allow_fixed_env_changes:
        locked_sections.append("fixed_env")
    for key in locked_sections:
        if baseline.get(key) != candidate.get(key):
            return False, f"Locked config section changed: {key}"

    grid = candidate.get("grid", [])
    if not isinstance(grid, list) or not grid:
        return False, "grid is empty or invalid"

    fixed_env_keys: set[str] = set()
    if allow_fixed_env_changes:
        cand_fixed_env = candidate.get("fixed_env", {})
        if isinstance(cand_fixed_env, dict):
            fixed_env_keys = {str(k) for k in cand_fixed_env.keys()}
        base_fixed_env = baseline.get("fixed_env", {})
        if not isinstance(base_fixed_env, dict):
            base_fixed_env = {}
        allowed_fixed_env = {str(k) for k in allowed_fixed_env_keys}
        for key in sorted(set(base_fixed_env.keys()) | set(cand_fixed_env.keys() if isinstance(cand_fixed_env, dict) else [])):
            base_val = base_fixed_env.get(key)
            cand_val = (cand_fixed_env or {}).get(key) if isinstance(cand_fixed_env, dict) else None
            if base_val != cand_val:
                if allowed_fixed_env and str(key) not in allowed_fixed_env:
                    return False, f"Disallowed fixed_env change: {key}"

    for entry in grid:
        env = entry.get("env", {})
        if not isinstance(env, dict):
            return False, "grid env must be dict"
        if allow_any_knob:
            continue
        for k, v in env.items():
            if allow_fixed_env_changes and str(k) in fixed_env_keys:
                # fixed_env keys may be overridden in grid when explicitly enabled
                continue
            if k not in allowed_knobs:
                return False, f"Disallowed knob: {k}"
            allowed_vals = [str(x) for x in allowed_knobs[k]]
            if str(v) not in allowed_vals:
                return False, f"Invalid value for {k}: {v}"

    return True, ""


def _summarize_results(comparison_path: Path) -> str:
    df = pd.read_csv(comparison_path)
    if df.empty:
        return "No runs completed."
    cols = [
        "name",
        "status",
        "trades_train",
        "trades_lock",
        "lock_expectancy_ticks",
        "lock_net_ticks",
        "lock_win_rate",
        "lock_avg_win_ticks",
        "lock_avg_loss_ticks",
        "lock_max_dd_ticks",
        "lock_worst_day_ticks",
        "daily_loss_breaches",
        "trailing_dd_breaches",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].head(5).to_string(index=False)


def _summarize_results_brief(comparison_path: Path) -> str:
    df = pd.read_csv(comparison_path)
    if df.empty:
        return "No runs completed."
    cols = [
        "name",
        "trades_train",
        "trades_lock",
        "lock_expectancy_ticks",
        "lock_pct_mfe_ge_1",
        "lock_pct_mfe_ge_4",
        "lock_max_dd_ticks",
        "lock_worst_day_ticks",
        "status",
    ]
    cols = [c for c in cols if c in df.columns]
    ranked = df.sort_values("lock_expectancy_ticks", ascending=False)
    return ranked[cols].head(4).to_string(index=False)


def _select_runs(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series | None, pd.Series]:
    ranked = df.sort_values("lock_expectancy_ticks", ascending=False)
    best = ranked.iloc[0]
    second = ranked.iloc[1] if len(ranked) > 1 else None
    worst = ranked.iloc[-1]
    return best, second, worst


def _run_dir(iter_dir: Path, run_name: str) -> Path:
    return iter_dir / "runs" / run_name


def _find_first(path: Path, pattern: str) -> Path | None:
    matches = list(path.rglob(pattern))
    return matches[0] if matches else None


def _block_reason_percentages(df: pd.DataFrame) -> Dict[str, float]:
    total = 0.0
    for col in ("gate_entry_candidates_when_flat", "base_entry_candidates_when_flat"):
        if col in df.columns:
            total = float(df[col].sum())
            if total:
                break
    if total <= 0:
        for col in ("entry_confirm_checked_gate", "entry_confirm_checked_base"):
            if col in df.columns:
                total = float(df[col].sum())
                if total:
                    break

    def pct(val: float) -> float:
        return 0.0 if total <= 0 else float(val) / total

    reasons: Dict[str, float] = {}
    if "gate_blocked_not_armed_gate" in df.columns:
        reasons["not_armed"] = pct(df["gate_blocked_not_armed_gate"].sum())
    if "entry_confirm_failed_gate" in df.columns:
        reasons["confirm_failed"] = pct(df["entry_confirm_failed_gate"].sum())
    if "blocked_entry_alpha" in df.columns:
        reasons["blocked_entry_alpha"] = pct(df["blocked_entry_alpha"].sum())
    return reasons


def _tail_stats(trades: pd.DataFrame) -> Dict[str, object]:
    mfe = trades.get("mfe_ticks", pd.Series(dtype=float)).astype(float).to_numpy()
    mae = trades.get("mae_ticks", pd.Series(dtype=float)).astype(float).to_numpy()
    def prob_ge(arr: pd.Series, val: float) -> float:
        if len(arr) == 0:
            return 0.0
        return float((arr >= val).mean())

    mfe_buckets = {
        "<1": float((mfe < 1).mean()) if len(mfe) else 0.0,
        "1-2": float(((mfe >= 1) & (mfe < 2)).mean()) if len(mfe) else 0.0,
        "2-3": float(((mfe >= 2) & (mfe < 3)).mean()) if len(mfe) else 0.0,
        "3-4": float(((mfe >= 3) & (mfe < 4)).mean()) if len(mfe) else 0.0,
        ">=4": float((mfe >= 4).mean()) if len(mfe) else 0.0,
    }
    mae_buckets = {
        "<1": float((mae < 1).mean()) if len(mae) else 0.0,
        "1-2": float(((mae >= 1) & (mae < 2)).mean()) if len(mae) else 0.0,
        "2-3": float(((mae >= 2) & (mae < 3)).mean()) if len(mae) else 0.0,
        ">=3": float((mae >= 3).mean()) if len(mae) else 0.0,
    }
    return {
        "p_mfe_ge_1": prob_ge(mfe, 1),
        "p_mfe_ge_2": prob_ge(mfe, 2),
        "p_mfe_ge_3": prob_ge(mfe, 3),
        "p_mfe_ge_4": prob_ge(mfe, 4),
        "p_mae_ge_1": prob_ge(mae, 1),
        "p_mae_ge_2": prob_ge(mae, 2),
        "p_mae_ge_3": prob_ge(mae, 3),
        "mfe_buckets": mfe_buckets,
        "mae_buckets": mae_buckets,
    }


def _trade_exemplars(trades: pd.DataFrame, limit: int = 20) -> List[Dict[str, object]]:
    if limit <= 0 or trades.empty:
        return []
    cols = [
        "date",
        "entry_time",
        "side",
        "pnl_ticks",
        "mfe_ticks",
        "mae_ticks",
        "pflft_flow_sum_signed",
        "pflft_flow_intensity",
        "pflft_lag_score",
        "pflft_dmid_ticks",
        "pflft_spread_ticks",
        "pflft_confirm_min_abs_ofid",
        "pflft_confirm_min_flow_sum",
        "pflft_proof_ticks",
        "pflft_proof_min_flow",
        "pflft_proof_bars",
    ]
    cols = [c for c in cols if c in trades.columns]
    if not cols:
        return []
    if "pnl_ticks" not in trades.columns:
        return trades.head(limit)[cols].to_dict(orient="records")
    half = max(1, limit // 2)
    winners = trades.sort_values("pnl_ticks", ascending=False).head(half)
    losers = trades.sort_values("pnl_ticks", ascending=True).head(max(0, limit - half))
    sample = pd.concat([winners, losers], ignore_index=True).head(limit)
    return sample[cols].to_dict(orient="records")


def _compact_trade_exemplars(exemplars: List[Dict[str, object]], limit: int = 6) -> List[Dict[str, object]]:
    keep = [
        "date",
        "side",
        "pnl_ticks",
        "mfe_ticks",
        "mae_ticks",
        "pflft_flow_sum_signed",
        "pflft_flow_intensity",
        "pflft_lag_score",
        "pflft_dmid_ticks",
        "pflft_spread_ticks",
        "pflft_proof_bars",
    ]
    out: List[Dict[str, object]] = []
    for row in exemplars[:limit]:
        if not isinstance(row, dict):
            continue
        out.append({k: row.get(k) for k in keep if k in row})
    return out


def _idea_prompt_digest(
    digest: Dict[str, object],
    exemplar_limit: int = 6,
    include_fixed_env_values: bool = False,
) -> Dict[str, object]:
    summary = digest.get("summary", {})
    constraints = dict(digest.get("constraints", {})) if isinstance(digest.get("constraints", {}), dict) else {}
    fixed_env = constraints.get("fixed_env", {})
    if isinstance(fixed_env, dict):
        if include_fixed_env_values:
            constraints["fixed_env"] = fixed_env
            constraints["fixed_env_keys"] = sorted(fixed_env.keys())
        else:
            constraints["fixed_env_keys"] = sorted(fixed_env.keys())
            constraints.pop("fixed_env", None)

    prompt_digest: Dict[str, object] = {
        "summary": summary,
        "constraints": constraints,
        "block_reasons": digest.get("block_reasons", {}),
        "tails": digest.get("tails", {}),
        "mfe_thresholds_root": digest.get("mfe_thresholds_root", {}),
        "failure_modes": digest.get("failure_modes", {}),
    }
    exemplars = digest.get("trade_exemplars", [])
    if isinstance(exemplars, list) and exemplar_limit > 0:
        prompt_digest["trade_exemplars"] = _compact_trade_exemplars(exemplars, limit=exemplar_limit)
    return prompt_digest


def _build_idea_digest(
    cfg: OrchestratorConfig,
    iter_dir: Path,
    comparison_path: Path,
) -> Dict[str, object]:
    df = pd.read_csv(comparison_path)
    if df.empty:
        return {"error": "comparison.csv empty"}

    cfg_data = _load_json(cfg.experiment_config)
    best, second, worst = _select_runs(df)
    comparison_root = comparison_path.parent

    def row_dict(row: pd.Series) -> Dict[str, object]:
        data: Dict[str, object] = {"name": row.get("name"), "status": row.get("status")}
        for col in df.columns:
            if col.endswith("_ticks") or col.endswith("_rate") or col.endswith("_trades") or col in {
                "trades_train",
                "trades_lock",
            }:
                data[col] = row.get(col)
        return data

    def run_dir_for(run_name: str) -> Path:
        run_name = str(run_name)
        candidates = [
            _run_dir(iter_dir, run_name),
            comparison_root / "runs" / run_name,
            iter_dir / "_screening" / "runs" / run_name,
            comparison_root / "_screening" / "runs" / run_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def load_diag(run_name: str) -> Dict[str, float]:
        run_dir = run_dir_for(str(run_name))
        diag_path = _find_first(run_dir, "strategy_diagnostics.csv")
        if not diag_path:
            return {}
        diag = pd.read_csv(diag_path)
        return _block_reason_percentages(diag)

    def load_trades(run_name: str) -> pd.DataFrame:
        run_dir = run_dir_for(str(run_name))
        trades_path = _find_first(run_dir, "trades_entry_alpha_v1_W*_gate_*_baseline_vs_gated.csv")
        if not trades_path:
            return pd.DataFrame()
        trades = pd.read_csv(trades_path)
        if "strategy" in trades.columns:
            trades = trades[trades["strategy"] == "gated"].copy()
        return trades

    def load_mfe_root(run_name: str) -> Dict[str, float]:
        run_dir = run_dir_for(str(run_name))
        mfe_path = _find_first(run_dir, "mfe_thresholds_root_entry_alpha_v1_W*_gate_*.csv")
        if not mfe_path:
            return {}
        mfe = pd.read_csv(mfe_path)
        if "strategy" in mfe.columns:
            mfe = mfe[mfe["strategy"] == "gated"].copy()
        if mfe.empty:
            return {}
        weights = mfe.get("count_trades", pd.Series(dtype=float)).astype(float)
        if weights.sum() <= 0:
            weights = None
        out: Dict[str, float] = {}
        for col in ("pct_mfe_ge_1", "pct_mfe_ge_2", "pct_mfe_ge_3"):
            if col in mfe.columns:
                vals = mfe[col].astype(float)
                if weights is None:
                    out[col] = float(vals.mean())
                else:
                    out[col] = float((vals * weights).sum() / weights.sum())
        return out

    best_trades = load_trades(best.get("name"))
    best_mfe_root = load_mfe_root(best.get("name"))
    digest = {
        "summary": {
            "best": row_dict(best),
            "second_best": row_dict(second) if second is not None else None,
            "worst": row_dict(worst),
        },
        "constraints": {
            "data_days": cfg_data.get("data_days", {}),
            "min_trades": cfg_data.get("min_trades", {}),
            "risk_limits": cfg_data.get("risk_limits", {}),
            "fixed_env": cfg_data.get("fixed_env", {}),
            "allow_any_knob": cfg.allow_any_knob,
        },
        "block_reasons": {
            "best": load_diag(best.get("name")),
            "second_best": load_diag(second.get("name")) if second is not None else {},
        },
        "tails": _tail_stats(best_trades),
        "mfe_thresholds_root": best_mfe_root,
        "failure_modes": {
            "worst_day_ticks": best.get("lock_worst_day_ticks"),
            "max_dd_ticks": best.get("lock_max_dd_ticks"),
        },
        "trade_exemplars": _trade_exemplars(best_trades, limit=20),
    }
    return digest


def _build_idea_prompt(
    cfg: OrchestratorConfig,
    digest: Dict[str, object],
    summary: str,
    idea_mode_override: str | None = None,
) -> str:
    idea_mode = (idea_mode_override or cfg.idea_mode).lower()
    allowed_note = (
        (
            "Any env key is allowed, including fixed_env keys."
            if cfg.allow_fixed_env_changes
            else "Any env key is allowed except those in fixed_env."
        )
        if cfg.allow_any_knob
        else (
            f"Allowed knobs: {json.dumps(cfg.allowed_knobs)}"
            + (
                " plus fixed_env keys may also be used."
                if cfg.allow_fixed_env_changes
                else ""
            )
        )
    )
    if cfg.allow_fixed_env_changes and cfg.allowed_fixed_env_keys:
        allowed_note += f" Fixed_env changes are restricted to: {json.dumps(cfg.allowed_fixed_env_keys)}."
    mode_note = (
        "Mode: GRID_ONLY. Proposal must only adjust grid env knobs."
        if idea_mode == "grid"
        else "Mode: STRUCTURAL. Proposal must include code patch(es) in changes."
    )

    if idea_mode == "grid":
        schema_block = (
            "{\n"
            '  "rationale": "...",\n'
            '  "expected_metrics": ["lock_expectancy_ticks", "trade_count"],\n'
            '  "research": {\n'
            '    "used_web_tools": true,\n'
            '    "queries": ["..."],\n'
            '    "sources": [{"url": "https://...", "why": "..."}],\n'
            '    "notes": "why tools were or were not used"\n'
            "  },\n"
            '  "run_id": "optional_short_name",\n'
            '  "grid": [\n'
            '    {"name": "A_variant", "env": {"ENTRY_MIN_PROGRESS_TICKS": "0", "ENTRY_COOLDOWN_BARS": "10"}}\n'
            "  ]\n"
            "}\n"
        )
    else:
        schema_block = (
            "{\n"
            '  "rationale": "...",\n'
            '  "expected_metrics": ["lock_expectancy_ticks", "trade_count"],\n'
            '  "research": {\n'
            '    "used_web_tools": true,\n'
            '    "queries": ["..."],\n'
            '    "sources": [{"url": "https://...", "why": "..."}],\n'
            '    "notes": "why tools were or were not used"\n'
            "  },\n"
            '  "changes": [\n'
            '    {"file": "src/entry_alpha.py", "patch": "...unified diff..."},\n'
            '    {"file": "configs/experiment_agent.json", "patch": "...unified diff..."}\n'
            "  ],\n"
            '  "grid": [\n'
            '    {"name": "A_variant", "env": {"ENTRY_MIN_PROGRESS_TICKS": "0"}}\n'
            "  ]\n"
            "}\n"
        )

    # Windows openclaw.cmd goes through cmd.exe, so keep the prompt comfortably small.
    prompt_digest = _idea_prompt_digest(
        digest,
        exemplar_limit=6,
        include_fixed_env_values=cfg.allow_fixed_env_changes,
    )
    digest_json = json.dumps(_json_safe(prompt_digest), separators=(",", ":"))
    if len(digest_json) > 4500:
        prompt_digest = _idea_prompt_digest(
            digest,
            exemplar_limit=2,
            include_fixed_env_values=cfg.allow_fixed_env_changes,
        )
        digest_json = json.dumps(_json_safe(prompt_digest), separators=(",", ":"))
    if len(digest_json) > 3500:
        prompt_digest = _idea_prompt_digest(
            digest,
            exemplar_limit=0,
            include_fixed_env_values=cfg.allow_fixed_env_changes,
        )
        digest_json = json.dumps(_json_safe(prompt_digest), separators=(",", ":"))

    fixed_env_rule = (
        "- You may change fixed_env values (including cost/exit assumptions) if justified by the digest.\n"
        "- You may also include an optional top-level \"fixed_env\" object in the JSON proposal.\n"
        if cfg.allow_fixed_env_changes
        else "- Do not modify cost model, exits, or fixed_env values.\n"
    )
    if cfg.allow_fixed_env_changes and cfg.allowed_fixed_env_keys:
        fixed_env_rule += (
            f"- Only these fixed_env keys may change: {json.dumps(cfg.allowed_fixed_env_keys)}.\n"
        )
    research_domain_note = ""
    if cfg.research_preferred_domains:
        research_domain_note += f"- Prefer high-quality sources from: {json.dumps(cfg.research_preferred_domains)}.\n"
    if cfg.research_blocked_domains:
        research_domain_note += f"- Avoid low-quality/blocked domains: {json.dumps(cfg.research_blocked_domains)}.\n"
    if cfg.research_allowed_domains:
        research_domain_note += f"- Source URLs must be from allowlisted domains: {json.dumps(cfg.research_allowed_domains)}.\n"
    strict_json_rule = (
        "- Strict JSON-only output is enforced by the orchestrator; extra prose will be rejected and retried.\n"
        if cfg.ideas_require_strict_json
        else ""
    )
    tool_budget_rule = (
        f"- Perform deep targeted research: use at least {cfg.research_min_web_search_count} web.search call(s) "
        f"and {cfg.research_min_web_fetch_count} web.fetch call(s). "
        "Use multiple distinct queries/sources when useful.\n"
    )
    source_count_rule = (
        f"- If used_web_tools=true, include {max(1, cfg.research_min_sources)}-"
        f"{max(cfg.research_min_sources, cfg.research_max_sources)} source URLs with short why notes.\n"
    )
    tool_min_rule = (
        f"- Research enforcement: at least {cfg.research_min_web_search_count} web.search call(s) and "
        f"{cfg.research_min_web_fetch_count} web.fetch call(s) are required.\n"
    )

    idea_prompt = (
        "You are the ideas agent. Propose the next experiment configuration.\n"
        "Return ONLY a single JSON object. No prose. No markdown.\n"
        "Schema:\n"
        f"{schema_block}"
        "Hard rules:\n"
        "- Output JSON only (no code fences, no commentary).\n"
        f"{strict_json_rule}"
        "- Do not ask for more inputs, clarifications, or confirmations.\n"
        "- If anything is ambiguous, infer reasonable defaults from the digest and still return a complete proposal JSON.\n"
        "- Prefer using web tools when they can improve the next hypothesis (especially new family ideas, structural changes, or ambiguous failure modes).\n"
        f"{tool_budget_rule}"
        "- Always include a research object with used_web_tools, queries, sources, and notes.\n"
        f"{source_count_rule}"
        f"{tool_min_rule}"
        "- If used_web_tools=false, set queries=[] and sources=[] and explain why in research.notes.\n"
        "- Proposals missing research.used_web_tools=true and at least one source URL will be rejected and retried.\n"
        f"{research_domain_note}"
        "- Do not modify data range.\n"
        f"{fixed_env_rule}"
        f"- {allowed_note}\n"
        f"- {mode_note}\n"
        "Digest JSON (compact):\n"
        f"{digest_json}\n"
        "Recent results (brief):\n"
        f"{summary}\n"
    )

    if len(idea_prompt) > 7000:
        # Last-resort reduction for Windows cmd.exe argument limits.
        prompt_digest = {
            "summary": digest.get("summary", {}),
            "tails": digest.get("tails", {}),
            "failure_modes": digest.get("failure_modes", {}),
            "block_reasons": digest.get("block_reasons", {}),
        }
        digest_json = json.dumps(_json_safe(prompt_digest), separators=(",", ":"))
        idea_prompt = (
            "You are the ideas agent. Propose the next experiment configuration.\n"
            "Return ONLY a single JSON object. No prose. No markdown.\n"
            "Schema:\n"
            f"{schema_block}"
            "Hard rules:\n"
            "- Output JSON only (no code fences, no commentary).\n"
            f"{strict_json_rule}"
            "- Prefer using web tools when they can improve the next hypothesis (especially new family ideas, structural changes, or ambiguous failure modes).\n"
            f"{tool_budget_rule}"
            "- Always include a research object with used_web_tools, queries, sources, and notes.\n"
            f"{source_count_rule}"
            f"{tool_min_rule}"
            "- If used_web_tools=false, set queries=[] and sources=[] and explain why in research.notes.\n"
            "- Proposals missing research.used_web_tools=true and at least one source URL will be rejected and retried.\n"
            f"{research_domain_note}"
            "- Do not modify data range.\n"
            f"{fixed_env_rule}"
            f"- {allowed_note}\n"
            f"- {mode_note}\n"
            "Digest JSON (ultra-compact):\n"
            f"{digest_json}\n"
            "Recent results (brief):\n"
            f"{summary}\n"
        )
    return idea_prompt


def _run_experiment_agent_once(experiment_config: Path, outdir: Path, log_name: str = "orchestrator_stdout.log") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(EXPERIMENT_AGENT),
        "--config",
        str(experiment_config),
        "--outdir",
        str(outdir),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path = outdir / log_name
    log_path.write_text(proc.stdout, encoding="utf-8")
    comparison = outdir / "comparison.csv"
    if not comparison.exists():
        raise RuntimeError(f"comparison.csv not found after run. See {log_path}")
    return comparison


def _run_experiment_agent(
    experiment_config: Path,
    outdir: Path,
    screening: ScreeningConfig | None = None,
) -> Tuple[Path, Dict[str, object]]:
    if screening is None or not screening.enabled:
        return _run_experiment_agent_once(experiment_config, outdir), {"screening_used": False}

    screen_dir = outdir / "_screening"
    screen_cfg_path, screen_meta = _build_screening_config(experiment_config, screen_dir, screening)
    screen_comparison = _run_experiment_agent_once(screen_cfg_path, screen_dir)
    screen_positive = _screening_positive_summary(
        screen_comparison,
        screening.positive_lock_ev_threshold,
    )
    trigger_full = (not screening.full_run_on_positive_only) or bool(screen_positive.get("has_positive", False))

    if not trigger_full:
        outdir.mkdir(parents=True, exist_ok=True)
        comparison = outdir / "comparison.csv"
        shutil.copy2(screen_comparison, comparison)
        best_quick = float(screen_positive.get("best_lock_expectancy", float("-inf")) or float("-inf"))
        threshold = float(screen_positive.get("threshold", screening.positive_lock_ev_threshold))
        print(
            f"[screening] Quick window best lock EV={best_quick:.4f} <= threshold {threshold:.4f}; "
            "skipping full 15-day backtest this iteration.",
            flush=True,
        )
        run_meta = {
            "screening_used": True,
            "screening_dir": str(screen_dir),
            "screening_config": str(screen_cfg_path),
            "screening_comparison": str(screen_comparison),
            "screening_settings": {
                "train_days": screening.train_days,
                "lock_days": screening.lock_days,
                "top_k": screening.top_k,
                "min_valid_runs": screening.min_valid_runs,
                "full_run_on_positive_only": screening.full_run_on_positive_only,
                "positive_lock_ev_threshold": screening.positive_lock_ev_threshold,
            },
            "screening_truncated_data_days": screen_meta.get("data_days", {}),
            "screening_positive_summary": screen_positive,
            "full_run_triggered": False,
            "full_run_reason": "screen_not_positive",
            "comparison_source": "screening",
            "full_grid_size_before": None,
            "full_grid_size_after": None,
        }
        _write_json(screen_dir / "screening_summary.json", run_meta)
        return comparison, run_meta

    base_cfg = _load_json(experiment_config)
    original_grid = base_cfg.get("grid", [])
    if not isinstance(original_grid, list):
        original_grid = []
    selected_grid, select_meta = _top_grid_from_comparison(
        screen_comparison,
        original_grid,
        screening.top_k,
        screening.min_valid_runs,
    )

    full_cfg = base_cfg
    if selected_grid and len(selected_grid) < len(original_grid):
        full_cfg = json.loads(json.dumps(base_cfg))
        full_cfg["grid"] = selected_grid
    selected_cfg_path = outdir / "_screening_selected_config.json"
    _write_json(selected_cfg_path, full_cfg)

    best_quick = float(screen_positive.get("best_lock_expectancy", float("-inf")) or float("-inf"))
    threshold = float(screen_positive.get("threshold", screening.positive_lock_ev_threshold))
    print(
        f"[screening] Quick window found positive lock EV (best={best_quick:.4f} > {threshold:.4f}); "
        "running full 15-day backtest.",
        flush=True,
    )
    comparison = _run_experiment_agent_once(selected_cfg_path, outdir)
    run_meta = {
        "screening_used": True,
        "screening_dir": str(screen_dir),
        "screening_config": str(screen_cfg_path),
        "screening_comparison": str(screen_comparison),
        "screening_settings": {
            "train_days": screening.train_days,
            "lock_days": screening.lock_days,
            "top_k": screening.top_k,
            "min_valid_runs": screening.min_valid_runs,
            "full_run_on_positive_only": screening.full_run_on_positive_only,
            "positive_lock_ev_threshold": screening.positive_lock_ev_threshold,
        },
        "screening_truncated_data_days": screen_meta.get("data_days", {}),
        "screening_positive_summary": screen_positive,
        "screening_selection": select_meta,
        "full_run_triggered": True,
        "full_run_reason": "screen_positive",
        "comparison_source": "full",
        "screening_selected_config": str(selected_cfg_path),
        "full_grid_size_before": len(original_grid),
        "full_grid_size_after": len(full_cfg.get("grid", [])) if isinstance(full_cfg.get("grid", []), list) else None,
    }
    _write_json(screen_dir / "screening_summary.json", run_meta)
    return comparison, run_meta


def _best_lock_expectancy(comparison_path: Path) -> float:
    df = pd.read_csv(comparison_path)
    if df.empty or "lock_expectancy_ticks" not in df.columns:
        return float("-inf")
    return float(df["lock_expectancy_ticks"].max())


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _best_row_metrics(comparison_path: Path, min_promote_trades_lock: int) -> Dict[str, object]:
    df = pd.read_csv(comparison_path)
    if df.empty:
        return {
            "name": None,
            "status": "EMPTY",
            "lock_expectancy_ticks": float("-inf"),
            "trades_lock": 0,
            "lock_max_dd_ticks": 0.0,
            "lock_worst_day_ticks": 0.0,
            "daily_loss_breaches": 0,
            "trailing_dd_breaches": 0,
            "breach_count": 0,
            "trade_shortfall": max(0, min_promote_trades_lock),
            "worst_day_loss_ticks": 0.0,
        }
    ranked = df.copy()
    if "status" in ranked.columns:
        ranked["__ok"] = (ranked["status"].astype(str) == "OK").astype(int)
    else:
        ranked["__ok"] = 0
    if "lock_expectancy_ticks" not in ranked.columns:
        ranked["lock_expectancy_ticks"] = float("-inf")
    ranked["lock_expectancy_ticks"] = pd.to_numeric(ranked["lock_expectancy_ticks"], errors="coerce").fillna(float("-inf"))
    ranked = ranked.sort_values(["__ok", "lock_expectancy_ticks"], ascending=[False, False])
    row = ranked.iloc[0]

    lock_ev = _as_float(row.get("lock_expectancy_ticks"), float("-inf"))
    trades_lock = _as_int(row.get("trades_lock"), 0)
    max_dd = max(0.0, _as_float(row.get("lock_max_dd_ticks"), 0.0))
    worst_day = _as_float(row.get("lock_worst_day_ticks"), 0.0)
    worst_day_loss = max(0.0, -worst_day)
    daily_b = max(0, _as_int(row.get("daily_loss_breaches"), 0))
    trailing_b = max(0, _as_int(row.get("trailing_dd_breaches"), 0))
    breach_count = daily_b + trailing_b
    trade_shortfall = max(0, int(min_promote_trades_lock) - trades_lock)

    return {
        "name": row.get("name"),
        "status": row.get("status"),
        "lock_expectancy_ticks": lock_ev,
        "trades_lock": trades_lock,
        "lock_max_dd_ticks": max_dd,
        "lock_worst_day_ticks": worst_day,
        "daily_loss_breaches": daily_b,
        "trailing_dd_breaches": trailing_b,
        "breach_count": breach_count,
        "trade_shortfall": trade_shortfall,
        "worst_day_loss_ticks": worst_day_loss,
    }


def _composite_score(metrics: Dict[str, object], cfg: OrchestratorConfig) -> float:
    ev = _as_float(metrics.get("lock_expectancy_ticks"), float("-inf"))
    if not math.isfinite(ev):
        return float("-inf")
    max_dd = max(0.0, _as_float(metrics.get("lock_max_dd_ticks"), 0.0))
    worst_day_loss = max(0.0, _as_float(metrics.get("worst_day_loss_ticks"), 0.0))
    breaches = max(0, _as_int(metrics.get("breach_count"), 0))
    shortfall = max(0, _as_int(metrics.get("trade_shortfall"), 0))
    return (
        ev
        - cfg.score_weight_max_dd * max_dd
        - cfg.score_weight_worst_day * worst_day_loss
        - cfg.score_weight_breach * float(breaches)
        - cfg.score_weight_trade_shortfall * float(shortfall)
    )


def _family_id(experiment_config: Path) -> str:
    cfg = _load_json(experiment_config)
    families = cfg.get("family_allowlist", [])
    if not families:
        return "UNKNOWN"
    return ",".join(sorted(map(str, families)))


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Orchestrator config JSON.")
    parser.add_argument("--outdir", required=True, help="Output directory for this run.")
    parser.add_argument(
        "--background",
        action="store_true",
        help="Spawn a detached orchestrator process and return immediately.",
    )
    return parser.parse_args(list(argv))


def _load_orchestrator_config(path: Path, outdir: Path) -> OrchestratorConfig:
    raw = _load_json(path)
    screening_raw = raw.get("screening", {})
    screening: ScreeningConfig | None = None
    if isinstance(screening_raw, dict) and bool(screening_raw.get("enabled", False)):
        screening = ScreeningConfig(
            enabled=True,
            train_days=max(1, int(screening_raw.get("train_days", 5))),
            lock_days=max(1, int(screening_raw.get("lock_days", 2))),
            top_k=max(1, int(screening_raw.get("top_k", 2))),
            min_valid_runs=max(1, int(screening_raw.get("min_valid_runs", 1))),
            full_run_on_positive_only=bool(screening_raw.get("full_run_on_positive_only", True)),
            positive_lock_ev_threshold=float(screening_raw.get("positive_lock_ev_threshold", 0.0)),
        )

    return OrchestratorConfig(
        experiment_config=Path(raw["experiment_config"]).expanduser().resolve(),
        outdir=outdir,
        ideas_agent=str(raw.get("ideas_agent", "ideas")),
        coding_agent=str(raw.get("coding_agent", "coding")),
        openclaw_timeout_seconds=int(raw.get("openclaw_timeout_seconds", 600)),
        idea_mode=str(raw.get("idea_mode", "grid")).lower(),
        max_iterations=int(raw.get("max_iterations", 5)),
        max_no_improve=int(raw.get("max_no_improve", 2)),
        pivot_min_lock_ev=float(raw.get("pivot_min_lock_ev", -0.25)),
        pivot_min_lock_ev_iterations=int(raw.get("pivot_min_lock_ev_iterations", 30)),
        pivot_family_fail_limit=int(raw.get("pivot_family_fail_limit", 3)),
        max_structural_variants_per_family=int(raw.get("max_structural_variants_per_family", 3)),
        pivot_min_right_tail_mfe3=float(raw.get("pivot_min_right_tail_mfe3", 0.01)),
        pivot_min_right_tail_mfe4=float(raw.get("pivot_min_right_tail_mfe4", 0.005)),
        scoring_enabled=bool(raw.get("scoring_enabled", True)),
        score_weight_max_dd=float(raw.get("score_weight_max_dd", 0.01)),
        score_weight_worst_day=float(raw.get("score_weight_worst_day", 0.01)),
        score_weight_breach=float(raw.get("score_weight_breach", 0.50)),
        score_weight_trade_shortfall=float(raw.get("score_weight_trade_shortfall", 0.02)),
        min_promote_score_delta=float(raw.get("min_promote_score_delta", 0.05)),
        min_promote_lock_ev_delta=float(raw.get("min_promote_lock_ev_delta", 0.05)),
        min_promote_trades_lock=max(0, int(raw.get("min_promote_trades_lock", 0))),
        revert_on_reject=bool(raw.get("revert_on_reject", True)),
        allowed_files=[str(x) for x in raw.get("allowed_files", [])],
        allowed_knobs={str(k): [str(v) for v in vals] for k, vals in raw.get("allowed_knobs", {}).items()},
        allow_any_knob=bool(raw.get("allow_any_knob", False)),
        allow_fixed_env_changes=bool(raw.get("allow_fixed_env_changes", False)),
        allowed_fixed_env_keys=[str(x) for x in raw.get("allowed_fixed_env_keys", [])],
        ideas_require_strict_json=bool(raw.get("ideas_require_strict_json", True)),
        ideas_allow_embedded_json=bool(raw.get("ideas_allow_embedded_json", True)),
        research_require_web_tools=bool(raw.get("research_require_web_tools", True)),
        research_require_source_url=bool(raw.get("research_require_source_url", True)),
        research_require_tool_evidence=bool(raw.get("research_require_tool_evidence", True)),
        research_min_web_search_count=max(0, int(raw.get("research_min_web_search_count", 1))),
        research_min_web_fetch_count=max(0, int(raw.get("research_min_web_fetch_count", 1))),
        research_min_sources=max(0, int(raw.get("research_min_sources", 1))),
        research_max_sources=max(1, int(raw.get("research_max_sources", 3))),
        research_allowed_domains=[str(x).lower() for x in raw.get("research_allowed_domains", [])],
        research_blocked_domains=[str(x).lower() for x in raw.get("research_blocked_domains", [])],
        research_preferred_domains=[str(x).lower() for x in raw.get("research_preferred_domains", [])],
        auto_mode_start=str(raw.get("auto_mode_start", "grid")).lower(),
        auto_grid_to_structural_no_improve=max(1, int(raw.get("auto_grid_to_structural_no_improve", 2))),
        auto_grid_to_structural_best_lock_below=float(raw.get("auto_grid_to_structural_best_lock_below", -0.25)),
        auto_grid_to_structural_require_tail_failure=bool(
            raw.get("auto_grid_to_structural_require_tail_failure", True)
        ),
        auto_structural_revert_on_improve=bool(raw.get("auto_structural_revert_on_improve", True)),
        screening=screening,
        max_proposal_attempts=int(raw.get("max_proposal_attempts", 2)),
        max_patch_attempts=int(raw.get("max_patch_attempts", 2)),
        rotation_request_limit=int(raw.get("rotation_request_limit", 2)),
    )


def run_orchestrator(config_path: Path, outdir: Path) -> None:
    cfg = _load_orchestrator_config(config_path, outdir)
    baseline_config = _load_json(cfg.experiment_config)
    baseline_min_trades_lock = 0
    if isinstance(baseline_config.get("min_trades", {}), dict):
        baseline_min_trades_lock = max(0, _as_int(baseline_config.get("min_trades", {}).get("lock"), 0))
    min_promote_trades_lock = cfg.min_promote_trades_lock if cfg.min_promote_trades_lock > 0 else baseline_min_trades_lock
    lock_path = _acquire_lock(outdir)
    history: List[Dict[str, object]] = []

    best_expectancy = float("-inf")
    best_score = float("-inf")
    no_improve = 0
    stop_reason = ""
    family_best: Dict[str, float] = {}
    family_order: List[str] = []
    consecutive_failed_families = 0
    last_family = _family_id(cfg.experiment_config)
    structural_variants: Dict[str, int] = {}
    family_modes: Dict[str, str] = {}
    best_config_text = cfg.experiment_config.read_text(encoding="utf-8")

    try:
        for iteration in range(1, cfg.max_iterations + 1):
            rotation_calls = 0
            current_family = _family_id(cfg.experiment_config)
            if current_family != last_family:
                prev_best = family_best.get(last_family, float("-inf"))
                if prev_best <= 0.0:
                    consecutive_failed_families += 1
                else:
                    consecutive_failed_families = 0
                last_family = current_family

            iter_dir = outdir / f"iter_{iteration:02d}"
            comparison_path = iter_dir / "comparison.csv"
            run_meta: Dict[str, object] = {"screening_used": False, "comparison_reused": comparison_path.exists()}
            if not comparison_path.exists():
                comparison_path, run_meta = _run_experiment_agent(cfg.experiment_config, iter_dir, screening=cfg.screening)
            digest = _json_safe(_build_idea_digest(cfg, iter_dir, comparison_path))
            (iter_dir / "idea_digest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
            summary = _summarize_results(comparison_path)
            summary_brief = _summarize_results_brief(comparison_path)
            current_best = _best_lock_expectancy(comparison_path)
            candidate_metrics = _best_row_metrics(comparison_path, min_promote_trades_lock)
            candidate_score = _composite_score(candidate_metrics, cfg)
            candidate_ev = _as_float(candidate_metrics.get("lock_expectancy_ticks"), float("-inf"))

            if not family_order or family_order[-1] != current_family:
                family_order.append(current_family)

            history.append(
                {
                    "iteration": iteration,
                    "family": current_family,
                    "comparison_path": str(comparison_path),
                    "best_lock_expectancy": current_best,
                    "screening": _json_safe(run_meta),
                }
            )
            iter_history = history[-1]
            iter_history["candidate_metrics"] = _json_safe(candidate_metrics)
            iter_history["composite_score"] = candidate_score

            promote_reason = "ev_only"
            if cfg.scoring_enabled:
                if not math.isfinite(best_score):
                    improved_this_iter = math.isfinite(candidate_ev)
                    promote_reason = "initial_baseline" if improved_this_iter else "initial_baseline_invalid"
                    iter_history["promote_checks"] = {
                        "score_delta": None,
                        "ev_delta": None,
                        "meets_score_delta": improved_this_iter,
                        "meets_ev_delta": improved_this_iter,
                        "meets_trades_floor": _as_int(candidate_metrics.get("trades_lock"), 0) >= min_promote_trades_lock,
                        "min_promote_score_delta": cfg.min_promote_score_delta,
                        "min_promote_lock_ev_delta": cfg.min_promote_lock_ev_delta,
                        "min_promote_trades_lock": min_promote_trades_lock,
                    }
                else:
                    score_delta = candidate_score - best_score
                    ev_delta = candidate_ev - best_expectancy
                    meets_score_delta = score_delta >= cfg.min_promote_score_delta
                    meets_ev_delta = ev_delta >= cfg.min_promote_lock_ev_delta
                    meets_trades_floor = _as_int(candidate_metrics.get("trades_lock"), 0) >= min_promote_trades_lock
                    improved_this_iter = meets_score_delta and meets_ev_delta and meets_trades_floor
                    promote_reason = "promoted_by_score_gate" if improved_this_iter else "rejected_by_score_gate"
                    iter_history["promote_checks"] = {
                        "score_delta": score_delta,
                        "ev_delta": ev_delta,
                        "meets_score_delta": meets_score_delta,
                        "meets_ev_delta": meets_ev_delta,
                        "meets_trades_floor": meets_trades_floor,
                        "min_promote_score_delta": cfg.min_promote_score_delta,
                        "min_promote_lock_ev_delta": cfg.min_promote_lock_ev_delta,
                        "min_promote_trades_lock": min_promote_trades_lock,
                    }
            else:
                improved_this_iter = current_best > best_expectancy

            if improved_this_iter:
                best_expectancy = candidate_ev
                best_score = candidate_score
                no_improve = 0
                family_best[current_family] = max(candidate_ev, family_best.get(current_family, float("-inf")))
                if cfg.revert_on_reject:
                    best_config_text = cfg.experiment_config.read_text(encoding="utf-8")
            else:
                no_improve += 1
                if cfg.revert_on_reject and best_config_text:
                    cfg.experiment_config.write_text(best_config_text, encoding="utf-8")
                    iter_history["reverted_to_best_config"] = True
            iter_history["promoted"] = improved_this_iter
            iter_history["promote_reason"] = promote_reason
            iter_history["improved"] = improved_this_iter
            iter_history["no_improve_streak"] = no_improve
            active_idea_mode = _effective_idea_mode(cfg, family_modes, current_family)
            iter_history["idea_mode"] = active_idea_mode

            if (
                iteration >= cfg.pivot_min_lock_ev_iterations
                and best_expectancy < cfg.pivot_min_lock_ev
            ):
                stop_reason = "pivot_min_lock_ev"
                break

            if active_idea_mode == "grid":
                tails = digest.get("tails", {})
                p_mfe_ge_3 = float(tails.get("p_mfe_ge_3", 0.0))
                p_mfe_ge_4 = float(tails.get("p_mfe_ge_4", 0.0))
                if cfg.idea_mode == "auto":
                    should_switch, switch_reason = _should_switch_grid_to_structural(
                        cfg,
                        best_expectancy,
                        no_improve,
                        tails if isinstance(tails, dict) else None,
                    )
                    if should_switch:
                        family_modes[current_family] = "structural"
                        active_idea_mode = "structural"
                        iter_history["idea_mode"] = active_idea_mode
                        iter_history["mode_switch_reason"] = switch_reason
                elif best_expectancy < cfg.pivot_min_lock_ev and (
                    p_mfe_ge_3 <= cfg.pivot_min_right_tail_mfe3
                    and p_mfe_ge_4 <= cfg.pivot_min_right_tail_mfe4
                ):
                    stop_reason = "grid_no_tail"
                    break

            if consecutive_failed_families >= cfg.pivot_family_fail_limit:
                stop_reason = "pivot_family_fail_limit"
                break

            if no_improve >= cfg.max_no_improve:
                if cfg.idea_mode == "auto" and active_idea_mode == "grid":
                    should_switch, switch_reason = _should_switch_grid_to_structural(
                        cfg,
                        best_expectancy,
                        no_improve,
                        digest.get("tails", {}) if isinstance(digest.get("tails", {}), dict) else None,
                    )
                    if should_switch:
                        family_modes[current_family] = "structural"
                        active_idea_mode = "structural"
                        iter_history["idea_mode"] = active_idea_mode
                        iter_history["mode_switch_reason"] = switch_reason or "grid_to_structural_no_improve"
                    else:
                        stop_reason = "no_improve"
                        break
                else:
                    stop_reason = "no_improve"
                    break

            idea_prompt = _build_idea_prompt(cfg, digest, summary_brief, idea_mode_override=active_idea_mode)
            (iter_dir / "ideas_prompt.txt").write_text(idea_prompt, encoding="utf-8")

            proposal = None
            proposal_text = ""
            proposal_reject_reason = ""
            proposal_tool_evidence: Dict[str, object] | None = None
            idea_attempts: List[Dict[str, object]] = []
            for attempt_idx in range(1, cfg.max_proposal_attempts + 1):
                if rotation_calls >= cfg.rotation_request_limit:
                    stop_reason = "rotation_request_limit"
                    break
                call_snapshot = _snapshot_agent_sessions(cfg.ideas_agent)
                call_started = time.time()
                idea_session_id = f"orchestrator-ideas-{outdir.name}-i{iteration:02d}-a{attempt_idx:02d}"
                proposal_text = _run_openclaw_agent(
                    cfg.ideas_agent,
                    idea_prompt,
                    timeout=cfg.openclaw_timeout_seconds,
                    json_only=False,
                    session_id=idea_session_id,
                )
                call_ended = time.time()
                proposal_tool_evidence = _collect_openclaw_tool_evidence(
                    cfg.ideas_agent,
                    call_started,
                    call_ended,
                    before_snapshot=call_snapshot,
                )
                proposal_text = _strip_openclaw_preamble(proposal_text)
                rotation_calls += 1
                response_path = iter_dir / f"ideas_response_attempt_{attempt_idx:02d}.txt"
                response_path.write_text(proposal_text, encoding="utf-8")
                (iter_dir / "ideas_response.txt").write_text(proposal_text, encoding="utf-8")
                evidence_path = iter_dir / f"ideas_tool_evidence_attempt_{attempt_idx:02d}.json"
                _write_json(evidence_path, proposal_tool_evidence)
                attempt_info: Dict[str, object] = {
                    "attempt": attempt_idx,
                    "session_id": idea_session_id,
                    "response_path": str(response_path),
                    "tool_evidence_path": str(evidence_path),
                    "tool_evidence_summary": {
                        "web_tools_used": proposal_tool_evidence.get("web_tools_used", False)
                        if isinstance(proposal_tool_evidence, dict)
                        else None,
                        "web_search_count": proposal_tool_evidence.get("web_search_count", 0)
                        if isinstance(proposal_tool_evidence, dict)
                        else None,
                        "web_fetch_count": proposal_tool_evidence.get("web_fetch_count", 0)
                        if isinstance(proposal_tool_evidence, dict)
                        else None,
                    },
                }
                if "The command line is too long" in proposal_text:
                    stop_reason = "ideas_cmdline_too_long"
                    attempt_info["reject_reason"] = "ideas_cmdline_too_long"
                    idea_attempts.append(_json_safe(attempt_info))  # type: ignore[arg-type]
                    break
                if _looks_like_input_request(proposal_text):
                    proposal = None
                    proposal_reject_reason = "Ideas asked for more input instead of returning a proposal."
                    attempt_info["reject_reason"] = proposal_reject_reason
                    idea_attempts.append(_json_safe(attempt_info))  # type: ignore[arg-type]
                    continue
                try:
                    strict_json_ok = _is_strict_json_only(proposal_text)
                    if cfg.ideas_require_strict_json and not strict_json_ok:
                        if not cfg.ideas_allow_embedded_json:
                            proposal = None
                            proposal_reject_reason = "Ideas response was not strict JSON-only."
                            attempt_info["reject_reason"] = proposal_reject_reason
                            idea_attempts.append(_json_safe(attempt_info))  # type: ignore[arg-type]
                            continue
                        attempt_info["strict_json_only"] = False
                    else:
                        attempt_info["strict_json_only"] = True
                    candidate = _extract_json_from_text(proposal_text)
                    ok_research, research_reason = _validate_ideas_research_rule(
                        candidate,
                        cfg,
                        tool_evidence=proposal_tool_evidence,
                    )
                    if not ok_research:
                        proposal = None
                        proposal_reject_reason = research_reason
                        attempt_info["reject_reason"] = research_reason
                        idea_attempts.append(_json_safe(attempt_info))  # type: ignore[arg-type]
                        continue
                    proposal_reject_reason = ""
                    attempt_info["accepted"] = True
                    idea_attempts.append(_json_safe(attempt_info))  # type: ignore[arg-type]
                    proposal = candidate
                    break
                except Exception as exc:
                    proposal = None
                    proposal_reject_reason = f"JSON parse error: {type(exc).__name__}"
                    attempt_info["reject_reason"] = proposal_reject_reason
                    idea_attempts.append(_json_safe(attempt_info))  # type: ignore[arg-type]
            iter_history["ideas_attempts"] = idea_attempts
            if proposal is None and not proposal_text.strip():
                break
            if proposal is None and proposal_reject_reason:
                iter_history["ideas_research"] = None
                iter_history["ideas_used_web_tools"] = None
                iter_history["ideas_research_error"] = proposal_reject_reason
                iter_history["proposal_rejected_reason"] = proposal_reject_reason
                iter_history["proposal_accepted"] = False
                stop_reason = (
                    "ideas_research_rule"
                    if "research" in proposal_reject_reason.lower() or "web" in proposal_reject_reason.lower()
                    else "ideas_proposal_rejected"
                )
                break

            if proposal is not None and active_idea_mode == "structural":
                changes = proposal.get("changes", [])
                if not isinstance(changes, list) or not changes:
                    proposal = None
                else:
                    target_files = [c.get("file") for c in changes if isinstance(c, dict)]
                    if any(f not in cfg.allowed_files for f in target_files):
                        break

            idea = None
            grid: List[Dict[str, object]] = []
            if proposal is not None:
                idea = proposal
                grid = proposal.get("grid", []) if isinstance(proposal.get("grid", []), list) else []
            else:
                idea, grid = _grid_from_idea_text(
                    proposal_text,
                    cfg.allowed_knobs,
                    cfg.allow_any_knob,
                )
            if isinstance(proposal_tool_evidence, dict):
                iter_history["ideas_tool_evidence"] = _json_safe(proposal_tool_evidence)
                iter_history["ideas_observed_web_tools"] = bool(proposal_tool_evidence.get("web_tools_used", False))
                iter_history["ideas_observed_source_urls"] = _json_safe(
                    proposal_tool_evidence.get("observed_urls", [])
                )
            else:
                iter_history["ideas_tool_evidence"] = None
                iter_history["ideas_observed_web_tools"] = None
                iter_history["ideas_observed_source_urls"] = []
            if idea is not None and isinstance(idea.get("research"), dict):
                research = idea.get("research", {})
                iter_history["ideas_research"] = _json_safe(research)
                iter_history["ideas_used_web_tools"] = bool(research.get("used_web_tools", False))
                iter_history["ideas_reported_source_urls"] = _research_source_urls(research)
                iter_history["ideas_research_error"] = None
                iter_history["proposal_rejected_reason"] = None
                iter_history["proposal_accepted"] = True
            else:
                iter_history["ideas_research"] = None
                iter_history["ideas_used_web_tools"] = None
                iter_history["ideas_reported_source_urls"] = []
                iter_history["ideas_research_error"] = "Missing research object."
                iter_history["proposal_accepted"] = False
            if active_idea_mode == "grid":
                if not grid:
                    iter_history["proposal_rejected_reason"] = "No valid grid produced."
                    break
            else:
                if idea is None or not isinstance(idea.get("changes", []), list):
                    iter_history["proposal_rejected_reason"] = "Structural mode requires changes[]."
                    break

            before_text = cfg.experiment_config.read_text(encoding="utf-8")
            updated_cfg = _load_json(cfg.experiment_config)
            before_fixed_env_snapshot = (
                dict(updated_cfg.get("fixed_env", {}))
                if isinstance(updated_cfg.get("fixed_env", {}), dict)
                else {}
            )
            if cfg.allow_fixed_env_changes and idea and isinstance(idea.get("fixed_env"), dict):
                base_fixed_env = updated_cfg.get("fixed_env", {})
                if not isinstance(base_fixed_env, dict):
                    base_fixed_env = {}
                merged_fixed_env = dict(base_fixed_env)
                allowed_fixed_env = {str(k) for k in cfg.allowed_fixed_env_keys}
                rejected_fixed_env_keys: List[str] = []
                for k, v in idea["fixed_env"].items():
                    key = str(k)
                    if allowed_fixed_env and key not in allowed_fixed_env:
                        rejected_fixed_env_keys.append(key)
                        continue
                    merged_fixed_env[key] = str(v)
                updated_cfg["fixed_env"] = merged_fixed_env
                iter_history["proposed_fixed_env_delta"] = _dict_delta(base_fixed_env, merged_fixed_env)
                iter_history["fixed_env_rejected_keys"] = sorted(set(rejected_fixed_env_keys))
            fixed_env = updated_cfg.get("fixed_env", {})
            locked_keys = (
                []
                if cfg.allow_fixed_env_changes
                else (fixed_env.keys() if isinstance(fixed_env, dict) else [])
            )
            if grid:
                cleaned_grid = _strip_locked_env(grid, locked_keys)
                if not cleaned_grid:
                    break
                updated_cfg["grid"] = cleaned_grid
                iter_history["proposed_grid_names"] = [str(g.get("name")) for g in cleaned_grid if isinstance(g, dict)]
            if idea and isinstance(idea.get("run_id"), str):
                updated_cfg["run_id"] = idea["run_id"]
                iter_history["proposed_run_id"] = idea["run_id"]

            after_text = json.dumps(updated_cfg, indent=2) + "\n"
            diff = _make_unified_diff(cfg.experiment_config, before_text, after_text)
            patches = {"patches": [{"file": str(cfg.experiment_config), "patch": diff}]}

            if active_idea_mode == "structural":
                changes = idea.get("changes", [])
                for patch_item in changes:
                    patch = patch_item.get("patch", "")
                    if not patch.strip():
                        continue
                    diff_files = _diff_files_from_patch(patch)
                    if any(f not in cfg.allowed_files for f in diff_files):
                        raise RuntimeError(f"Patch touches disallowed files: {diff_files}")
                    _apply_patch(patch)
                structural_variants[current_family] = structural_variants.get(current_family, 0) + 1
                iter_history["structural_variants_for_family"] = structural_variants[current_family]
                if structural_variants[current_family] >= cfg.max_structural_variants_per_family:
                    if cfg.idea_mode == "auto":
                        family_modes[current_family] = "grid"
                        iter_history["mode_switch_reason"] = "structural_variant_limit_revert_to_grid"
                    else:
                        stop_reason = "structural_variant_limit"
                        break

            patch_list = patches.get("patches", [])
            if not isinstance(patch_list, list) or not patch_list:
                break

            before_config_text = cfg.experiment_config.read_text(encoding="utf-8")
            try:
                for patch_item in patch_list:
                    patch = patch_item.get("patch", "")
                    if not patch.strip():
                        continue
                    diff_files = _diff_files_from_patch(patch)
                    if any(f not in cfg.allowed_files for f in diff_files):
                        raise RuntimeError(f"Patch touches disallowed files: {diff_files}")
                    _apply_patch(patch)

                updated_config = _load_json(cfg.experiment_config)
                ok, reason = _validate_config_guardrails(
                    baseline_config,
                    updated_config,
                    cfg.allowed_knobs,
                    cfg.allow_any_knob,
                    cfg.allow_fixed_env_changes,
                    cfg.allowed_fixed_env_keys,
                )
                updated_fixed_env = (
                    dict(updated_config.get("fixed_env", {}))
                    if isinstance(updated_config.get("fixed_env", {}), dict)
                    else {}
                )
                iter_history["applied_fixed_env_delta"] = _dict_delta(before_fixed_env_snapshot, updated_fixed_env)
                iter_history["applied_grid_names"] = [
                    str(x.get("name"))
                    for x in updated_config.get("grid", [])
                    if isinstance(x, dict)
                ] if isinstance(updated_config.get("grid", []), list) else []
                iter_history["applied_run_id"] = updated_config.get("run_id")
                if not ok:
                    cfg.experiment_config.write_text(before_config_text, encoding="utf-8")
                    iter_history["proposal_rejected_reason"] = f"guardrail: {reason}"
                    break
                if cfg.idea_mode == "auto" and active_idea_mode == "structural" and cfg.auto_structural_revert_on_improve and improved_this_iter:
                    family_modes[current_family] = "grid"
                    iter_history["mode_switch_reason"] = "structural_improve_revert_to_grid"
            except Exception:
                cfg.experiment_config.write_text(before_config_text, encoding="utf-8")
                iter_history["proposal_rejected_reason"] = "apply_or_validate_exception"
                break
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    report = {
        "best_lock_expectancy": best_expectancy,
        "best_composite_score": best_score,
        "iterations": history,
        "stop_reason": stop_reason,
        "family_best": family_best,
        "family_order": family_order,
        "consecutive_failed_families": consecutive_failed_families,
        "policies": {
            "idea_mode": cfg.idea_mode,
            "auto_mode_start": cfg.auto_mode_start,
            "scoring_enabled": cfg.scoring_enabled,
            "score_weight_max_dd": cfg.score_weight_max_dd,
            "score_weight_worst_day": cfg.score_weight_worst_day,
            "score_weight_breach": cfg.score_weight_breach,
            "score_weight_trade_shortfall": cfg.score_weight_trade_shortfall,
            "min_promote_score_delta": cfg.min_promote_score_delta,
            "min_promote_lock_ev_delta": cfg.min_promote_lock_ev_delta,
            "min_promote_trades_lock": min_promote_trades_lock,
            "revert_on_reject": cfg.revert_on_reject,
            "ideas_require_strict_json": cfg.ideas_require_strict_json,
            "ideas_allow_embedded_json": cfg.ideas_allow_embedded_json,
            "research_require_web_tools": cfg.research_require_web_tools,
            "research_require_source_url": cfg.research_require_source_url,
            "research_require_tool_evidence": cfg.research_require_tool_evidence,
            "research_min_web_search_count": cfg.research_min_web_search_count,
            "research_min_web_fetch_count": cfg.research_min_web_fetch_count,
            "research_min_sources": cfg.research_min_sources,
            "research_max_sources": cfg.research_max_sources,
            "research_allowed_domains": cfg.research_allowed_domains,
            "research_blocked_domains": cfg.research_blocked_domains,
            "research_preferred_domains": cfg.research_preferred_domains,
            "allow_fixed_env_changes": cfg.allow_fixed_env_changes,
            "allowed_fixed_env_keys": cfg.allowed_fixed_env_keys,
            "screening": _json_safe(cfg.screening.__dict__) if cfg.screening else {"enabled": False},
        },
    }
    _write_json(outdir / "orchestrator_report.json", report)


def main(argv: Iterable[str]) -> int:
    args = _parse_args(argv)
    cfg_path = Path(args.config).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    if args.background:
        outdir.mkdir(parents=True, exist_ok=True)
        log_path = outdir / "orchestrator_background.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(cfg_path),
            "--outdir",
            str(outdir),
        ]
        with log_path.open("w", encoding="utf-8") as fh:
            creation_flags = 0
            if os.name == "nt":
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=fh,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
        print(f"Spawned orchestrator pid={proc.pid}. Log: {log_path}")
        return 0
    run_orchestrator(cfg_path, outdir)
    print(f"Wrote orchestrator_report.json to {outdir / 'orchestrator_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
