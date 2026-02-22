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
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
            check=False,
        )
        return "ProcessName" in proc.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_lock(outdir: Path) -> Path:
    lock_path = outdir / ".experiment_orchestrator.lock"
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
class OrchestratorConfig:
    experiment_config: Path
    outdir: Path
    ideas_agent: str
    coding_agent: str
    max_iterations: int
    max_no_improve: int
    pivot_min_lock_ev: float
    pivot_min_lock_ev_iterations: int
    pivot_family_fail_limit: int
    allowed_files: List[str]
    allowed_knobs: Dict[str, List[str]]
    max_proposal_attempts: int
    max_patch_attempts: int
    rotation_request_limit: int


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _extract_json_block(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in agent response.")
    return json.loads(match.group(0))


def _run_openclaw_agent(agent_id: str, message: str, timeout: int = 600) -> str:
    cmd = ["openclaw.cmd", "agent", "--agent", agent_id, "--message", message]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.stdout


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
) -> Tuple[bool, str]:
    for key in ("data_days", "min_trades", "risk_limits", "fixed_env"):
        if baseline.get(key) != candidate.get(key):
            return False, f"Locked config section changed: {key}"

    grid = candidate.get("grid", [])
    if not isinstance(grid, list) or not grid:
        return False, "grid is empty or invalid"

    for entry in grid:
        env = entry.get("env", {})
        if not isinstance(env, dict):
            return False, "grid env must be dict"
        for k, v in env.items():
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


def _run_experiment_agent(experiment_config: Path, outdir: Path) -> Path:
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
        check=False,
    )
    log_path = outdir / "orchestrator_stdout.log"
    log_path.write_text(proc.stdout, encoding="utf-8")
    comparison = outdir / "comparison.csv"
    if not comparison.exists():
        raise RuntimeError(f"comparison.csv not found after run. See {log_path}")
    return comparison


def _best_lock_expectancy(comparison_path: Path) -> float:
    df = pd.read_csv(comparison_path)
    if df.empty or "lock_expectancy_ticks" not in df.columns:
        return float("-inf")
    return float(df["lock_expectancy_ticks"].max())


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
    return parser.parse_args(list(argv))


def _load_orchestrator_config(path: Path, outdir: Path) -> OrchestratorConfig:
    raw = _load_json(path)
    return OrchestratorConfig(
        experiment_config=Path(raw["experiment_config"]).expanduser().resolve(),
        outdir=outdir,
        ideas_agent=str(raw.get("ideas_agent", "ideas")),
        coding_agent=str(raw.get("coding_agent", "coding")),
        max_iterations=int(raw.get("max_iterations", 5)),
        max_no_improve=int(raw.get("max_no_improve", 2)),
        pivot_min_lock_ev=float(raw.get("pivot_min_lock_ev", -0.25)),
        pivot_min_lock_ev_iterations=int(raw.get("pivot_min_lock_ev_iterations", 30)),
        pivot_family_fail_limit=int(raw.get("pivot_family_fail_limit", 3)),
        allowed_files=[str(x) for x in raw.get("allowed_files", [])],
        allowed_knobs={str(k): [str(v) for v in vals] for k, vals in raw.get("allowed_knobs", {}).items()},
        max_proposal_attempts=int(raw.get("max_proposal_attempts", 2)),
        max_patch_attempts=int(raw.get("max_patch_attempts", 2)),
        rotation_request_limit=int(raw.get("rotation_request_limit", 2)),
    )


def run_orchestrator(config_path: Path, outdir: Path) -> None:
    cfg = _load_orchestrator_config(config_path, outdir)
    baseline_config = _load_json(cfg.experiment_config)
    lock_path = _acquire_lock(outdir)
    history: List[Dict[str, object]] = []

    best_expectancy = float("-inf")
    no_improve = 0
    stop_reason = ""
    family_best: Dict[str, float] = {}
    family_order: List[str] = []
    consecutive_failed_families = 0
    last_family = _family_id(cfg.experiment_config)

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
            comparison_path = _run_experiment_agent(cfg.experiment_config, iter_dir)
            summary = _summarize_results(comparison_path)
            current_best = _best_lock_expectancy(comparison_path)

            family_best[current_family] = max(current_best, family_best.get(current_family, float("-inf")))
            if not family_order or family_order[-1] != current_family:
                family_order.append(current_family)

            history.append(
                {
                    "iteration": iteration,
                    "family": current_family,
                    "comparison_path": str(comparison_path),
                    "best_lock_expectancy": current_best,
                }
            )

            if current_best > best_expectancy:
                best_expectancy = current_best
                no_improve = 0
            else:
                no_improve += 1

            if (
                iteration >= cfg.pivot_min_lock_ev_iterations
                and best_expectancy < cfg.pivot_min_lock_ev
            ):
                stop_reason = "pivot_min_lock_ev"
                break

            if consecutive_failed_families >= cfg.pivot_family_fail_limit:
                stop_reason = "pivot_family_fail_limit"
                break

            if no_improve >= cfg.max_no_improve:
                stop_reason = "no_improve"
                break

            idea_prompt = (
                "You are the ideas agent. Propose a new experiment configuration.\n"
                "Return ONLY JSON matching this schema:\n"
                "{\n"
                '  "rationale": "...",\n'
                '  "expected_metrics": ["lock_expectancy_ticks", "trade_count"],\n'
                '  "changes": [\n'
                '    {"file": "src/entry_alpha.py", "patch": ""},\n'
                '    {"file": "configs/experiment_agent.json", "patch": ""}\n'
                "  ],\n"
                '  "grid": ["A_baseline", "B_proof"]\n'
                "}\n"
                "Guardrails:\n"
                "- Do not modify cost model, exits, or data range.\n"
                "- Only adjust allowed knobs in grid env.\n"
                f"- Allowed knobs: {json.dumps(cfg.allowed_knobs)}\n"
                "Recent results:\n"
                f"{summary}\n"
            )

            proposal = None
            proposal_text = ""
            for _ in range(cfg.max_proposal_attempts):
                if rotation_calls >= cfg.rotation_request_limit:
                    stop_reason = "rotation_request_limit"
                    break
                proposal_text = _run_openclaw_agent(cfg.ideas_agent, idea_prompt)
                rotation_calls += 1
                try:
                    proposal = _extract_json_block(proposal_text)
                    break
                except Exception:
                    proposal = None
            if proposal is None:
                break

            changes = proposal.get("changes", [])
            if not isinstance(changes, list) or not changes:
                break

            target_files = [c.get("file") for c in changes if isinstance(c, dict)]
            if any(f not in cfg.allowed_files for f in target_files):
                break

            coding_prompt = (
                "You are the coding agent. Produce unified diffs for the proposal below.\n"
                "Return ONLY JSON:\n"
                "{ \"patches\": [ {\"file\": \"path\", \"patch\": \"...\"} ] }\n\n"
                f"Proposal JSON:\n{json.dumps(proposal, indent=2)}\n"
            )

            patches = None
            for _ in range(cfg.max_patch_attempts):
                if rotation_calls >= cfg.rotation_request_limit:
                    stop_reason = "rotation_request_limit"
                    break
                patch_text = _run_openclaw_agent(cfg.coding_agent, coding_prompt)
                rotation_calls += 1
                try:
                    patches = _extract_json_block(patch_text)
                    break
                except Exception:
                    patches = None
            if patches is None or "patches" not in patches:
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
                ok, reason = _validate_config_guardrails(baseline_config, updated_config, cfg.allowed_knobs)
                if not ok:
                    cfg.experiment_config.write_text(before_config_text, encoding="utf-8")
                    break
            except Exception:
                cfg.experiment_config.write_text(before_config_text, encoding="utf-8")
                break
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    report = {
        "best_lock_expectancy": best_expectancy,
        "iterations": history,
        "stop_reason": stop_reason,
        "family_best": family_best,
        "family_order": family_order,
        "consecutive_failed_families": consecutive_failed_families,
    }
    _write_json(outdir / "orchestrator_report.json", report)


def main(argv: Iterable[str]) -> int:
    args = _parse_args(argv)
    cfg_path = Path(args.config).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    run_orchestrator(cfg_path, outdir)
    print(f"Wrote orchestrator_report.json to {outdir / 'orchestrator_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
