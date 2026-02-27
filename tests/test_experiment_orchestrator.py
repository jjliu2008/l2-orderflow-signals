from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import scripts.experiment_orchestrator as eo


def _cfg(tmp_path: Path) -> eo.OrchestratorConfig:
    return eo.OrchestratorConfig(
        experiment_config=tmp_path / "experiment_agent.json",
        outdir=tmp_path / "out",
        ideas_agent="ideas",
        coding_agent="coding",
        openclaw_timeout_seconds=600,
        idea_mode="grid",
        max_iterations=5,
        max_no_improve=2,
        pivot_min_lock_ev=-0.25,
        pivot_min_lock_ev_iterations=30,
        pivot_family_fail_limit=3,
        max_structural_variants_per_family=3,
        pivot_min_right_tail_mfe3=0.01,
        pivot_min_right_tail_mfe4=0.005,
        scoring_enabled=True,
        score_weight_max_dd=0.01,
        score_weight_worst_day=0.01,
        score_weight_breach=5.0,
        score_weight_trade_shortfall=0.001,
        min_promote_score_delta=0.0,
        min_promote_lock_ev_delta=0.0,
        min_promote_trades_lock=0,
        revert_on_reject=True,
        allowed_files=["src/entry_alpha.py", "configs/experiment_agent.json"],
        allowed_knobs={"TRADE_SESSION": ["all", "rth"]},
        allow_any_knob=True,
        allow_fixed_env_changes=True,
        allowed_fixed_env_keys=["ENTRY_CONFIRM_BARS"],
        ideas_require_strict_json=True,
        ideas_allow_embedded_json=True,
        research_require_web_tools=True,
        research_require_source_url=True,
        research_require_tool_evidence=True,
        research_min_web_search_count=1,
        research_min_web_fetch_count=1,
        research_min_sources=1,
        research_max_sources=3,
        research_allowed_domains=[],
        research_blocked_domains=["blocked.example"],
        research_preferred_domains=["ninjatrader.com"],
        auto_mode_start="grid",
        auto_grid_to_structural_no_improve=2,
        auto_grid_to_structural_best_lock_below=-0.25,
        auto_grid_to_structural_require_tail_failure=True,
        auto_structural_revert_on_improve=True,
        screening=None,
        max_proposal_attempts=2,
        max_patch_attempts=1,
        rotation_request_limit=2,
    )


class TestExperimentOrchestrator(unittest.TestCase):
    def test_extract_json_from_text_and_strict_json_only(self) -> None:
        text = 'Prose before\n```json\n{"a": 1}\n```\n'
        self.assertEqual(eo._extract_json_from_text(text), {"a": 1})
        self.assertTrue(eo._is_strict_json_only('{"a":1}'))
        self.assertFalse(eo._is_strict_json_only(text))

    def test_validate_research_rule_requires_tool_evidence_and_domain_quality(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(Path(d))
            proposal = {
                "research": {
                    "used_web_tools": True,
                    "queries": ["es futures order flow"],
                    "sources": [
                        {
                            "url": "https://ninjatrader.com/futures/blogs/ninjatrader-order-flow/",
                            "why": "order flow",
                        }
                    ],
                    "notes": "Used targeted search",
                }
            }
            evidence = {
                "web_tools_used": True,
                "observed_urls": ["https://ninjatrader.com/futures/blogs/ninjatrader-order-flow/"],
                "observed_domains": ["ninjatrader.com"],
                "web_search_count": 1,
                "web_fetch_count": 1,
            }
            ok, reason = eo._validate_ideas_research_rule(proposal, cfg, tool_evidence=evidence)
            self.assertTrue(ok, reason)

            blocked = {
                "research": {
                    "used_web_tools": True,
                    "queries": ["x"],
                    "sources": [{"url": "https://blocked.example/post", "why": "bad"}],
                    "notes": "x",
                }
            }
            ok, reason = eo._validate_ideas_research_rule(blocked, cfg, tool_evidence=evidence)
            self.assertFalse(ok)
            self.assertIn("Blocked source domain", reason)

            no_tools_evidence = {"web_tools_used": False, "web_search_count": 0, "web_fetch_count": 0}
            ok, reason = eo._validate_ideas_research_rule(proposal, cfg, tool_evidence=no_tools_evidence)
            self.assertFalse(ok)
            self.assertIn("No observed web_search/web_fetch", reason)

    def test_validate_config_guardrails_respects_fixed_env_allowlist(self) -> None:
        baseline = {
            "data_days": {"train_start": "2025-12-01"},
            "min_trades": {"train": 100, "lock": 50},
            "risk_limits": {"daily_loss_limit_ticks": 80},
            "fixed_env": {"ENTRY_CONFIRM_BARS": "2", "PNL_SLIPPAGE_TICKS": "1"},
            "grid": [{"name": "A", "env": {"TRADE_SESSION": "all"}}],
        }
        candidate_ok = {
            **baseline,
            "fixed_env": {"ENTRY_CONFIRM_BARS": "3", "PNL_SLIPPAGE_TICKS": "1"},
            "grid": [{"name": "A", "env": {"ENTRY_CONFIRM_BARS": "3"}}],
        }
        ok, reason = eo._validate_config_guardrails(
            baseline,
            candidate_ok,
            allowed_knobs={"TRADE_SESSION": ["all", "rth"]},
            allow_any_knob=True,
            allow_fixed_env_changes=True,
            allowed_fixed_env_keys=["ENTRY_CONFIRM_BARS"],
        )
        self.assertTrue(ok, reason)

        candidate_bad = {
            **baseline,
            "fixed_env": {"ENTRY_CONFIRM_BARS": "2", "PNL_SLIPPAGE_TICKS": "2"},
            "grid": [{"name": "A", "env": {"TRADE_SESSION": "all"}}],
        }
        ok, reason = eo._validate_config_guardrails(
            baseline,
            candidate_bad,
            allowed_knobs={"TRADE_SESSION": ["all", "rth"]},
            allow_any_knob=True,
            allow_fixed_env_changes=True,
            allowed_fixed_env_keys=["ENTRY_CONFIRM_BARS"],
        )
        self.assertFalse(ok)
        self.assertIn("Disallowed fixed_env change", reason)

    def test_top_grid_from_comparison_selects_best_runs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)
            comparison = tmp_path / "comparison.csv"
            pd.DataFrame(
                [
                    {"name": "A", "status": "OK", "lock_expectancy_ticks": -2.5},
                    {"name": "B", "status": "OK", "lock_expectancy_ticks": -2.0},
                    {"name": "C", "status": "INVALID", "lock_expectancy_ticks": -1.0},
                    {"name": "D", "status": "OK", "lock_expectancy_ticks": -3.0},
                ]
            ).to_csv(comparison, index=False)
            grid = [
                {"name": "A", "env": {}},
                {"name": "B", "env": {}},
                {"name": "C", "env": {}},
                {"name": "D", "env": {}},
            ]
            selected, meta = eo._top_grid_from_comparison(comparison, grid, top_k=2, min_valid_runs=1)
            self.assertEqual([x["name"] for x in selected], ["B", "A"])
            self.assertEqual(meta["selected_names"], ["B", "A"])

    def test_trade_exemplars_handles_empty_or_missing_pnl_ticks(self) -> None:
        self.assertEqual(eo._trade_exemplars(pd.DataFrame()), [])

        trades = pd.DataFrame(
            [
                {"date": "2025-12-01", "side": "BUY", "mfe_ticks": 2.0, "mae_ticks": 1.0},
                {"date": "2025-12-02", "side": "SELL", "mfe_ticks": 1.0, "mae_ticks": 2.0},
            ]
        )
        exemplars = eo._trade_exemplars(trades, limit=2)
        self.assertEqual(len(exemplars), 2)
        self.assertIn("date", exemplars[0])
        self.assertNotIn("pnl_ticks", exemplars[0])

    def test_build_idea_digest_finds_screening_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)
            (tmp_path / "experiment_agent.json").write_text(
                json.dumps(
                    {
                        "data_days": {},
                        "min_trades": {},
                        "risk_limits": {},
                        "fixed_env": {},
                    }
                ),
                encoding="utf-8",
            )
            cfg = _cfg(tmp_path)
            iter_dir = tmp_path / "iter_01"
            comparison_path = iter_dir / "comparison.csv"
            screening_run = iter_dir / "_screening" / "runs" / "A"
            screening_run.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {"name": "A", "status": "OK", "lock_expectancy_ticks": -1.0, "trades_train": 10, "trades_lock": 5},
                    {"name": "B", "status": "OK", "lock_expectancy_ticks": -2.0, "trades_train": 8, "trades_lock": 4},
                ]
            ).to_csv(comparison_path, index=False)
            pd.DataFrame(
                [
                    {
                        "gate_entry_candidates_when_flat": 10,
                        "entry_confirm_failed_gate": 3,
                    }
                ]
            ).to_csv(screening_run / "strategy_diagnostics.csv", index=False)

            digest = eo._build_idea_digest(cfg, iter_dir, comparison_path)
            best_block_reasons = digest.get("block_reasons", {}).get("best", {})
            self.assertIn("confirm_failed", best_block_reasons)
            self.assertGreater(best_block_reasons["confirm_failed"], 0.0)


if __name__ == "__main__":
    unittest.main()
