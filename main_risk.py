#!/usr/bin/env python3
"""
Combined portfolio risk view.

  python main_risk.py                    # compare ALL strategies (ranked table)
  python main_risk.py --strategy one.yaml  # detailed output for one strategy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BASE_DIR      = Path(__file__).resolve().parent
STRATEGIES_DIR = BASE_DIR / "configs" / "strategies"


def _imports():
    import run as tail_run
    from calculate_CRS import DECIMALS as CRS_DECIMALS
    from calculate_CRS import (
        compute_crs, compute_crs_portfolio, compute_ors,
        compute_s_cdiv, required_param as crs_req,
    )
    from calculate_PRS import compute_prs, compute_psr, is_asset, load_inputs, required_param
    from run_liquidation import (
        compute_lrc_vault_from_params,
        liquidation_params_path_from_portfolio_strategy,
        load_params as load_liq_params,
    )
    return (tail_run, CRS_DECIMALS, compute_crs, compute_crs_portfolio, compute_ors,
            compute_s_cdiv, crs_req, compute_prs, compute_psr, is_asset, load_inputs,
            required_param, compute_lrc_vault_from_params,
            liquidation_params_path_from_portfolio_strategy, load_liq_params)


def score_strategy(strategy_path: Path) -> dict:
    (tail_run, CRS_DECIMALS, compute_crs, compute_crs_portfolio, compute_ors,
     compute_s_cdiv, crs_req, compute_prs, compute_psr, is_asset, load_inputs,
     required_param, compute_lrc_vault_from_params,
     liquidation_params_path_from_portfolio_strategy, load_liq_params) = _imports()

    params_path   = BASE_DIR / "configs" / "general" / "params.yaml"
    protocols_dir = BASE_DIR / "configs" / "protocols"

    params, protocols, _ = load_inputs(strategy_path, protocols_dir, params_path)

    # PSR
    gamma = required_param(params, "gamma")
    prs_by_name = {}
    for name, proto in protocols.items():
        if is_asset(proto):
            continue
        prs, _ = compute_prs(proto, params)
        prs_by_name[name] = prs
    psr, _, _ = compute_psr(prs_by_name, protocols, gamma)

    # CRS
    delta_crs  = float(params.get("delta_crs", params.get("delta_pcr", 0.5)))
    omega_cdiv = crs_req(params, "omega_cdiv")
    crs_by_name = {}
    for name, proto in protocols.items():
        ors_i, _   = compute_ors(proto, params)
        crs_i, _, _ = compute_crs(proto, params, ors_i)
        s_cdiv, _  = compute_s_cdiv(proto)
        crs_by_name[name] = round(crs_i + omega_cdiv * s_cdiv, CRS_DECIMALS)
    crs_portfolio, _ = compute_crs_portfolio(crs_by_name, protocols, delta_crs)

    # CVaR
    p_sim      = tail_run.load_params(strategy_path)
    jump_model = tail_run.build_protocol_jump_model(strategy_path, p_sim["jump_probability"])
    rng_base   = np.random.default_rng(p_sim.get("seed"))
    rng1 = np.random.default_rng(rng_base.integers(0, 2**32 - 1))
    rng2 = np.random.default_rng(rng_base.integers(0, 2**32 - 1))
    _, cvar_nj = tail_run.var_cvar(
        tail_run.simulate_losses(p_sim, rng1, jump_model, jump_probability_override=0.0),
        p_sim["confidence"])
    _, cvar_wj = tail_run.var_cvar(
        tail_run.simulate_losses(p_sim, rng2, jump_model),
        p_sim["confidence"])
    cvar_sum = cvar_nj + cvar_wj

    # Liquidation
    lrc_vault = 0.0
    try:
        liq_path = liquidation_params_path_from_portfolio_strategy(strategy_path)
        liq_p    = load_liq_params(liq_path)
        lrc_vault, _, _ = compute_lrc_vault_from_params(liq_p)
    except (ValueError, FileNotFoundError):
        pass

    proto_lines = [
        f"{name}  {proto.get('alloc_max_pct', '?')}%"
        for name, proto in protocols.items()
    ]
    return {
        "name":        p_sim.get("strategy_name", strategy_path.stem),
        "file":        strategy_path.stem,
        "n_proto":     len(protocols),
        "proto_lines": proto_lines,
        "psr":         psr,
        "crs":         crs_portfolio,
        "cvar_mkt":    cvar_nj,
        "cvar_str":    cvar_wj,
        "cvar_sum":    cvar_sum,
        "lrc":         lrc_vault,
        "total":       psr + crs_portfolio + cvar_sum + lrc_vault,
    }


def print_detail(r: dict) -> None:
    print(f"\n=== {r['name']}  ({r['file']}.yaml) ===")
    print(f"  Protocols ({r['n_proto']}):")
    for line in r["proto_lines"]:
        print(f"    • {line}")
    print(f"  PSR                   : {r['psr']:.6f}")
    print(f"  CRS                   : {r['crs']:.6f}")
    print(f"  CVaR market           : {r['cvar_mkt']:.2%}")
    print(f"  CVaR structural       : {r['cvar_str']:.2%}")
    print(f"  CVaR sum              : {r['cvar_sum']:.2%}")
    print(f"  LRC vault             : {r['lrc']:.6f}")
    print(f"  ─────────────────────────────────")
    print(f"  Final risk score      : {r['total']:.6f}")


def print_table(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda x: x["total"])
    hdr = f"{'Strategy':<28} {'#':<3} {'PSR':>7} {'CRS':>7} {'CVaR mkt':>9} {'CVaR str':>9} {'CVaR sum':>9} {'LRC':>7} {'TOTAL':>9}"
    sep = "─" * len(hdr)
    print("\n=== Strategy comparison (sorted by total risk) ===\n")
    print(hdr)
    print(sep)
    for r in rows:
        print(
            f"{r['name']:<28} {r['n_proto']:<3} "
            f"{r['psr']:>7.4f} {r['crs']:>7.4f} "
            f"{r['cvar_mkt']:>9.2%} {r['cvar_str']:>9.2%} {r['cvar_sum']:>9.2%} "
            f"{r['lrc']:>7.4f} {r['total']:>9.4f}"
        )
        print(f"  └─ {', '.join(r['proto_lines'])}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", default=None,
        help="Single strategy YAML to score in detail (omit to compare all)",
    )
    args = parser.parse_args()

    if args.strategy:
        p = Path(args.strategy)
        if not p.is_absolute() and p.parent == Path("."):
            p = STRATEGIES_DIR / p
        if not p.is_file():
            raise FileNotFoundError(p)
        print_detail(score_strategy(p))
    else:
        yamls = sorted(STRATEGIES_DIR.glob("*.yaml"))
        rows  = []
        for y in yamls:
            print(f"Scoring {y.stem}…", flush=True)
            try:
                rows.append(score_strategy(y))
            except Exception as e:
                print(f"  ⚠ skipped: {e}")
        print_table(rows)


if __name__ == "__main__":
    main()
