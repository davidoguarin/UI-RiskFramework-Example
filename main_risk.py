#!/usr/bin/env python3
"""
Combined portfolio risk view for a strategy YAML.

Computes:
  - PSR   — portfolio protocol score (calculate_PRS_V2: Σ (γ/N)·PRS_i)
  - CRS   — portfolio counterparty aggregate (calculate_CRS: CRS_portfolio)
  - CVaR_sum — tail simulation from run.py (sum of no-jump and with-jump CVaR)
  - LRC_vault — optional liquidation vault coefficient (run_liquidation) when
    strategy.liquidation.enabled is true

Final risk score (default): PSR + CRS_portfolio + CVaR_sum + LRC_vault
  (components use different units; treat as a composite index.)

Usage:
    python main_risk.py
    python main_risk.py --strategy Leveraged_Stake.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# Default strategy file (under configs/strategies/ unless you pass a path).

STRATEGY_YAML_FLAG = "Leveraged_Stake.yaml"
#STRATEGY_YAML_FLAG = "Leveraged_Stake_hedged.yaml"
#STRATEGY_YAML_FLAG = "default_strategy.yaml"
#STRATEGY_YAML_FLAG = "Morpho_Gauntlet_Core.yaml"
#STRATEGY_YAML_FLAG = "Morpho_Gauntlet_PstExp.yaml"
#STRATEGY_YAML_FLAG = "Morpho_Steakhouse.yaml"
#STRATEGY_YAML_FLAG = "ALL.yaml"
#STRATEGY_YAML_FLAG = "one.yaml"


def resolve_strategy_path(arg: str) -> Path:
    p = Path(arg)
    if p.is_absolute():
        return p
    if p.parent != Path("."):
        return BASE_DIR / p
    return BASE_DIR / "configs" / "strategies" / p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate PSR, CRS, CVaR_sum, and optional liquidation LRC into one score",
    )
    parser.add_argument(
        "--strategy",
        default=STRATEGY_YAML_FLAG,
        help="Strategy YAML filename in configs/strategies/ or a relative/absolute path",
    )
    args = parser.parse_args()
    strategy_path = resolve_strategy_path(args.strategy)

    # Local imports keep CLI startup light and avoid import cycles.
    import run as tail_run
    from calculate_CRS import DECIMALS as CRS_DECIMALS
    from calculate_CRS import (
        compute_crs,
        compute_crs_portfolio,
        compute_ors,
        compute_s_cdiv,
        required_param as crs_required_param,
    )
    from calculate_PRS_V2 import compute_prs, compute_psr, is_asset, load_inputs, required_param

    from run_liquidation import (
        compute_lrc_vault_from_params,
        liquidation_params_path_from_portfolio_strategy,
        load_params as load_liquidation_params,
    )

    params_path = BASE_DIR / "configs" / "general" / "params.yaml"
    protocols_dir = BASE_DIR / "configs" / "protocols"

    if not strategy_path.is_file():
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")

    params, protocols, _portfolio = load_inputs(strategy_path, protocols_dir, params_path)

    gamma = required_param(params, "gamma")
    prs_by_name: dict[str, float] = {}
    for name, protocol in protocols.items():
        if is_asset(protocol):
            continue
        prs, _, _ = compute_prs(protocol, params)
        prs_by_name[name] = prs
    psr, _psr_contrib, _n = compute_psr(prs_by_name, protocols, gamma)

    delta_crs = (
        float(params["delta_crs"])
        if "delta_crs" in params
        else crs_required_param(params, "delta_pcr")
    )
    omega_cdiv = crs_required_param(params, "omega_cdiv")
    crs_by_name: dict[str, float] = {}
    for name, protocol in protocols.items():
        ors_i, _ = compute_ors(protocol, params)
        crs_i, _, _ = compute_crs(protocol, params, ors_i)
        s_cdiv, _ = compute_s_cdiv(protocol)
        crs_full = round(crs_i + omega_cdiv * s_cdiv, CRS_DECIMALS)
        crs_by_name[name] = crs_full
    crs_portfolio, _crs_contrib = compute_crs_portfolio(crs_by_name, protocols, delta_crs)

    p_sim = tail_run.load_params(strategy_path)
    jump_model = tail_run.build_protocol_jump_model(strategy_path, p_sim["jump_probability"])
    rng_base = np.random.default_rng(p_sim.get("seed"))
    rng_no_jump = np.random.default_rng(rng_base.integers(0, 2**32 - 1))
    losses_no_jump = tail_run.simulate_losses(
        p_sim, rng_no_jump, jump_model, jump_probability_override=0.0
    )
    _var_no_jump, cvar_no_jump = tail_run.var_cvar(losses_no_jump, p_sim["confidence"])
    rng_with_jump = np.random.default_rng(rng_base.integers(0, 2**32 - 1))
    losses_with_jump = tail_run.simulate_losses(
        p_sim, rng_with_jump, jump_model, jump_probability_override=None
    )
    _var_with_jump, cvar_with_jump = tail_run.var_cvar(losses_with_jump, p_sim["confidence"])
    cvar_sum = cvar_no_jump + cvar_with_jump

    liq_path: Path | None
    try:
        liq_path = liquidation_params_path_from_portfolio_strategy(strategy_path)
    except ValueError:
        liq_path = None

    lrc_vault = 0.0
    liq_detail = ""
    if liq_path is not None:
        liq_p = load_liquidation_params(liq_path)
        lrc_vault, liq_names, liq_probs = compute_lrc_vault_from_params(liq_p)
        try:
            rel = liq_path.relative_to(BASE_DIR)
        except ValueError:
            rel = liq_path
        parts = [f"{n}: P(Liq)={float(liq_probs[i]):.4f}" for i, n in enumerate(liq_names)]
        liq_detail = f"{rel}  |  " + "; ".join(parts)

    final_risk_score = psr + crs_portfolio + cvar_sum + lrc_vault

    sim_name = p_sim.get("strategy_name", strategy_path.stem)
    try:
        strategy_display = strategy_path.relative_to(BASE_DIR)
    except ValueError:
        strategy_display = strategy_path
    print("=== Combined risk (main_risk.py) ===")
    print(f"Strategy file: {strategy_display}")
    print(f"Simulation label: {sim_name}")
    print()
    print(f"  PSR (portfolio protocol score):     {psr:.6f}")
    print(f"  CRS (portfolio, calculate_CRS):      {crs_portfolio:.6f}")
    print(f"  CVaR_sum (run.py, no-jump + w-jump): {cvar_sum:.6f}  (fraction of NAV)")
    if liq_path is None:
        print("  LRC_vault (liquidation):             0.000000  (not enabled or no strategy.liquidation block)")
    else:
        print(f"  LRC_vault (liquidation):             {lrc_vault:.6f}")
        print(f"       {liq_detail}")
    print()
    print(f"  Final risk score (sum):              {final_risk_score:.6f}")
    print()
    print("  Components: PSR + CRS + CVaR_sum + LRC_vault (liquidation term 0 if disabled).")


if __name__ == "__main__":
    main()
