"""
Liquidation risk simulation.

Usage:
    Set STRATEGY_LIQUIDATION_FLAG at the top of this file (or pass --strategy), then:

        python run_liquidation.py

    Or use a portfolio strategy YAML that declares `strategy.liquidation` (see configs/strategies/*.yaml):

        python run_liquidation.py --from-strategy configs/strategies/Leveraged_Stake.yaml

    Optional: point to any YAML with the same schema:

        python run_liquidation.py --params path/to/params.yaml

This script:
  - Loads liquidation parameters from configs/general/params_liquidation_<strategy>.yaml
  - Each strategy/position has one borrowed asset and one or more collateral
    legs; each leg has its own target LTV, liquidation LTV, volatility/APY, and
    correlation vs the borrowed asset.
  - For each leg we simulate the price ratio R_t = P_borrowed / P_collateral
    using a correlated Student-t process, compute LTV_j,t, and estimate that
    leg's liquidation probability. Position-level liquidation is "any leg
    liquidates", approximated as 1 - Π_j (1 - P_leg_j).
  - Outputs per-position P(Liq_i) = LRC_i and vault LRC_vault = Σ_i w_i * LRC_i.
"""

from __future__ import annotations

import argparse
import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


BASE_DIR = Path(__file__).parent

# Default strategy key — change this to switch runs without CLI args.
# Keys must match LIQUIDATION_STRATEGY_PATHS.
STRATEGY_LIQUIDATION_FLAG = "usdc_lend_eth_borrow"

LIQUIDATION_STRATEGY_PATHS: Dict[str, str] = {
    "usdc_lend_eth_borrow": "configs/general/params_liquidation_usdc_lend_eth_borrow.yaml",
    "eth_lending_b": "configs/general/params_liquidation_eth_lending_b.yaml",
}


def load_params(path: Path | str | None = None) -> Dict[str, Any]:
    resolved = Path(path) if path is not None else _default_params_path()
    if not resolved.is_absolute():
        resolved = BASE_DIR / resolved
    with open(resolved, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_params_path() -> Path:
    return BASE_DIR / LIQUIDATION_STRATEGY_PATHS[STRATEGY_LIQUIDATION_FLAG]


def resolve_liquidation_params_path(
    strategy: str | None,
    explicit_path: Path | str | None,
) -> Path:
    """Pick YAML path: explicit --params wins, else strategy key."""
    if explicit_path is not None:
        p = Path(explicit_path)
        return p if p.is_absolute() else BASE_DIR / p
    key = strategy if strategy is not None else STRATEGY_LIQUIDATION_FLAG
    if key not in LIQUIDATION_STRATEGY_PATHS:
        raise KeyError(
            f"Unknown liquidation strategy {key!r}. "
            f"Choose one of: {', '.join(sorted(LIQUIDATION_STRATEGY_PATHS))}"
        )
    return BASE_DIR / LIQUIDATION_STRATEGY_PATHS[key]


GENERAL_CONFIG_DIR = BASE_DIR / "configs" / "general"


def liquidation_params_path_from_portfolio_strategy(strategy_yaml: Path | str) -> Optional[Path]:
    """
    Read configs/strategies/<file>.yaml `strategy.liquidation`:
    - enabled: false → None (caller should skip run)
    - enabled: true → require params_file (filename under configs/general/)
    """
    path = Path(strategy_yaml)
    if not path.is_absolute():
        path = BASE_DIR / path
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    liq = (data.get("strategy") or {}).get("liquidation")
    if not isinstance(liq, dict):
        raise ValueError(f"Missing strategy.liquidation block in {path}")
    enabled = bool(liq.get("enabled", False))
    if not enabled:
        return None
    fname = liq.get("params_file")
    if not fname or not str(fname).strip():
        raise ValueError(
            f"strategy.liquidation.enabled is true in {path} but params_file is missing or empty"
        )
    return GENERAL_CONFIG_DIR / str(fname).strip()


def correlated_student_t_returns(
    rng: np.random.Generator,
    vol_coll: float,
    vol_borr: float,
    nu: float,
    rho: float,
    apy_coll_min: float,
    apy_coll_max: float,
    apy_borr_min: float,
    apy_borr_max: float,
    dt: float,
    n_paths: int,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw correlated Student-t simple returns for collateral and borrowed assets.

    Returns two arrays of shape (n_paths, n_steps): (rets_collateral, rets_borrowed).
    """
    # Correlated standard normals Z ~ N(0, Σ) with corr = rho
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    chol = np.linalg.cholesky(cov)

    z = rng.standard_normal(size=(n_paths, n_steps, 2))
    z_corr = z @ chol.T  # shape: (n_paths, n_steps, 2)

    # Shared chi-square scaling for multivariate t
    g = rng.chisquare(nu, size=(n_paths, n_steps, 1))
    t_std = z_corr / np.sqrt(g / nu)

    drift_coll = (apy_coll_min + apy_coll_max) / 2.0
    drift_borr = (apy_borr_min + apy_borr_max) / 2.0

    scale_coll = vol_coll * np.sqrt(dt) * np.sqrt((nu - 2.0) / nu)
    scale_borr = vol_borr * np.sqrt(dt) * np.sqrt((nu - 2.0) / nu)

    rets_coll = drift_coll * dt + scale_coll * t_std[:, :, 0]
    rets_borr = drift_borr * dt + scale_borr * t_std[:, :, 1]
    return rets_coll, rets_borr


def simulate_terminal_ratio_for_leg(
    p: Dict[str, Any],
    rng: np.random.Generator,
    pos: Dict[str, Any],
    leg: Dict[str, Any],
) -> np.ndarray:
    """
    Simulate the terminal price ratio R_T = (P_borrowed / P_collateral)_T
    for one collateral leg of a position. R_0 is normalized to 1.
    """
    n_paths = int(p["n_paths"])
    dt = p["step_days"] / 365.25
    n_steps = max(1, int(p["horizon_days"] // p["step_days"]))

    nu = float(pos["nu"])

    coll = leg["collateral"]
    # Volatility in config is per month; convert to annual: vol_annual = vol_monthly * sqrt(12)
    coll_vol_monthly = float(coll["volatility_monthly"])
    coll_vol = coll_vol_monthly * np.sqrt(12.0)
    coll_apy_min = float(coll["apy_min"])
    coll_apy_max = float(coll["apy_max"])
    borr = pos["borrowed"]
    borr_vol_monthly = float(borr["volatility_monthly"])
    borr_vol = borr_vol_monthly * np.sqrt(12.0)

    rets_coll, rets_borr = correlated_student_t_returns(
        rng=rng,
        vol_coll=coll_vol,
        vol_borr=borr_vol,
        nu=nu,
        rho=float(leg["rho"]),
        apy_coll_min=coll_apy_min,
        apy_coll_max=coll_apy_max,
        apy_borr_min=float(borr["apy_min"]),
        apy_borr_max=float(borr["apy_max"]),
        dt=dt,
        n_paths=n_paths,
        n_steps=n_steps,
    )

    ratio = np.ones(n_paths, dtype=np.float64)
    for t in range(n_steps):
        numer = 1.0 + rets_borr[:, t]
        denom = np.clip(1.0 + rets_coll[:, t], 1e-12, None)
        ratio *= numer / denom
    return ratio


def health_factors_0(positions: List[Dict[str, Any]]) -> np.ndarray:
    """
    Initial health factor HF_i,0 per position.

    We take the most conservative leg in each position:
        HF_0,pos = min_j (ltv_liq_j / ltv_target_j).
    """
    hfs = []
    tiny = 1e-12
    for pos in positions:
        legs = pos.get("legs", [])
        if not legs:
            hfs.append(1.0)
            continue
        hf_legs = []
        for leg in legs:
            ltv_target = float(leg["ltv_target"])
            ltv_liq = float(leg["ltv_liq"])
            hf_legs.append(ltv_liq / max(ltv_target, tiny))
        hfs.append(min(hf_legs))
    return np.array(hfs, dtype=np.float64)


def liquidation_probability_for_leg(
    ratio_terminal: np.ndarray,
    leg: Dict[str, Any],
) -> float:
    """P(Liq_leg) = fraction of paths where LTV_T = ltv_target * R_T ≥ ltv_liq (R_0 = 1)."""
    ltv_target = float(leg["ltv_target"])
    ltv_liq = float(leg["ltv_liq"])
    ltv_T = ratio_terminal * ltv_target
    return float(np.mean(ltv_T >= ltv_liq))


def liquidation_probability_for_position(
    p: Dict[str, Any],
    rng: np.random.Generator,
    pos: Dict[str, Any],
) -> float:
    """
    Approximate position-level P(Liq) assuming leg-level liquidation events are
    independent:

        P_pos = 1 - Π_j (1 - P_leg_j).
    """
    legs = pos.get("legs", [])
    if not legs:
        return 0.0

    leg_ps = []
    for leg in legs:
        ratio = simulate_terminal_ratio_for_leg(p, rng, pos, leg)
        leg_ps.append(liquidation_probability_for_leg(ratio, leg))

    leg_ps_arr = np.array(leg_ps, dtype=np.float64)
    return float(1.0 - np.prod(1.0 - leg_ps_arr))


def compute_lrc_vault_from_params(p: Dict[str, Any]) -> Tuple[float, List[str], np.ndarray]:
    """
    Run the same Monte Carlo as the CLI: per-position P(Liq) and vault LRC_vault = Σ w_i * LRC_i.
    Returns (lrc_vault, position_names, p_liq_per_position).
    """
    rng = np.random.default_rng(p.get("seed"))
    positions: List[Dict[str, Any]] = p["positions"]
    names = [str(pos["name"]) for pos in positions]
    weights = np.array([float(pos["weight"]) for pos in positions], dtype=np.float64)
    p_liq = np.zeros(len(positions), dtype=np.float64)
    for i, pos in enumerate(positions):
        p_liq[i] = liquidation_probability_for_position(p, rng, pos)
    lrc_vault = float(np.sum(weights * p_liq))
    return lrc_vault, names, p_liq


def main() -> None:
    parser = argparse.ArgumentParser(description="Liquidation risk simulation")
    parser.add_argument(
        "--strategy",
        choices=sorted(LIQUIDATION_STRATEGY_PATHS.keys()),
        default=None,
        help=(
            "Which built-in liquidation params YAML to load (default: STRATEGY_LIQUIDATION_FLAG at top of run_liquidation.py)"
        ),
    )
    parser.add_argument(
        "--from-strategy",
        dest="from_strategy",
        default=None,
        metavar="YAML",
        help=(
            "Portfolio strategy YAML (configs/strategies/…): use strategy.liquidation.enabled "
            "and params_file to pick the liquidation params file"
        ),
    )
    parser.add_argument(
        "--params",
        dest="params_file",
        default=None,
        help="Override: path to a liquidation params YAML (same schema as configs/general/params_liquidation_*.yaml)",
    )
    args = parser.parse_args()

    if args.params_file and args.from_strategy:
        parser.error("Use only one of --params and --from-strategy")
    if args.from_strategy is not None and args.strategy is not None:
        parser.error("Use only one of --from-strategy and --strategy")

    params_path: Path
    if args.from_strategy is not None:
        resolved = liquidation_params_path_from_portfolio_strategy(args.from_strategy)
        if resolved is None:
            print(
                "Liquidation is disabled for this portfolio strategy "
                f"(strategy.liquidation.enabled: false in {args.from_strategy}). Nothing to run.",
                file=sys.stderr,
            )
            sys.exit(0)
        params_path = resolved
    else:
        params_path = resolve_liquidation_params_path(args.strategy, args.params_file)
    p = load_params(params_path)

    positions: List[Dict[str, Any]] = p["positions"]
    names = [pos["name"] for pos in positions]
    weights = np.array([float(pos["weight"]) for pos in positions], dtype=np.float64)

    hf0 = health_factors_0(positions)

    lrc_vault, _, p_liq = compute_lrc_vault_from_params(p)
    lrc_i = p_liq

    print("=== Liquidation Risk Simulation ===")
    try:
        params_display = params_path.relative_to(BASE_DIR)
    except ValueError:
        params_display = params_path
    print(f"Params file: {params_display}")
    print(f"Paths: {p['n_paths']}, Horizon: {p['horizon_days']} days ({p['step_days']}-day steps)")
    print()
    print(
        "Position".ljust(20),
        "HF_0".rjust(10),
        "P(Liq)".rjust(10),
        "Weight".rjust(10),
        "LRC_i".rjust(10),
        sep="",
    )
    for i, name in enumerate(names):
        print(
            name.ljust(20),
            f"{hf0[i]:.2f}".rjust(10),
            f"{p_liq[i]:.4f}".rjust(10),
            f"{weights[i]:.3f}".rjust(10),
            f"{lrc_i[i]:.4f}".rjust(10),
            sep="",
        )

    print()
    print(f"Vault liquidation coefficient (LRC_vault): {lrc_vault:.4f}")


if __name__ == "__main__":
    main()

