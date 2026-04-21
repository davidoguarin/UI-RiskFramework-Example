#!/usr/bin/env python3
"""
Counterparty Risk Score (CRS) calculator.

Per-protocol equations:
  S_OH,i        = rho * (1 - exp(-oracle_heartbeat_hours_i))
  S_OD,i        = 1 - exp(-xi * oracle_deviation_i)
  ORS_i         = S_OH,i + S_OD,i   (0 for category: asset — no oracle risk)
  S_ODiv,i      = exp(-psi_oracle_div * max(0, n_oracle_providers_i - 1))  (not used for assets)
  S_VDec,i      = exp(-n_operators_i / N_ref)
  S_VOps,i      = 1 - exp(-tau_v * slashing_events_i / age_years_i)
                  (slashing intensity per year of protocol life; age_years from protocol YAML)
  S_CDiv,i      = Σ_j w²_div * Q_j   (equal weights w_div = 1/n_counterparties)
                  Q_j from COUNTERPARTY_QUALITY_SCORES; 0 if no counterparty_protocols defined
                  Self-references and oracle/validator dependencies excluded (tracked elsewhere)
  S_DVN,i       = exp(-phi_dvn * max(0, n_dvn_i - 1))
                  n_dvn_i from protocol YAML field n_dvn (null → 0 contribution, not applicable)
                  More required DVNs → lower score → lower cross-chain verification risk
  CRS_i         = ORS_i + chi * S_ODiv,i + delta_val,i * (S_VDec,i + S_VOps,i)
                    + omega_cdiv * S_CDiv,i + lambda_dvn * S_DVN,i
                    delta_val,i = iota if has_slashing_insurance else 1.0  (iota from params.yaml)
                    validator terms = 0 if protocol has no validator_services
                    lambda_dvn * S_DVN,i = 0 when n_dvn is null (no cross-chain messaging)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, Union

import yaml

#STRATEGY_YAML_FLAG = "default_strategy.yaml"
#STRATEGY_YAML_FLAG = "Morpho_Gauntlet_Core.yaml"
#STRATEGY_YAML_FLAG = "Morpho_Steakhouse.yaml"
#STRATEGY_YAML_FLAG = "one.yaml"
#STRATEGY_YAML_FLAG = "Leveraged_Stake.yaml"
STRATEGY_YAML_FLAG = "Leveraged_Stake_hedged.yaml"

DECIMALS = 4

COUNTERPARTY_QUALITY_SCORES: Dict[str, float] = {
    "idle_blue_chip_token": 0.0,           # BTC, ETH
    "idle_blue_chip_lending": 0.0,         # Aave, Compound
    "idle_wrapped_blue_chip": 0.0,         # WETH, WBTC
    "custodial_wrapper": 0.05,             # cbBTC, centralized custody wrappers
    "major_lst": 0.05,                     # Lido, Rocket Pool
    "established_cdp_stablecoin": 0.3,     # MakerDAO, Liquity
    "rwa_credit": 0.4,                     # Maple, Centrifuge
    "centralized_exchange": 0.5,           # Binance, OKX, Bybit, Deribit
    "delta_neutral_stable": 0.5,           # Ethena
    "yield_aggregator": 0.5,               # Pendle, vault wrappers
    "non_established_cdp_synthetic": 0.6,
    "idle_meme_token": 0.7,
    "reflexive_algorithmic_stablecoin": 0.9,
    "new_or_unaudited": 1.0,
    "recently_depegged_unstable": 1.0,
}


def is_asset(protocol: dict) -> bool:
    return str(protocol.get("category", "protocol")).strip().lower() == "asset"


def required_param(params: Dict, key: str) -> float:
    if key not in params:
        raise KeyError(f"Missing required parameter: {key}")
    return float(params[key])


def load_yaml_file(path: Union[str, Path]) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict YAML content in: {p}")
    return data


def load_inputs(strategy_path: Union[str, Path], protocols_dir: Union[str, Path], params_path: Union[str, Path]):
    strategy_data = load_yaml_file(strategy_path)
    general_data = load_yaml_file(params_path)
    params = dict(general_data.get("parameters", {}))
    params.update(strategy_data.get("parameters", {}))

    portfolio = strategy_data.get("portfolio", {})
    strategy = strategy_data.get("strategy", {})
    selected = strategy.get("protocols", [])
    if not isinstance(selected, list) or len(selected) == 0:
        raise ValueError("strategy.protocols must be a non-empty list")

    protocols: Dict[str, Dict] = {}
    pdir = Path(protocols_dir)
    for item in selected:
        name = item.get("name")
        alloc_max_pct = item.get("alloc_max_pct", 0.0)
        if not name:
            raise ValueError("Each strategy.protocols item needs a name")
        proto_path = pdir / f"{str(name).lower().replace(' ', '_')}.yaml"
        proto = load_yaml_file(proto_path)
        proto["alloc_max_pct"] = alloc_max_pct
        protocols[name] = proto
    return params, protocols, portfolio


def s_oh(oracle_heartbeat_hours: float, rho: float) -> float:
    """ρ scales outside the exponential: ρ * (1 - e^{-h})."""
    return round(rho * (1.0 - math.exp(-oracle_heartbeat_hours)), DECIMALS)


def s_od(oracle_deviation: float, xi: float) -> float:
    return round(1.0 - math.exp(-xi * oracle_deviation), DECIMALS)


def compute_ors(protocol: dict, params: dict) -> Tuple[float, Dict[str, float]]:
    if is_asset(protocol):
        return 0.0, {"S_OH": 0.0, "S_OD": 0.0}
    if protocol.get("oracle_heartbeat_hours") is None:
        raise ValueError("oracle_heartbeat_hours is required for non-asset protocols")
    if protocol.get("oracle_deviation") is None:
        raise ValueError("oracle_deviation is required for non-asset protocols")
    rho = required_param(params, "rho")
    xi = required_param(params, "xi")
    h = float(protocol["oracle_heartbeat_hours"])
    d = float(protocol["oracle_deviation"])
    components = {"S_OH": s_oh(h, rho), "S_OD": s_od(d, xi)}
    ors = round(sum(components.values()), DECIMALS)
    return ors, components


def _n_oracle_providers(protocol: dict) -> int:
    od = protocol.get("oracle_diversification") or {}
    if not isinstance(od, dict):
        return 1
    n = od.get("provider_count")
    if n is not None:
        try:
            return max(1, int(n))
        except (TypeError, ValueError):
            pass
    providers = od.get("providers")
    if isinstance(providers, list) and len(providers) > 0:
        return max(1, len(providers))
    return 1


def s_odiv(n_providers: int, psi: float) -> float:
    return round(math.exp(-psi * max(0, int(n_providers) - 1)), DECIMALS)


def s_vdec(n_operators: int, N_ref: float) -> float:
    """Validator decentralisation: risk decays as operator count grows."""
    return round(math.exp(-n_operators / N_ref), DECIMALS)


def s_vops(slashing_per_year: float, tau_v: float) -> float:
    """Validator operational risk from slashing rate (events per year of maturity)."""
    return round(1.0 - math.exp(-tau_v * float(slashing_per_year)), DECIMALS)


def compute_s_validator(protocol: dict, params: dict) -> Tuple[float, float, float, Dict[str, float]]:
    vs = protocol.get("validator_services")
    if vs is None:
        return 0.0, 0.0, 1.0, {}

    N_ref = required_param(params, "N_ref")
    tau_v = required_param(params, "tau_v")
    iota = required_param(params, "iota")

    n_operators = int(vs.get("n_operators", 1))
    slashing = float(vs.get("slashing_events", 0))
    has_insurance = bool(vs.get("has_slashing_insurance", False))
    maturity_years = max(float(protocol.get("age_years", 1.0)), 1e-9)
    slashing_per_year = slashing / maturity_years

    vdec = s_vdec(n_operators, N_ref)
    vops = s_vops(slashing_per_year, tau_v)
    discount = float(iota) if has_insurance else 1.0

    details = {
        "S_VDec": vdec,
        "S_VOps": vops,
        "discount": discount,
        "iota": float(iota),
        "slashing_per_year": round(slashing_per_year, DECIMALS),
    }
    return vdec, vops, discount, details


def compute_crs(protocol: dict, params: dict, ors_i: float) -> Tuple[float, float, Dict[str, float]]:
    chi = required_param(params, "chi")
    psi = required_param(params, "psi_oracle_div")
    vdec, vops, discount, val_details = compute_s_validator(protocol, params)
    validator_term = round(discount * (vdec + vops), DECIMALS)
    dvn_term, n_dvn = compute_s_dvn(protocol, params)
    if is_asset(protocol):
        crs = round(ors_i + validator_term + dvn_term, DECIMALS)
        detail = {
            "n_oracle_providers": 0,
            "S_ODiv": 0.0,
            "chi": chi,
            "psi_oracle_div": psi,
            "validator_term": validator_term,
            "dvn_term": dvn_term,
            "n_dvn": n_dvn,
            **{f"val_{k}": v for k, v in val_details.items()},
        }
        return crs, 0.0, detail
    n = _n_oracle_providers(protocol)
    sodiv = s_odiv(n, psi)
    crs = round(ors_i + chi * sodiv + validator_term + dvn_term, DECIMALS)
    detail = {
        "n_oracle_providers": n,
        "S_ODiv": sodiv,
        "chi": chi,
        "psi_oracle_div": psi,
        "validator_term": validator_term,
        "dvn_term": dvn_term,
        "n_dvn": n_dvn,
        **{f"val_{k}": v for k, v in val_details.items()},
    }
    return crs, sodiv, detail


def compute_crs_portfolio(crs_by_name: Dict[str, float], protocols: Dict, delta_crs: float) -> Tuple[float, Dict[str, float]]:
    n = len(protocols)
    if n <= 0:
        raise ValueError("protocols must be non-empty")
    don = float(delta_crs) / n
    contrib: Dict[str, float] = {}
    s = 0.0
    for name in protocols:
        c = round(don * float(crs_by_name.get(name, 0.0)), DECIMALS)
        contrib[name] = c
        s += c
    return round(s, DECIMALS), contrib


def compute_allocation_weights(protocols: Dict) -> Tuple[Dict[str, float], float]:
    alloc_values = {name: float(proto.get("alloc_max_pct", 0.0)) for name, proto in protocols.items()}
    total_alloc = sum(alloc_values.values())
    if total_alloc <= 0:
        raise ValueError("Sum of alloc_max_pct across strategy.protocols must be positive")
    weights = {name: (alloc / total_alloc) for name, alloc in alloc_values.items()}
    pds = sum(w * w for w in weights.values())
    return {k: round(v, DECIMALS) for k, v in weights.items()}, round(pds, DECIMALS)


def s_dvn(n_dvn: int, phi_dvn: float) -> float:
    """DVN risk: exp(-phi_dvn * max(0, n-1)). More required DVNs → lower risk."""
    return round(math.exp(-phi_dvn * max(0, int(n_dvn) - 1)), DECIMALS)


def compute_s_dvn(protocol: dict, params: dict) -> Tuple[float, int]:
    """
    Returns (lambda_dvn * S_DVN, n_dvn).
    When n_dvn is null/absent: returns (0.0, None) — not applicable.
    """
    n_dvn = protocol.get("n_dvn")
    if n_dvn is None:
        return 0.0, None
    phi_dvn = required_param(params, "phi_dvn")
    lambda_dvn = required_param(params, "lambda_dvn")
    score = s_dvn(int(n_dvn), phi_dvn)
    return round(lambda_dvn * score, DECIMALS), int(n_dvn)


def compute_s_cdiv(protocol: dict) -> Tuple[float, list]:
    """
    S_CDiv = Σ_j w²_div * Q_j  with equal weights w_div = 1/n.
    Returns (score, list of (name, category, Q_j) tuples) for display.
    0.0 when no counterparty_protocols are defined.
    """
    cps = protocol.get("counterparty_protocols")
    if not cps or not isinstance(cps, list) or len(cps) == 0:
        return 0.0, []
    n = len(cps)
    w = 1.0 / n
    total = 0.0
    details = []
    for cp in cps:
        name = cp.get("name", "unknown")
        cat = str(cp.get("category", "new_or_unaudited")).strip().lower()
        q = COUNTERPARTY_QUALITY_SCORES.get(cat, 1.0)
        total += w * w * q
        details.append((name, cat, q))
    return round(total, DECIMALS), details


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterparty Risk Score (CRS) calculator")
    parser.add_argument(
        "--strategy",
        default=STRATEGY_YAML_FLAG,
        help=f"Strategy YAML file name inside configs/strategies/ (default: {STRATEGY_YAML_FLAG})",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    strategy_path = base / "configs" / "strategies" / args.strategy
    protocols_dir = base / "configs" / "protocols"
    params_path = base / "configs" / "general" / "params.yaml"

    params, protocols, _ = load_inputs(strategy_path, protocols_dir, params_path)
    delta_crs = float(params["delta_crs"]) if "delta_crs" in params else required_param(params, "delta_pcr")
    omega_cdiv = required_param(params, "omega_cdiv")
    weights, pds = compute_allocation_weights(protocols)

    print(f"Using strategy file: {args.strategy}\n")
    print("Oracle Risk (ORS_i) and Counterparty Risk Score (CRS_i)")
    print("=" * 60)
    print("  S_ODiv,i = exp(-psi_oracle_div * max(0, n_i - 1)), n_i = oracle provider count")
    print(
        "  S_VDec,i = exp(-n_operators / N_ref)  |  "
        "S_VOps,i = 1 - exp(-tau_v * slashing_events / age_years)"
    )
    print(f"  S_CDiv,i = Σ_j w²_div * Q_j  (equal weights; Q_j by counterparty category)")
    print(f"  CRS_i includes omega_cdiv * S_CDiv,i (omega_cdiv={omega_cdiv})")

    crs_by_name: Dict[str, float] = {}
    for name, protocol in protocols.items():
        ors_i, ors_comp = compute_ors(protocol, params)
        crs_i, sodiv, detail = compute_crs(protocol, params, ors_i)
        s_cdiv_i, cdiv_details = compute_s_cdiv(protocol)
        cdiv_term = round(omega_cdiv * s_cdiv_i, DECIMALS)
        crs_i_full = round(crs_i + cdiv_term, DECIMALS)
        crs_by_name[name] = crs_i_full
        print(f"\n{name}")
        print("-" * 40)
        for comp_name, value in ors_comp.items():
            print(f"  {comp_name}: {value:.{DECIMALS}f}")
        print(f"  ORS_i: {ors_i:.{DECIMALS}f}")
        print(
            f"  n_oracle_providers: {detail['n_oracle_providers']}  "
            f"S_ODiv: {sodiv:.{DECIMALS}f}  (chi={detail['chi']}, psi={detail['psi_oracle_div']})"
        )
        if detail["validator_term"] > 0:
            print(
                f"  S_VDec: {detail['val_S_VDec']:.{DECIMALS}f}"
                f"  S_VOps: {detail['val_S_VOps']:.{DECIMALS}f}"
                f"  (discount={detail['val_discount']}, iota={detail['val_iota']:.{DECIMALS}f}, "
                f"slashing_events/age_years={detail['val_slashing_per_year']:.{DECIMALS}f}, "
                f"validator_term={detail['validator_term']:.{DECIMALS}f})"
            )
        if cdiv_details:
            n_cp = len(cdiv_details)
            cp_str = ", ".join(f"{cp_name}({cat}→Q={q})" for cp_name, cat, q in cdiv_details)
            print(f"  S_CDiv: {s_cdiv_i:.{DECIMALS}f}  n={n_cp}  [{cp_str}]")
        else:
            print(f"  S_CDiv: 0.0000  (no counterparty protocols)")
        print(f"  omega_cdiv * S_CDiv: {cdiv_term:.{DECIMALS}f}")
        dvn_term_i = detail.get("dvn_term", 0.0)
        n_dvn_i = detail.get("n_dvn")
        if n_dvn_i is not None:
            print(f"  n_dvn: {n_dvn_i}  lambda_dvn * S_DVN: {dvn_term_i:.{DECIMALS}f}")
        else:
            print(f"  n_dvn: null  (no cross-chain DVN messaging)")
        print(f"  CRS_i (incl. S_CDiv + DVN terms): {crs_i_full:.{DECIMALS}f}")

    crs_port, crs_contrib = compute_crs_portfolio(crs_by_name, protocols, delta_crs)

    print("\nCRS portfolio aggregate")
    print("=" * 60)
    print(
        f"  CRS_portfolio = (delta_crs / N) * Σ CRS_i, with delta_crs={delta_crs}"
        f"  (CRS_i includes oracle + validator + omega_cdiv * S_CDiv,i)"
    )
    for name, c in crs_contrib.items():
        print(f"    {name}: {c:.{DECIMALS}f}")
    print(f"  CRS_portfolio = {crs_port:.{DECIMALS}f}")

    print("\nAllocation concentration (HHI, informational)")
    print("=" * 60)
    print("  PDS = Σ w_i^2  (portfolio allocation HHI; lower = more diversified)")
    for name, w in weights.items():
        print(f"    {name}: w_i = {w:.{DECIMALS}f}")
    print(f"  PDS = {pds:.{DECIMALS}f}")

    print()
    print("=" * 60)
    print(f"Final CRS (portfolio): {crs_port:.{DECIMALS}f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
