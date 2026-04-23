#!/usr/bin/env python3
"""
Protocol Risk Score (PRS) calculator.

Definitions (N = number of line items in the strategy, including assets):

  PRS_i = mu*S_Maturity,i + S_sigmaTVL,i + S_NAudits,i + zeta*S_TAudit,i + S_CE,i + nu*S_MS,i
          + Delta_Critical,i + mu_bd*S_BD,i + mu_bdp*S_BDP,i
        (only for category protocol; assets are skipped)

  S_BD,i  = 1 - exp(-kappa_bd * bad_debt_ratio_i)
             bad_debt_ratio = bad_debt_usd / total_loans_usd; null → 0 (non-lending)
  S_BDP,i = (1 - exp(-eta_ltv * max_ltv_i)) * delta_sf,i
             delta_sf,i = iota_sf if has_safety_module else 1.0; null → 0 (non-lending)

  PSR   = Σ_{i not asset} gamma * w_i * PRS_i
  w_i   = alloc_max_pct_i / (Σ_j alloc_max_pct_j)  over all strategy lines (allocation share)

Assets (category: asset, e.g. cbBTC) are not scored (PRS_i omitted); they still shape weights w via their alloc_max_pct.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml

#STRATEGY_YAML_FLAG = "default_strategy.yaml"
#STRATEGY_YAML_FLAG = "Morpho_Gauntlet_Core.yaml"
#STRATEGY_YAML_FLAG = "Morpho_Gauntlet_PstExp.yaml"
#STRATEGY_YAML_FLAG = "Morpho_Steakhouse.yaml"
#STRATEGY_YAML_FLAG = "ALL.yaml"
STRATEGY_YAML_FLAG = "one.yaml"


DECIMALS = 4

AUDITOR_TIERS = {
    "trail of bits": 1.0,
    "openzeppelin": 1.0,
    "sigma prime": 1.0,
    "consensys diligence": 1.0,
    "certora": 1.0,
    "quantstamp": 0.7,
    "certik": 0.7,
    "chainsecurity": 0.7,
    "hacken": 0.4,
    "peckshield": 0.4,
    "slowmist": 0.4,
}
DEFAULT_AUDITOR_Q = 0.2
_MS_TN_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


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


def s_maturity(age_years: float, lambda_: float) -> float:
    return round(math.exp(-lambda_ * age_years), DECIMALS)


def s_sigma_tvl(tvl_vol_pct: float, eta: float) -> float:
    return round(1.0 - math.exp(-eta * tvl_vol_pct), DECIMALS)


def get_auditor_quality(auditor_name: str) -> float:
    return AUDITOR_TIERS.get(auditor_name.strip().lower(), DEFAULT_AUDITOR_Q)


def compute_nq_audit(protocol: dict) -> float:
    audits = protocol.get("audits_by_auditor", [])
    if audits:
        total = 0.0
        for item in audits:
            auditor = str(item.get("auditor", "")).strip()
            n_audits = float(item.get("n_audits", 0))
            total += n_audits * get_auditor_quality(auditor)
        return round(total, DECIMALS)
    return round(float(protocol.get("n_audits", 0)), DECIMALS)


def s_n_audits(nq_audit: float) -> float:
    return round(1.0 / (1.0 + nq_audit), DECIMALS)


def s_t_audit(months_since_audit: float, t_max_months: float) -> float:
    if t_max_months <= 0:
        return round(1.0, DECIMALS)
    return round(min(months_since_audit / t_max_months, 1.0), DECIMALS)


def s_ce(n_critical_exploits: int, kappa: float) -> float:
    return round(1.0 - math.exp(-kappa * n_critical_exploits), DECIMALS)


def parse_multisig_tn(value: Optional[str], unknown_t: int, unknown_n: int) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    upper = s.upper()
    if upper in ("N/A", "NA"):
        return None
    if upper == "UNKNOWN" or upper.startswith("UNKNOWN"):
        return max(1, unknown_t), max(1, unknown_n)
    m = _MS_TN_RE.match(s)
    if not m:
        return max(1, unknown_t), max(1, unknown_n)
    t, n = int(m.group(1)), int(m.group(2))
    if n <= 0 or t <= 0 or t > n:
        return max(1, unknown_t), max(1, unknown_n)
    return t, n


def compute_sms(protocol: dict, params: dict) -> Tuple[float, Dict[str, float]]:
    phi = required_param(params, "phi")
    w_u = required_param(params, "w_u")
    w_f = required_param(params, "w_f")
    w_p = required_param(params, "w_p")
    w_m = required_param(params, "w_m")
    ut = int(required_param(params, "sms_unknown_t"))
    un = int(required_param(params, "sms_unknown_n"))

    ms = protocol.get("multisig_requirements") or {}
    spec: List[Tuple[str, float, str]] = [
        ("contract_changes", w_u, "U"),
        ("funds_movement", w_f, "F"),
        ("parameter_changes", w_p, "P"),
        ("minting", w_m, "M"),
    ]
    inner = 0.0
    details: Dict[str, float] = {}
    for field, w, label in spec:
        tn = parse_multisig_tn(ms.get(field, "unknown"), ut, un)
        if tn is None:
            details[f"term_{label}"] = 0.0
            continue
        t_i, n_i = tn
        term = w * (t_i / (float(n_i) ** 2))
        inner += term
        details[f"term_{label}"] = round(term, DECIMALS)

    details["inner_sum"] = round(inner, DECIMALS)
    sms = 1.0 - math.exp(-phi * inner)
    details["S_MS_raw"] = round(sms, DECIMALS)
    return round(sms, DECIMALS), details


def s_bd(bad_debt_ratio: float, kappa_bd: float) -> float:
    """Bad debt incidence: 1 - exp(-kappa_bd * ratio). Higher ratio → higher risk."""
    return round(1.0 - math.exp(-kappa_bd * bad_debt_ratio), DECIMALS)


def s_bdp(max_ltv: float, eta_ltv: float, has_safety_module: bool, iota_sf: float) -> float:
    """Bad debt prevention: structural risk from aggressive LTV, discounted by safety module."""
    delta_sf = iota_sf if has_safety_module else 1.0
    return round((1.0 - math.exp(-eta_ltv * max_ltv)) * delta_sf, DECIMALS)


def compute_bad_debt_scores(protocol: dict, params: dict) -> Tuple[float, float, Dict]:
    """
    Returns (mu_bd * S_BD, mu_bdp * S_BDP, details).
    Both terms are 0 when the protocol has no bad_debt_ratio / bad_debt_prevention fields (non-lending).
    """
    kappa_bd = required_param(params, "kappa_bd")
    eta_ltv  = required_param(params, "eta_ltv")
    iota_sf  = required_param(params, "iota_sf")
    mu_bd    = required_param(params, "mu_bd")
    mu_bdp   = required_param(params, "mu_bdp")

    bd_ratio = protocol.get("bad_debt_ratio")
    bdp      = protocol.get("bad_debt_prevention")

    bd_term, bdp_term = 0.0, 0.0
    details: Dict = {"S_BD": 0.0, "S_BDP": 0.0, "bd_ratio": None, "max_ltv": None, "has_safety_module": None}

    if bd_ratio is not None:
        score = s_bd(float(bd_ratio), kappa_bd)
        bd_term = round(mu_bd * score, DECIMALS)
        details["S_BD"] = score
        details["bd_ratio"] = float(bd_ratio)

    if bdp is not None and isinstance(bdp, dict):
        max_ltv = float(bdp.get("max_ltv", 0.0))
        has_sf  = bool(bdp.get("has_safety_module", False))
        score   = s_bdp(max_ltv, eta_ltv, has_sf, iota_sf)
        bdp_term = round(mu_bdp * score, DECIMALS)
        details["S_BDP"] = score
        details["max_ltv"] = max_ltv
        details["has_safety_module"] = has_sf

    return bd_term, bdp_term, details


def compute_prs(protocol: dict, params: dict) -> Tuple[float, Dict[str, float]]:
    lambda_ = required_param(params, "lambda")
    eta = required_param(params, "eta")
    kappa = required_param(params, "kappa")
    mu = required_param(params, "mu")
    zeta = required_param(params, "zeta")
    nu = required_param(params, "nu")

    t_max_months = float(protocol["age_years"]) * 12.0
    s_maturity_raw = s_maturity(float(protocol["age_years"]), lambda_)
    nq_audit = compute_nq_audit(protocol)
    s_t_audit_raw = s_t_audit(float(protocol["months_since_audit"]), t_max_months)
    s_ms, _ = compute_sms(protocol, params)

    dc_flag = float(protocol.get("Delta_Critical", 0.0))
    delta_critical_term = (
        round(required_param(params, "delta_critical"), DECIMALS) if dc_flag > 0.0 else 0.0
    )

    bd_term, bdp_term, _ = compute_bad_debt_scores(protocol, params)

    components = {
        "S_Maturity":    round(s_maturity_raw * mu, DECIMALS),
        "S_sigmaTVL":    s_sigma_tvl(float(protocol["tvl_vol_pct"]), eta),
        "S_NAudits":     s_n_audits(nq_audit),
        "S_TAudit":      round(zeta * s_t_audit_raw, DECIMALS),
        "S_CE":          s_ce(int(protocol["n_critical_exploits"]), kappa),
        "S_MS":          round(nu * s_ms, DECIMALS),
        "Delta_Critical": delta_critical_term,
        "S_BD":          bd_term,
        "S_BDP":         bdp_term,
    }
    prs = round(sum(components.values()), DECIMALS)
    return prs, components


def compute_psr(prs_by_name: Dict[str, float], protocols: Dict, gamma: float) -> Tuple[float, Dict[str, float], int]:
    n = len(protocols)
    if n <= 0:
        raise ValueError("protocols must be non-empty")
    total_alloc = sum(float(p.get("alloc_max_pct", 0.0)) for p in protocols.values())
    if total_alloc <= 0:
        raise ValueError("Sum of alloc_max_pct across strategy lines must be positive")
    gamma_f = float(gamma)
    contrib: Dict[str, float] = {}
    s = 0.0
    for name, protocol in protocols.items():
        if is_asset(protocol):
            contrib[name] = 0.0
            continue
        w_i = float(protocol.get("alloc_max_pct", 0.0)) / total_alloc
        c = round(gamma_f * w_i * float(prs_by_name.get(name, 0.0)), DECIMALS)
        contrib[name] = c
        s += c
    return round(s, DECIMALS), contrib, n


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol Risk Score (PRS) calculator")
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
    gamma = required_param(params, "gamma")

    prs_by_name: Dict[str, float] = {}
    print(f"Using strategy file: {args.strategy}\n")
    print("Protocol Risk Score (PRS_i)")
    print("=" * 50)
    for name, protocol in protocols.items():
        print(f"\n{name}")
        print("-" * 32)
        if is_asset(protocol):
            print("  category: asset — no PRS; alloc_max_pct still affects peer allocation weights w_i")
            continue
        prs, components = compute_prs(protocol, params)
        prs_by_name[name] = prs
        for comp_name, value in components.items():
            print(f"  {comp_name}: {value:.{DECIMALS}f}")
        print(f"  PRS_i: {prs:.{DECIMALS}f}")

    psr, psr_contrib, n_proto = compute_psr(prs_by_name, protocols, gamma)
    print("\nPSR (portfolio protocol score)")
    print("=" * 50)
    print(
        f"  PSR = Σ γ * w_i * PRS_i with w_i = alloc_max_pct_i / (Σ alloc_max_pct); "
        f"lines = {n_proto}. γ = {gamma}"
    )
    for name, c in psr_contrib.items():
        print(f"    {name}: {c:.{DECIMALS}f}")
    print(f"  PSR = {psr:.{DECIMALS}f}")


if __name__ == "__main__":
    main()
