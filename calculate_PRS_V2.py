#!/usr/bin/env python3
"""
Risk scoring (V2): PSR, concentration risk, and counterparty risk as separate aggregates.

Definitions (N = number of protocols in the strategy):

  PSR (portfolio protocol score)
      PSR = Σ_{i not asset} (γ / N) * PRS_i
      (assets: no PRS; counted in N)

  Concentration risk (independent of PSR)
      CR = Σ_i (α * w_i²) + ω * Σ_i θ_i
      w_i = AllocMax_i / P_tot,  θ_i = AllocMax_i / TVL_i

  Per-protocol Oracle Risk Score
      ORS_i = S_OH,i + S_OD,i
      S_OH,i = ρ * (1 - exp(-oracle_heartbeat_hours_i))
      (category: asset → ORS_i = 0; oracle fields not used)

  Oracle diversification risk (per protocol, fewer distinct oracle providers → higher term; not applied to assets)
      S_ODiv,i = exp(-ψ * max(0, n_i - 1))
      n_i = oracle provider count (protocol `oracle_diversification.provider_count`, else len(providers), else 1)

  Counterparty Risk Score (per protocol)
      CRS_i = ORS_i + χ * S_ODiv,i  (for assets: χ * S_ODiv not applied; ORS_i = 0)

  Portfolio counterparty aggregate (optional reporting)
      CRS_portfolio = (δ_crs / N) * Σ_i CRS_i

Strategy parameters: α `alpha`, ω `omega` (defaults to `beta` if `omega` omitted), γ `gamma`,
χ `chi`, ψ `psi_oracle_div`, δ_crs `delta_crs` (defaults to legacy `delta_pcr` if omitted).
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml")

# Default strategy YAML filename under configs/strategies/ (override with --strategy)
STRATEGY_YAML_FLAG = "ALL.yaml"
# All numeric results rounded to this many decimal places
DECIMALS = 4


def is_asset(protocol: dict) -> bool:
    return str(protocol.get("category", "protocol")).strip().lower() == "asset"


AUDITOR_TIERS = {
    # Tier 1 (Q = 1.0)
    "trail of bits": 1.0,
    "openzeppelin": 1.0,
    "sigma prime": 1.0,
    "consensys diligence": 1.0,
    "certora": 1.0,
    # Tier 2 (Q = 0.7)
    "quantstamp": 0.7,
    "certik": 0.7,
    "chainsecurity": 0.7,
    # Tier 3 (Q = 0.4)
    "hacken": 0.4,
    "peckshield": 0.4,
    "slowmist": 0.4,
}
DEFAULT_AUDITOR_Q = 0.2

_MS_TN_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def required_param(params: Dict, key: str) -> float:
    """Read a required numeric parameter from YAML."""
    if key not in params:
        raise KeyError(f"Missing required parameter: {key}")
    return float(params[key])


def load_yaml_file(path: Union[str, Path]) -> Dict:
    """Load a YAML file and return a dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict YAML content in: {path}")
    return data


def load_inputs(
    strategy_path: Union[str, Path],
    protocols_dir: Union[str, Path],
    params_path: Union[str, Path],
) -> Tuple[Dict, Dict, Dict]:
    """Load shared params, strategy config, and selected protocol configs."""
    strategy_data = load_yaml_file(strategy_path)
    general_data = load_yaml_file(params_path)
    params = dict(general_data.get("parameters", {}))
    # Keep backward compatibility: allow optional per-strategy parameter overrides.
    params.update(strategy_data.get("parameters", {}))
    portfolio = strategy_data.get("portfolio", {})
    strategy = strategy_data.get("strategy", {})
    selected = strategy.get("protocols", [])

    if not isinstance(selected, list) or len(selected) == 0:
        raise ValueError("strategy.protocols must be a non-empty list")

    protocols: Dict[str, Dict] = {}
    protocols_dir = Path(protocols_dir)
    for item in selected:
        name = item.get("name")
        alloc_max_pct = item.get("alloc_max_pct", 0.0)
        if not name:
            raise ValueError("Each strategy.protocols item needs a name")

        file_name = name.lower().replace(" ", "_") + ".yaml"
        proto_path = protocols_dir / file_name
        proto_data = load_yaml_file(proto_path)
        proto_data["alloc_max_pct"] = alloc_max_pct
        protocols[name] = proto_data

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
        raw = ms.get(field, "unknown")
        tn = parse_multisig_tn(str(raw) if raw is not None else None, ut, un)
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
    components = {
        "S_OH": s_oh(h, rho),
        "S_OD": s_od(d, xi),
    }
    ors = round(sum(components.values()), DECIMALS)
    return ors, components


def compute_prs(protocol: dict, params: dict) -> Tuple[float, Dict[str, float], Dict[str, float]]:
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
    s_ms, sms_details = compute_sms(protocol, params)
    components = {
        "S_Maturity": round(s_maturity_raw * mu, DECIMALS),
        "S_sigmaTVL": s_sigma_tvl(float(protocol["tvl_vol_pct"]), eta),
        "S_NAudits": s_n_audits(nq_audit),
        "S_TAudit": round(zeta * s_t_audit_raw, DECIMALS),
        "S_CE": s_ce(int(protocol["n_critical_exploits"]), kappa),
        "S_MS": round(nu * s_ms, DECIMALS),
        "Delta_Critical": round(float(protocol.get("Delta_Critical", 0.0)), DECIMALS),
    }
    prs = round(sum(components.values()), DECIMALS)
    return prs, components, sms_details


def _n_oracle_providers(protocol: dict) -> int:
    """
    Number of distinct oracle providers for diversification.
    Uses `oracle_diversification.provider_count` if set, else len(providers), else 1.
    """
    od = protocol.get("oracle_diversification") or {}
    if not isinstance(od, dict):
        return 1
    n = od.get("provider_count")
    if n is not None:
        try:
            return max(1, int(n))
        except (TypeError, ValueError):
            pass
    prov = od.get("providers")
    if isinstance(prov, list) and len(prov) > 0:
        return max(1, len(prov))
    return 1


def s_odiv(n_providers: int, psi: float) -> float:
    """
    Oracle diversification term: single provider (n=1) → 1.0; more providers → lower (bounded in (0,1]).
    S_ODiv = exp(-ψ * max(0, n - 1)).
    """
    psi = float(psi)
    return round(math.exp(-psi * max(0, int(n_providers) - 1)), DECIMALS)


def compute_crs(protocol: dict, params: dict, ors_i: float) -> Tuple[float, float, Dict[str, float]]:
    """
    CRS_i = ORS_i + χ * S_ODiv,i (no oracle terms for category: asset)
    Returns (CRS_i, S_ODiv, detail dict).
    """
    chi = required_param(params, "chi")
    psi = required_param(params, "psi_oracle_div")
    if is_asset(protocol):
        crs = round(ors_i, DECIMALS)
        detail = {
            "n_oracle_providers": 0,
            "S_ODiv": 0.0,
            "chi": chi,
            "psi_oracle_div": psi,
        }
        return crs, 0.0, detail
    n = _n_oracle_providers(protocol)
    sodiv = s_odiv(n, psi)
    crs = round(ors_i + chi * sodiv, DECIMALS)
    detail = {
        "n_oracle_providers": n,
        "S_ODiv": sodiv,
        "chi": chi,
        "psi_oracle_div": psi,
    }
    return crs, sodiv, detail


def compute_psr(prs_by_name: Dict[str, float], protocols: Dict, gamma: float) -> Tuple[float, Dict[str, float], int]:
    """PSR = Σ (γ/N) PRS_i over protocol lines only; assets counted in N but contribute 0."""
    n = len(protocols)
    if n <= 0:
        raise ValueError("protocols must be non-empty")
    gon = float(gamma) / n
    contrib: Dict[str, float] = {}
    s = 0.0
    for name, protocol in protocols.items():
        if is_asset(protocol):
            contrib[name] = 0.0
            continue
        prs = float(prs_by_name.get(name, 0.0))
        c = round(gon * prs, DECIMALS)
        contrib[name] = c
        s += c
    return round(s, DECIMALS), contrib, n


def compute_concentration_risk(
    protocols: Dict,
    p_tot: float,
    alpha: float,
    omega: float,
) -> Tuple[float, float, float, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    CR = Σ (α w_i²) + ω Σ θ_i.
    Returns (cr, term_alpha_w2, term_omega_theta, w_i, theta_i, contrib_w2, contrib_omega_theta).
    """
    if p_tot <= 0:
        raise ValueError("portfolio.p_tot must be positive")
    if len(protocols) <= 0:
        raise ValueError("protocols must be non-empty")
    w_i: Dict[str, float] = {}
    theta_i: Dict[str, float] = {}
    contrib_w2: Dict[str, float] = {}
    contrib_omega_theta: Dict[str, float] = {}
    sum_theta = 0.0
    sum_w2 = 0.0
    for name, proto in protocols.items():
        alloc_max_pct = float(proto.get("alloc_max_pct", 0.0))
        tvl = float(proto.get("tvl", 1.0)) or 1.0
        alloc_max = (alloc_max_pct / 100.0) * p_tot
        w = alloc_max / p_tot
        theta = alloc_max / tvl if tvl > 0 else 0.0
        w_i[name] = round(w, DECIMALS)
        theta_i[name] = round(theta, DECIMALS)
        aw2 = alpha * w * w
        ot = omega * theta
        contrib_w2[name] = round(aw2, DECIMALS)
        contrib_omega_theta[name] = round(ot, DECIMALS)
        sum_w2 += aw2
        sum_theta += theta
    term_alpha_w2 = round(sum_w2, DECIMALS)
    term_omega_theta = round(omega * sum_theta, DECIMALS)
    cr = round(term_alpha_w2 + term_omega_theta, DECIMALS)
    return cr, term_alpha_w2, term_omega_theta, w_i, theta_i, contrib_w2, contrib_omega_theta


def compute_crs_portfolio(
    crs_by_name: Dict[str, float],
    protocols: Dict,
    delta_crs: float,
) -> Tuple[float, Dict[str, float]]:
    """CRS_portfolio = (δ_crs / N) * Σ CRS_i."""
    n = len(protocols)
    if n <= 0:
        raise ValueError("protocols must be non-empty")
    don = float(delta_crs) / n
    contrib: Dict[str, float] = {}
    s = 0.0
    for name in protocols:
        crs = float(crs_by_name.get(name, 0.0))
        c = round(don * crs, DECIMALS)
        contrib[name] = c
        s += c
    return round(s, DECIMALS), contrib


def main() -> None:
    parser = argparse.ArgumentParser(description="PRS V2: PSR, concentration, counterparty (CRS)")
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

    print(f"Using strategy file: {args.strategy}")
    print()

    params, protocols, portfolio = load_inputs(strategy_path, protocols_dir, params_path)
    p_tot = float(portfolio.get("p_tot", 1.0))
    alpha = required_param(params, "alpha")
    omega = float(params["omega"]) if "omega" in params else required_param(params, "beta")
    gamma = required_param(params, "gamma")
    delta_crs = float(params["delta_crs"]) if "delta_crs" in params else required_param(params, "delta_pcr")

    print("Parameters (V2-relevant):")
    for k in (
        "alpha",
        "omega",
        "beta",
        "gamma",
        "chi",
        "psi_oracle_div",
        "delta_crs",
        "delta_pcr",
    ):
        if k in params:
            v = params[k]
            if isinstance(v, float):
                print(f"  {k}: {round(v, DECIMALS):.{DECIMALS}f}")
            else:
                print(f"  {k}: {v}")
    print(f"  portfolio.p_tot: {round(p_tot, DECIMALS):.{DECIMALS}f}")
    print(f"  (omega defaults to beta if `omega` not in YAML)")
    print()

    # Per-protocol PRS
    prs_by_name: Dict[str, float] = {}
    print("Protocol Risk Score (PRS_i) — includes Delta_Critical")
    print("=" * 60)
    for name, protocol in protocols.items():
        print(f"\n{name}")
        print("-" * 40)
        if is_asset(protocol):
            print("  category: asset — no PRS; excluded from PSR sum; counted in N")
            continue
        prs, components, sms_details = compute_prs(protocol, params)
        prs_by_name[name] = prs
        for comp_name, value in components.items():
            print(f"  {comp_name}: {value:.{DECIMALS}f}")
        print(f"  PRS_i: {prs:.{DECIMALS}f}")
    print()

    # PSR
    psr, psr_contrib, n_proto = compute_psr(prs_by_name, protocols, gamma)
    print("PSR (portfolio protocol score)")
    print("=" * 60)
    print(
        f"  PSR = Σ (γ/N) * PRS_i over protocol lines only; "
        f"N = {n_proto} (includes assets). γ = {gamma}"
    )
    for name, c in psr_contrib.items():
        print(f"    {name}: (γ/N)*PRS_i = {c:.{DECIMALS}f}")
    print(f"  PSR = {psr:.{DECIMALS}f}")
    print()

    # Concentration (α, ω)
    cr, term_w2, term_oth, w_i, theta_i, cw2, cot = compute_concentration_risk(
        protocols, p_tot, alpha, omega
    )
    print("Concentration risk (independent of PSR)")
    print("=" * 60)
    print(f"  CR = Σ (α * w_i²) + ω * Σ θ_i")
    print(f"  α = {alpha}, ω = {omega}")
    print(f"\n  w_i = AllocMax_i / P_tot:")
    for name, w in w_i.items():
        pct = protocols[name].get("alloc_max_pct", 0)
        print(f"    {name}: {pct}% → w_i = {w:.{DECIMALS}f}")
    print(f"\n  θ_i = AllocMax_i / TVL_i:")
    for name, th in theta_i.items():
        print(f"    {name}: {th:.{DECIMALS}f}")
    print(f"\n  Per-protocol:")
    print(f"    {'Protocol':<14}  {'α*w_i²':<12}  {'ω*θ_i':<12}")
    for name in w_i:
        print(f"    {name:<14}  {cw2[name]:.{DECIMALS}f}    {cot[name]:.{DECIMALS}f}")
    print(f"    Σ (α*w_i²) = {term_w2:.{DECIMALS}f}    ω * Σ θ_i = {term_oth:.{DECIMALS}f}")
    print(f"  CR = {term_w2:.{DECIMALS}f} + {term_oth:.{DECIMALS}f} = {cr:.{DECIMALS}f}")
    print()

    # ORS + CRS
    ors_by_name: Dict[str, float] = {}
    crs_by_name: Dict[str, float] = {}
    print("Oracle Risk (ORS_i) and Counterparty Risk Score (CRS_i = ORS_i + χ * S_ODiv)")
    print("=" * 60)
    print("  S_ODiv,i = exp(-ψ * max(0, n_i - 1)), n_i = oracle provider count")
    for name, protocol in protocols.items():
        ors_i, ors_comp = compute_ors(protocol, params)
        ors_by_name[name] = ors_i
        crs_i, sodiv, odiv_detail = compute_crs(protocol, params, ors_i)
        crs_by_name[name] = crs_i
        print(f"\n{name}")
        print("-" * 40)
        for comp_name, value in ors_comp.items():
            print(f"  {comp_name}: {value:.{DECIMALS}f}")
        print(f"  ORS_i: {ors_i:.{DECIMALS}f}")
        print(
            f"  n_oracle_providers: {odiv_detail['n_oracle_providers']}  "
            f"S_ODiv: {sodiv:.{DECIMALS}f}  (χ={odiv_detail['chi']}, ψ={odiv_detail['psi_oracle_div']})"
        )
        print(f"  CRS_i: {crs_i:.{DECIMALS}f}")
    print()

    crs_port, crs_contrib = compute_crs_portfolio(crs_by_name, protocols, delta_crs)
    print("Portfolio counterparty aggregate (optional)")
    print("=" * 60)
    print(f"  CRS_portfolio = (δ_crs / N) * Σ CRS_i   with N = {n_proto}, δ_crs = {delta_crs}")
    for name, c in crs_contrib.items():
        print(f"    {name}: (δ_crs/N)*CRS_i = {c:.{DECIMALS}f}")
    print(f"  CRS_portfolio = {crs_port:.{DECIMALS}f}")
    print()

    print("Summary (V2)")
    print("=" * 60)
    print(f"  PSR                  = {psr:.{DECIMALS}f}")
    print(f"  Concentration risk   = {cr:.{DECIMALS}f}")
    print(f"  CRS_portfolio        = {crs_port:.{DECIMALS}f}")
    print()


if __name__ == "__main__":
    main()
