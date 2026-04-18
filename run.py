"""
Tail risk simulation: one script.
Set STRATEGY_YAML_FLAG below for a quick default switch, then run: python run.py
Optional override: python run.py --strategy <strategy>.yaml

Architecture:
  configs/general/params_simulation.yaml  →  shared simulation parameters
  configs/strategies/<strategy>.yaml      →  strategy.simulation parameters
  run.py       →  load params, simulate NAV paths (Student-t + jump), compute VaR/CVaR, print
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from calculate_PRS import load_inputs as load_prs_inputs, compute_prs


BASE_DIR = Path(__file__).parent
DEFAULT_SHARED_SIM_PATH = BASE_DIR / "configs" / "general" / "params_simulation.yaml"
#STRATEGY_YAML_FLAG = "Morpho_Gauntlet_PstExp.yaml"
#STRATEGY_YAML_FLAG = "default_strategy.yaml"
STRATEGY_YAML_FLAG = "Leveraged_Stake.yaml"
#STRATEGY_YAML_FLAG = "Leveraged_Stake_hedged.yaml"
#STRATEGY_YAML_FLAG = "Morpho_Gauntlet_Core.yaml"
#STRATEGY_YAML_FLAG = "Morpho_Gauntlet_PstExp.yaml"
#STRATEGY_YAML_FLAG = "Morpho_Steakhouse.yaml"
#STRATEGY_YAML_FLAG = "ALL.yaml"
#STRATEGY_YAML_FLAG = "one.yaml"

def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_params(strategy_path, shared_path=DEFAULT_SHARED_SIM_PATH):
    shared = load_yaml(shared_path)
    strategy_yaml = load_yaml(strategy_path)
    strategy = strategy_yaml.get("strategy", {})
    strategy_sim = strategy.get("simulation", {})

    if not isinstance(shared, dict):
        raise ValueError(f"Expected dict in shared simulation YAML: {shared_path}")
    if not isinstance(strategy_sim, dict):
        raise ValueError(f"Expected strategy.simulation dict in strategy YAML: {strategy_path}")

    params = {**shared, **strategy_sim}
    required = (
        "nu",
        "horizon_days",
        "step_days",
        "n_paths",
        "confidence",
        "seed",
        "volatility_annual",
        "apy_min",
        "apy_max",
        "jump_probability",
        "jump_volatility",
        "strategy_name",
    )
    missing = [k for k in required if k not in params]
    if missing:
        raise KeyError(f"Missing simulation parameters: {', '.join(missing)}")
    return params


def build_protocol_jump_model(strategy_path, base_jump_probability):
    params_path = BASE_DIR / "configs" / "general" / "params.yaml"
    protocols_dir = BASE_DIR / "configs" / "protocols"
    prs_params, protocols, _ = load_prs_inputs(strategy_path, protocols_dir, params_path)

    jump_model = []
    for name, protocol in protocols.items():
        category = str(protocol.get("category", "protocol")).strip().lower()
        if category == "asset":
            multiplier = float(protocol["monthly_volatility"])
            multiplier_source = "monthly_volatility"
        else:
            multiplier, _ = compute_prs(protocol, prs_params)
            multiplier_source = "PRS_i"

        alloc_pct = float(protocol.get("alloc_max_pct", 0.0))
        weight = max(0.0, alloc_pct / 100.0)
        jump_probability_i = float(np.clip(base_jump_probability * multiplier, 0.0, 1.0))
        jump_mean_severity_i = float(np.clip(weight, 0.0, 1.0))
        jump_model.append(
            {
                "name": name,
                "category": category,
                "multiplier": multiplier,
                "multiplier_source": multiplier_source,
                "weight": weight,
                "jump_probability": jump_probability_i,
                "jump_mean_severity": jump_mean_severity_i,
            }
        )

    if not jump_model:
        raise ValueError("No protocols found to build protocol jump model.")
    return jump_model


def student_t_return(rng, vol, nu, drift, dt, n):
    """One step return: mean = drift*dt, scale = vol*sqrt(dt), Student-t(nu)."""
    scale = vol * np.sqrt(dt) * np.sqrt((nu - 2) / nu)
    return drift * dt + scale * rng.standard_t(nu, size=n)


def jump_loss(rng, prob, mean_sev, sev_vol, n):
    """Jump loss per path: 0 w.p. (1-p), else lognormal(mean_sev, sev_vol)."""
    u = rng.random(n)
    jump = u < prob
    tiny = 1e-12
    s2 = np.log(1 + (sev_vol / (mean_sev + tiny)) ** 2)
    m = np.log(mean_sev + tiny) - 0.5 * s2
    sev = np.clip(rng.lognormal(m, np.sqrt(s2), n), 0.0, 1.0)
    return np.where(jump, sev, 0.0)


def simulate_losses(p, rng, protocol_jump_model, jump_probability_override=None):
    """Run Monte Carlo; return array of terminal losses (fraction of NAV)."""
    n_paths = p["n_paths"]
    dt = p["step_days"] / 365.25
    n_steps = max(1, p["horizon_days"] // p["step_days"])
    drift = (p["apy_min"] + p["apy_max"]) / 2.0
    vol, nu = p["volatility_annual"], p["nu"]
    jv = p["jump_volatility"]

    nav = np.ones((n_paths, n_steps + 1), dtype=np.float64)
    for t in range(n_steps):
        ret = student_t_return(rng, vol, nu, drift, dt, n_paths)
        total_jump = np.zeros(n_paths, dtype=np.float64)
        for cfg in protocol_jump_model:
            jp_i = 0.0 if jump_probability_override == 0.0 else cfg["jump_probability"]
            jm_i = cfg["jump_mean_severity"]
            total_jump += cfg["weight"] * jump_loss(rng, jp_i, jm_i, jv, n_paths)
        nav[:, t + 1] = nav[:, t] * np.clip(1.0 + ret - total_jump, 1e-12, None)
    return np.clip(1.0 - nav[:, -1], 0.0, 1.0)


def var_cvar(losses, confidence):
    """VaR = quantile at confidence; CVaR = mean of losses >= VaR."""
    var = np.quantile(losses, confidence)
    tail = losses[losses >= var]
    cvar = float(np.mean(tail)) if len(tail) > 0 else var
    return float(var), cvar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tail risk simulation")
    parser.add_argument(
        "--strategy-file",
        "--strategy",
        dest="strategy_file",
        default=STRATEGY_YAML_FLAG,
        help=(
            "Strategy YAML file name (in configs/strategies/) or full path "
            f"(default: {STRATEGY_YAML_FLAG})"
        ),
    )
    args = parser.parse_args()

    strategy_arg = Path(args.strategy_file)
    strategy_path = (
        strategy_arg
        if strategy_arg.is_absolute() or strategy_arg.parent != Path(".")
        else BASE_DIR / "configs" / "strategies" / strategy_arg
    )
    print(f"Using strategy file: {strategy_path}")
    p = load_params(strategy_path=strategy_path)
    protocol_jump_model = build_protocol_jump_model(strategy_path, p["jump_probability"])
    seed = p.get("seed")
    rng_base = np.random.default_rng(seed)

    # Run 1: Student-t only (no jumps)
    rng_no_jump = np.random.default_rng(rng_base.integers(0, 2**32 - 1))
    losses_no_jump = simulate_losses(p, rng_no_jump, protocol_jump_model, jump_probability_override=0.0)
    var_no_jump, cvar_no_jump = var_cvar(losses_no_jump, p["confidence"])

    # Run 2: Student-t + jumps (as in YAML)
    rng_with_jump = np.random.default_rng(rng_base.integers(0, 2**32 - 1))
    losses_with_jump = simulate_losses(p, rng_with_jump, protocol_jump_model, jump_probability_override=None)
    var_with_jump, cvar_with_jump = var_cvar(losses_with_jump, p["confidence"])

    # Requested "sum both" (note: this is a simple arithmetic sum of the reported risk numbers)
    var_sum = var_no_jump + var_with_jump
    cvar_sum = cvar_no_jump + cvar_with_jump

    print(f"Strategy: {p['strategy_name']}")
    print(f"Paths: {p['n_paths']}, Confidence: {p['confidence']:.0%}")
    print(f"Horizon: {p['horizon_days']} days ({p['step_days']}-day steps)")
    print(f"Jump base probability constant: {p['jump_probability']}")
    print(f"Jump volatility (shared): {p['jump_volatility']}")
    print()
    print("Per-protocol jump inputs:")
    print("  Protocol            Category   Multiplier(source)          jump_prob_i   jump_mean_severity_i")
    for cfg in protocol_jump_model:
        print(
            f"  {cfg['name']:<18}  {cfg['category']:<8}  {cfg['multiplier']:<7.4f} ({cfg['multiplier_source']:<17})  "
            f"{cfg['jump_probability']:<12.6f}  {cfg['jump_mean_severity']:<.4f}"
        )
    print()
    print(f"[No jumps: jump_probability=0]")
    print(f"VaR{p['confidence']:.0%}:  {var_no_jump:.2%} of NAV")
    print(f"CVaR{p['confidence']:.0%} (Tail Risk Severity): {cvar_no_jump:.2%} of NAV")
    print()
    print(
        f"[With protocol-level jumps: jump_probability_i = {p['jump_probability']} * "
        "multiplier (PRS_i for protocols, monthly_volatility for assets)]"
    )
    print(f"VaR{p['confidence']:.0%}:  {var_with_jump:.2%} of NAV")
    print(f"CVaR{p['confidence']:.0%} (Tail Risk Severity): {cvar_with_jump:.2%} of NAV")
    print()
    print(f"[Sum both arithmetic]")
    print(f"VaR_sum{p['confidence']:.0%}:  {var_sum:.2%} of NAV")
    print(f"CVaR_sum{p['confidence']:.0%}:  {cvar_sum:.2%} of NAV")
