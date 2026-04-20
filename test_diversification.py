"""
Quick test: proves that concentration raises CVaR structural and
diversification lowers it, using the current jump model.

Run:  python test_diversification.py
"""
import numpy as np
from run import simulate_losses, var_cvar
from calculate_PRS import load_yaml_file
from pathlib import Path

BASE = Path(__file__).parent
sim = load_yaml_file(BASE / "configs" / "general" / "params_simulation.yaml")

p = {
    "n_paths":           100_000,
    "step_days":         1,
    "horizon_days":      30,
    "volatility_annual": 0.30,   # realistic crypto vol
    "apy_min":           0.05,
    "apy_max":           0.15,
    "nu":                4.0,
    "jump_volatility":   float(sim.get("jump_volatility", 0.3)),
    "confidence":        0.95,
}
base_jp  = float(sim.get("jump_probability", 0.00055))
PRS      = 2.5          # same PRS for every protocol in both scenarios
seed     = 42

def run(label, jump_model):
    rng = np.random.default_rng(seed)
    r1  = np.random.default_rng(rng.integers(0, 2**32 - 1))
    r2  = np.random.default_rng(rng.integers(0, 2**32 - 1))
    nj  = simulate_losses(p, r1, jump_model, jump_probability_override=0.0)
    wj  = simulate_losses(p, r2, jump_model)
    _, cvar_nj = var_cvar(nj, p["confidence"])
    _, cvar_wj = var_cvar(wj, p["confidence"])
    print(f"  {label}")
    print(f"    CVaR market     (no jumps):   {cvar_nj:.2%}")
    print(f"    CVaR structural (with jumps): {cvar_wj:.2%}")
    print(f"    CVaR sum:                     {cvar_nj+cvar_wj:.2%}")
    print()

# Scenario A: 1 protocol, 100% allocation
jm_1 = [{"weight": 1.0,
          "jump_probability": base_jp * PRS * 1.0,   # jp × prs × weight
          "jump_mean_severity": 1.0}]

# Scenario B: 4 equal protocols, 25% each — same PRS
w = 0.25
jm_4 = [{"weight": w,
          "jump_probability": base_jp * PRS * w,
          "jump_mean_severity": 1.0}
         for _ in range(4)]

print("=== Diversification test ===")
print(f"  base_jp={base_jp}, PRS={PRS}, horizon=30d, vol=30%, n=100k paths\n")
run("1 protocol  — weight=100%  (concentrated)", jm_1)
run("4 protocols — weight=25% each (diversified)", jm_4)
print("Expected: CVaR structural should be LOWER for the diversified case.")
