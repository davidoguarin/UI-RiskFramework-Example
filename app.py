"""
Risk Analysis Framework — Streamlit dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from calculate_CRS import (
    COUNTERPARTY_QUALITY_SCORES,
    compute_allocation_weights,
    compute_crs,
    compute_crs_portfolio,
    compute_ors,
    compute_s_cdiv,
    compute_s_dvn,
)
from calculate_PRS import (
    compute_bad_debt_scores,
    compute_prs,
    compute_psr,
    is_asset,
    load_yaml_file,
    required_param,
)
from run import simulate_losses, var_cvar
from data_fetcher import (
    fetch_protocol_tvl,
    fetch_stablecoin_price_history,
    fetch_top_usde_pools,
    fetch_usde_price_coingecko,
    find_stablecoin_id,
)

PROTOCOLS_DIR = BASE / "configs" / "protocols"
PARAMS_PATH   = BASE / "configs" / "general" / "params.yaml"
SIM_PARAMS_PATH = BASE / "configs" / "general" / "params_simulation.yaml"

DISPLAY_NAMES: dict[str, str] = {
    "aave":               "Aave",
    "cbbtc":              "cbBTC",
    "clearpool":          "Clearpool",
    "compound":           "Compound",
    "ethena":             "Ethena",
    "euler":              "Euler",
    "fluid":              "Fluid",
    "idle":               "Idle",
    "lido":               "Lido",
    "maple":              "Maple",
    "morpho_blue":        "Morpho Blue",
    "pendle":             "Pendle",
    "resolv_post_exploit":"Resolv (Post-Exploit)",
    "resolv_pre_exploit": "Resolv (Pre-Exploit)",
    "silo":               "Silo",
    "wbtc":               "WBTC",
}

PRS_META: dict[str, tuple[str, str]] = {
    "S_Maturity":     ("Protocol Maturity",      "Age-based maturity risk — lower = more established"),
    "S_sigmaTVL":     ("TVL Volatility",          "TVL instability risk"),
    "S_NAudits":      ("Audit Coverage",          "Inverse of weighted audit count — lower = more audits"),
    "S_TAudit":       ("Time Since Audit",        "Recency of last security audit"),
    "S_CE":           ("Critical Exploits",       "Historical exploit severity"),
    "S_MS":           ("Multisig Governance",     "Centralisation of privileged actions"),
    "Delta_Critical": ("Recent Critical Event",   "Depeg / freeze / governance attack in last month"),
    "S_BD":           ("Bad Debt Incidence",      "Current bad debt ratio — 0 for non-lending protocols"),
    "S_BDP":          ("Bad Debt Prevention",     "Structural risk from LTV settings; discounted by safety module"),
}

CRS_META: dict[str, str] = {
    "S_OH":      "Oracle Heartbeat Risk",
    "S_OD":      "Oracle Deviation Risk",
    "ORS":       "Oracle Risk (total)",
    "S_ODiv×χ":  "Oracle Diversity (chi×S_ODiv)",
    "Validator": "Validator Risk (scaled)",
    "S_CDiv×ω":  "Counterparty Div. (omega×S_CDiv)",
    "DVN×λ":     "DVN Risk (lambda×S_DVN) — 0 if not cross-chain",
}

# ── Colour helpers ────────────────────────────────────────────────────────────

def _risk_color(v: float, hi: float = 0.6, mid: float = 0.25) -> str:
    return "#e74c3c" if v >= hi else "#f39c12" if v >= mid else "#2ecc71"

def _risk_emoji(v: float, hi: float = 0.6, mid: float = 0.25) -> str:
    return "🔴" if v >= hi else "🟡" if v >= mid else "🟢"

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data
def load_params() -> dict:
    data = load_yaml_file(PARAMS_PATH)
    return dict(data.get("parameters", {}))

@st.cache_data
def load_sim_params() -> dict:
    data = load_yaml_file(SIM_PARAMS_PATH)
    # merge top-level keys (params_simulation.yaml has no "parameters" wrapper)
    return dict(data)

@st.cache_data
def load_all_protocols() -> dict[str, dict]:
    return {p.stem: load_yaml_file(p) for p in sorted(PROTOCOLS_DIR.glob("*.yaml"))}

@st.cache_data(ttl=1800)
def cached_ethena_tvl() -> pd.DataFrame:
    return fetch_protocol_tvl("ethena")

@st.cache_data(ttl=1800)
def cached_usde_price() -> pd.DataFrame:
    """Try DeFiLlama stablecoins first, fall back to CoinGecko."""
    try:
        coin_id = find_stablecoin_id("USDe")
        if coin_id:
            df = fetch_stablecoin_price_history(coin_id)
            if not df.empty:
                return df
    except Exception:
        pass
    return fetch_usde_price_coingecko()

@st.cache_data(ttl=1800)
def cached_top_usde_pools() -> pd.DataFrame:
    return fetch_top_usde_pools(4)

# ── CVaR simulation ───────────────────────────────────────────────────────────

def run_strategy_cvar(
    protocol_stems: tuple[str, ...],
    alloc_values: tuple[float, ...],
    volatility_annual: float = 0.30,
) -> dict:
    """Monte Carlo CVaR for the given strategy (protocol stems + allocations %)."""
    sim = load_sim_params()
    params = load_params()
    all_protocols = load_all_protocols()

    n_paths = min(int(sim.get("n_paths", 10_000)), 10_000)   # cap for web speed
    p = {
        "n_paths":           n_paths,
        "step_days":         int(sim.get("step_days", 1)),
        "horizon_days":      int(sim.get("horizon_days", 30)),
        "volatility_annual": float(volatility_annual),
        "apy_min":           float(sim.get("apy_min", 0.05)),
        "apy_max":           float(sim.get("apy_max", 0.15)),
        "nu":                float(sim.get("nu", 4.0)),
        "jump_volatility":   float(sim.get("jump_volatility", 0.1)),
        "confidence":        float(sim.get("confidence", 0.95)),
    }
    base_jp = float(sim.get("jump_probability", 0.00055))

    jump_model = []
    for stem, alloc in zip(protocol_stems, alloc_values):
        proto = all_protocols.get(stem, {})
        if is_asset(proto):
            mult = 1.0  # jump probability not scaled by asset volatility
        else:
            mult, _ = compute_prs(proto, params)
        weight = max(0.0, alloc / 100.0)
        jp_i   = float(np.clip(base_jp * mult * weight, 0.0, 1.0))
        jump_model.append({"weight": weight, "jump_probability": jp_i, "jump_mean_severity": 1.0})

    seed = int(sim.get("seed", 32))
    rng_base = np.random.default_rng(seed)

    rng1 = np.random.default_rng(rng_base.integers(0, 2**32 - 1))
    losses_nj = simulate_losses(p, rng1, jump_model, jump_probability_override=0.0)
    var_nj, cvar_nj = var_cvar(losses_nj, p["confidence"])

    rng2 = np.random.default_rng(rng_base.integers(0, 2**32 - 1))
    losses_wj = simulate_losses(p, rng2, jump_model)
    var_wj, cvar_wj = var_cvar(losses_wj, p["confidence"])

    return {
        "var_no_jump":    var_nj,
        "cvar_no_jump":   cvar_nj,
        "var_with_jump":  var_wj,
        "cvar_with_jump": cvar_wj,
        "cvar_sum":       cvar_nj + cvar_wj,
        "var_sum":        var_nj + var_wj,
        "confidence":     p["confidence"],
        "horizon_days":   p["horizon_days"],
        "n_paths":        n_paths,
    }

# ── Score computation ─────────────────────────────────────────────────────────

def compute_all_scores(protocols: dict, params: dict) -> tuple[dict, dict, dict, dict]:
    prs_by_name, prs_comps = {}, {}
    crs_by_name, crs_comps = {}, {}
    omega_cdiv = required_param(params, "omega_cdiv")
    weights, _  = compute_allocation_weights(protocols)

    for name, proto in protocols.items():
        if is_asset(proto):
            continue
        prs, components = compute_prs(proto, params)
        prs_by_name[name] = prs
        prs_comps[name]   = components

        ors_i, ors_comp  = compute_ors(proto, params)
        crs_i, _, detail = compute_crs(proto, params, ors_i)
        s_cdiv_i, cdiv_d = compute_s_cdiv(proto)
        cdiv_term  = round(omega_cdiv * s_cdiv_i, 4)
        chi = detail.get("chi", 1.0)
        sodiv = detail.get("S_ODiv", 0.0)
        dvn_term, _ = compute_s_dvn(proto, params)
        crs_comps[name] = {
            "S_OH":      ors_comp.get("S_OH", 0),
            "S_OD":      ors_comp.get("S_OD", 0),
            "ORS":       ors_i,
            "S_ODiv×χ":  round(chi * sodiv, 4),
            "Validator": detail.get("validator_term", 0),
            "S_CDiv×ω":  cdiv_term,
            "DVN×λ":     dvn_term,
            "CRS":       round(crs_i + cdiv_term + dvn_term, 4),
            "_cdiv_details": cdiv_d,
        }
        crs_by_name[name] = round(crs_i + cdiv_term + dvn_term, 4)

    return prs_by_name, prs_comps, crs_by_name, crs_comps

# ── Chart helpers ─────────────────────────────────────────────────────────────

def _hbar(labels, values, title, height=320) -> go.Figure:
    colors = [_risk_color(v) for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Risk score",
        xaxis_range=[0, max(values) * 1.35 + 0.01] if values else [0, 1],
        height=height,
        margin=dict(l=200, r=60, t=40, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def prs_bar_chart(components: dict, prs_total: float) -> go.Figure:
    keys   = [k for k in components if not k.startswith("_")]
    labels = [PRS_META.get(k, (k,))[0] for k in keys]
    values = [float(components[k]) for k in keys]
    return _hbar(labels, values, f"PRS components — Total: {prs_total:.4f}")

def crs_bar_chart(crs_comps: dict, crs_total: float) -> go.Figure:
    keys   = [k for k in crs_comps if not k.startswith("_") and k != "CRS"]
    labels = [CRS_META.get(k, k) for k in keys]
    values = [float(crs_comps[k]) for k in keys]
    return _hbar(labels, values, f"CRS components — Total: {crs_total:.4f}")

# ── Ethena charts ─────────────────────────────────────────────────────────────

def show_ethena_charts() -> None:
    st.markdown("---")
    st.subheader("Market data — Ethena (USDe)")
    col_tvl, col_depeg = st.columns(2)

    with col_tvl:
        st.markdown("**Protocol TVL history**")
        with st.spinner("Loading TVL…"):
            try:
                tvl_df = cached_ethena_tvl()
                if tvl_df.empty:
                    st.warning("No TVL data from DeFiLlama.")
                else:
                    fig = px.area(
                        tvl_df, x="date", y="tvl",
                        labels={"tvl": "TVL (USD)", "date": ""},
                        color_discrete_sequence=["#6c63ff"],
                    )
                    fig.update_layout(
                        yaxis_tickformat="$.3s",
                        hovermode="x unified",
                        showlegend=False,
                        height=340,
                        margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"TVL fetch failed: {exc}")

    with col_depeg:
        st.markdown("**USDe peg deviation — 4 biggest DEX pools**")
        with st.spinner("Loading peg data…"):
            try:
                price_df = cached_usde_price()
                pools_df  = cached_top_usde_pools()

                if price_df.empty:
                    st.warning("No USDe price data available.")
                else:
                    price_df["deviation_pct"] = (price_df["price"] - 1.0) * 100
                    fig = go.Figure()
                    fig.add_hline(y=0, line_dash="dot", line_color="gray",
                                  annotation_text="$1.00 peg", annotation_position="right")
                    fig.add_trace(go.Scatter(
                        x=price_df["date"],
                        y=price_df["deviation_pct"],
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color="#e74c3c", width=1.5),
                        fillcolor="rgba(231,76,60,0.12)",
                        name="USDe deviation",
                        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}%<extra></extra>",
                    ))

                    # Overlay pool TVL as bubble annotations if available
                    if not pools_df.empty:
                        pool_labels = (
                            pools_df["symbol"].tolist()
                            if "symbol" in pools_df.columns else []
                        )
                        tvl_vals = (
                            pools_df["tvlUsd"].tolist()
                            if "tvlUsd" in pools_df.columns else []
                        )
                        if pool_labels:
                            # show as a compact legend line in title
                            pool_str = " | ".join(
                                f"{sym}: ${tvl/1e6:.1f}M"
                                for sym, tvl in zip(pool_labels[:4], tvl_vals[:4])
                                if isinstance(tvl, (int, float))
                            )
                            fig.update_layout(title=dict(
                                text=f"<sup>Top pools — {pool_str}</sup>",
                                font=dict(size=11),
                            ))

                    fig.update_layout(
                        yaxis_title="% deviation from peg",
                        hovermode="x unified",
                        showlegend=False,
                        height=340,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if not pools_df.empty:
                    display = pools_df.copy()
                    display["TVL"] = display["tvlUsd"].apply(
                        lambda x: f"${float(x):,.0f}" if pd.notna(x) else "—"
                    )
                    if "apy" in display.columns:
                        display["APY"] = display["apy"].apply(
                            lambda x: f"{float(x):.2f}%" if pd.notna(x) else "—"
                        )
                    rename = {"chain": "Chain", "project": "Protocol", "symbol": "Pool"}
                    display = display.rename(columns=rename)
                    show_cols = [c for c in ["Chain", "Protocol", "Pool", "TVL", "APY"] if c in display.columns]
                    st.dataframe(display[show_cols], hide_index=True, use_container_width=True)
                else:
                    st.info("Pool breakdown not available right now.")
            except Exception as exc:
                st.error(f"Depeg data fetch failed: {exc}")

# ── Protocol detail page ──────────────────────────────────────────────────────

def show_protocol_detail(stem: str, all_protocols: dict, params: dict) -> None:
    if st.button("← Back to protocols"):
        st.session_state.selected_protocol = None
        st.rerun()

    proto   = all_protocols[stem]
    display = DISPLAY_NAMES.get(stem, stem.replace("_", " ").title())
    st.title(display)

    if is_asset(proto):
        st.info("This entry is an **asset** (wrapped token) — PRS / CRS not applicable.")
        return

    # ── PRS + CRS side by side ────────────────────────────────────────────────
    prs, prs_comp = compute_prs(proto, params)

    omega_cdiv       = required_param(params, "omega_cdiv")
    ors_i, ors_comp  = compute_ors(proto, params)
    crs_i, _, detail = compute_crs(proto, params, ors_i)
    s_cdiv_i, cdiv_d = compute_s_cdiv(proto)
    cdiv_term        = round(omega_cdiv * s_cdiv_i, 4)
    chi              = detail.get("chi", 1.0)
    sodiv            = detail.get("S_ODiv", 0.0)
    dvn_term, _ = compute_s_dvn(proto, params)
    crs_full         = round(crs_i + cdiv_term + dvn_term, 4)

    crs_comp_display = {
        "S_OH":      ors_comp.get("S_OH", 0),
        "S_OD":      ors_comp.get("S_OD", 0),
        "ORS":       ors_i,
        "S_ODiv×χ":  round(chi * sodiv, 4),
        "Validator": detail.get("validator_term", 0),
        "S_CDiv×ω":  cdiv_term,
        "DVN×λ":     dvn_term,
    }

    col_prs, col_crs = st.columns(2)
    with col_prs:
        st.markdown("### Protocol Risk Score (PRS)")
        st.plotly_chart(prs_bar_chart(prs_comp, prs), use_container_width=True)

    with col_crs:
        st.markdown("### Counterparty Risk Score (CRS)")
        st.plotly_chart(crs_bar_chart(crs_comp_display, crs_full), use_container_width=True)

    if cdiv_d:
        n_cp = len(cdiv_d)
        st.caption(
            f"Counterparty protocols (n={n_cp}, equal weight): "
            + " | ".join(f"**{nm}** ({cat}, Q={q})" for nm, cat, q in cdiv_d)
        )

    with st.expander("Component descriptions"):
        st.markdown("**PRS**")
        for k, (lbl, desc) in PRS_META.items():
            st.markdown(f"- **{lbl}**: {desc}")
        st.markdown("**CRS**")
        for k, desc in CRS_META.items():
            st.markdown(f"- **{k}**: {desc}")

    if stem == "ethena":
        show_ethena_charts()

# ── Main page ─────────────────────────────────────────────────────────────────

def show_protocol_grid(all_protocols: dict) -> None:
    COLS  = 4
    stems = [s for s, p in all_protocols.items() if not is_asset(p)]
    stems = sorted(stems, key=lambda s: (0 if s == "ethena" else 1, s))
    rows  = [stems[i : i + COLS] for i in range(0, len(stems), COLS)]
    for row in rows:
        cols = st.columns(COLS)
        for col, stem in zip(cols, row):
            label = DISPLAY_NAMES.get(stem, stem.replace("_", " ").title())
            with col:
                if st.button(f"🔷 {label}", key=f"btn_{stem}", use_container_width=True):
                    st.session_state.selected_protocol = stem
                    st.rerun()


def show_vault_strategy_section(all_protocols: dict, params: dict) -> None:
    st.markdown("---")
    st.header("Vault Strategy Risk Score")
    st.markdown(
        "Use this dedicated tool to assess the risk of a diversified investment strategy or a specific vault. "
        "To begin, select the protocols involved from the list and assign an approximate maximum percentage allocation to each."
    )

    non_assets = [s for s, p in all_protocols.items() if not is_asset(p)]
    selected: list[str] = st.multiselect(
        "Which protocols are in the strategy?",
        options=non_assets,
        format_func=lambda s: DISPLAY_NAMES.get(s, s.replace("_", " ").title()),
    )

    allocations: dict[str, float] = {}
    if selected:
        st.markdown("**Max allocation per protocol (%)**")
        alloc_cols = st.columns(min(len(selected), 4))
        for i, stem in enumerate(selected):
            with alloc_cols[i % 4]:
                allocations[stem] = st.slider(
                    DISPLAY_NAMES.get(stem, stem),
                    min_value=0, max_value=100,
                    value=max(5, 100 // len(selected)),
                    step=5,
                    key=f"alloc_{stem}",
                    format="%d%%",
                )

        def _guard_strategy_type():
            st.session_state["_strategy_type"] = "Rebalancing"

        st.radio(
            "Strategy type",
            options=["Rebalancing", "Leveraged Staking/Supply"],
            index=0,
            horizontal=True,
            key="_strategy_type",
            on_change=_guard_strategy_type,
            help="Leveraged Staking/Supply (involves liquidation risk) is not available in this version.",
        )

        volatility_pct = st.slider(
            "Portfolio main asset volatility — year-average of monthly realized vol from daily TVL (%)",
            min_value=1, max_value=100, value=30, step=1,
            help="Used as the annualized volatility input for the CVaR Monte Carlo simulation.",
        )

        if st.button("Calculate Risk Score", type="primary"):
            _show_strategy_results(selected, allocations, all_protocols, params,
                                   volatility_annual=volatility_pct / 100.0)


def _show_strategy_results(
    selected: list[str],
    allocations: dict[str, float],
    all_protocols: dict,
    params: dict,
    volatility_annual: float = 0.30,
) -> None:
    sim_params = load_sim_params()

    # Build protocols dict with alloc_max_pct
    strategy_protocols: dict[str, dict] = {}
    for stem in selected:
        proto = dict(all_protocols[stem])
        proto["alloc_max_pct"] = float(allocations.get(stem, 0))
        strategy_protocols[DISPLAY_NAMES.get(stem, stem)] = proto

    prs_by_name, prs_comps, crs_by_name, crs_comps = compute_all_scores(
        strategy_protocols, params
    )

    gamma     = required_param(params, "gamma")
    delta_crs = required_param(params, "delta_crs")
    psr, _, n_proto = compute_psr(prs_by_name, strategy_protocols, gamma)
    crs_port, _     = compute_crs_portfolio(crs_by_name, strategy_protocols, delta_crs)

    # ── CVaR simulation ───────────────────────────────────────────────────────
    with st.spinner(f"Running CVaR simulation ({min(int(sim_params.get('n_paths', 10000)), 10000):,} paths)…"):
        stems_tuple  = tuple(selected)
        allocs_tuple = tuple(allocations.get(s, 0) for s in selected)
        cvar_res = run_strategy_cvar(stems_tuple, allocs_tuple, volatility_annual)

    conf    = cvar_res["confidence"]
    horizon = cvar_res["horizon_days"]
    cvar_sum = cvar_res["cvar_sum"]

    # ── VSRS ─────────────────────────────────────────────────────────────────
    vsrs = round(psr + crs_port + cvar_sum, 4)

    st.markdown("### Vault Strategy Risk Score (VSRS)")
    st.caption(f"VSRS = PSR + CRS_portfolio + CVaR_sum  |  {conf:.0%} confidence, {horizon}-day horizon")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PSR", f"{psr:.4f}",     help="Aggregated protocol quality risk")
    m2.metric("CRS", f"{crs_port:.4f}", help="Aggregated counterparty/oracle risk")
    m3.metric(f"CVaR_sum ({conf:.0%})", f"{cvar_sum:.2%}",
               help=f"CVaR market + CVaR structural over {horizon} days")
    m4.metric("VSRS", f"{vsrs:.4f}", delta=None, help="PSR + CRS_portfolio + CVaR_sum")

    # ── CVaR detail ───────────────────────────────────────────────────────────
    with st.expander("CVaR simulation detail"):
        c1, c2 = st.columns(2)
        c1.metric(f"VaR {conf:.0%}  (market)",     f"{cvar_res['var_no_jump']:.2%}")
        c1.metric(f"CVaR {conf:.0%} (market)",     f"{cvar_res['cvar_no_jump']:.2%}")
        c2.metric(f"VaR {conf:.0%}  (structural)", f"{cvar_res['var_with_jump']:.2%}")
        c2.metric(f"CVaR {conf:.0%} (structural)", f"{cvar_res['cvar_with_jump']:.2%}")
        st.caption(
            f"Simulation: {cvar_res['n_paths']:,} paths · Student-t ν={sim_params.get('nu', 4)} · "
            f"base jump prob={sim_params.get('jump_probability', 0.00055)}"
        )

    # ── Per-protocol comparison chart ─────────────────────────────────────────
    if prs_by_name:
        st.markdown("### Risk score per protocol")
        compare_df = pd.DataFrame({
            "Protocol": list(prs_by_name.keys()),
            "PRS":      [prs_by_name[n] for n in prs_by_name],
            "CRS":      [crs_by_name.get(n, 0) for n in prs_by_name],
        })
        fig = px.bar(
            compare_df.melt(id_vars="Protocol", var_name="Score", value_name="Value"),
            x="Protocol", y="Value", color="Score", barmode="group",
            color_discrete_map={"PRS": "#6c63ff", "CRS": "#e74c3c"},
            labels={"Value": "Risk score"},
            title="PRS vs CRS per protocol",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Full per-protocol tables ───────────────────────────────────────────────
    with st.expander("Full PRS breakdown per protocol"):
        rows = []
        for name in strategy_protocols:
            if name not in prs_by_name:
                continue
            row = {"Protocol": name,
                   **{k: round(v, 4) for k, v in prs_comps[name].items()},
                   "PRS": round(prs_by_name[name], 4)}
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Protocol"), use_container_width=True)

    with st.expander("Full CRS breakdown per protocol"):
        rows = []
        for name in strategy_protocols:
            if name not in crs_by_name:
                continue
            comps = {k: v for k, v in crs_comps[name].items() if not k.startswith("_")}
            row = {"Protocol": name, **{k: round(float(v), 4) for k, v in comps.items()}}
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Protocol"), use_container_width=True)

# ── Protocol AI agent ────────────────────────────────────────────────────────

def show_protocol_agent_section() -> None:
    st.markdown("---")
    st.header("Add New Protocol via AI Agent")
    st.markdown(
        "Type a protocol name and the AI agent will search the web to gather all required data "
        "and generate a YAML config ready for risk scoring."
    )

    # API key — from Streamlit secrets or manual input
    api_key = None
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    if not api_key:
        api_key = st.text_input(
            "Anthropic API key",
            type="password",
            help="Set ANTHROPIC_API_KEY in Streamlit secrets to avoid entering it manually.",
        )

    # Chat history
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Protocol name (e.g. 'Spark', 'Kamino', 'Sky')…"):
        if not api_key:
            st.warning("Enter your Anthropic API key above first.")
            st.stop()

        # Show user message
        st.session_state.agent_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            search_log = []

            status_placeholder = st.empty()
            yaml_placeholder   = st.empty()

            def on_search(query: str) -> None:
                search_log.append(query)
                status_placeholder.markdown(
                    "**Searching…**\n" + "\n".join(f"- `{q}`" for q in search_log)
                )

            try:
                from agent_protocol import run_agent
                yaml_text = run_agent(
                    protocol_name=prompt,
                    api_key=api_key,
                    on_search=on_search,
                )
                status_placeholder.empty()
                yaml_placeholder.code(yaml_text, language="yaml")

                # Save button
                stem = prompt.strip().lower().replace(" ", "_").replace("-", "_")
                save_path = BASE / "configs" / "protocols" / f"{stem}.yaml"
                col_save, col_info = st.columns([1, 4])
                with col_save:
                    if st.button("Save to protocols", key=f"save_{stem}"):
                        save_path.write_text(yaml_text, encoding="utf-8")
                        st.success(f"Saved as `configs/protocols/{stem}.yaml`. Reload the app to see it.")
                with col_info:
                    st.caption(
                        f"Review all values before saving — especially ⚠ flagged fields. "
                        f"Will be saved as `{stem}.yaml`."
                    )

                full_reply = f"Here is the generated YAML for **{prompt}**:\n\n```yaml\n{yaml_text}\n```"
            except Exception as e:
                status_placeholder.empty()
                full_reply = f"Error running agent: {e}"
                st.error(full_reply)

            st.session_state.agent_messages.append({"role": "assistant", "content": full_reply})


# ── Entry point ───────────────────────────────────────────────────────────────

LOGO_PATH = BASE / "assets" / "p2p_logo.png"

def main() -> None:
    _icon = Image.open(LOGO_PATH) if LOGO_PATH.exists() else "🛡️"
    st.set_page_config(
        page_title="Risk Analysis Framework",
        page_icon=_icon,
        layout="wide",
    )
    if "selected_protocol" not in st.session_state:
        st.session_state.selected_protocol = None

    params        = load_params()
    all_protocols = load_all_protocols()

    if st.session_state.selected_protocol:
        show_protocol_detail(st.session_state.selected_protocol, all_protocols, params)
        return

    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=80)
    with col_title:
        st.title("Risk Analysis Framework")

    st.header("Protocol Risk")
    st.markdown(
        "This tool evaluates protocol risk and continuously monitors protocol health, "
        "helping you assess risk when allocating capital to a particular protocol. "
        "Use it to understand its condition and avoid hidden vulnerabilities before committing funds."
    )
    with st.container(border=True):
        show_protocol_grid(all_protocols)

    show_vault_strategy_section(all_protocols, params)
    show_protocol_agent_section()


if __name__ == "__main__":
    main()
