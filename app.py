"""
Risk Analysis Framework — Streamlit dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

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
)
from calculate_PRS import (
    compute_prs,
    compute_psr,
    is_asset,
    load_yaml_file,
    required_param,
)
from data_fetcher import (
    fetch_protocol_tvl,
    fetch_stablecoin_price_history,
    fetch_top_usde_pools,
    find_stablecoin_id,
)

PROTOCOLS_DIR = BASE / "configs" / "protocols"
PARAMS_PATH   = BASE / "configs" / "general" / "params.yaml"

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
    "S_Maturity":     ("Protocol Maturity",     "Age-based maturity risk — lower = more established"),
    "S_sigmaTVL":     ("TVL Volatility",         "TVL instability risk"),
    "S_NAudits":      ("Audit Coverage",         "Inverse of weighted audit count — lower = more audits"),
    "S_TAudit":       ("Time Since Audit",        "Recency of last security audit"),
    "S_CE":           ("Critical Exploits",       "Historical exploit severity"),
    "S_MS":           ("Multisig Governance",     "Centralisation of privileged actions"),
    "Delta_Critical": ("Recent Critical Event",   "Depeg / freeze / governance attack in last month"),
}

# ── Colour helpers ────────────────────────────────────────────────────────────

def _risk_color(value: float, high: float = 0.6, mid: float = 0.25) -> str:
    if value >= high:
        return "#e74c3c"
    if value >= mid:
        return "#f39c12"
    return "#2ecc71"


def _risk_emoji(value: float, high: float = 0.6, mid: float = 0.25) -> str:
    if value >= high:
        return "🔴"
    if value >= mid:
        return "🟡"
    return "🟢"


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data
def load_params() -> dict:
    data = load_yaml_file(PARAMS_PATH)
    return dict(data.get("parameters", {}))


@st.cache_data
def load_all_protocols() -> dict[str, dict]:
    out = {}
    for p in sorted(PROTOCOLS_DIR.glob("*.yaml")):
        out[p.stem] = load_yaml_file(p)
    return out


@st.cache_data(ttl=1800)
def cached_ethena_tvl() -> pd.DataFrame:
    return fetch_protocol_tvl("ethena")


@st.cache_data(ttl=1800)
def cached_usde_price() -> pd.DataFrame:
    coin_id = find_stablecoin_id("USDe")
    if coin_id is None:
        return pd.DataFrame(columns=["date", "price"])
    return fetch_stablecoin_price_history(coin_id)


@st.cache_data(ttl=1800)
def cached_top_usde_pools() -> pd.DataFrame:
    return fetch_top_usde_pools(4)


# ── Score computation helpers ─────────────────────────────────────────────────

def compute_all_scores(
    protocols: dict[str, dict],
    params: dict,
) -> tuple[dict, dict, dict, dict]:
    """Returns (prs_by_name, prs_components, crs_by_name, crs_components)."""
    prs_by_name: dict[str, float]   = {}
    prs_comps:   dict[str, dict]    = {}
    crs_by_name: dict[str, float]   = {}
    crs_comps:   dict[str, dict]    = {}

    omega_cdiv = required_param(params, "omega_cdiv")
    weights, _ = compute_allocation_weights(protocols)

    for name, proto in protocols.items():
        if is_asset(proto):
            continue

        prs, components = compute_prs(proto, params)
        prs_by_name[name] = prs
        prs_comps[name]   = components

        ors_i, ors_comp = compute_ors(proto, params)
        crs_i, sodiv, detail = compute_crs(proto, params, ors_i)
        s_cdiv_i, cdiv_details = compute_s_cdiv(proto)
        cdiv_term   = round(omega_cdiv * s_cdiv_i, 4)
        crs_i_full  = round(crs_i + cdiv_term, 4)
        crs_by_name[name] = crs_i_full
        crs_comps[name]   = {
            "S_OH":       ors_comp.get("S_OH", 0),
            "S_OD":       ors_comp.get("S_OD", 0),
            "ORS":        ors_i,
            "S_ODiv":     detail.get("S_ODiv", 0),
            "Validator":  detail.get("validator_term", 0),
            "S_CDiv×ω":  cdiv_term,
            "CRS":        crs_i_full,
            "_cdiv_details": cdiv_details,
        }

    return prs_by_name, prs_comps, crs_by_name, crs_comps


# ── UI helpers ────────────────────────────────────────────────────────────────

def prs_bar_chart(components: dict, prs_total: float, title: str = "") -> go.Figure:
    keys   = [k for k in components if not k.startswith("_")]
    labels = [PRS_META.get(k, (k, ""))[0] for k in keys]
    values = [components[k] for k in keys]
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
        title=f"{title}PRS = {prs_total:.4f}",
        xaxis_title="Risk score (higher = riskier)",
        xaxis_range=[0, max(values) * 1.3 + 0.01],
        height=300,
        margin=dict(l=180, r=80, t=40, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def crs_table_df(crs_comps: dict) -> pd.DataFrame:
    rows = []
    for key, val in crs_comps.items():
        if key.startswith("_"):
            continue
        rows.append({
            "Component": key,
            "Score": round(float(val), 4),
            "Risk": _risk_emoji(float(val)),
        })
    return pd.DataFrame(rows)


# ── Ethena charts ─────────────────────────────────────────────────────────────

def show_ethena_charts() -> None:
    st.markdown("---")
    st.subheader("Market data — Ethena (USDe)")
    col_tvl, col_depeg = st.columns(2)

    with col_tvl:
        st.markdown("**Protocol TVL history**")
        with st.spinner("Loading…"):
            try:
                tvl_df = cached_ethena_tvl()
                if tvl_df.empty:
                    st.warning("No TVL data returned from DeFiLlama.")
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
                        margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"TVL fetch failed: {exc}")

    with col_depeg:
        st.markdown("**USDe peg history**")
        with st.spinner("Loading…"):
            try:
                price_df = cached_usde_price()
                if price_df.empty:
                    st.warning("No price data returned from DeFiLlama.")
                else:
                    price_df["deviation_pct"] = (price_df["price"] - 1.0) * 100

                    fig = go.Figure()
                    fig.add_hline(
                        y=0, line_dash="dot", line_color="gray",
                        annotation_text="Peg ($1.00)", annotation_position="right",
                    )
                    fig.add_trace(go.Scatter(
                        x=price_df["date"],
                        y=price_df["deviation_pct"],
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color="#e74c3c", width=1.5),
                        fillcolor="rgba(231,76,60,0.15)",
                        name="USDe deviation (%)",
                        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}%<extra></extra>",
                    ))
                    fig.update_layout(
                        yaxis_title="Deviation from peg (%)",
                        hovermode="x unified",
                        showlegend=False,
                        margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Price fetch failed: {exc}")

        st.markdown("**Top 4 USDe DEX pools (current TVL)**")
        with st.spinner("Loading…"):
            try:
                pools_df = cached_top_usde_pools()
                if pools_df.empty:
                    st.warning("No pool data found.")
                else:
                    display = pools_df.copy()
                    display["TVL"] = display["tvlUsd"].apply(lambda x: f"${x:,.0f}")
                    if "apy" in display.columns:
                        display["APY"] = display["apy"].apply(
                            lambda x: f"{x:.2f}%" if pd.notna(x) else "—"
                        )
                    rename = {"chain": "Chain", "project": "Protocol", "symbol": "Pool"}
                    display = display.rename(columns=rename)
                    show_cols = [c for c in ["Chain", "Protocol", "Pool", "TVL", "APY"] if c in display.columns]
                    st.dataframe(display[show_cols], hide_index=True, use_container_width=True)
            except Exception as exc:
                st.error(f"Pool fetch failed: {exc}")


# ── Protocol detail page ──────────────────────────────────────────────────────

def show_protocol_detail(stem: str, all_protocols: dict, params: dict) -> None:
    if st.button("← Back to protocols"):
        st.session_state.selected_protocol = None
        st.rerun()

    proto = all_protocols[stem]
    display = DISPLAY_NAMES.get(stem, stem.replace("_", " ").title())
    st.title(display)

    if is_asset(proto):
        st.info("This entry is classified as an **asset** (e.g. wrapped token). PRS/CRS risk scores are not applicable.")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Category", "Asset")
        with col2:
            tvl = proto.get("tvl")
            if tvl:
                st.metric("TVL", f"${tvl:,.0f}")
        return

    # ── PRS ──────────────────────────────────────────────────────────────────
    st.markdown("### Protocol Risk Score (PRS)")
    prs, prs_comp = compute_prs(proto, params)
    st.plotly_chart(prs_bar_chart(prs_comp, prs), use_container_width=True)

    with st.expander("Component descriptions"):
        for key, (label, desc) in PRS_META.items():
            st.markdown(f"- **{label}**: {desc}")

    # ── CRS ──────────────────────────────────────────────────────────────────
    st.markdown("### Counterparty Risk Score (CRS)")
    omega_cdiv = required_param(params, "omega_cdiv")
    ors_i, ors_comp    = compute_ors(proto, params)
    crs_i, sodiv, detail = compute_crs(proto, params, ors_i)
    s_cdiv_i, cdiv_det = compute_s_cdiv(proto)
    cdiv_term  = round(omega_cdiv * s_cdiv_i, 4)
    crs_full   = round(crs_i + cdiv_term, 4)

    crs_data = {
        "Oracle Heartbeat (S_OH)":        ors_comp.get("S_OH", 0),
        "Oracle Deviation (S_OD)":        ors_comp.get("S_OD", 0),
        "Oracle Risk (ORS)":              ors_i,
        "Oracle Diversity (chi×S_ODiv)":  round(detail.get("chi", 1) * detail.get("S_ODiv", 0), 4),
        "Validator risk":                 detail.get("validator_term", 0),
        f"Counterparty Div. (ω×S_CDiv)": cdiv_term,
    }
    crs_rows = [
        {"Component": k, "Score": round(v, 4), "": _risk_emoji(v)}
        for k, v in crs_data.items()
    ]
    st.dataframe(pd.DataFrame(crs_rows), hide_index=True, use_container_width=True)
    st.metric("Total CRS", f"{crs_full:.4f}")

    if cdiv_det:
        n_cp = len(cdiv_det)
        st.caption(
            f"Counterparty protocols (n={n_cp}, equal weight 1/{n_cp} each): "
            + " | ".join(f"**{nm}** ({cat}, Q={q})" for nm, cat, q in cdiv_det)
        )

    # ── Ethena-specific charts ────────────────────────────────────────────────
    if stem == "ethena":
        show_ethena_charts()


# ── Main page ─────────────────────────────────────────────────────────────────

def show_protocol_grid(all_protocols: dict) -> None:
    COLS = 4
    stems = list(all_protocols.keys())
    rows  = [stems[i : i + COLS] for i in range(0, len(stems), COLS)]

    for row in rows:
        cols = st.columns(COLS)
        for col, stem in zip(cols, row):
            proto  = all_protocols[stem]
            label  = DISPLAY_NAMES.get(stem, stem.replace("_", " ").title())
            icon   = "📦" if is_asset(proto) else "🔷"
            with col:
                if st.button(f"{icon} {label}", key=f"btn_{stem}", use_container_width=True):
                    st.session_state.selected_protocol = stem
                    st.rerun()


def show_vault_strategy_section(all_protocols: dict, params: dict) -> None:
    st.markdown("---")
    st.header("Vault Strategy Risk Score")
    st.markdown(
        "If you want to check the risk of a given vault or investment strategy, "
        "select the protocols involved, set a maximum allocation for each, and calculate."
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

        st.radio(
            "Strategy type",
            options=["Rebalancing"],
            horizontal=True,
            help="Staked looping (which involves liquidation risk) is disabled in this version.",
        )

        if st.button("Calculate Risk Score", type="primary", disabled=not selected):
            _show_strategy_results(selected, allocations, all_protocols, params)


def _show_strategy_results(
    selected: list[str],
    allocations: dict[str, float],
    all_protocols: dict,
    params: dict,
) -> None:
    # Build protocols dict with alloc_max_pct injected
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

    # ── Portfolio-level summary ───────────────────────────────────────────────
    st.markdown("#### Portfolio risk summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Protocols in strategy", n_proto)
    m2.metric("PSR  (Protocol Score Risk)", f"{psr:.4f}",
               help="(γ/N)×Σ PRS_i — aggregated protocol quality risk")
    m3.metric("CRS  (Counterparty Risk Score)", f"{crs_port:.4f}",
               help="(δ/N)×Σ CRS_i — aggregated counterparty/oracle risk")

    # ── Per-protocol PRS table ────────────────────────────────────────────────
    st.markdown("#### PRS per protocol")
    summary_rows = []
    for name in strategy_protocols:
        if name not in prs_by_name:
            continue
        comps = prs_comps[name]
        row = {"Protocol": name, **{k: round(v, 4) for k, v in comps.items()},
               "PRS": round(prs_by_name[name], 4)}
        summary_rows.append(row)

    if summary_rows:
        df = pd.DataFrame(summary_rows).set_index("Protocol")
        st.dataframe(df, use_container_width=True)

    # ── Per-protocol CRS table ────────────────────────────────────────────────
    st.markdown("#### CRS per protocol")
    crs_rows = []
    for name in strategy_protocols:
        if name not in crs_by_name:
            continue
        comps = {k: v for k, v in crs_comps[name].items() if not k.startswith("_")}
        row   = {"Protocol": name, **{k: round(float(v), 4) for k, v in comps.items()}}
        crs_rows.append(row)

    if crs_rows:
        df = pd.DataFrame(crs_rows).set_index("Protocol")
        st.dataframe(df, use_container_width=True)

    # ── PRS comparison bar chart ──────────────────────────────────────────────
    if prs_by_name:
        fig = px.bar(
            x=list(prs_by_name.keys()),
            y=list(prs_by_name.values()),
            labels={"x": "Protocol", "y": "PRS"},
            color=list(prs_by_name.values()),
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            title="Protocol Risk Score comparison",
        )
        fig.update_layout(
            coloraxis_showscale=False,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ── App entry point ───────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Risk Analysis Framework",
        page_icon="🛡️",
        layout="wide",
    )

    if "selected_protocol" not in st.session_state:
        st.session_state.selected_protocol = None

    params       = load_params()
    all_protocols = load_all_protocols()

    if st.session_state.selected_protocol:
        show_protocol_detail(st.session_state.selected_protocol, all_protocols, params)
        return

    # ── Main page ─────────────────────────────────────────────────────────────
    st.title("🛡️ Risk Analysis Framework")

    st.header("Protocol Risk")
    st.markdown(
        "This tool rates protocol risk and monitors protocol health "
        "to help you allocate funds into the more reliable protocols."
    )

    with st.container(border=True):
        show_protocol_grid(all_protocols)

    show_vault_strategy_section(all_protocols, params)


if __name__ == "__main__":
    main()
