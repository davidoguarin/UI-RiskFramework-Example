"""
DeFiLlama API helpers for the Streamlit dashboard.
All functions return DataFrames or raise on network failure.
"""
from __future__ import annotations

import requests
import pandas as pd
from typing import Optional

_LLAMA_API    = "https://api.llama.fi"
_LLAMA_STABLE = "https://stablecoins.llama.fi"
_LLAMA_YIELDS = "https://yields.llama.fi"
_TIMEOUT = 15


def fetch_protocol_tvl(slug: str) -> pd.DataFrame:
    """Historical TVL for a DeFiLlama protocol slug (e.g. 'ethena')."""
    r = requests.get(f"{_LLAMA_API}/protocol/{slug}", timeout=_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    records = data.get("tvl", [])
    if not records:
        return pd.DataFrame(columns=["date", "tvl"])
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], unit="s")
    df = df.rename(columns={"totalLiquidityUSD": "tvl"})
    return df[["date", "tvl"]].sort_values("date").reset_index(drop=True)


def find_stablecoin_id(symbol: str) -> Optional[str]:
    """Return DeFiLlama stablecoin ID for the given symbol, or None."""
    r = requests.get(f"{_LLAMA_STABLE}/stablecoins", timeout=_TIMEOUT)
    r.raise_for_status()
    for asset in r.json().get("peggedAssets", []):
        if asset.get("symbol", "").upper() == symbol.upper():
            return str(asset["id"])
    return None


def fetch_stablecoin_price_history(coin_id: str) -> pd.DataFrame:
    """Daily price history for a DeFiLlama stablecoin ID."""
    r = requests.get(f"{_LLAMA_STABLE}/stablecoin/{coin_id}", timeout=_TIMEOUT)
    r.raise_for_status()
    prices = r.json().get("price", [])
    if not prices:
        return pd.DataFrame(columns=["date", "price"])
    df = pd.DataFrame(prices)
    # DeFiLlama returns date as "YYYY-MM-DD" strings
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "price"]].sort_values("date").reset_index(drop=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna(subset=["price"])


def fetch_top_usde_pools(n: int = 4) -> pd.DataFrame:
    """Top N DeFiLlama yield pools containing USDe, sorted by TVL."""
    r = requests.get(f"{_LLAMA_YIELDS}/pools", timeout=60)
    r.raise_for_status()
    raw = r.json()
    # Handle both {"data": [...]} and direct list
    pools = raw.get("data", raw) if isinstance(raw, dict) else raw
    if not isinstance(pools, list):
        return pd.DataFrame()
    filtered = [
        p for p in pools
        if "usde" in p.get("symbol", "").lower()
        and float(p.get("tvlUsd") or 0) > 0
    ]
    filtered.sort(key=lambda x: float(x.get("tvlUsd") or 0), reverse=True)
    top = filtered[:n]
    if not top:
        return pd.DataFrame()
    cols = ["chain", "project", "symbol", "tvlUsd", "apy"]
    df = pd.DataFrame(top)
    return df[[c for c in cols if c in df.columns]].head(n)


def fetch_usde_price_coingecko(days: int = 365) -> pd.DataFrame:
    """USDe daily price from CoinGecko (free API, no key needed)."""
    url = (
        "https://api.coingecko.com/api/v3/coins/ethena-usde/market_chart"
        f"?vs_currency=usd&days={days}&interval=daily"
    )
    r = requests.get(url, timeout=_TIMEOUT, headers={"Accept": "application/json"})
    r.raise_for_status()
    prices = r.json().get("prices", [])
    if not prices:
        return pd.DataFrame(columns=["date", "price"])
    df = pd.DataFrame(prices, columns=["ts_ms", "price"])
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms")
    return df[["date", "price"]].sort_values("date").reset_index(drop=True)
