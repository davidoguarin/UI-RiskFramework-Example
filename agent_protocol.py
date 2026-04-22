"""
AI agent that searches for DeFi protocol data and generates a risk-framework YAML.

Uses Claude claude-sonnet-4-6 with web_search tool. The caller drives the agentic loop and
receives streaming updates via callbacks so the Streamlit UI can show progress.
"""
from __future__ import annotations

import json
from typing import Callable

import anthropic

SYSTEM_PROMPT = """\
You are a DeFi protocol risk researcher for P2P.org's Risk Analysis Framework.

Your job: given a protocol name, search the web for accurate data and produce a YAML
configuration file that exactly matches the schema below. Add inline comments
citing your sources (URL or publication). Mark uncertain values with # ⚠.

━━━ YAML SCHEMA ━━━
# Protocol display name
age_years: <float>          # Years since mainnet launch
tvl_vol_pct: <float>        # Year-avg of monthly realized TVL vol (stddev of daily log-returns × sqrt(30), averaged over 12 months). Source: DeFiLlama
audits_by_auditor:
  - auditor: <name>
    n_audits: <int>         # Count per auditor; list all distinct security firms
months_since_audit: <int>   # Months since the most recent completed security audit
n_critical_exploits: <int>  # Number of confirmed smart-contract exploits causing loss of funds
Delta_Critical: <0 or 1>    # 1 if depeg < 0.9, withdrawal freeze, or governance attack in last 30 days; else 0
oracle_heartbeat_hours: <float>   # Worst-case Chainlink (or equivalent) heartbeat in hours
oracle_deviation: <float>         # Worst-case oracle deviation threshold (e.g. 0.005 = 0.5%)
tvl: <int>                  # Current TVL in USD (integer)
validator_services: null    # OR:
  # n_operators: <int>
  # slashing_events: <int>
  # has_slashing_insurance: <bool>
counterparty_protocols: null   # OR list of external protocols this protocol depends on:
  # - name: <str>
  #   category: <one of: idle_blue_chip_token | idle_blue_chip_lending | idle_wrapped_blue_chip |
  #              custodial_wrapper | major_lst | established_cdp_stablecoin | rwa_credit |
  #              centralized_exchange | delta_neutral_stable | yield_aggregator |
  #              non_established_cdp_synthetic | idle_meme_token |
  #              reflexive_algorithmic_stablecoin | new_or_unaudited | recently_depegged_unstable>
oracle_diversification:
  provider_count: <int>     # Number of distinct oracle providers
  providers:
    - <provider name>
multisig_requirements:
  funds_movement: "<m>/<n>"       # e.g. "3/5"; use "unknown" if not found
  contract_changes: "<m>/<n>"
  parameter_changes: "<m>/<n>"
  minting: "<m>/<n> or N/A"
n_dvn: <int or null>        # Required LayerZero DVN count for cross-chain OFT; null if not applicable

━━━ SEARCH STRATEGY ━━━
Search in this order:
1. Protocol docs / whitepaper for oracle, multisig, and validator details
2. DeFiLlama (defillama.com) for TVL and age
3. Audit databases (Solodit, Code4rena, Sherlock, protocol GitHub) for audit history
4. Rekt.news / DefiHacks.io for exploit history
5. LayerZero scan / governance forum for DVN config
6. Protocol governance forum for multisig Safe addresses

Output ONLY the raw YAML block (no markdown fences, no extra text before or after).
"""


def run_agent(
    protocol_name: str,
    api_key: str,
    on_search: Callable[[str], None] | None = None,
    on_result: Callable[[str], None] | None = None,
) -> str:
    """
    Run the agent loop. Calls on_search(query) for each web search performed
    and on_result(yaml_text) once the final YAML is ready.
    Returns the final YAML string.
    """
    client = anthropic.Anthropic(api_key=api_key)

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Research the DeFi protocol **{protocol_name}** and generate its YAML config "
                f"for the risk analysis framework. Search for all required fields."
            ),
        }
    ]

    tools = [
        {
            "name": "web_search",
            "description": "Search the web for up-to-date information about a DeFi protocol.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    }
                },
                "required": ["query"],
            },
        }
    ]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            yaml_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    yaml_text += block.text
            yaml_text = yaml_text.strip()
            if on_result:
                on_result(yaml_text)
            return yaml_text

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                query = block.input.get("query", "")
                if on_search:
                    on_search(query)
                search_output = _web_search(query)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": search_output,
                    }
                )
            messages.append({"role": "user", "content": tool_results})


def _web_search(query: str) -> str:
    """Execute a web search and return results as a formatted string."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(
                    f"Title: {r.get('title', '')}\n"
                    f"URL: {r.get('href', '')}\n"
                    f"Snippet: {r.get('body', '')}\n"
                )
        return "\n---\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {e}"
