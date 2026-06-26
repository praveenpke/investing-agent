"""
Config-driven runner for the TradingAgents multi-agent stock analyst (powered by GLM 5.2).

Edit `config.json` to change the ticker, date, analysts, model, etc. — or override on the CLI:

    python run_analysis.py                 # uses config.json as-is
    python run_analysis.py AAPL            # override ticker
    python run_analysis.py AAPL 2026-06-25 # override ticker + date
    python run_analysis.py AAPL 2026-06-25 market,fundamentals,news

Secrets (your GLM API key) live in `.env`, never in config.json.
Outputs a clean markdown report to the configured output_dir (default: reports/).
"""
import json
import os
import sys

# Windows: force UTF-8 so debug pretty_print() can emit Unicode (e.g. the − minus sign)
# without a cp1252 UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

# ---- Load config.json ----
with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
    CONF = json.load(f)

# CLI overrides (optional): ticker, date, analysts
ticker = sys.argv[1] if len(sys.argv) > 1 else CONF["ticker"]
date = sys.argv[2] if len(sys.argv) > 2 else CONF["date"]
analysts = tuple((sys.argv[3].split(",") if len(sys.argv) > 3 else CONF["analysts"]))

llm = CONF["llm"]

# ---- Reliability patch: GLM can stall on the very large researcher/risk-stage prompts.
# Inject a per-request timeout + retries so a hung call fails fast and retries. ----
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

_orig_kwargs = TradingAgentsGraph._get_provider_kwargs
def _patched_kwargs(self):
    k = _orig_kwargs(self)
    k.setdefault("timeout", float(CONF.get("request_timeout_seconds", 150)))
    k.setdefault("max_retries", int(CONF.get("max_retries", 3)))
    return k
TradingAgentsGraph._get_provider_kwargs = _patched_kwargs

# ---- Build config ----
cfg = DEFAULT_CONFIG.copy()
cfg["llm_provider"] = llm["provider"]
cfg["deep_think_llm"] = llm["deep_think_model"]
cfg["quick_think_llm"] = llm["quick_think_model"]
cfg["backend_url"] = llm.get("backend_url")
cfg["max_debate_rounds"] = CONF.get("max_debate_rounds", 1)

print(f">>> Analyzing {ticker} as of {date} | model={cfg['deep_think_llm']} | analysts={analysts}")
ta = TradingAgentsGraph(selected_analysts=analysts, debug=True, config=cfg)
final_state, decision = ta.propagate(ticker, date)

print("\n================ FINAL DECISION ================")
print(decision)

# ---- Write a clean markdown report ----
def _sub(d, key):
    return (d or {}).get(key, "") if isinstance(d, dict) else ""

inv = final_state.get("investment_debate_state", {})
risk = final_state.get("risk_debate_state", {})

sections = [
    ("📈 Market / Technical Analysis", final_state.get("market_report")),
    ("💼 Fundamentals Analysis", final_state.get("fundamentals_report")),
    ("📰 News Analysis", final_state.get("news_report")),
    ("💬 Sentiment Analysis", final_state.get("sentiment_report")),
    ("🐂 Bull Researcher", _sub(inv, "bull_history")),
    ("🐻 Bear Researcher", _sub(inv, "bear_history")),
    ("⚖️ Research Manager — Verdict", _sub(inv, "judge_decision") or final_state.get("investment_plan")),
    ("🧮 Trader — Investment Plan", final_state.get("trader_investment_plan")),
    ("🔥 Risk: Aggressive", _sub(risk, "aggressive_history") or _sub(risk, "risky_history")),
    ("🛡️ Risk: Conservative", _sub(risk, "conservative_history") or _sub(risk, "safe_history")),
    ("➖ Risk: Neutral", _sub(risk, "neutral_history")),
    ("🏛️ Portfolio Manager — Final Trade Decision", _sub(risk, "judge_decision") or final_state.get("final_trade_decision")),
]

out_dir = CONF.get("output_dir", "reports")
os.makedirs(out_dir, exist_ok=True)
outfile = os.path.join(out_dir, f"{ticker}_{date}.md")
with open(outfile, "w", encoding="utf-8") as f:
    f.write(f"# {ticker} — TradingAgents Report\n\n")
    f.write(f"**Date analyzed:** {date}  \n**Model:** {cfg['deep_think_llm']} (GLM)  \n")
    f.write(f"**Analysts:** {', '.join(analysts)}\n\n")
    f.write(f"## ✅ FINAL DECISION: {str(decision).strip()}\n\n---\n\n")
    for title, body in sections:
        if body and str(body).strip():
            f.write(f"## {title}\n\n{str(body).strip()}\n\n---\n\n")

print(f"\nMarkdown report written -> {outfile}")
