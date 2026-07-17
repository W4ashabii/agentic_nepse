import os
import sys
import json
import requests
import pandas as pd

import streamlit as st

# Load LLM config from app.py
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

def load_stock_data(symbol: str) -> pd.DataFrame:
    """Load CSV data for a given symbol from the data/ directory."""
    csv_path = f"/home/sid/agentic_nepse/data/{symbol.upper()}.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            df.sort_values('Date', inplace=True)
            return df
        except Exception:
            pass
    return pd.DataFrame()

def get_stock_info(symbol: str) -> dict:
    """Get recent stock info including latest close, 5d return, volume."""
    df = load_stock_data(symbol)
    if df.empty:
        return None
    latest = df.iloc[-1]
    recent_5 = df.tail(6)
    if len(recent_5) > 1:
        prev_close = recent_5['Close'].iloc[0]
        curr_close = latest['Close']
        return_5d = (curr_close - prev_close) / prev_close * 100 if prev_close > 0 else 0
    else:
        return_5d = 0
    return {
        'symbol': symbol.upper(),
        'date': latest['Date'].strftime('%Y-%m-%d') if 'Date' in latest else 'N/A',
        'open': latest['Open'],
        'high': latest['High'],
        'low': latest['Low'],
        'close': latest['Close'],
        'volume': latest['Volume'],
        'return_5d': return_5d
    }

def ask_qwen(question: str) -> str:
    """Send a question to the Qwen LLM and return the answer."""
    # Extract stock symbol from question if present
    symbols_in_question = []
    question_upper = question.upper()
    possible_symbols = ['BANDIPUR', 'SBL', 'NABIL', 'EBL', 'SCB', 'KBL', 'SBI', 'NICA', 'HBL', 'ADBL', 'PCBL', 'GBIME', 'DBL', 'SANIMA', 'CZBIL', 'MBL', 'PRVU', 'NBL']
    
    for sym in possible_symbols:
        if sym in question_upper:
            symbols_in_question.append(sym)
    
    # Build context with stock data
    context = ""
    if symbols_in_question:
        for sym in symbols_in_question[:3]:  # Limit to 3 symbols
            info = get_stock_info(sym)
            if info:
                context += f"\n\nStock Data for {info['symbol']}:\n"
                context += f"- Date: {info['date']}\n"
                context += f"- Close: Rs {info['close']:.2f}\n"
                context += f"- High: Rs {info['high']:.2f}\n"
                context += f"- Low: Rs {info['low']:.2f}\n"
                context += f"- Open: Rs {info['open']:.2f}\n"
                context += f"- Volume: {int(info['volume']):,}\n"
                context += f"- 5-Day Return: {info['return_5d']:.2f}%\n"
    
    prompt = f"Answer this question about NEPSE stocks using the context provided below. If stock data is available, use it for accurate analysis.\n\nQuestion: {question}\n{context if context else ''}"
    
    if LLM_PROVIDER:
        headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
        data = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        url = LLM_BASE_URL if "/chat/completions" in LLM_BASE_URL else f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {e}"
    else:
        try:
            resp = requests.post(LLM_BASE_URL, json={"model": LLM_MODEL, "prompt": prompt, "stream": False}, timeout=60)
            if resp.status_code == 200:
                return resp.json().get("response", "")
        except Exception as e:
            return f"Error: {e}"
    
    return "LLM unavailable. Please check your connection or API keys."

# CSS styling – blue themed version of the provided CSS
CUSTOM_CSS = """
/* Base colors */
:root {
    --border: #e0e0e0;
    --text: #0d47a1;           /* deep blue */
    --text-dim: #1565c0;       /* lighter blue */
    --text-muted: #42a5f5;    /* accent blue */
    --red: #d32f2f;            /* keep error red */
    --err: #c62828;
    --card: #e3f2fd;           /* very light blue background */
}

/* ---- inline approval (in a chat ledger entry) --------------------------- */
.apc {
  margin: 10px 0;
  max-width: 560px;
}
.apc-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  font-family: var(--mono, monospace);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1.4px;
  color: var(--text-dim);
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
.apc-label.hot { color: var(--red); }
.apc-id { letter-spacing: 0.4px; text-transform: none; }
.apc-call { border-bottom: 1px solid var(--border); }
.apc-call-head {
  display: flex;
  align-items: baseline;
  gap: 12px;
  width: 100%;
  padding: 9px 2px;
  background: none;
  border: none;
  text-align: left;
  color: var(--text);
}
.apc-call-head:hover .apc-caret { color: var(--text-muted); }
.apc-call-head.as-text { cursor: default; }
.apc-tool { font-size: 12px; color: var(--text); white-space: nowrap; }
.apc-args { flex: 1; min-width: 0; font-size: 11px; color: var(--text-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.apc-caret { font-size: 10px; color: var(--text-dim); }
.apc-call-body { padding: 0 2px 10px; }
.apc-kv { display: flex; gap: 12px; align-items: baseline; padding: 2px 0; }
.apc-k { font-size: 10.5px; color: var(--text-dim); min-width: 72px; }
.apc-v { font-size: 11px; color: var(--text-muted); word-break: break-word; white-space: pre-wrap; }
.apc-actions { display: flex; align-items: center; gap: 8px; padding: 10px 2px 2px; }
.apc-note { font-size: 11px; color: var(--text-dim); padding: 8px 2px 2px; }
.apc-note.err { color: var(--err); }

/* ---- artifact ledger (under an answer) ----------------------------------- */
.afc { margin: 10px 0 2px; max-width: 560px; }
.afc-head { font-family: var(--mono, monospace); font-size: 10px; text-transform: uppercase; letter-spacing: 1.4px; color: var(--text-dim); padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.afc-row { display: flex; align-items: baseline; gap: 12px; padding: 9px 2px; border-bottom: 1px solid var(--border); font-size: 13px; }
.afc-row:hover { background: var(--card); }
.afc-name { white-space: nowrap; }
.afc-meta { flex: 1; min-width: 0; font-size: 10.5px; color: var(--text-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.afc-get { background: none; border: none; padding: 0; font-size: 10.5px; letter-spacing: 0.6px; color: var(--text-dim); cursor: pointer; }
.afc-get:hover { color: var(--text); }
.afc-get:disabled { cursor: default; opacity: 0.6; }
.afc-get.err, .afc-state.err { color: var(--err); }
.afc-state { font-size: 10.5px; letter-spacing: 0.6px; }
"""

st.set_page_config(page_title="Qwen Terminal Chat", layout="centered")
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

st.title("💬 Qwen Knowledge‑Base Terminal")
st.write("Ask a question about NEPSE stocks with access to live CSV data. Try: 'How will BANDIPUR stock do?'")

question = st.text_input("Your question:", "")
if st.button("Send") and question.strip():
    with st.spinner("Thinking…"):
        answer = ask_qwen(question.strip())
    # Render answer in a styled block
    st.markdown(f"""
    <div class=\"apc\">
      <div class=\"apc-head\"><span>Question</span></div>
      <div class=\"apc-call-head as-text\">{question}</div>
      <div class=\"apc-head\"><span>Answer</span></div>
      <div class=\"apc-call-body\">{answer}</div>
    </div>
    """, unsafe_allow_html=True)
