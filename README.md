# Agentic NEPSE Predictor

A production-ready, closed-loop agentic AI system for predicting NEPSE (Nepal Stock Exchange) stock percentage changes. The system predicts target returns, optimizes a diversified Markowitz portfolio, runs entirely locally, and iteratively improves via a Large Language Model (LLM) self-reflection loop with secure sandboxed feature invention.

## Core Features

- **Universal Cross-Sectional Model:** Pools all NEPSE symbols into a single massive dataset to solve the short-history and thin-liquidity problems. Learns universal market patterns simultaneously using XGBoost and LightGBM.
- **Secure Sandboxed Feature Invention:** An LLM (Qwen2.5:3b via Ollama, or an external provider) acts as a quantitative researcher. It reviews past iterations and invents completely novel mathematical features dynamically. This utilizes `pandas.eval()` to ensure strict mathematical security and prevent arbitrary code execution.
- **Volume-Aware Sharpe Backtester:** Models are evaluated via a realistic trading simulator that penalizes low-volume stocks with dynamic slippage (up to 1%+).
- **Hedge-Fund Level Metrics:** Implements strict risk-management and trading realism. It enforces T+2 settlement capital lockups, applies realistic bid-ask spreads, and filters models using a rigid **GT-Score** (Win Rate + Edge + Profit Factor) before promotion.
- **Diversified Portfolio Allocation:** Uses `PyPortfolioOpt` to compute portfolio weights based on cross-sectional predicted returns and 60-day historical covariance, heavily stabilized using **L2 Regularization** to prevent risky, concentrated allocations.
- **Advanced Quant Features:** Generates 184+ indicators (using TA-Lib with a pure-Python `pandas_ta` fallback), continuous regime scores (20-day volatility percentiles), and incorporates macroeconomic data scraped from the Nepal Rastra Bank (NRB).
- **Nepal Trading Calendar:** Intelligently schedules background tasks and trading logic by respecting Nepal's local trading schedule (closed on Fridays, Saturdays, and public holidays) using `holidays.Nepal()`.

## Enhanced Quant Features (from nepse-quant-terminal)

- **Market Regime Detection:** Automatically detects bull/bear/neutral regimes using 60-day rolling NEPSE return, adjusting position limits and capital deployment accordingly.
- **Gold/Silver Regime Overlay:** Determines risk-on/risk-off/neutral based on gold price volatility, adjusting capital deployment (90% risk-off, 97% neutral, 100% risk-on).
- **Cross-Sectional Momentum:** Calculates 6-month minus 1-month momentum with momentum crash protection (skips last month).
- **Quality Scoring:** Composite score based on ROE, debt-to-equity ratio, and earnings stability (coefficient of variation).
- **Quarterly Fundamental Analysis:** Tracks EPS growth and revenue growth from quarterly filings for fundamental strength assessment.
- **Regime-Aware Position Limits:** Reduces max positions in bear markets (3), maintains 8 in neutral, increases to 10 in bull markets.
- **Walk-Forward Validation:** Slides train/test windows across 6+ years of history to validate out-of-sample performance robustness.
- **Signal Strength Scoring:** Combines multiple signals (momentum, quality, fundamentals) to produce more robust buy/sell signals.

## Headless & UI

Features a Streamlit interactive dashboard and a headless mode designed for automated execution via GitHub Actions.

## Interactive Tools

### Qwen Terminal Chat
Interactive terminal-style chat with access to live stock CSV data. Just ask about any NEPSE stock (e.g., "How will BANDIPUR stock do?") and get answers based on actual data:
```bash
streamlit run terminal_chat.py
```
- Loads real-time CSV data from `data/{symbol}.csv`
- Provides stock info: close, high, low, volume, 5-day return
- Uses the same LLM backend as the main agent
- Supports multiple symbols in one question

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) (If running the LLM locally)
- [TA-Lib](https://ta-lib.org/) (Optional, C-library; system will gracefully fallback to `pandas_ta` if missing)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd agentic_nepse
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and run Ollama (Optional - for local LLM usage):**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve &
   ollama run qwen2.5:3b
   ```

## Usage

### 1. Interactive UI (Streamlit)
To visualize data, trigger the agent loop manually, and view the allocation tables:
```bash
streamlit run app.py
```
Click **Run Full Upgrade Loop** in the sidebar to start the training iterations.
*New:* The promotion logic now incorporates a **GT‑Score ≥ 0.85** threshold (factoring in Win Rate, Edge, and Profit Factor) to guarantee that only strictly robust models are promoted to production.

### 2. Qwen Knowledge‑Base Terminal
Interact with the Qwen model via a customized terminal‑style Streamlit UI featuring a distinct blue tech-aesthetic. Now with live CSV data access:
```bash
streamlit run terminal_chat.py
```

### 3. Terminal Reader
To view a quick summary of the top 5 predicted gainers and losers, along with their suggested portfolio allocations:
```bash
python terminal_reader.py
```

### 4. Headless Mode
To run the agent non-interactively (e.g. on a server cron job):
```bash
export HEADLESS_MODE=1
python app.py
```

## Docker Deployment (Optional)

You can run the Agentic NEPSE system within a Docker container.

```bash
# 1. Build the image
docker build -t agentic-nepse -f docker/Dockerfile .

# 2. Run the Streamlit terminal chat interface
docker run -p 8501:8501 agentic-nepse

# 3. Alternatively, run the headless model training
docker run -e HEADLESS_MODE=1 agentic-nepse python app.py
```

## Running Tests

Unit tests are included to verify feature generation, model ensembling, agent memory I/O, backtesting with dynamic slippage, and terminal reading logic.
```bash
python -m unittest discover -s test
```

## GitHub Actions

The system is configured with `.github/workflows/agent.yml` to automatically run every trading day at 10:00 AM UTC. 

**Using Remote LLMs in Actions:**
Since standard GitHub runners have limited resources, you can configure GitHub Repository Secrets to route the agent LLM queries to a remote provider (e.g., OpenRouter, Groq, or OpenAI):
- `LLM_PROVIDER`: Set to any non-empty string (e.g. `openrouter`)
- `LLM_API_KEY`: Your provider API Key
- `LLM_BASE_URL`: Base URL (e.g. `https://openrouter.ai/api/v1/chat/completions`)
- `LLM_MODEL`: Model ID (e.g. `qwen/qwen-2.5-7b-instruct`)

## Recent Updates

- **Infinity/Overflow Handling:** Added robust data cleaning to replace infinity values and clip extreme values before scaling to prevent sklearn validation errors
- **Terminal Chat Enhancement:** Now loads and analyzes real CSV data from `data/{symbol}.csv` for accurate stock predictions
- **Feature Generation Improvements:** Added idempotent `log` function replacement to prevent `np.np.log` errors
- **Enhanced Quant Features (v2.0):** Integrated market regime detection, gold/silver regime overlay, cross-sectional momentum, quality scoring, quarterly fundamental analysis, walk-forward validation, and signal strength scoring from nepse-quant-terminal
- **Regime-Aware Capital Deployment:** Auto-adjusts capital deployment based on market and gold regimes (90%-100%)
- **Enhanced Signal Generation:** Combines multiple signals for more robust predictions with XSec Momentum, Quality Score, and Fund Score columns in predictions

## Usage: Enhanced Quant Features

The enhanced quant features can be used programmatically:

```python
from enhanced_quant import (
    get_regime_score, get_gold_regime, calculate_xsec_momentum,
    calculate_quality_score, apply_regime_filter, walk_forward_validation
)

# Detect market regime
regime = get_regime_score(df, window=60)

# Get gold regime for capital deployment
gold_regime = get_gold_regime()

# Calculate momentum scores
momentum = calculate_xsec_momentum(prices_df, symbols, lookback=180)

# Calculate quality scores
quality = calculate_quality_score(fundamentals_df)

# Apply regime filter
filtered = apply_regime_filter(predictions, regime)

# Walk-forward validation
results = walk_forward_validation(df, model_trainer, train_start, train_end)
```

## Limitations & Risks

1. **Macro Scraping Fragility:** The scraper fetching data from the Nepal Rastra Bank (NRB) relies on the specific HTML structure of their public website. If NRB updates their frontend, the scraper may fail and silently revert to default placeholder macroeconomic values.
2. **Python Version Compatibility:** The system avoids dependencies with highly rigid C-extensions (like `tensorflow` or `numba`/`shap`) to ensure it builds successfully even on very modern interpreters (e.g., Python 3.14+). Ensure your environment is fully compatible with standard scientific Python stack tools.
