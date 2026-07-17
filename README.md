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
- **Headless & UI:** Features a Streamlit interactive dashboard and a headless mode designed for automated execution via GitHub Actions.

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
Interact with the Qwen model via a customized terminal‑style Streamlit UI featuring a distinct blue tech-aesthetic:
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

## Limitations & Risks

1. **Macro Scraping Fragility:** The scraper fetching data from the Nepal Rastra Bank (NRB) relies on the specific HTML structure of their public website. If NRB updates their frontend, the scraper may fail and silently revert to default placeholder macroeconomic values.
2. **Python Version Compatibility:** The system avoids dependencies with highly rigid C-extensions (like `tensorflow` or `numba`/`shap`) to ensure it builds successfully even on very modern interpreters (e.g., Python 3.14+). Ensure your environment is fully compatible with standard scientific Python stack tools.
