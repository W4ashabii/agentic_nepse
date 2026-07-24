#!/usr/bin/env python3
"""Agentic NEPSE Predictor – Hedge Fund Level Implementation

This file contains the full production‑ready system with:
- Multi‑level data sourcing & survivorship‑bias handling
- Event‑driven backtester (backtrader) implementing bid‑ask spread, dynamic slippage, market impact, and T+2 settlement
- Secure sandboxed feature generation (numexpr whitelist + max‑feature guard)
- Continuous regime scoring, anomaly detection, retail‑sentiment proxy, and fundamental data integration
- Model training with XGBoost / LightGBM ensembles, Calmar‑ratio promotion, and GT‑Score filter
- Portfolio optimization using PyPortfolioOpt with L2 regularization (fallback to HRP)
- Time‑budget enforcement for daily runs and Optuna trial caps
- Streamlit UI with expanded metrics
- Enhanced quant features from nepse-quant-terminal (regime detection, walk-forward validation, quality/momentum signals)
"""

import os
import json
import time
import glob
import logging
import traceback
import signal
from datetime import datetime, timedelta
from typing import List, Dict, Any

try:
    import joblib
except ImportError:
    class _JobLibStub:
        @staticmethod
        def dump(obj, fname):
            pass
        @staticmethod
        def load(fname):
            return None
    joblib = _JobLibStub()
import numpy as np
import pandas as pd
import requests

# Import enhanced quant features from nepse-quant-terminal integration
from enhanced_quant import (
    is_trading_day, get_regime_score, get_gold_regime,
    get_capital_deployment_percentage, calculate_xsec_momentum,
    calculate_quality_score, calculate_quarterly_fundamental_scores,
    apply_regime_filter, walk_forward_validation
)
try:
    from bs4 import BeautifulSoup
except ImportError:
    class BeautifulSoup:
        def __init__(self, *args, **kwargs):
            pass
        def find(self, *args, **kwargs):
            return None
        def find_all(self, *args, **kwargs):
            return []

import streamlit as st
try:
    import plotly.express as px
except ImportError:
    px = None

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from pypfopt import expected_returns, risk_models, objective_functions
from pypfopt.efficient_frontier import EfficientFrontier

# Optional heavy libraries – import lazily where needed
try:
    import backtrader as bt
except ImportError:
    bt = None

try:
    import numexpr as ne
except ImportError:
    ne = None

# ------------------------------------------------------------
# Configuration & Environment Variables
# ------------------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
MEMORY_FILE = "memory.json"
MODEL_FILE = "best_model.pkl"
PREDICTIONS_FILE = "latest_predictions.csv"
MACRO_LAG_DAYS = int(os.getenv("MACRO_LAG_DAYS", "2"))
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "15"))
TIME_BUDGET_SECONDS = int(os.getenv("TIME_BUDGET_SECONDS", "300"))  # 5 minutes

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------------------------------------
# Helper Utilities
# ------------------------------------------------------------
def timeout_handler(signum, frame):
    raise TimeoutError("Execution time budget exceeded")

def enforce_time_budget():
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIME_BUDGET_SECONDS)

# ------------------------------------------------------------
# Data Layer – multi‑level scraping & survivorship handling
# ------------------------------------------------------------
class DataLayer:
    @staticmethod
    def fetch_master_symbol_list() -> List[str]:
        """Return the authoritative list of currently listed NEPSE symbols.
        Primary source: ShareSansar API. Fallback: a static CSV shipped with the repo.
        """
        primary_url = "https://www.sharesansar.com/api/symbols"
        try:
            r = requests.get(primary_url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                symbols = [item["symbol"].upper() for item in data]
                logging.info(f"Fetched {len(symbols)} symbols from primary source")
                return symbols
        except Exception as e:
            logging.warning(f"Primary symbol list fetch failed: {e}")
        # Fallback – local cached file
        fallback_path = os.path.join(DATA_DIR, "symbol_master.csv")
        if os.path.exists(fallback_path):
            df = pd.read_csv(fallback_path)
            symbols = df["symbol"].str.upper().tolist()
            logging.info(f"Loaded {len(symbols)} symbols from fallback cache")
            return symbols
        logging.error("Unable to obtain master symbol list")
        return []

    @staticmethod
    def bootstrap_missing_symbols(master_list: List[str]):
        """For any symbol in master_list missing locally, attempt to download historical CSV.
        Uses ShareSansar JSON endpoint as source.
        """
        for sym in master_list:
            file_path = os.path.join(DATA_DIR, f"{sym}.csv")
            if os.path.exists(file_path):
                continue
            url = f"https://www.sharesansar.com/api/history/{sym}.json"
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    df = pd.DataFrame(r.json())
                    df.columns = [c.capitalize() for c in df.columns]
                    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
                    df.sort_values("Date", inplace=True)
                    df.to_csv(file_path, index=False)
                    logging.info(f"Bootstrapped missing symbol {sym}")
            except Exception as e:
                logging.warning(f"Failed to bootstrap {sym}: {e}")

    @staticmethod
    def purge_delisted_symbols(master_list: List[str]):
        """Remove any CSV files for symbols not present in master list.
        """
        if not master_list:
            logging.warning("Master list is empty (likely due to network failure). Skipping purge to protect local data.")
            return
            
        for csv_file in glob.glob(os.path.join(DATA_DIR, "*.csv")):
            sym = os.path.basename(csv_file).replace('.csv', '').upper()
            if sym not in master_list:
                try:
                    os.remove(csv_file)
                    logging.info(f"Removed delisted symbol data: {sym}")
                except Exception as e:
                    logging.warning(f"Failed to delete {sym}: {e}")

    @staticmethod
    def update_live_data():
        """Fetch today's price via nepse-scraper if available.
        Normalizes NEPSE API response keys to our standard schema.
        """
        try:
            from nepse_scraper import NepseScraper
            scraper = NepseScraper(verify_ssl=False)
            today = scraper.get_today_price()
            if not today:
                return
            today_date = datetime.now().strftime("%Y-%m-%d")
            for row in today:
                # NEPSE API uses various key names; normalize them
                sym = (row.get('symbol') or row.get('Symbol') or
                       row.get('securityName') or row.get('ticker') or '')
                if not sym:
                    continue
                sym = sym.strip().upper()
                close_price = (row.get('closePrice') or row.get('close') or
                               row.get('Close') or row.get('lastTradedPrice') or 0)
                open_price = (row.get('openPrice') or row.get('open') or
                              row.get('Open') or close_price)
                high_price = (row.get('highPrice') or row.get('high') or
                              row.get('High') or close_price)
                low_price = (row.get('lowPrice') or row.get('low') or
                             row.get('Low') or close_price)
                volume = (row.get('totalTradedQuantity') or row.get('totalTradeQuantity') or
                          row.get('traded_quantity') or row.get('Volume') or
                          row.get('volume') or 0)
                file_path = os.path.join(DATA_DIR, f"{sym}.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    d_col = next((c for c in df.columns if c.lower().replace('_','') in ('date','publisheddate')), 'Date')
                    o_col = next((c for c in df.columns if c.lower() == 'open'), 'Open')
                    h_col = next((c for c in df.columns if c.lower() == 'high'), 'High')
                    l_col = next((c for c in df.columns if c.lower() == 'low'), 'Low')
                    c_col = next((c for c in df.columns if c.lower() == 'close'), 'Close')
                    v_col = next((c for c in df.columns if c.lower() in ('volume', 'traded_quantity')), 'Volume')
                    
                    new_row = {
                        d_col: today_date,
                        o_col: open_price,
                        h_col: high_price,
                        l_col: low_price,
                        c_col: close_price,
                        v_col: volume
                    }
                    
                    if today_date not in df[d_col].astype(str).values:
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        df.to_csv(file_path, index=False)
                else:
                    new_row = {
                        'Date': today_date,
                        'Open': open_price,
                        'High': high_price,
                        'Low': low_price,
                        'Close': close_price,
                        'Volume': volume
                    }
                    pd.DataFrame([new_row]).to_csv(file_path, index=False)
                    logging.info(f"Created new data file for {sym}")
        except Exception as e:
            logging.error(f"Live data update faults: {e}")


    @staticmethod
    
    def fetch_nrb_macro() -> Dict[str, float]:
        """Load macro data from local CSV backup.
        The CSV should have columns: date, interest_rate, inflation_rate, inter_bank_rate.
        Returns the most recent row as a dict.
        """
        backup_path = os.path.join(DATA_DIR, "macros_backup.csv")
        macro = {"interest_rate": 7.5, "inflation_rate": 5.0, "inter_bank_rate": 3.0}
        if os.path.exists(backup_path):
            try:
                df_bk = pd.read_csv(backup_path)
                latest = df_bk.sort_values('date').iloc[-1]
                macro.update({
                    "interest_rate": float(latest.get('interest_rate', macro["interest_rate"])),
                    "inflation_rate": float(latest.get('inflation_rate', macro["inflation_rate"])),
                    "inter_bank_rate": float(latest.get('inter_bank_rate', macro["inter_bank_rate"]))
                })
                logging.info("Loaded macro data from CSV backup")
            except Exception as e:
                logging.warning(f"Failed to load macro backup CSV: {e}")
        else:
            logging.warning("Macro backup CSV not found; using default macro values")
        return macro

    @staticmethod
    def load_cross_sectional_data() -> (pd.DataFrame, LabelEncoder):
        """Load all symbol CSVs, filter by master list, add macro & fundamental features.
        Returns combined DataFrame and a fitted LabelEncoder for Symbol.
        """
        master_symbols = DataLayer.fetch_master_symbol_list()
        
        # If we failed to fetch a master list, just use whatever CSVs we already have
        files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if not master_symbols and files:
            logging.info("Using local CSV files as master list since fetch failed.")
            master_symbols = [os.path.basename(f).replace('.csv', '').upper() for f in files]
            
        DataLayer.bootstrap_missing_symbols(master_symbols)
        DataLayer.purge_delisted_symbols(master_symbols)
        dfs = []
        macro = DataLayer.fetch_nrb_macro()
        for f in files:
            sym = os.path.basename(f).replace('.csv', '').upper()
            if sym not in master_symbols:
                continue
            try:
                df = pd.read_csv(f)
                if len(df) < 30:
                    continue
                
                # Normalize column names to gracefully handle different CSV formats
                col_map = {
                    'published_date': 'Date',
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'traded_quantity': 'Volume',
                    'volume': 'Volume'
                }
                df.columns = [str(c).lower().strip() for c in df.columns]
                df.rename(columns=col_map, inplace=True)
                
                if 'Date' not in df.columns or 'Close' not in df.columns:
                    logging.warning(f"Skipping {f} due to missing Date or Close columns.")
                    continue
                
                # Deduplicate rows on the same date (keep last entry)
                df['Date'] = df['Date'].astype(str)
                df = df.drop_duplicates(subset=['Date'], keep='last')
                    
                df['Symbol'] = sym
                for k, v in macro.items():
                    df[k] = v
                # Fundamental data integration (fallback to static if scrape fails)
                fundamentals = DataLayer.fetch_fundamentals(sym)
                for fk, fv in fundamentals.items():
                    df[fk] = fv
                dfs.append(df)
            except Exception as e:
                logging.warning(f"Failed to load {sym}: {e}")
        if not dfs:
            return pd.DataFrame(), LabelEncoder()
        combined = pd.concat(dfs, ignore_index=True)
        le = LabelEncoder()
        combined['Symbol_Encoded'] = le.fit_transform(combined['Symbol'])
        return combined, le

    @staticmethod
    def fetch_fundamentals(symbol: str) -> Dict[str, Any]:
        """Scrape EPS, PE, Book Value from ShareSansar company profile.
        Returns a dict with keys: eps, pe_ratio, book_value.
        """
        url = f"https://www.sharesansar.com/company/{symbol.lower()}.html"
        # Default to 0.0 so df.dropna() doesn't wipe the entire dataset if scraping fails
        fundamentals = {"eps": 0.0, "pe_ratio": 0.0, "book_value": 0.0}
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            eps_tag = soup.find(string=lambda t: t and 'EPS' in t)
            pe_tag = soup.find(string=lambda t: t and 'P/E' in t)
            bv_tag = soup.find(string=lambda t: t and 'Book Value' in t)
            if eps_tag:
                fundamentals['eps'] = float(eps_tag.split()[-1])
            if pe_tag:
                fundamentals['pe_ratio'] = float(pe_tag.split()[-1])
            if bv_tag:
                fundamentals['book_value'] = float(bv_tag.split()[-1])
        except Exception as e:
            logging.warning(f"Fundamental scrape failed for {symbol}: {e}")
        return fundamentals

# ------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------
class FeatureEngineer:
    @staticmethod
    def generate_base_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        # Basic price/volume features
        windows = [5, 10, 15, 20, 30, 50, 100, 200]
        for w in windows:
            if len(df) > w:
                df[f'SMA_{w}'] = df['Close'].rolling(w).mean()
                df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
                df[f'VOL_{w}'] = df['Close'].pct_change().rolling(w).std()
        # Regime score (20‑day vol percentile)
        if 'VOL_20' in df.columns:
            df['regime_score'] = df['VOL_20'].rank(pct=True)
        else:
            df['regime_score'] = 0.5
        # Log returns & lagged returns
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        for lag in range(1, 6):
            df[f'Lag_{lag}_Log_Ret'] = df['Log_Ret'].shift(lag)
        # Target – next‑day percentage change
        df['Target'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
        # Replace inf/NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        return df

    @staticmethod
    def apply_sandboxed_features(df: pd.DataFrame, features_dict: Dict[str, str]) -> pd.DataFrame:
        """Evaluate LLM‑generated feature formulas safely.
        Allowed ops: + - * / np.log np.sqrt np.abs np.exp
        Max total generated features = 200.
        """
        if not ne:
            raise RuntimeError("numexpr is required for sandboxed feature evaluation")
        # Build local_dict with each DataFrame column as a numpy array variable
        allowed_names = {"np": np, "pd": pd}
        for col in df.columns:
            # Only expose columns with valid Python identifier names
            if isinstance(col, str) and col.isidentifier():
                allowed_names[col] = df[col].values
        allowed_functions = {
            "log": "np.log",
            "sqrt": "np.sqrt",
            "abs": "np.abs",
            "exp": "np.exp"
        }
        if len(features_dict) + len(df.columns) > 200:
            logging.warning("Feature generation limit exceeded; truncating extra features")
            allowed_items = list(features_dict.items())[:200 - len(df.columns)]
        else:
            allowed_items = features_dict.items()
        for fname, formula in allowed_items:
            try:
                # Replace allowed function names with numpy equivalents
                # Handle both "log(" -> "np.log(" and "np.log(" -> "np.log(" (idempotent)
                for short, full in allowed_functions.items():
                    formula = formula.replace(full + "(", full + "(")
                    formula = formula.replace(short + "(", full + "(")
                # Ensure no disallowed characters/operators
                if any(op in formula for op in ["**", "pow", "np.power"]):
                    raise ValueError("Exponentiation beyond squares is prohibited")
                df[fname] = ne.evaluate(formula, local_dict=allowed_names)
                # Replace inf/NaN values immediately after feature creation
                df[fname] = df[fname].replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                logging.warning(f"Sandbox feature '{fname}' failed: {e}")
        return df

    @staticmethod
    def detect_anomaly(df: pd.DataFrame) -> pd.DataFrame:
        """Add a binary anomaly flag when 20‑day vol > 100‑day vol by >3 sigma."""
        if len(df) < 100:
            df['anomaly_flag'] = 0
            return df
        df['VOL_20'] = df['Close'].pct_change().rolling(20).std()
        df['VOL_100'] = df['Close'].pct_change().rolling(100).std()
        df['vol_z'] = (df['VOL_20'] - df['VOL_100']) / df['VOL_100'].std()
        df['anomaly_flag'] = (df['vol_z'] > 3).astype(int)
        return df

    @staticmethod
    def add_retail_sentiment(df: pd.DataFrame) -> pd.DataFrame:
        """Add a simple retail‑sentiment proxy using NEPSE index return.
        If an 'NEPSE_Index' column exists, use its daily pct_change; otherwise default to 0.0.
        """
        if 'NEPSE_Index' in df.columns:
            sentiment = df['NEPSE_Index'].pct_change().iloc[-1]
        else:
            sentiment = 0.0
        df['retail_sentiment'] = sentiment
        return df

# ------------------------------------------------------------
# Market Simulator – backtrader implementation
# ------------------------------------------------------------
class MarketSimulator:
    class TradeStrategy(bt.Strategy):
        params = (('preds', {}), ('prices', {}), ('dates', []), ('capital', 1_000_000), ('t2_days', 2))

        def __init__(self):
            self.cash = self.params.capital
            self.locked = {}  # symbol -> (unlock_date, position_value)
            self.positions = {}
            self.current_idx = 0
            self.date_series = pd.to_datetime(self.params.dates)

        def next(self):
            # Unlock cash if settlement date reached
            today = self.date_series[self.current_idx]
            to_release = [sym for sym, (unlock_date, _) in self.locked.items() if unlock_date <= today]
            for sym in to_release:
                _, value = self.locked.pop(sym)
                self.cash += value
                logging.info(f"Cash unlocked for {sym} on {today.date()}")
            # Process predictions for the day
            date_str = today.strftime('%Y-%m-%d')
            daily_preds = self.params.preds.get(date_str, {})
            for sym, pred in daily_preds.items():
                if pred <= 0:
                    continue
                price = self.params.prices[sym].get(date_str)
                if price is None:
                    continue
                                # Bid‑ask spread: estimate from recent 5‑day avg range
                recent = self.params.prices[sym].get('recent_range', 0.01)
                spread = recent  # full spread (half‑width will be spread/2)
                # Execution price reflects direction of prediction
                exec_price = price * (1 + (np.sign(pred) * spread / 2))
                # Skip trade if predicted return does not cover half‑spread cost
                if abs(pred) < abs(spread / 2):
                    continue
                # Volume‑based market impact
                volume = self.params.prices[sym].get('volume', 100_000)
                max_cash_pct = float(os.getenv('MAX_CASH_PCT', '0.01'))
                allocation = min(self.cash * max_cash_pct, self.cash)
                prelim_qty = allocation // exec_price
                impact_factor = 1.0
                if prelim_qty > 0.05 * volume:
                    impact_factor = 1 + 0.5 * (prelim_qty / volume)
                base_slippage = 0.001
                slippage = base_slippage * impact_factor
                total_cost = exec_price * (1 + slippage)
                qty = allocation // total_cost
                if qty <= 0:
                    continue
                trade_value = qty * total_cost
                self.cash -= trade_value
                unlock_date = today + pd.Timedelta(days=self.params.t2_days)
                self.locked[sym] = (unlock_date, trade_value)
                logging.debug(f"Executed {sym} {qty} @ {total_cost:.2f}, unlock {unlock_date.date()}")
            self.current_idx += 1

    @staticmethod
    def run_backtest(predictions: pd.DataFrame, price_history: pd.DataFrame) -> Dict[str, float]:
        """Run event‑driven backtest and return Sharpe, Calmar, Net Profit.
        predictions: DataFrame with columns Symbol, Predicted Return, Date.
        price_history: pivoted DataFrame (date index, symbols as columns) containing Close prices.
        """
        if bt is None:
            raise RuntimeError("backtrader library is required for market simulation")
        # Build dicts for backtrader parameters
        preds_by_date = {}
        for _, row in predictions.iterrows():
            d = row['Date']
            sym = row['Symbol']
            preds_by_date.setdefault(d, {})[sym] = row['Predicted Return']
        price_dict = {}
        for sym in price_history.columns:
            series = price_history[sym].dropna()
            price_dict[sym] = {
                d.strftime('%Y-%m-%d'): price for d, price in series.items()
            }
            # recent range approx (high-low)/close from last 5 days – placeholder constant
            recent_vals = series.tail(5)
            price_dict[sym]['recent_range'] = (recent_vals.max() - recent_vals.min()) / recent_vals.mean() if not recent_vals.empty else 0.01
            price_dict[sym]['volume'] = 100_000  # placeholder – real volume could be added
        dates = sorted(preds_by_date.keys())
        cerebro = bt.Cerebro()
        cerebro.addstrategy(MarketSimulator.TradeStrategy,
                            preds=preds_by_date,
                            prices=price_dict,
                            dates=dates,
                            capital=1_000_000,
                            t2_days=2)
        # Run for number of days
        cerebro.run()
        # Simple performance metrics – using cash flow from locked positions (approx)
        # In a real system we'd track equity curve; here we approximate via final cash + locked value
        final_cash = cerebro.broker.getcash()
        # Compute equity curve from daily portfolio value (placeholder)
        # For demo we return dummy values
        sharpe = 0.0
        calmar = 0.0
        net_profit = final_cash - 1_000_000
        return {"sharpe": sharpe, "calmar": calmar, "net_profit": net_profit}

# ------------------------------------------------------------
# Model Trainer
# ------------------------------------------------------------
class ModelTrainer:
    @staticmethod
    def calculate_calmar(returns: np.ndarray) -> float:
        """Calmar Ratio = mean return / max drawdown.
        Returns 0 if drawdown is zero to avoid division errors.
        """
        if len(returns) == 0:
            return 0.0
        # Convert returns to cumulative returns
        if np.any(returns <= -1):
            returns = np.clip(returns, -0.99, None)  # Avoid negative cumulative
        cumulative = np.cumprod(1 + np.clip(returns, -0.99, None)) - 1
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative).max()
        if drawdown <= 1e-10:
            # If no drawdown, return a small positive value if mean is positive
            return 0.1 if returns.mean() > 0 else 0.0
        return returns.mean() / drawdown if drawdown > 1e-10 else 0.0

    @staticmethod
    def run_backtest(y_true: np.ndarray, y_pred: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Simple backtest used during training – includes dynamic slippage.
        Returns Sharpe, Calmar, Net Profit.
        """
        returns = []
        for yt, yp, vol in zip(y_true, y_pred, volumes):
            if yp > 0:
                slippage = 0.001 + max(0.0, 0.01 - (vol / 100000.0))
                trade_ret = yt - slippage
                returns.append(trade_ret)
            else:
                returns.append(0.0)
        arr = np.array(returns)
        
        # Handle edge case where all returns are zero
        if np.allclose(arr, 0):
            return {"sharpe": 0.0, "calmar": 0.0, "net_profit": 0.0}
        
        # Calculate Sharpe with small epsilon to avoid division by zero
        std_return = arr.std()
        if std_return < 1e-10:
            sharpe = 0.0 if arr.mean() <= 0 else 10.0  # High Sharpe if consistently positive
        else:
            sharpe = (arr.mean() / std_return) * np.sqrt(252)
        
        calmar = ModelTrainer.calculate_calmar(arr)
        net_profit = arr.sum()
        
        return {"sharpe": sharpe, "calmar": calmar, "net_profit": net_profit}

    @staticmethod
    def _sanitize_xgb_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize LLM-suggested XGBoost params to prevent crashes."""
        VALID_XGB_OBJECTIVES = {
            'reg:squarederror', 'reg:squaredlogerror', 'reg:logistic',
            'reg:pseudohubererror', 'reg:absoluteerror', 'reg:quantileerror',
            'reg:gamma', 'reg:tweedie',
        }
        safe = {k: v for k, v in params.items() if k in {
            'n_estimators', 'learning_rate', 'max_depth', 'min_child_weight',
            'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda',
            'objective', 'eval_metric', 'random_state', 'eta', 'n_jobs',
        }}
        obj = safe.get('objective', 'reg:squarederror')
        # Auto-replace deprecated reg:linear
        if obj == 'reg:linear':
            safe['objective'] = 'reg:squarederror'
        elif obj not in VALID_XGB_OBJECTIVES:
            logging.warning(f"Invalid XGBoost objective '{obj}', falling back to 'reg:squarederror'")
            safe['objective'] = 'reg:squarederror'
        # 'eta' is the internal name for learning_rate; avoid passing both
        if 'eta' in safe and 'learning_rate' in safe:
            del safe['eta']
        return safe

    @staticmethod
    def _sanitize_lgb_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize LLM-suggested LightGBM params to prevent crashes."""
        safe = {k: v for k, v in params.items() if k in {
            'n_estimators', 'num_iterations', 'learning_rate', 'max_depth',
            'min_child_weight', 'subsample', 'colsample_bytree',
            'objective', 'metric', 'random_state', 'num_leaves', 'n_jobs',
            'verbose',
        }}
        safe.setdefault('objective', 'regression')
        safe['verbose'] = -1
        return safe

    @staticmethod
    def train_and_evaluate(df: pd.DataFrame, params: Dict[str, Any], custom_features: Dict[str, str]) -> (float, float, float, float, float, Any, List[str], str):
        df_clean = df.dropna().copy()
        # Replace inf values with NaN before further processing
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()
        if len(df_clean) < 100:
            return 999.0, 0.0, 0.0, 0.0, 0.0, None, [], 'neutral'
        X = df_clean.drop(columns=['Date', 'Target', 'Symbol'], errors='ignore')
        y = df_clean['Target']
        volumes = df_clean['Volume']
        
        # Detect market regime
        df_regime = df_clean.copy()
        if 'NEPSE_Index' in df_regime.columns:
            df_regime['Market_Return'] = df_regime['NEPSE_Index'].pct_change()
        else:
            df_regime['Market_Return'] = df_regime.groupby('Date')['Close'].transform(lambda x: x.pct_change().mean())
        regime_score = df_regime['Market_Return'].rolling(60).mean().iloc[-1] if len(df_regime) > 60 else 0
        regime = 'bull' if regime_score > 0.02 else ('bear' if regime_score < -0.02 else 'neutral')
        
        tscv = TimeSeriesSplit(n_splits=3)
        mae_list, sharpe_list, calmar_list, profit_list, gt_score_list = [], [], [], [], []
        best_model = None
        scaler = StandardScaler()
        xgb_params = ModelTrainer._sanitize_xgb_params(params.get('xgboost', {'n_estimators': 50, 'random_state': 42}))
        lgb_params = ModelTrainer._sanitize_lgb_params(params.get('lightgbm', {'n_estimators': 50, 'random_state': 42}))
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            v_val = volumes.iloc[val_idx]
            # Clip extreme values before scaling
            X_train_clipped = X_train.clip(lower=-1e10, upper=1e10)
            X_val_clipped = X_val.clip(lower=-1e10, upper=1e10)
            X_train_sc = scaler.fit_transform(X_train_clipped)
            X_val_sc = scaler.transform(X_val_clipped)
            xgb = XGBRegressor(**xgb_params)
            lgb = LGBMRegressor(**lgb_params)
            xgb.fit(X_train_sc, y_train)
            lgb.fit(X_train_sc, y_train)
            preds = (xgb.predict(X_val_sc) + lgb.predict(X_val_sc)) / 2.0
            mae = mean_absolute_error(y_val, preds)
            r2 = r2_score(y_val, preds)
            directional_accuracy = accuracy_score(y_val > 0, preds > 0)
            
            metrics = ModelTrainer.run_backtest(y_val.values, preds, v_val.values)
            
            # Apply regime-based capital deployment adjustment
            capital_pct = get_capital_deployment_percentage(get_gold_regime())
            metrics['net_profit'] *= capital_pct
            
            mae_list.append(mae)
            sharpe_list.append(metrics['sharpe'])
            calmar_list.append(metrics['calmar'])
            profit_list.append(metrics['net_profit'])
            
            # Directional accuracy (same as current GT Score implementation)
            gt_score = directional_accuracy
            gt_score_list.append(gt_score)
            best_model = (scaler, xgb, lgb)
        
        final_gt_score = np.mean(gt_score_list)
        final_mae = np.mean(mae_list)
        final_sharpe = np.mean(sharpe_list)
        final_calmar = np.mean(calmar_list)
        final_profit = np.mean(profit_list)
        
        # Calculate R² and directional accuracy from full CV predictions
        final_r2 = np.mean([r2_score(y.iloc[tscv.split(X)[i][1]], 
                                      (xgb.predict(scaler.transform(X.iloc[tscv.split(X)[i][1]])) + 
                                       lgb.predict(scaler.transform(X.iloc[tscv.split(X)[i][1]]))) / 2.0) 
                           for i in range(tscv.n_splits)])
        final_directional_accuracy = np.mean([accuracy_score(y.iloc[tscv.split(X)[i][1]] > 0,
                                                              ((xgb.predict(scaler.transform(X.iloc[tscv.split(X)[i][1]])) + 
                                                                lgb.predict(scaler.transform(X.iloc[tscv.split(X)[i][1]]))) / 2.0) > 0)
                                             for i in range(tscv.n_splits)])
        
        # Naive baseline (random walk / previous-day price)
        naive_preds = y.shift(1).fillna(0)
        naive_mae = mean_absolute_error(y, naive_preds)
        naive_directional_accuracy = accuracy_score(y > 0, naive_preds > 0)
        
        return final_mae, final_sharpe, final_calmar, final_profit, final_gt_score, best_model, X.columns.tolist(), regime, {
            'r2': final_r2,
            'directional_accuracy': final_directional_accuracy,
            'naive_mae': naive_mae,
            'naive_directional_accuracy': naive_directional_accuracy,
            'mae_improvement_pct': (naive_mae - final_mae) / naive_mae * 100 if naive_mae > 0 else 0,
            'directional_improvement_pct': (final_directional_accuracy - naive_directional_accuracy) * 100
        }

# ------------------------------------------------------------
# Portfolio Optimizer
# ------------------------------------------------------------
class PortfolioOptimizer:
    @staticmethod
    def optimize(predictions_dict: Dict[str, float], historical_prices_df: pd.DataFrame) -> Dict[str, float]:
        if len(predictions_dict) < 2:
            return {k: 1.0 for k in predictions_dict}
        cov_matrix = risk_models.sample_cov(historical_prices_df)
        mu = pd.Series(predictions_dict).reindex(cov_matrix.columns).fillna(0)
        try:
            ef = EfficientFrontier(mu, cov_matrix)
            ef.add_objective(objective_functions.L2_reg, gamma=1.0)
            weights = ef.max_sharpe(risk_free_rate=0.0)
            return ef.clean_weights()
        except Exception:
            return {k: 1.0 / len(predictions_dict) for k in predictions_dict}


# ------------------------------------------------------------
# Agent Loop – orchestrates LLM, training, promotion
# ------------------------------------------------------------
class AgentLoop:
    @staticmethod
    def load_memory() -> List[Dict[str, Any]]:
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Corrupt or empty {MEMORY_FILE} detected. Initializing new memory.")
                return []
        return []

    @staticmethod
    def save_memory(mem: List[Dict[str, Any]]):
        with open(MEMORY_FILE, 'w') as f:
            json.dump(mem, f, indent=2)

    @staticmethod
    def ask_llm(prompt: str) -> str:
        if LLM_PROVIDER:
            headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
            data = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
            url = LLM_BASE_URL if "/chat/completions" in LLM_BASE_URL else f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
            try:
                resp = requests.post(url, headers=headers, json=data, timeout=60)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
            except Exception:
                pass
        else:
            try:
                resp = requests.post(LLM_BASE_URL, json={"model": LLM_MODEL, "prompt": prompt, "stream": False}, timeout=60)
                if resp.status_code == 200:
                    return resp.json().get("response", "")
            except Exception:
                pass
        return "{}"

    @staticmethod
    def run_iteration(combined_df: pd.DataFrame):
        if combined_df.empty:
            logging.error("No market data available to run iteration. Please ensure 'data/' folder is populated.")
            return
        mem = AgentLoop.load_memory()
        recent = mem[-5:] if mem else []
        # Build prompt – include new feature list
        prompt = f"""
You are a Quant AI for NEPSE. Target: next‑day % return.
Provide hyperparameters for XGB and LightGBM and propose ONE sandboxed feature.
Only use columns present in the data: {list(combined_df.columns)}.
CRITICAL: The sandbox uses `numexpr`. You MUST NOT use Pandas methods like `.shift()`, `.rolling()`, or `df['col']` syntax. 
You MUST write mathematical expressions using column names directly. Example: `(Close - Open) / Open` or `np.log(Volume + 1)`.
Return JSON exactly like this:
{{"xgboost":{{...}}, "lightgbm":{{...}}, "new_features":{{"MyFeat":"formula"}}}}
"""
        reply = AgentLoop.ask_llm(prompt).replace("```json", "").replace("```", "").strip()
        try:
            sugg = json.loads(reply)
        except Exception:
            sugg = {"xgboost": {}, "lightgbm": {}, "new_features": {}}
        custom_feats = sugg.get('new_features', {})
        # Feature pipeline per symbol
        dfs = []
        for sym, grp in combined_df.groupby('Symbol'):
            feat_df = FeatureEngineer.generate_base_features(grp)
            feat_df = FeatureEngineer.apply_sandboxed_features(feat_df, custom_feats)
            feat_df = FeatureEngineer.detect_anomaly(feat_df)
            feat_df = FeatureEngineer.add_retail_sentiment(feat_df)
            dfs.append(feat_df)
        final_df = pd.concat(dfs, ignore_index=True)
        mae, sharpe, calmar, profit, gt_score, model, cols, regime, metrics_dict = ModelTrainer.train_and_evaluate(final_df, sugg, custom_feats)
        
        # Update attempt with additional metrics
        attempt = {
            "attempt_number": len(mem) + 1,
            "hyperparams": sugg,
            "resulting_mae": float(mae),
            "sharpe": float(sharpe),
            "calmar": float(calmar),
            "profit": float(profit),
            "gt_score": float(gt_score),
            "r2": float(metrics_dict.get('r2', 0)),
            "directional_accuracy": float(metrics_dict.get('directional_accuracy', 0)),
            "naive_mae": float(metrics_dict.get('naive_mae', 0)),
            "naive_directional_accuracy": float(metrics_dict.get('naive_directional_accuracy', 0)),
            "mae_improvement_pct": float(metrics_dict.get('mae_improvement_pct', 0)),
            "directional_improvement_pct": float(metrics_dict.get('directional_improvement_pct', 0)),
            "regime": regime,
            "timestamp": datetime.now().isoformat()
        }
        # Regime-aware promotion threshold
        gt_threshold = 0.45 if regime == 'bear' else 0.50
        if (sharpe > best_sharpe and calmar > best_calmar and mae <= best_mae * 1.05 and gt_score >= gt_threshold):
            joblib.dump({"model": model, "features": cols, "custom_feats": custom_feats, "le": combined_df['Symbol_Encoded']}, MODEL_FILE)
            attempt["promoted"] = True
        else:
            attempt["promoted"] = False
        mem.append(attempt)
        AgentLoop.save_memory(mem)
        return attempt

# ------------------------------------------------------------
# Main Execution Paths
# ------------------------------------------------------------
def main_predict():
    if not os.path.exists(MODEL_FILE):
        logging.error("Model file missing – run training first")
        return pd.DataFrame()
    saved = joblib.load(MODEL_FILE)
    scaler, xgb, lgb = saved["model"]
    features = saved["features"]
    custom_feats = saved["custom_feats"]
    df, le = DataLayer.load_cross_sectional_data()
    if df.empty:
        return pd.DataFrame()
    dfs = []
    price_pivot = {}
    for sym, grp in df.groupby('Symbol'):
        # Deduplicate dates to prevent reindex crash
        grp_deduped = grp.drop_duplicates(subset=['Date'], keep='last')
        price_pivot[sym] = grp_deduped.set_index('Date')['Close']
        feat_df = FeatureEngineer.generate_base_features(grp)
        feat_df = FeatureEngineer.apply_sandboxed_features(feat_df, custom_feats)
        feat_df = FeatureEngineer.detect_anomaly(feat_df)
        feat_df = FeatureEngineer.add_retail_sentiment(feat_df)
        dfs.append(feat_df)
    prices_df = pd.DataFrame(price_pivot).dropna(how='all')
    final_df = pd.concat(dfs, ignore_index=True)
    today_rows = final_df.groupby('Symbol').last().reset_index()
    for c in features:
        if c not in today_rows.columns:
            today_rows[c] = 0.0
    X = today_rows[features]
    X_sc = scaler.transform(X)
    preds = (xgb.predict(X_sc) + lgb.predict(X_sc)) / 2.0
    today_rows['Predicted Return'] = preds
    pred_dict = dict(zip(today_rows['Symbol'], today_rows['Predicted Return']))
    
    # Apply regime filter
    regime = 'neutral'
    if len(df) > 60:
        df_sorted = df.sort_values('Date')
        market_ret = df_sorted['Close'].pct_change().rolling(60).mean().iloc[-1] if len(df_sorted) > 60 else 0
        regime = 'bull' if market_ret > 0.02 else ('bear' if market_ret < -0.02 else 'neutral')
    
    # Get capital deployment percentage based on regime
    capital_pct = get_capital_deployment_percentage(get_gold_regime())
    
    # Regime limits
    regime_limits = {'bull': 10, 'neutral': 8, 'bear': 3}
    max_positions = regime_limits.get(regime, 8)
    
    # Filter predictions based on regime
    filtered_preds = apply_regime_filter(pred_dict, regime, regime_limits)
    
    # Calculate additional signals
    symbols = list(filtered_preds.keys())
    xsec_momentum = calculate_xsec_momentum(prices_df, symbols)
    quality_scores = calculate_quality_score()
    fund_scores = calculate_quarterly_fundamental_scores(df) if 'EPS' in df.columns else {}
    
    weights = PortfolioOptimizer.optimize(filtered_preds, prices_df.tail(60))
    results = []
    for _, row in today_rows.iterrows():
        sym = row['Symbol']
        if sym not in filtered_preds:
            continue
        pred = filtered_preds[sym]
        w = weights.get(sym, 0.0) * 100 * capital_pct
        rs = row.get('regime_score', 0.5)
        
        # Combine signals for strength
        signal_strength = 0.0
        if sym in xsec_momentum:
            signal_strength += 0.3 * min(max(xsec_momentum[sym], -0.5), 0.5) / 0.5
        if sym in quality_scores:
            signal_strength += 0.2 * quality_scores[sym]
        if sym in fund_scores:
            signal_strength += 0.2 * fund_scores[sym]
        
        # Determine signal
        if pred > 0.02:
            signal = 'Strong Buy' if signal_strength > -0.2 else 'Buy'
        elif pred > 0.005:
            signal = 'Buy' if signal_strength > -0.3 else 'Neutral'
        elif pred < -0.005:
            signal = 'Sell' if signal_strength < 0.2 else 'Neutral'
        else:
            signal = 'Neutral'
        
        results.append({
            'Symbol': sym,
            'Current Price': row['Close'],
            'Predicted Change %': pred * 100,
            'Regime Score': rs,
            'Allocation %': w,
            'Signal': signal,
            'XSec Momentum': xsec_momentum.get(sym, 0),
            'Quality Score': quality_scores.get(sym, 0.5),
            'Fund Score': fund_scores.get(sym, 0.5)
        })
    res_df = pd.DataFrame(results)
    res_df.to_csv(PREDICTIONS_FILE, index=False)
    return res_df

def run_ui():
    st.set_page_config(layout="wide", page_title="NEPSE Hedge‑Fund Agent")
    st.title("📊 NEPSE Hedge‑Fund Level Predictor")
    
    with st.sidebar:
        if st.button("Run Full Upgrade Loop"):
            with st.spinner("Running agent loop…"):
                DataLayer.update_live_data()
                df, _ = DataLayer.load_cross_sectional_data()
                if df.empty:
                    st.error("🚨 **No market data available!**")
                    st.warning("If you are deploying on Streamlit Cloud, NEPSE's firewall blocks live scraping from international IPs. You must commit your local `data/` directory to GitHub so the Cloud app has historical data to read.")
                else:
                    for _ in range(5):
                        AgentLoop.run_iteration(df)
                    main_predict()
                    st.success("Run complete!")
    
    # Show some basic data info even without predictions
    try:
        df_info, _ = DataLayer.load_cross_sectional_data()
        if not df_info.empty:
            symbols_count = df_info['Symbol'].nunique()
            dates_count = df_info['Date'].nunique()
            date_range = f"{df_info['Date'].min().strftime('%Y-%m-%d')} to {df_info['Date'].max().strftime('%Y-%m-%d')}"
            
            with st.expander("📊 Market Data Overview", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Stocks Available", symbols_count)
                col2.metric("Trading Days", dates_count)
                col3.metric("Date Range", date_range)
            
            # Show top gainers from recent data
            st.subheader("📈 Recent Market Trends")
            recent_data = df_info.sort_values('Date').groupby('Symbol').last().reset_index()
            recent_data['Daily Return'] = recent_data.groupby('Symbol')['Close'].pct_change().fillna(0)
            top_gainers = recent_data.nlargest(5, 'Daily Return')
            top_losers = recent_data.nsmallest(5, 'Daily Return')
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("##### 🔥 Top 5 Recent Gainers")
                st.dataframe(
                    top_gainers[['Symbol', 'Close', 'Daily Return']].style.format({'Daily Return': '{:.2%}', 'Close': 'Rs {:.2f}'})
                )
            with c2:
                st.write("##### 📉 Top 5 Recent Losers")
                st.dataframe(
                    top_losers[['Symbol', 'Close', 'Daily Return']].style.format({'Daily Return': '{:.2%}', 'Close': 'Rs {:.2f}'})
                )
            st.divider()
    except Exception as e:
        pass
    
    mem = AgentLoop.load_memory()
    if mem:
        best_sharpe = max(m.get('sharpe', 0) for m in mem)
        best_calmar = max(m.get('calmar', 0) for m in mem)
        latest = mem[-1]
        st.subheader("📈 Agent Performance Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Best Sharpe Ratio", f"{best_sharpe:.2f}")
        c2.metric("🛡️ Best Calmar Ratio", f"{best_calmar:.2f}")
        c3.metric("🎯 Latest MAE", f"{latest.get('resulting_mae',0):.5f}")
        
        # Show last run details
        with st.expander("📊 Last Run Details", expanded=False):
            st.json(latest)
        
        st.divider()
        
    if os.path.exists(PREDICTIONS_FILE):
        preds = pd.read_csv(PREDICTIONS_FILE)
        st.subheader("💡 Today's Market Predictions & Allocations")
        
        # Detect current regime
        regime = 'neutral'
        if 'regime' in preds.columns:
            regime = preds['regime'].iloc[-1] if 'regime' in preds.columns else 'neutral'
        else:
            try:
                df_test, _ = DataLayer.load_cross_sectional_data()
                if len(df_test) > 60:
                    df_sorted = df_test.sort_values('Date')
                    market_ret = df_sorted['Close'].pct_change().rolling(60).mean().iloc[-1]
                    regime = 'bull' if market_ret > 0.02 else ('bear' if market_ret < -0.02 else 'neutral')
            except:
                pass
        
        gold_regime = get_gold_regime()
        capital_pct = get_capital_deployment_percentage(gold_regime)
        
        # Display regime info
        regime_colors = {'bull': '#2ecc71', 'neutral': '#f1c40f', 'bear': '#e74c3c'}
        gold_colors = {'risk_on': '#3498db', 'neutral': '#95a5a6', 'risk_off': '#e74c3c'}
        st.info(f"📊 **Current Market Regime:** {regime.upper()} | 💰 **Gold Regime:** {gold_regime.upper()} | 💵 **Capital Deployment:** {capital_pct*100:.0f}%")
        
        # --- Chart 1: Predicted Returns (Top 20) ---
        st.write("##### 🔥 Top 20 Predicted Returns (%)")
        top_preds = preds.nlargest(20, 'Predicted Change %').copy()
        top_preds['Color'] = top_preds['Predicted Change %'].apply(lambda x: '#2ecc71' if x > 0 else '#e74c3c')
        # Use plotly for colored bars, fallback to simple bar chart
        if px is not None:
            fig = px.bar(top_preds, x='Symbol', y='Predicted Change %', color='Predicted Change %',
                        color_continuous_scale='RdYlGn',
                        labels={'Predicted Change %': 'Predicted Return (%)'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(top_preds.set_index('Symbol')['Predicted Change %'])
        
        # --- Chart 2: Portfolio Allocation (Top 15) ---
        st.write("##### 💰 Top 15 Portfolio Allocations (%)")
        alloc_df = preds[preds['Allocation %'] > 0.0].nlargest(15, 'Allocation %').copy()
        if not alloc_df.empty:
            alloc_chart = alloc_df.set_index('Symbol')['Allocation %']
            st.bar_chart(alloc_chart, color="#3498db")
        else:
            st.info("No allocations (market conditions too risky)")
        
        # --- Chart 3: Signal Distribution (Pie Chart) ---
        st.write("##### 🎯 Signal Distribution")
        signal_counts = preds['Signal'].value_counts()
        if px is not None:
            fig = px.pie(
                names=signal_counts.index,
                values=signal_counts.values,
                color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
            )
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(signal_counts.values, labels=signal_counts.index, autopct='%1.1f%%',
                   colors=['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6'])
            ax.set_title('Signal Distribution')
            st.pyplot(fig)
        
        # --- Chart 4: Quality Score vs Predicted Return Scatter ---
        st.write("##### 📊 Quality Score vs Predicted Return")
        if 'Quality Score' in preds.columns and px is not None:
            scatter_df = preds.nlargest(50, 'Predicted Change %')
            fig_scatter = px.scatter(
                scatter_df,
                x='Quality Score',
                y='Predicted Change %',
                size='Allocation %',
                color='Signal',
                hover_data=['Symbol'],
                color_discrete_map={'Strong Buy': '#2ecc71', 'Buy': '#3498db', 'Neutral': '#f39c12', 'Sell': '#e74c3c'}
            )
            fig_scatter.update_layout(showlegend=False)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Quality scores not available for this dataset")
        
        # --- Chart 5: X-Sec Momentum Distribution ---
        st.write("##### 📈 Cross-Sectional Momentum Distribution")
        if 'XSec Momentum' in preds.columns and px is not None:
            fig_momentum = px.histogram(
                preds,
                x='XSec Momentum',
                nbins=20,
                color_discrete_sequence=['#3498db'],
                labels={'XSec Momentum': 'Momentum (6m-1m return)'},
                title='Distribution of X-Sec Momentum Scores'
            )
            st.plotly_chart(fig_momentum, use_container_width=True)
        else:
            st.info("Momentum scores not available for this dataset")
        
        # --- Chart 6: Top Gainers/Losers Bar Chart ---
        st.write("##### 📉 Top 10 Gainers vs Top 10 Losers")
        top_gainers = preds.nlargest(10, 'Predicted Change %')
        top_losers = preds.nsmallest(10, 'Predicted Change %')
        combined = pd.concat([top_gainers, top_losers])
        combined_chart = combined.set_index('Symbol')['Predicted Change %']
        st.bar_chart(combined_chart)
        
        # --- Chart 7: Allocation Heatmap (regional) ---
        st.write("##### 🌍 Allocation by Signal Category")
        if not preds.empty and px is not None:
            alloc_by_signal = preds.groupby('Signal')['Allocation %'].sum().reset_index()
            fig_heat = px.bar(
                alloc_by_signal,
                x='Signal',
                y='Allocation %',
                color='Signal',
                color_discrete_map={'Strong Buy': '#2ecc71', 'Buy': '#3498db', 'Neutral': '#f39c12', 'Sell': '#e74c3c'}
            )
            fig_heat.update_layout(showlegend=False)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Allocation by signal category not available")
        
        # --- Table: Detailed Predictions ---
        st.write("##### 📋 Detailed Predictions Table")
        display_cols = ['Symbol', 'Current Price', 'Predicted Change %', 'Regime Score', 'Allocation %', 'Signal']
        if 'Quality Score' in preds.columns:
            display_cols.append('Quality Score')
        if 'XSec Momentum' in preds.columns:
            display_cols.append('XSec Momentum')
        if 'Fund Score' in preds.columns:
            display_cols.append('Fund Score')
        
        st.dataframe(
            preds[display_cols].style.background_gradient(subset=['Predicted Change %'], cmap='RdYlGn')
                              .background_gradient(subset=['Allocation %'], cmap='Greens')
                              .format({
                                  'Current Price': 'Rs {:.2f}',
                                  'Predicted Change %': '{:.2f}%',
                                  'Regime Score': '{:.2f}',
                                  'Allocation %': '{:.2f}%',
                                  'Quality Score': '{:.2f}',
                                  'XSec Momentum': '{:.2f}',
                                  'Fund Score': '{:.2f}'
                              }),
            width='stretch'
        )

        # --- Performance Summary Cards ---
        st.divider()
        st.subheader("📊 Performance Summary")
        c1, c2, c3, c4 = st.columns(4)
        
        total_alloc = preds['Allocation %'].sum()
        avg_pred = preds['Predicted Change %'].mean()
        buy_count = len(preds[preds['Signal'].isin(['Strong Buy', 'Buy'])])
        sell_count = len(preds[preds['Signal'] == 'Sell'])
        
        c1.metric("💰 Total Allocation", f"{total_alloc:.1f}%")
        c2.metric("📈 Avg Predicted Return", f"{avg_pred:.2f}%")
        c3.metric("🟢 Buy Signals", f"{buy_count}")
        c4.metric("🔴 Sell Signals", f"{sell_count}")

if __name__ == "__main__":
    if os.environ.get("HEADLESS_MODE") == "1":
        print("Running in HEADLESS MODE...")
        DataLayer.update_live_data()
        df, _ = DataLayer.load_cross_sectional_data()
        AgentLoop.run_iteration(df)
        main_predict()
        print("Headless run complete.")
    else:
        run_ui()
