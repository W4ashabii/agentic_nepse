"""
Enhanced Quant Features from nepse-quant-terminal
- Market Regime Detection (bull/bear/neutral)
- Gold/Silver Regime Overlay
- Additional Signal Types
- Enhanced Validation
"""
import numpy as np
import pandas as pd
from datetime import date
from typing import Tuple, Optional, Dict

# NEPSE Calendar constants
TRADING_WEEK_MODE = "mon_fri"  # Default: Monday-Friday

# Known NEPSE holidays
KNOWN_HOLIDAYS_2025 = {
    date(2025, 1, 11),  # Prithvi Jayanti
    date(2025, 1, 14),  # Maghe Sankranti
    date(2025, 1, 30),  # Martyrs' Memorial Day
    date(2025, 2, 19),  # National Democracy Day
    date(2025, 2, 26),  # Maha Shivaratri
    date(2025, 3, 8),   # Nari Dibas
    date(2025, 4, 6),   # Ram Nawami
    date(2025, 4, 14),  # Nepali New Year
    date(2025, 5, 1),   # Majdoor Divas
    date(2025, 5, 12),  # Buddha Jayanti
    date(2025, 5, 29),  # Ganatantra Diwas
    date(2025, 6, 7),   # Eid ul-Adha
    date(2025, 8, 9),   # Janai Purnima
    date(2025, 8, 16),  # Krishna Janmashtami
    date(2025, 9, 17),  # National Mourning Day
    date(2025, 9, 19),  # Constitution Day
    date(2025, 9, 22),  # Ghatasthapana (Dashain start)
    date(2025, 9, 29),  # Phulpati
    date(2025, 9, 30),  # Astami
    date(2025, 10, 1),  # Nawami
    date(2025, 10, 2),  # Dashami
    date(2025, 10, 3),  # Ekadashi
    date(2025, 10, 4),  # Duwadashi
    date(2025, 10, 5),  # Post-Dashain
    date(2025, 10, 6),  # Post-Dashain
    date(2025, 10, 20), # Laxmi Puja (Tihar)
    date(2025, 10, 21), # Gai Tihar
    date(2025, 10, 22), # Gobhardan Pujan
    date(2025, 10, 23), # Bhai Tika
    date(2025, 10, 24), # Tihar Holiday
    date(2025, 10, 27), # Chhath Parwa
}

KNOWN_HOLIDAYS_2026 = {
    date(2026, 1, 15),  # Maghe Sankranti
    date(2026, 1, 19),  # Sonam Losar
    date(2026, 2, 15),  # Maha Shivaratri
    date(2026, 2, 19),  # National Democracy Day
    date(2026, 3, 8),   # Nari Dibas
    date(2026, 5, 1),   # Majdoor Divas
    date(2026, 9, 19),  # Constitution Day
    date(2026, 10, 17), # Phulpati (Dashain)
    date(2026, 10, 18), # Astami
    date(2026, 10, 20), # Nawami
    date(2026, 10, 21), # Dashami
    date(2026, 10, 22), # Ekadashi
    date(2026, 11, 8),  # Laxmi Puja (Tihar)
    date(2026, 11, 9),  # Gobhardan Pujan
    date(2026, 11, 10), # Bhai Tika
}

KNOWN_HOLIDAYS = KNOWN_HOLIDAYS_2025 | KNOWN_HOLIDAYS_2026

# Dashain/Tihar rally periods
DASHAIN_START_DATES = {
    2025: date(2025, 9, 22),
    2026: date(2026, 10, 17),
}

TIHAR_START_DATES = {
    2025: date(2025, 10, 20),
    2026: date(2026, 11, 8),
}


def is_trading_day(dt: date) -> bool:
    """Check if a date is a NEPSE trading day."""
    if TRADING_WEEK_MODE == "sun_thu":
        if dt.weekday() in (4, 5):  # Friday, Saturday
            return False
    else:
        if dt.weekday() in (5, 6):  # Saturday, Sunday
            return False
    
    if dt in KNOWN_HOLIDAYS:
        return False
    
    return True


def get_regime_score(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Calculate market regime score based on rolling NEPSE return.
    Returns: 'bull', 'neutral', 'bear' or numeric score
    """
    if 'NEPSE_Index' not in df.columns:
        df = df.copy()
        df['Market_Return'] = df.groupby('Date')['Close'].transform(lambda x: x.pct_change().mean())
        returns = df['Market_Return'].rolling(window).mean()
    else:
        returns = df['NEPSE_Index'].pct_change().rolling(window).mean()
    
    bear_threshold = -0.02
    bull_threshold = 0.02
    
    regime = returns.apply(
        lambda x: 'bull' if x > bull_threshold 
        else ('bear' if x < bear_threshold else 'neutral')
    )
    return regime.fillna('neutral')


def get_gold_regime(gold_price_df: Optional[pd.DataFrame] = None) -> str:
    """
    Determine gold/silver regime for capital deployment.
    Returns: 'risk_on', 'neutral', 'risk_off'
    """
    if gold_price_df is not None and 'Close' in gold_price_df.columns:
        vol = gold_price_df['Close'].pct_change().rolling(20).std()
        avg_vol = vol.mean()
        
        if avg_vol < 0.01:
            return 'risk_on'
        elif avg_vol > 0.03:
            return 'risk_off'
    
    return 'neutral'


def get_capital_deployment_percentage(regime: str) -> float:
    """Adjust capital deployment based on gold regime."""
    deployment_map = {
        'risk_on': 1.0,
        'neutral': 0.97,
        'risk_off': 0.90,
    }
    return deployment_map.get(regime, 0.97)


def calculate_xsec_momentum(prices_df: pd.DataFrame, symbols: list, 
                           lookback: int = 180) -> Dict[str, float]:
    """Calculate cross-sectional momentum: 6m minus 1m return."""
    momentum_scores = {}
    
    for sym in symbols:
        if sym not in prices_df.columns or prices_df[sym].isna().all():
            momentum_scores[sym] = 0.0
            continue
        
        prices = prices_df[sym].dropna()
        if len(prices) < lookback:
            momentum_scores[sym] = 0.0
            continue
        
        six_month_return = (prices.iloc[-1] / prices.iloc[-lookback]) - 1 if len(prices) >= lookback else 0.0
        one_month_return = (prices.iloc[-1] / prices.iloc[-20]) - 1 if len(prices) >= 20 else 0.0
        
        momentum = six_month_return - one_month_return
        momentum_scores[sym] = momentum
    
    return momentum_scores


def calculate_quality_score(fundamentals_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Calculate quality score based on ROE, debt-to-equity, earnings stability."""
    if fundamentals_df is None:
        return {}
    
    quality_scores = {}
    
    for sym, grp in fundamentals_df.groupby('Symbol'):
        if len(grp) < 4:
            quality_scores[sym] = 0.5
            continue
        
        scores = []
        
        if 'ROE' in grp.columns:
            avg_roe = grp['ROE'].mean()
            scores.append(min(avg_roe / 0.20, 1.0))
        
        if 'Debt_to_Equity' in grp.columns:
            dte = grp['Debt_to_Equity'].mean()
            scores.append(max(0, 1 - dte / 2.0))
        
        if 'EPS' in grp.columns:
            cv = grp['EPS'].std() / grp['EPS'].mean() if grp['EPS'].mean() != 0 else 1
            scores.append(max(0, 1 - cv))
        
        if scores:
            quality_scores[sym] = np.mean(scores)
        else:
            quality_scores[sym] = 0.5
    
    return quality_scores


def calculate_quarterly_fundamental_scores(fundamentals_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate scores based on quarterly earnings growth and revenue growth."""
    scores = {}
    
    for sym, grp in fundamentals_df.groupby('Symbol'):
        if len(grp) < 2:
            scores[sym] = 0.5
            continue
        
        grp = grp.sort_values('Date', ascending=False)
        
        if len(grp) >= 4:
            current_eps = grp['EPS'].iloc[0]
            prior_eps = grp['EPS'].iloc[4] if len(grp) > 4 else grp['EPS'].iloc[-1]
            
            eps_growth = (current_eps - prior_eps) / abs(prior_eps) if prior_eps != 0 else 0
            revenue_growth = (grp['Revenue'].iloc[0] - grp['Revenue'].iloc[4]) / abs(grp['Revenue'].iloc[4]) if len(grp) > 4 and grp['Revenue'].iloc[4] != 0 else 0
            
            score = (eps_growth + revenue_growth) / 2
            normalized = min(max(score + 0.5, 0), 1)
        else:
            normalized = 0.5
        
        scores[sym] = normalized
    
    return scores


def apply_regime_filter(predictions: Dict[str, float], regime: str, 
                       regime_limits: Dict[str, int] = None) -> Dict[str, float]:
    """Apply regime-based filtering to predictions."""
    if regime_limits is None:
        regime_limits = {'bull': 10, 'neutral': 8, 'bear': 3}
    
    max_positions = regime_limits.get(regime, 8)
    
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_n = sorted_preds[:max_positions]
    
    return dict(top_n)


def walk_forward_validation(df: pd.DataFrame, model_trainer, 
                           train_start: str, train_end: str,
                           test_period: int = 30, step: int = 30) -> Dict:
    """Perform walk-forward validation on the model."""
    results = {
        'train_start': train_start,
        'train_end': train_end,
        'test_periods': [],
        'sharpe_scores': [],
        'profit_scores': [],
    }
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    train_dates = df['Date'].unique()
    train_idx_start = 0
    train_idx_end = len(train_dates) - test_period
    
    while train_idx_end < len(train_dates):
        train_start_date = train_dates[train_idx_start]
        train_end_date = train_dates[train_idx_end]
        
        test_start_date = train_end_date
        test_end_date = train_end_date + pd.Timedelta(days=test_period)
        
        train_df = df[(df['Date'] >= train_start_date) & (df['Date'] <= train_end_date)]
        test_df = df[(df['Date'] > train_end_date) & (df['Date'] <= test_end_date)]
        
        if len(train_df) < 100 or len(test_df) < 10:
            train_idx_end += step
            continue
        
        try:
            mae, sharpe, calmar, profit, gt_score, model, cols = model_trainer(
                train_df, test_params={'xgboost': {}, 'lightgbm': {}}, custom_features={}
            )
            
            if model is not None and len(test_df) > 10:
                results['sharpe_scores'].append(sharpe)
                results['profit_scores'].append(profit)
                results['test_periods'].append({
                    'train_start': str(train_start_date.date()),
                    'train_end': str(train_end_date.date()),
                    'test_start': str(test_start_date.date()),
                    'test_end': str(test_end_date.date()),
                    'sharpe': sharpe,
                    'profit': profit,
                })
        except Exception as e:
            pass
        
        train_idx_end += step
    
    if results['sharpe_scores']:
        results['avg_sharpe'] = np.mean(results['sharpe_scores'])
        results['avg_profit'] = np.mean(results['profit_scores'])
        results['sharpe_std'] = np.std(results['sharpe_scores'])
        results['success_rate'] = sum(1 for s in results['sharpe_scores'] if s > 0) / len(results['sharpe_scores'])
    else:
        results['avg_sharpe'] = 0
        results['avg_profit'] = 0
        results['sharpe_std'] = 0
        results['success_rate'] = 0
    
    return results
