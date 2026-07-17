import unittest
import pandas as pd
import numpy as np
import sys
import os

# Ensure the project root is on the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import ModelTrainer

class TestModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 150
        self.df = pd.DataFrame({
            'Target': np.random.randn(n_samples) * 0.02,
            'Close': np.cumsum(np.random.randn(n_samples)) + 100,
            'Open': np.cumsum(np.random.randn(n_samples)) + 100,
            'Volume': np.random.randint(10000, 500000, n_samples),
            'SMA_10': np.random.randn(n_samples),
            'VOL_10': np.random.rand(n_samples)
        })

    def test_backtest(self):
        y_true = np.array([0.02, -0.01, 0.03])
        y_pred = np.array([0.01, -0.05, 0.02])
        open_next = np.array([100, 105, 110])
        metrics = ModelTrainer.run_backtest(y_true, y_pred, open_next)
        self.assertIsInstance(metrics, dict)
        self.assertIn('sharpe', metrics)
        self.assertIn('net_profit', metrics)
        self.assertIsInstance(metrics['sharpe'], float)
        self.assertIsInstance(metrics['net_profit'], float)

    def test_train_and_evaluate(self):
        params = {
            'xgboost': {'n_estimators': 10, 'max_depth': 2},
            'lightgbm': {'n_estimators': 10, 'max_depth': 2}
        }
        # Expect 7 return values now (including calmar and gt_score)
        mae, sharpe, calmar, profit, gt_score, model, cols = ModelTrainer.train_and_evaluate(self.df, params, {})
        self.assertLess(mae, 1.0)
        self.assertIsNotNone(model)
        self.assertTrue(0.0 <= gt_score <= 1.0)

if __name__ == '__main__':
    unittest.main()
