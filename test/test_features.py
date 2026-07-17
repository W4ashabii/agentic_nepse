import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import FeatureEngineer

class TestFeatures(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
        np.random.seed(42)
        close_prices = np.cumsum(np.random.randn(250) + 0.1) + 100
        high_prices = close_prices + np.random.rand(250)
        low_prices = close_prices - np.random.rand(250)
        open_prices = close_prices + np.random.randn(250) * 0.5
        volume = np.random.randint(100, 1000, 250)
        
        self.df = pd.DataFrame({
            'Date': dates,
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        })
        
    def test_base_features(self):
        df_feat = FeatureEngineer.generate_base_features(self.df)
        self.assertFalse(df_feat.empty)
        self.assertIn('Target', df_feat.columns)
        self.assertIn('regime_score', df_feat.columns)
        self.assertFalse(df_feat['Target'].isna().any())

    def test_sandbox_eval(self):
        df_feat = FeatureEngineer.generate_base_features(self.df)
        custom = {"MyNewFeat": "df['Close'] * 2.5"}
        df_new = FeatureEngineer.apply_sandboxed_features(df_feat, custom)
        
        self.assertIn('MyNewFeat', df_new.columns)
        self.assertEqual(df_new['MyNewFeat'].iloc[0], df_feat['Close'].iloc[0] * 2.5)

if __name__ == '__main__':
    unittest.main()
