import unittest
import pandas as pd
import sys
import os
from io import StringIO
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from terminal_reader import main

class TestTerminalReader(unittest.TestCase):
    def setUp(self):
        # Create a dummy predictions file
        self.test_file = "latest_predictions.csv"
        # Temporarily change working directory to test dir or just mock os.path.exists
        
        # We will mock the predictions file reading by patching pandas and os
        self.dummy_data = pd.DataFrame({
            'Symbol': ['NICA', 'NABIL', 'GBIME', 'CIT', 'NMB'],
            'Current Price': [500, 600, 700, 800, 900],
            'Predicted Change %': [2.5, -1.2, 0.5, -0.8, 3.0],
            'Signal': ['Strong Buy', 'Sell', 'Buy', 'Sell', 'Strong Buy']
        })
        
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch('sys.stdout', new_callable=StringIO)
    def test_terminal_reader_output(self, mock_stdout, mock_read_csv, mock_exists):
        mock_exists.return_value = True
        mock_read_csv.return_value = self.dummy_data
        
        main()
        
        output = mock_stdout.getvalue()
        
        # Check if output contains expected sections and data
        self.assertIn("NEPSE CROSS-SECTIONAL PREDICTIONS", output)
        self.assertIn("TOP 5 PREDICTED GAINERS", output)
        self.assertIn("TOP 5 PREDICTED LOSERS", output)
        self.assertIn("NMB", output)
        self.assertIn("NABIL", output)
        self.assertIn("+ 3.00%", output)

    @patch('os.path.exists')
    @patch('sys.stdout', new_callable=StringIO)
    def test_terminal_reader_no_file(self, mock_stdout, mock_exists):
        mock_exists.return_value = False
        
        main()
        
        output = mock_stdout.getvalue()
        self.assertIn("not found. Run the agent first", output)

if __name__ == '__main__':
    unittest.main()
