import unittest
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import AgentLoop, MEMORY_FILE

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.original_mem = None
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                self.original_mem = f.read()
                
    def tearDown(self):
        if self.original_mem is not None:
            with open(MEMORY_FILE, 'w') as f:
                f.write(self.original_mem)
        else:
            if os.path.exists(MEMORY_FILE):
                os.remove(MEMORY_FILE)
                
    def test_memory_io(self):
        dummy_mem = [{"attempt_number": 1, "resulting_mae": 0.5, "sharpe": 1.2}]
        AgentLoop.save_memory(dummy_mem)
        loaded = AgentLoop.load_memory()
        
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["attempt_number"], 1)
        self.assertEqual(loaded[0]["sharpe"], 1.2)

if __name__ == '__main__':
    unittest.main()
