import os
import pandas as pd

PREDICTIONS_FILE = "latest_predictions.csv"

def main():
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"Error: {PREDICTIONS_FILE} not found. Run the agent first to generate predictions.")
        return
        
    try:
        df = pd.read_csv(PREDICTIONS_FILE)
        
        if df.empty:
            print("Predictions file is empty.")
            return
            
        print("="*80)
        print(" " * 25 + "NEPSE CROSS-SECTIONAL PREDICTIONS")
        print("="*80)
        
        print("\n🟢 TOP 5 PREDICTED GAINERS:")
        print("-" * 80)
        gainers = df.nlargest(5, 'Predicted Change %')
        for idx, row in gainers.iterrows():
            sym = row['Symbol']
            price = row['Current Price']
            change = row['Predicted Change %']
            alloc = row.get('Allocation %', 0.0)
            rs = row.get('Regime Score', 0.5)
            signal = row['Signal']
            print(f"{sym:<8} | Price: {price:>7.2f} | Change: +{change:>5.2f}% | Alloc: {alloc:>5.1f}% | Regime: {rs:.2f} | Sig: {signal}")
            
        print("\n🔴 TOP 5 PREDICTED LOSERS:")
        print("-" * 80)
        losers = df.nsmallest(5, 'Predicted Change %')
        for idx, row in losers.iterrows():
            sym = row['Symbol']
            price = row['Current Price']
            change = row['Predicted Change %']
            alloc = row.get('Allocation %', 0.0)
            rs = row.get('Regime Score', 0.5)
            signal = row['Signal']
            sign = "+" if change > 0 else ""
            print(f"{sym:<8} | Price: {price:>7.2f} | Change: {sign}{change:>5.2f}% | Alloc: {alloc:>5.1f}% | Regime: {rs:.2f} | Sig: {signal}")
            
        print("\n" + "="*80 + "\n")
            
    except Exception as e:
        print(f"Error reading predictions: {e}")

if __name__ == "__main__":
    main()
