# demo_generate_sample.py
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

def generate_demo(path="data/sample_online_retail.csv", n_tx=2000, n_customers=500, seed=42):
    np.random.seed(seed)
    if not os.path.exists('data'):
        os.makedirs('data')
    customers = [f"C{str(i).zfill(5)}" for i in range(1, n_customers+1)]
    invoices = [f"INV{100000+i}" for i in range(n_tx)]
    df = pd.DataFrame({
        "InvoiceNo": np.random.choice(invoices, n_tx),
        "StockCode": np.random.randint(10000,99999, n_tx).astype(str),
        "Description": np.random.choice(["Shirt","Lipstick","Candle","Bag","Shoe","Perfume","Towel","Brush"], n_tx),
        "Quantity": np.random.randint(1,6, n_tx),
        "InvoiceDate": [datetime(2024,1,1) + timedelta(days=int(x)) for x in np.random.randint(0,365, n_tx)],
        "UnitPrice": np.round(np.random.exponential(scale=20, size=n_tx)+1, 2),
        "CustomerID": np.random.choice(customers, n_tx),
        "Country": np.random.choice(["United Kingdom","France","Germany","EIRE"], n_tx)
    })
    df["Amount"] = df["Quantity"] * df["UnitPrice"]
    df.to_csv(path, index=False)
    print(f"✅ Demo sample saved to {path}")

if __name__ == "__main__":
    generate_demo()
