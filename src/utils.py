# src/utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# ------------ Column normalization ------------
def normalize_columns(df):
    """
    Map common column name variants to canonical names used by the app.
    Returns a copy of df with renamed columns and robust InvoiceDate parsing.
    """
    df = df.copy()
    # Standardize whitespace and strip column names
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    mapping = {}
    # Invoice
    for x in ['InvoiceNo', 'Invoice No', 'Invoice', 'OrderID', 'Order_ID', 'Order Id']:
        if x in df.columns:
            mapping[x] = 'InvoiceNo'
            break
    # InvoiceDate
    for x in ['InvoiceDate', 'Invoice Date', 'OrderDate', 'Order Date', 'Date']:
        if x in df.columns:
            mapping[x] = 'InvoiceDate'
            break
    # CustomerID
    for x in ['CustomerID', 'Customer Id', 'Customer ID', 'CustID', 'cust_id', 'Customer_Id']:
        if x in df.columns:
            mapping[x] = 'CustomerID'
            break
    # Quantity
    for x in ['Quantity', 'Qty', 'quantity']:
        if x in df.columns:
            mapping[x] = 'Quantity'
            break
    # UnitPrice / Price
    for x in ['UnitPrice', 'Unit Price', 'Price', 'price', 'Unitprice']:
        if x in df.columns:
            mapping[x] = 'UnitPrice'
            break
    # Amount / Revenue
    for x in ['Amount', 'Revenue', 'Total', 'LineTotal']:
        if x in df.columns:
            mapping[x] = 'Amount'
            break
    # Description
    for x in ['Description', 'Product', 'ProductDescription', 'Desc']:
        if x in df.columns:
            mapping[x] = 'Description'
            break
    # Country
    for x in ['Country', 'country', 'CountryName']:
        if x in df.columns:
            mapping[x] = 'Country'
            break

    if mapping:
        df = df.rename(columns=mapping)

    # ---------- Robust date parsing ----------
    if 'InvoiceDate' in df.columns:
        # Try the known Online Retail II format first (fast, exact)
        # Example formats in that dataset look like: "12/1/10 8:26" meaning day/month/year hour:minute
        try:
            parsed = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%y %H:%M', errors='coerce')
        except Exception:
            parsed = pd.to_datetime(df['InvoiceDate'], errors='coerce')

        # If any rows failed (NaT), fallback to a general parse with dayfirst=True
        # Note: we do NOT use infer_datetime_format (deprecated) — pandas will use its consistent parsing.
        if parsed.isna().any():
            fallback = pd.to_datetime(df['InvoiceDate'], dayfirst=True, errors='coerce')
            # prefer parsed values, use fallback where parsed is NaT
            parsed = parsed.combine_first(fallback)

        df['InvoiceDate'] = parsed

    return df


# ------------ Loading helpers ------------
def load_retail_data(path):
    """
    Load CSV/Excel retail file and normalize columns.
    """
    if str(path).lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(path)
    else:
        try:
            # attempt to parse InvoiceDate column while reading (but safe fallback below)
            df = pd.read_csv(path, parse_dates=['InvoiceDate'], low_memory=False)
        except Exception:
            df = pd.read_csv(path, encoding='ISO-8859-1', low_memory=False)

    df = normalize_columns(df)

    # Compute Amount if possible
    if 'Amount' not in df.columns and 'UnitPrice' in df.columns and 'Quantity' in df.columns:
        try:
            df['Amount'] = df['UnitPrice'].astype(float) * df['Quantity'].astype(float)
        except Exception:
            pass

    return df


# ------------ Cleaning & Feature engineering ------------
def clean_data(df):
    """
    Remove negative/zero quantities and cancellations (InvoiceNo starting with C).
    Will first normalize columns (safe to call on uploaded df).
    """
    df = normalize_columns(df)
    required = []
    if 'InvoiceNo' in df.columns:
        required.append('InvoiceNo')
    if 'InvoiceDate' in df.columns:
        required.append('InvoiceDate')
    if 'CustomerID' in df.columns:
        required.append('CustomerID')

    if required:
        df = df.dropna(subset=required, how='any')

    if 'Quantity' in df.columns:
        # ensure numeric and remove non-positive
        try:
            df = df[df['Quantity'].astype(float) > 0]
        except Exception:
            # if conversion fails, keep rows where Quantity is truthy and numeric-like
            df = df[df['Quantity'].notna()]

    if 'InvoiceNo' in df.columns:
        try:
            df = df[~df['InvoiceNo'].astype(str).str.startswith('C', na=False)]
        except Exception:
            pass

    if 'CustomerID' in df.columns:
        df['CustomerID'] = df['CustomerID'].astype(str)

    if 'Amount' not in df.columns and 'UnitPrice' in df.columns and 'Quantity' in df.columns:
        try:
            df['Amount'] = df['UnitPrice'].astype(float) * df['Quantity'].astype(float)
        except Exception:
            df['Amount'] = np.nan

    return df


def compute_rfm(df):
    """
    Compute RFM: Recency (days), Frequency (#invoices), Monetary (sum).
    """
    df = normalize_columns(df)
    if 'InvoiceDate' not in df.columns or 'CustomerID' not in df.columns:
        raise ValueError("Missing InvoiceDate or CustomerID required to compute RFM.")

    snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    invoice_col = 'InvoiceNo' if 'InvoiceNo' in df.columns else df.columns[0]

    agg = df.groupby('CustomerID').agg(
        recency=('InvoiceDate', lambda x: (snapshot - x.max()).days),
        frequency=(invoice_col, 'nunique'),
        monetary=('Amount', 'sum') if 'Amount' in df.columns else ('UnitPrice', 'sum')
    ).reset_index().rename(columns={'recency': 'Recency', 'frequency': 'Frequency', 'monetary': 'Monetary'})
    return agg


def add_behavioral_metrics(df):
    """
    Create customer-level features: Frequency, Monetary, Recency, AvgOrderValue, CategoryDiversity, AvgItemPerOrder
    """
    df = normalize_columns(df)
    if 'InvoiceDate' not in df.columns or 'CustomerID' not in df.columns:
        raise ValueError("Missing InvoiceDate or CustomerID required to compute features.")

    snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    inv_col = 'InvoiceNo' if 'InvoiceNo' in df.columns else df.columns[0]
    g = df.groupby('CustomerID')

    out = pd.DataFrame({'CustomerID': g.size().index})
    out = out.set_index('CustomerID')
    out['Frequency'] = g[inv_col].nunique()

    if 'Amount' in df.columns:
        out['Monetary'] = g['Amount'].sum()
    elif 'UnitPrice' in df.columns:
        out['Monetary'] = g['UnitPrice'].sum()
    else:
        out['Monetary'] = g.size()

    out['Recency'] = g['InvoiceDate'].max().apply(lambda d: (snapshot - d).days).values
    out['AvgOrderValue'] = out['Monetary'] / out['Frequency']

    if 'Quantity' in df.columns:
        out['AvgItemPerOrder'] = g['Quantity'].sum() / out['Frequency']
    else:
        out['AvgItemPerOrder'] = 1

    if 'Description' in df.columns:
        out['CategoryDiversity'] = g['Description'].nunique()
    else:
        out['CategoryDiversity'] = 1

    out = out.reset_index()
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


# ------------ ML helpers ------------
def scale_features(df, feature_cols):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    return X, scaler


def run_kmeans(X, n_clusters=4, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
    return km, labels, sil


def reduce_pca(X, n_components=3):
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)
    return coords, pca


def save_model(model, path='models/kmeans_model.joblib'):
    joblib.dump(model, path)
