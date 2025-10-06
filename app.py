# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os

from src.utils import (
    load_retail_data,
    clean_data,
    add_behavioral_metrics,
    scale_features,
    run_kmeans,
    reduce_pca,
    save_model,
)

import plotly.express as px
import plotly.graph_objects as go

# ----------------- Page config & small style -----------------
st.set_page_config(page_title="✨ Consumer Segmentation Studio", layout="wide", page_icon="🧩")

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(90deg, #ffffff, #f7fbff); }
      .title { font-size:28px; font-weight:700; }
      .muted { color: #6c757d; }
      .kpi { font-size:20px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Header -----------------
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.markdown("<div class='title'>🧩 AI-Driven Consumer Segmentation Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Upload, cluster, and visualize customer behavior — RFM segmentation powered by K-Means.</div>", unsafe_allow_html=True)

st.info("💡 Quick Start for Recruiters: Keep 'Use demo/sample data' checked for instant testing. Uncheck to load the full dataset or upload your own CSV.")
st.write("---")

# ----------------- Sidebar controls -----------------
st.sidebar.header("⚙️ Data & Model Controls")
uploaded = st.sidebar.file_uploader("Upload transactional CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use demo/sample data (data/sample_online_retail.csv)", value=True)
n_clusters = st.sidebar.slider("Clusters (k)", min_value=2, max_value=10, value=4)
min_monetary = st.sidebar.number_input("Min monetary filter", min_value=0.0, value=0.0, step=10.0)
run_button = st.sidebar.button("Run segmentation ▶️")

# ----------------- Paths -----------------
data_path = "data/online_retail_II.csv"         # your full main file (optional)
sample_path = "data/sample_online_retail.csv"  # demo file created by demo_generate_sample.py

# ----------------- Data loaders (cached) -----------------
@st.cache_data(show_spinner=False)
def load_csv_path(path):
    """Load CSV robustly and normalize column names using utils loader if available."""
    try:
        df = load_retail_data(path)
        return df
    except Exception:
        try:
            df = pd.read_csv(path, parse_dates=["InvoiceDate"], low_memory=False)
        except Exception:
            df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)
        return df

@st.cache_data(show_spinner=False)
def load_uploaded_csv(ufile):
    try:
        df = pd.read_csv(ufile, parse_dates=["InvoiceDate"], low_memory=False, encoding="ISO-8859-1")
    except Exception:
        df = pd.read_csv(ufile, low_memory=False)
    return df

# ----------------- Decide source & load -----------------
df = None
if uploaded:
    try:
        df = load_uploaded_csv(uploaded)
        st.sidebar.success("Uploaded CSV loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded CSV: {e}")
        st.stop()
elif use_sample and os.path.exists(sample_path):
    df = load_csv_path(sample_path)
    st.sidebar.info("Using demo/sample dataset (recommended for recruiters).")
elif not use_sample and os.path.exists(data_path):
    df = load_csv_path(data_path)
    st.sidebar.info("Using main dataset: data/online_retail_II.csv")
else:
    st.error("❌ No dataset found. Upload a CSV or generate demo/sample data first (run demo_generate_sample.py).")
    st.stop()

# Basic column normalization
df.columns = [c.strip() for c in df.columns]

# ----------------- Country filter -----------------
country_opt = ["All"]
if "Country" in df.columns:
    country_opt = ["All"] + sorted(df["Country"].astype(str).dropna().unique().tolist())
country_choice = st.sidebar.selectbox("🌍 Filter by Country", options=country_opt, index=0)

if country_choice != "All":
    df = df[df["Country"] == country_choice]

# ----------------- Show raw data sample -----------------
st.subheader("Dataset Overview — sample transactions")
st.dataframe(df.head(8))

# ----------------- Clean & feature engineer -----------------
df = clean_data(df)
cust = add_behavioral_metrics(df)

if "Monetary" in cust.columns and min_monetary > 0:
    cust = cust[cust["Monetary"] >= min_monetary]

# ----------------- KPI cards -----------------
total_customers = cust.shape[0]
avg_ltv = cust["Monetary"].mean() if "Monetary" in cust.columns else 0
recent_threshold = np.percentile(cust["Recency"].dropna(), 50) if "Recency" in cust.columns else 0
active_pct = round((cust[cust["Recency"] <= recent_threshold].shape[0] / max(1, total_customers)) * 100, 1)

k1, k2, k3 = st.columns(3)
k1.metric("Total Customers", f"{total_customers:,}")
k2.metric("Avg Customer LTV", f"${avg_ltv:,.2f}")
k3.metric("Active Customers (%)", f"{active_pct}%")

st.write("---")

# ----------------- Tabs -----------------
tabs = st.tabs(["Overview", "Clusters", "Recommendations", "User Manual", "Download"])

# ---------- Overview ----------
with tabs[0]:
    st.header("📊 Data overview")
    st.write("Transactions (top rows):")
    st.dataframe(df.head(10))
    if "Monetary" in cust.columns:
        st.subheader("Monetary distribution")
        fig = px.histogram(cust, x="Monetary", nbins=50, marginal="box", title="Customer monetary distribution")
        st.plotly_chart(fig, use_container_width=True)
    if "Description" in df.columns:
        st.subheader("Top product descriptions")
        top_desc = df["Description"].value_counts().head(10).reset_index()
        top_desc.columns = ["Description", "Count"]
        st.dataframe(top_desc)

# ---------- Clusters ----------
with tabs[1]:
    st.header("🎯 Clustering & Visualization")
    available_feats = [c for c in ["Recency", "Frequency", "Monetary", "AvgOrderValue", "CategoryDiversity", "AvgItemPerOrder"] if c in cust.columns]
    selected_feats = st.multiselect("Select features for clustering", options=available_feats, default=(available_feats[:4] if len(available_feats) >= 4 else available_feats))

    if len(selected_feats) < 2:
        st.warning("Pick at least 2 features for meaningful clusters.")
    if run_button:
        X, scaler = scale_features(cust, selected_feats)
        model, labels, sil = run_kmeans(X, n_clusters=n_clusters)
        cust["Cluster"] = labels
        st.sidebar.metric("Silhouette Score", round(sil, 3))

        coords, _ = reduce_pca(X)
        cust["pc1"], cust["pc2"], cust["pc3"] = coords[:, 0], coords[:, 1], coords[:, 2]

        st.subheader("Cluster sizes")
        cluster_counts = cust["Cluster"].value_counts().sort_index().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        st.plotly_chart(px.bar(cluster_counts, x="Cluster", y="Count", title="Customers per cluster"), use_container_width=True)

        st.subheader("2D cluster scatter (PCA)")
        fig2d = px.scatter(cust, x="pc1", y="pc2", color="Cluster", hover_data=["CustomerID", "Monetary", "Frequency"], title="PC1 vs PC2 by cluster")
        st.plotly_chart(fig2d, use_container_width=True)

        st.subheader("Cluster profiles (means)")
        profile = cust.groupby("Cluster")[selected_feats].mean().round(2)
        st.dataframe(profile)

        for c in profile.index:
            vals = profile.loc[c].values.tolist()
            theta = profile.columns.tolist()
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=theta + [theta[0]], fill="toself", name=f"Cluster {c}"))
            fig_r.update_layout(title_text=f"Cluster {c} profile", polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig_r, use_container_width=True)

        os.makedirs("models", exist_ok=True)
        save_model(model, path=f"models/kmeans_k{n_clusters}.joblib")
        st.success(f"Model saved: models/kmeans_k{n_clusters}.joblib")
    else:
        st.info("Adjust features/filters & press 'Run segmentation' in the sidebar.")

# ---------- Recommendations ----------
with tabs[2]:
    st.header("💡 Marketing recommendations")
    if "Cluster" not in cust.columns:
        st.info("Run clustering first to generate recommendations.")
    else:
        prof = cust.groupby("Cluster").agg({"Monetary": "mean", "Frequency": "mean", "Recency": "mean"}).reset_index()
        recs = []
        for _, row in prof.iterrows():
            cid = int(row["Cluster"])
            if row["Monetary"] > prof["Monetary"].mean() and row["Frequency"] > prof["Frequency"].mean():
                label = "High-Value Loyalists"
                actions = "- VIP perks\n- Premium cross-sell\n- Early access"
            elif row["Frequency"] > prof["Frequency"].mean() and row["Monetary"] <= prof["Monetary"].mean():
                label = "Deal Seekers"
                actions = "- Coupons & bundles\n- Loyalty points"
            elif row["Recency"] <= prof["Recency"].quantile(0.25):
                label = "Recent Buyers"
                actions = "- Repeat purchase offer\n- Review request"
            else:
                label = "Occasional / Dormant"
                actions = "- Win-back campaign\n- Personalized discounts"
            recs.append((cid, label, actions))
        rec_df = pd.DataFrame(recs, columns=["Cluster", "Segment label", "Recommended actions"])
        st.dataframe(rec_df)

# ---------- User Manual ----------
with tabs[3]:
    st.header("📘 User Manual — How to use this app")
    st.markdown(
        """
**Purpose**  
This app performs RFM + behavioral segmentation on transaction data and provides marketing recommendations.

**Which files work? (Required columns)**  
- `InvoiceNo` (or `Invoice`, `OrderID`)  
- `InvoiceDate` (or `OrderDate`) — parseable date  
- `CustomerID` (or `Customer ID`)  
- `Quantity`  
- `UnitPrice` (or `Price`) — or `Amount` (precomputed)

**Optional**: `Description` (for category diversity), `Country` (filtering).

**Quick steps**  
1. Upload a CSV or toggle 'Use demo/sample data' (sample at `data/sample_online_retail.csv`).  
2. (Optional) Filter by Country and set min monetary threshold.  
3. Choose features and `k` then click **Run segmentation**.  
4. Explore Overview, Clusters, Recommendations.  
5. Download clustered customer CSV in the **Download** tab.

**Demo dataset**  
Use `demo_generate_sample.py` to create `data/sample_online_retail.csv` if you don't have a file.
"""
    )

# ---------- Download ----------
with tabs[4]:
    st.header("📥 Download / Export")
    if "Cluster" in cust.columns:
        out_csv = cust.to_csv(index=False).encode("utf-8")
        st.download_button("Download clustered customers (CSV)", out_csv, file_name="clustered_customers.csv", mime="text/csv")
        st.write("Model file(s) saved to `models/` after running clustering.")
    else:
        st.info("Run clustering to enable download.")
