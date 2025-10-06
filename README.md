# 🧠 AI-Driven Consumer Segmentation Studio

An **interactive Streamlit web app** that performs **AI-driven customer segmentation** using behavioral and RFM (Recency, Frequency, Monetary) analytics.  
Built with Python, Pandas, Scikit-learn, and Plotly — this app helps businesses uncover **customer personas** like *High-Value Loyalists*, *Deal Seekers*, and *Dormant Users* for smarter marketing strategies.

---

## 🚀 Live Demo

👉 https://ai-driven-consumer-segmentation-studio-w7nmv5fxikojcnkgszvivv.streamlit.app/ 
*(Note: It may take a minute to load on first run)*

---

## 🎯 Project Overview

This project demonstrates how to combine **machine learning (K-Means clustering)** with **behavioral metrics** to analyze customer shopping patterns.

**Core idea:**  
We take raw transactional data → clean & engineer features → cluster customers into segments → generate **actionable marketing insights**.

---

## 🧩 Features

✅ Upload or use the pre-loaded **demo dataset**  
✅ Automatic data cleaning and normalization  
✅ Compute **RFM + behavioral features**  
✅ Interactive **K-Means clustering**  
✅ **2D & 3D visualizations** (Plotly scatter, radar, histograms)  
✅ Personalized **marketing recommendations** for each segment  
✅ Export segmented customer CSV and trained model  
✅ Country-level filtering and KPI dashboard  

---

## 🧠 How It Works (App Flow)

1. **Upload your data** or use the demo dataset (`data/sample_online_retail.csv`).  
2. The app automatically:
   - Cleans your data (removes nulls, cancellations, negative quantities)
   - Parses invoice dates properly (`InvoiceDate`)
   - Computes behavioral metrics:
     - `Recency` → days since last purchase  
     - `Frequency` → total purchases per customer  
     - `Monetary` → total spending  
     - `AvgOrderValue`, `AvgItemPerOrder`, `CategoryDiversity`  
3. Apply **country filter** (optional)  
4. Choose number of clusters (`k`) and features for segmentation.  
5. Visualize the clusters:
   - Cluster sizes (bar chart)
   - PCA scatter plot
   - Radar chart for behavioral comparison  
6. See **auto-generated marketing recommendations** for each segment.  
7. **Download** clustered dataset (`clustered_customers.csv`) and model file.

---

## 🧾 Which CSV to Use

You have **two options** when running or testing the app:

| Option | File | Path | Description |
|---------|------|------|--------------|
| ✅ **Demo Dataset (Recommended)** | `sample_online_retail.csv` | `/data/sample_online_retail.csv` | Use this for recruiters/testing (auto-created by `demo_generate_sample.py`). |
| 🛒 **Main Dataset** | `online_retail_II.csv` | `/data/online_retail_II.csv` | Use this for real customer segmentation — full data from your Kaggle retail dataset. |
| 🧍 **Custom Dataset** | *(upload your own)* | via app upload | Must contain required columns listed below. |

---

## 📋 Required Columns

Your CSV **must contain** the following fields (case-insensitive):

| Required Column | Accepted Alternatives | Description |
|-----------------|-----------------------|-------------|
| `InvoiceNo` | `Invoice`, `OrderID`, `Order No` | Unique transaction ID |
| `InvoiceDate` | `OrderDate`, `Date` | Date of transaction (DD/MM/YY HH:MM) |
| `CustomerID` | `Customer ID`, `CustID` | Unique customer identifier |
| `Quantity` | `Qty` | Units purchased |
| `UnitPrice` | `Price` | Item price |
| `Country` *(optional)* | — | Enables country-level filter |
| `Description` *(optional)* | `Product` | For category diversity metric |

💡 The app automatically detects and renames common variants — you don’t need to change your file’s column names manually.

---

## ⚙️ Setup & Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/dikshajain24/AI-Driven-Consumer-Segmentation-Studio.git
cd AI-Driven-Consumer-Segmentation-Studio
2️⃣ Create Virtual Environment
python -m venv .venv
Activate it:
Windows PowerShell
.\.venv\Scripts\Activate.ps1

Mac/Linux
source .venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ (Optional) Generate Demo Dataset
python demo_generate_sample.py

5️⃣ Run the App
streamlit run app.py


Streamlit will start the app — open the local URL shown in your terminal (e.g. http://localhost:8501).
💻 Requirements

Your requirements.txt includes:
streamlit>=1.26.0
pandas>=2.2
numpy>=1.23
scikit-learn>=1.2
plotly>=5.14
joblib>=1.2

📊 Example Output
Metric	Example
Total Customers	483
Avg Customer LTV	$268.41
Active Customers	50.1%

Clusters Found (Example)
Cluster	Segment	Description
0	High-Value Loyalists	Frequent, high-spending, low-recency buyers
1	Deal Seekers	Medium spend, high frequency
2	Dormant Users	Infrequent, long time since last purchase
3	New Buyers	Recent one-time shoppers

📦 Output Files
File	Description
clustered_customers.csv	Exported customer data with cluster assignments
models/kmeans_model.joblib	Saved K-Means model
data/sample_online_retail.csv	Demo dataset
data/online_retail_II.csv	Full dataset (optional)

🧭 App Navigation
Tab	Purpose
Overview	Dataset snapshot, top products, spend histogram
Clusters	Select features → visualize clusters (bar, scatter, radar)
Recommendations	Auto-generated marketing actions per segment
User Manual	Step-by-step guide for recruiters
Download	Export results as CSV and save models

🧠 Built With
Python 3.11
Streamlit – interactive app framework
Pandas & NumPy – data wrangling
Scikit-learn – K-Means clustering, PCA
Plotly – interactive data visualization

📸 Screenshots

❤️ About
Developed by Diksha Jain. 
