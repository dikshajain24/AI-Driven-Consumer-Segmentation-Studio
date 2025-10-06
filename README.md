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

---

## 📸 Screenshots

Explore the **AI-Driven Consumer Segmentation Studio** interface — an end-to-end view of how customer insights come to life.

### 🏠 Main Analytics Dashboard
Get a quick summary of your dataset, KPIs, and behavioral metrics before clustering.
<br>
<img width="1890" height="876" alt="Main Analytics" src="https://github.com/user-attachments/assets/1984d3c1-01f6-4b19-8318-3b2f7aaba904" />

---

### 🔍 Interactive Filtering
Filter by **Country**, **Monetary Threshold**, or toggle between demo and main datasets to focus your analysis.
<br>
<img width="1920" height="809" alt="Filtering by Clustering" src="https://github.com/user-attachments/assets/2103d97c-cba8-4a5b-a5c4-f57acbf519b7" />

---

### 🎯 Cluster Visualization
Visualize customer segments in 2D and 3D space using **PCA-based scatter plots** and explore behavior patterns at a glance.
<br>
<img width="1920" height="904" alt="Clusters & Visualization" src="https://github.com/user-attachments/assets/1ebaf868-e5c4-4296-a870-919b80027d8f" />

---

### 🧭 Cluster Filtering & Insights
Fine-tune the number of clusters or select different feature combinations (Recency, Frequency, Monetary, etc.) for new insights.
<br>
<img width="1920" height="902" alt="Clusters by filters" src="https://github.com/user-attachments/assets/76350535-0f2a-4778-b210-cf45a04a827d" />

---

### 💡 Marketing Recommendations
Automatically generated **actionable insights** for each cluster — identify Loyalists, Deal Seekers, and Dormant Customers with suggested strategies.
<br>
<img width="1912" height="902" alt="Recommendations" src="https://github.com/user-attachments/assets/bcc211f2-2999-478a-8f0b-ce4218924758" />

---

### 📥 Download Segmentation Results
Export clustered customer data as CSV and save trained models for further analysis or integration.
<br>
<img width="1854" height="390" alt="Download Consumer Segmentation" src="https://github.com/user-attachments/assets/7bc0fa4b-29f0-4c3e-b0b1-8c5c1dfb094a" />

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


❤️ About
Developed by Diksha Jain. 
