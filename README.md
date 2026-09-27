# 🛍️ Retail Sales Analytics & Customer Segmentation Dashboard

An interactive Data Mining & Data Warehousing (DMDW) case study dashboard built with **Python** and **Streamlit**. This application analyzes transactional e-commerce data from the UCI Online Retail dataset to reveal sales trends, top-performing products, and customer segments using **RFM Analysis** combined with **K-Means Clustering**.

---

## Key Features

- **Dataset Overview:** Quick metrics (Total Revenue, Order Count, Customer Base, Product Count) and country-wise revenue distribution via a custom donut chart.
- **Sales Trends Analysis:** Monthly revenue trends, order volume patterns, revenue breakdown by day of week, and Month-over-Month (MoM) growth tracking.
- **Product & Geographical Insights:** Top 10 best-selling products by quantity and revenue, top purchasing countries, and unit price distribution analysis.
- **Customer Segmentation (RFM + K-Means):**
  - Interactive **Elbow Curve** to select the optimal number of clusters ($K$).
  - **Silhouette Score** evaluation for cluster validation.
  - Automated segment labeling (*Champions*, *Loyal Customers*, *Potential Loyalists*, *At-Risk / Lost*).
  - Multi-dimensional visualizations: Cluster distribution bars, RFM Scatter Plots, and Box Plots.
  - Filterable customer RFM data table.
- **Dark Mode UI:** Sleek, consistent dark theme design system (`#0d1117`) across both Streamlit components and Matplotlib charts.

---

## 🛠️ Tech Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (StandardScaler, KMeans, Silhouette Score)
- **Data Source:** UCI Machine Learning Repository — *Online Retail Dataset*

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Installation & Setup

Clone this repository:
```bash
git clone https://github.com/your-username/retail-sales-rfm-segmentation.git
cd retail-sales-rfm-segmentation
