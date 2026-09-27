import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# --------------------------------------------------
# COLOUR PALETTE  (single source of truth)
# --------------------------------------------------

BG       = "#0d1117"   # chart / figure background
AX_BG    = "#161b22"   # axes background
GRID_CLR = "#30363d"   # grid lines
TEXT_CLR = "#e6edf3"   # labels / titles
TICK_CLR = "#8b949e"   # tick labels

C1 = "#6366f1"   # indigo  — primary
C2 = "#38bdf8"   # sky     — accent
C3 = "#10b981"   # emerald — positive / growth
C4 = "#f59e0b"   # amber
C5 = "#f43f5e"   # rose    — negative / loss

PALETTE_7 = [C1, C2, C3, C4, C5, "#a78bfa", "#34d399"]

# --------------------------------------------------
# GLOBAL MATPLOTLIB STYLE
# --------------------------------------------------

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    AX_BG,
    "axes.edgecolor":    GRID_CLR,
    "axes.labelcolor":   TEXT_CLR,
    "axes.titlecolor":   TEXT_CLR,
    "axes.grid":         True,
    "grid.color":        GRID_CLR,
    "grid.linestyle":    "--",
    "grid.alpha":        0.45,
    "xtick.color":       TICK_CLR,
    "ytick.color":       TICK_CLR,
    "text.color":        TEXT_CLR,
    "legend.facecolor":  AX_BG,
    "legend.edgecolor":  GRID_CLR,
    "legend.labelcolor": TEXT_CLR,
    "legend.title_fontsize": 9,
    "font.family":       "sans-serif",
    "font.size":         10,
    "savefig.facecolor": BG,
    "figure.dpi":        110,
})

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Retail Sales & Customer Segmentation",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* App background */
    .stApp { background-color: #0d1117; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.85rem; }
    [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.6rem; font-weight: 700; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #010409 !important; }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* Headings */
    h1 { color: #e6edf3 !important; }
    h2 { color: #6366f1 !important; }
    h3 { color: #38bdf8 !important; }
    hr { border-color: #30363d; }
    [data-testid="stDivider"] { border-color: #30363d !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("Retail Sales Trends & Customer Segmentation")
st.markdown(
    "Comprehensive analysis of retail transaction data: "
    "identifying **sales trends**, **top products**, and segmenting "
    "customers using **RFM Analysis + K-Means Clustering**."
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_data():
    # Read with explicit engine to avoid openpyxl warnings
    df = pd.read_excel("Online Retail.xlsx", engine="openpyxl")

    # Remove missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove cancelled transactions (InvoiceNo starts with 'C')
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove invalid Quantity / UnitPrice
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # Derived columns
    df["Revenue"]     = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["MonthYear"]   = df["InvoiceDate"].dt.to_period("M")
    df["DayOfWeek"]   = df["InvoiceDate"].dt.day_name()

    return df


df = load_data()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Analysis",
    [
        "Overview",
        "Sales Trends",
        "Product Analysis",
        "Customer Segmentation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset:** Online Retail (UCI ML Repository)  \n"
    "**Subject:** DMDW Case Study  \n"
    "**Methods:** RFM + K-Means"
)


# Helper — render and immediately free figure memory
def show_plot(fig):
    st.pyplot(fig)
    plt.close(fig)


# Unified axis styler
def style_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_CLR, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_CLR)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_CLR)
    ax.tick_params(colors=TICK_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    return ax


# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"\u00a3{df['Revenue'].sum():,.0f}")
    with col2:
        st.metric("Total Orders", f"{df['InvoiceNo'].nunique():,}")
    with col3:
        st.metric("Unique Customers", f"{df['CustomerID'].nunique():,}")
    with col4:
        st.metric("Unique Products", f"{df['StockCode'].nunique():,}")

    st.divider()
    date_min = df["InvoiceDate"].min().strftime("%d %b %Y")
    date_max = df["InvoiceDate"].max().strftime("%d %b %Y")
    st.markdown(f"**Data Range:** {date_min} to {date_max}")
    st.divider()

    # Revenue by Country pie chart
    st.subheader("Revenue Distribution by Top Countries")
    top_c  = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(6)
    other  = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False).iloc[6:].sum()
    pie_data = pd.concat([top_c, pd.Series({"Others": other})])

    total         = pie_data.sum()
    pct_labels    = [f"{v / total * 100:.1f}%" for v in pie_data.values]
    legend_labels = [f"{c}  —  {p}" for c, p in zip(pie_data.index, pct_labels)]

    fig, ax = plt.subplots(figsize=(9, 5))
    wedges, _ = ax.pie(
        pie_data,
        startangle=140,
        colors=PALETTE_7[:len(pie_data)],
        wedgeprops={"edgecolor": BG, "linewidth": 2.5, "width": 0.55},
        counterclock=False,
    )
    ax.text(0, 0, "Revenue\nShare", ha="center", va="center",
            fontsize=11, fontweight="bold", color=TEXT_CLR)
    ax.set_title("Revenue Share by Country", fontsize=13, fontweight="bold",
                 color=TEXT_CLR, pad=14)
    ax.legend(
        wedges, legend_labels,
        title="Country",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title_fontsize=10,
        frameon=True,
    )
    plt.tight_layout()
    show_plot(fig)

    st.divider()
    st.subheader("Dataset Preview (first 20 rows)")
    st.dataframe(df.head(20), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Rows:** {len(df):,}")
        st.markdown(f"**Columns:** {len(df.columns)}")
    with col_b:
        st.markdown(f"**Countries:** {df['Country'].nunique()}")
        avg = df.groupby("InvoiceNo")["Revenue"].sum().mean()
        st.markdown(f"**Avg Order Value:** \u00a3{avg:,.2f}")


# ==================================================
# SALES TRENDS
# ==================================================

elif page == "Sales Trends":

    st.header("Sales Trends Analysis")

    # ---- Monthly Revenue ----
    st.subheader("Monthly Revenue Trend")
    monthly_revenue = df.groupby("MonthYear")["Revenue"].sum()
    idx = list(range(len(monthly_revenue)))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(idx, monthly_revenue.values, alpha=0.18, color=C1)
    ax.plot(idx, monthly_revenue.values, marker="o", color=C1, linewidth=2.5, markersize=6)
    ax.set_xticks(idx)
    ax.set_xticklabels(monthly_revenue.index.astype(str), rotation=45, ha="right", fontsize=9)
    style_ax(ax, "Monthly Revenue Trend", "Month", "Revenue (GBP)")
    sns.despine(ax=ax)
    plt.tight_layout()
    show_plot(fig)

    # ---- Monthly Orders ----
    st.subheader("Monthly Order Volume")
    monthly_orders = df.groupby("MonthYear")["InvoiceNo"].nunique()
    idx2 = list(range(len(monthly_orders)))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(idx2, monthly_orders.values, alpha=0.18, color=C2)
    ax.plot(idx2, monthly_orders.values, marker="s", color=C2, linewidth=2.5, markersize=6)
    ax.set_xticks(idx2)
    ax.set_xticklabels(monthly_orders.index.astype(str), rotation=45, ha="right", fontsize=9)
    style_ax(ax, "Monthly Order Volume", "Month", "Number of Orders")
    sns.despine(ax=ax)
    plt.tight_layout()
    show_plot(fig)

    # ---- Day of Week ----
    st.subheader("Revenue by Day of Week")
    dow_order   = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
    dow_revenue = df.groupby("DayOfWeek")["Revenue"].sum()
    dow_revenue = dow_revenue.reindex([d for d in dow_order if d in dow_revenue.index])

    bar_colors = [C1, C2, C3, C4, C5, "#a78bfa"][:len(dow_revenue)]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(dow_revenue.index, dow_revenue.values,
                  color=bar_colors, edgecolor=BG, linewidth=0.8)
    for bar, val in zip(bars, dow_revenue.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(dow_revenue.values) * 0.01,
                f"\u00a3{val:,.0f}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT_CLR)
    style_ax(ax, "Revenue by Day of Week", "Day of Week", "Revenue (GBP)")
    sns.despine(ax=ax)
    plt.tight_layout()
    show_plot(fig)

    # ---- MoM Growth ----
    st.subheader("Month-over-Month Revenue Growth (%)")
    growth  = (monthly_revenue.pct_change() * 100).dropna()
    gidx    = list(range(len(growth)))

    colors_g = [C3 if v >= 0 else C5 for v in growth.values]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(gidx, growth.values, color=colors_g, edgecolor=BG, linewidth=0.5)
    ax.axhline(0, color=GRID_CLR, linewidth=1.2, linestyle="--")
    ax.set_xticks(gidx)
    ax.set_xticklabels(growth.index.astype(str), rotation=45, ha="right", fontsize=9)
    style_ax(ax, "Month-over-Month Revenue Growth", ylabel="Growth (%)")
    sns.despine(ax=ax)
    plt.tight_layout()
    show_plot(fig)


# ==================================================
# PRODUCT ANALYSIS
# ==================================================

elif page == "Product Analysis":

    st.header("Product Analysis")

    col_l, col_r = st.columns(2)

    # Top products by quantity
    with col_l:
        st.subheader("Top 10 by Quantity Sold")
        top_products = (
            df.groupby("Description")["Quantity"].sum()
            .sort_values(ascending=False).head(10)
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(top_products)),
                top_products.sort_values().values,
                color=C1, edgecolor=BG, linewidth=0.6)
        ax.set_yticks(range(len(top_products)))
        ax.set_yticklabels([t[:30] for t in top_products.sort_values().index],
                           fontsize=8, color=TEXT_CLR)
        style_ax(ax, "Top 10 Best-Selling Products", "Quantity Sold")
        sns.despine(ax=ax)
        plt.tight_layout()
        show_plot(fig)

    # Top products by revenue
    with col_r:
        st.subheader("Top 10 by Revenue")
        top_rv = (
            df.groupby("Description")["Revenue"].sum()
            .sort_values(ascending=False).head(10)
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(top_rv)),
                top_rv.sort_values().values,
                color=C2, edgecolor=BG, linewidth=0.6)
        ax.set_yticks(range(len(top_rv)))
        ax.set_yticklabels([t[:30] for t in top_rv.sort_values().index],
                           fontsize=8, color=TEXT_CLR)
        style_ax(ax, "Top 10 Products by Revenue", "Revenue (GBP)")
        sns.despine(ax=ax)
        plt.tight_layout()
        show_plot(fig)

    st.divider()

    # Country analysis
    st.subheader("Top 10 Countries by Revenue")
    country_revenue = (
        df.groupby("Country")["Revenue"].sum()
        .sort_values(ascending=False).head(10)
    )
    clrs = [PALETTE_7[i % len(PALETTE_7)] for i in range(len(country_revenue))]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(range(len(country_revenue)),
            country_revenue.sort_values().values,
            color=clrs[::-1], edgecolor=BG, linewidth=0.6)
    ax.set_yticks(range(len(country_revenue)))
    ax.set_yticklabels(country_revenue.sort_values().index, fontsize=9, color=TEXT_CLR)
    style_ax(ax, "Top 10 Countries by Revenue", "Revenue (GBP)")
    sns.despine(ax=ax)
    plt.tight_layout()
    show_plot(fig)

    st.divider()

    # Unit Price distribution
    st.subheader("Unit Price Distribution (products < 50)")
    price_data = df[df["UnitPrice"] < 50]["UnitPrice"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(price_data, bins=60, color=C1, edgecolor=BG, alpha=0.9, linewidth=0.4)
    style_ax(ax, "Distribution of Unit Prices (< 50)", "Unit Price", "Frequency")
    sns.despine(ax=ax)
    plt.tight_layout()
    show_plot(fig)


# ==================================================
# CUSTOMER SEGMENTATION
# ==================================================

elif page == "Customer Segmentation":

    st.header("Customer Segmentation -  RFM + K-Means")

    st.markdown("""
    Customers are segmented using **RFM Analysis**:

    | Metric | Definition |
    |--------|------------|
    | **Recency (R)** | Days since last purchase |
    | **Frequency (F)** | Number of unique invoices |
    | **Monetary (M)** | Total revenue contributed |
    """)

    # ------------------------------------------------
    # RFM COMPUTATION (cached so slider doesn't re-read Excel)
    # ------------------------------------------------

    @st.cache_data(show_spinner="Computing RFM...")
    def compute_rfm(_df):
        ref       = _df["InvoiceDate"].max() + pd.Timedelta(days=1)
        recency   = _df.groupby("CustomerID")["InvoiceDate"].max().apply(
                        lambda x: (ref - x).days)
        frequency = _df.groupby("CustomerID")["InvoiceNo"].nunique()
        monetary  = _df.groupby("CustomerID")["Revenue"].sum()
        return pd.DataFrame({"Recency": recency, "Frequency": frequency, "Monetary": monetary})

    rfm_base   = compute_rfm(df)
    rfm_log    = np.log1p(rfm_base.copy())
    scaler     = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # ------------------------------------------------
    # ELBOW CURVE
    # ------------------------------------------------

    st.subheader("Elbow Method - Optimal Number of Clusters")

    @st.cache_data(show_spinner="Computing Elbow curve...")
    def elbow_curve_data(_scaled):
        k_list   = list(range(2, 11))
        inertias = []
        for ki in k_list:
            km = KMeans(n_clusters=ki, random_state=42, n_init=10)
            km.fit(_scaled)
            inertias.append(km.inertia_)
        return k_list, inertias

    k_range, inertias = elbow_curve_data(rfm_scaled)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(k_range, inertias, marker="o", color="#6366f1", linewidth=2.5, markersize=7)
    ax.fill_between(k_range, inertias, alpha=0.15, color="#6366f1")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Elbow Method for Optimal K", fontsize=14, fontweight="bold")
    ax.set_xticks(k_range)
    ax.grid(linestyle="--", alpha=0.35)
    sns.despine(ax=ax)
    plt.tight_layout()
    show_plot(fig)

    # ------------------------------------------------
    # K SELECTION
    # ------------------------------------------------

    st.subheader("Select Number of Customer Segments")
    k = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4,
                  help="Choose K based on the Elbow curve above.")

    # ------------------------------------------------
    # K-MEANS
    # ------------------------------------------------

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm    = rfm_base.copy()
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # ------------------------------------------------
    # SILHOUETTE SCORE
    # ------------------------------------------------

    score = silhouette_score(rfm_scaled, rfm["Cluster"])
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Silhouette Score", f"{score:.3f}",
                  help="Closer to 1.0 = better-defined clusters")
    with col_m2:
        st.metric("Number of Clusters", k)
    with col_m3:
        st.metric("Customers Segmented", f"{len(rfm):,}")

    # ------------------------------------------------
    # AUTO CLUSTER LABELS
    # ------------------------------------------------

    cluster_summary = (
        rfm.groupby("Cluster")
           .agg({"Recency": "mean", "Frequency": "mean", "Monetary": "mean"})
           .round(2)
    )

    def label_cluster(row):
        rec_med = cluster_summary["Recency"].median()
        frq_med = cluster_summary["Frequency"].median()
        mon_med = cluster_summary["Monetary"].median()
        if row["Recency"] < rec_med and row["Frequency"] > frq_med:
            return "Champions"
        elif row["Recency"] < rec_med:
            return "Loyal Customers"
        elif row["Monetary"] < mon_med and row["Frequency"] < frq_med:
            return "At-Risk / Lost"
        else:
            return "Potential Loyalists"

    cluster_summary["Segment Label"] = cluster_summary.apply(label_cluster, axis=1)

    st.divider()
    st.subheader("Customer Segment Summary")
    st.dataframe(cluster_summary, use_container_width=True)

    # ------------------------------------------------
    # CLUSTER DISTRIBUTION BAR
    # ------------------------------------------------

    st.subheader("Customer Distribution Across Segments")
    cluster_counts = rfm["Cluster"].value_counts().sort_index()

    seg_colors = [PALETTE_7[i % len(PALETTE_7)] for i in range(len(cluster_counts))]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        [f"Cluster {i}" for i in cluster_counts.index],
        cluster_counts.values,
        color=seg_colors,
        edgecolor=BG, linewidth=0.8
    )
    for bar, val in zip(bars, cluster_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(cluster_counts.values) * 0.01,
                str(val), ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=TEXT_CLR)
    style_ax(ax, "Customers per Segment", ylabel="Number of Customers")
    sns.despine(ax=ax)
    plt.tight_layout()
    show_plot(fig)

    # ------------------------------------------------
    # RFM SCATTER PLOTS (2 panels)
    # ------------------------------------------------

    st.subheader("RFM Scatter Plots")
    scatter_pal = PALETTE_7[:k]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(data=rfm, x="Frequency", y="Monetary",
                    hue="Cluster", palette=scatter_pal, s=60, alpha=0.75, ax=axes[0])
    style_ax(axes[0], "Frequency vs Monetary", "Frequency", "Monetary (GBP)")
    axes[0].legend(title="Cluster", fontsize=8)
    sns.despine(ax=axes[0])

    sns.scatterplot(data=rfm, x="Recency", y="Monetary",
                    hue="Cluster", palette=scatter_pal, s=60, alpha=0.75, ax=axes[1])
    style_ax(axes[1], "Recency vs Monetary", "Recency (days)", "Monetary (GBP)")
    axes[1].legend(title="Cluster", fontsize=8)
    sns.despine(ax=axes[1])

    plt.tight_layout()
    show_plot(fig)

    # ------------------------------------------------
    # RFM BOX PLOTS
    # ------------------------------------------------

    st.subheader("RFM Distribution per Cluster")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
        data_by_cluster = [rfm[rfm["Cluster"] == c][col].values
                           for c in sorted(rfm["Cluster"].unique())]
        bp = ax.boxplot(
            data_by_cluster, patch_artist=True,
            medianprops=dict(color=TEXT_CLR, linewidth=2),
            whiskerprops=dict(color=TICK_CLR),
            capprops=dict(color=TICK_CLR),
            flierprops=dict(marker="o", color=C5, alpha=0.4, markersize=3),
        )
        box_colors = PALETTE_7[:len(data_by_cluster)]
        for patch, clr in zip(bp["boxes"], box_colors):
            patch.set_facecolor(clr)
            patch.set_alpha(0.75)
        ax.set_xticklabels([f"C{c}" for c in sorted(rfm["Cluster"].unique())])
        style_ax(ax, col, "Cluster", col)
        sns.despine(ax=ax)
    fig.suptitle("RFM Distributions by Cluster", fontsize=14,
                 fontweight="bold", color=TEXT_CLR)
    plt.tight_layout()
    show_plot(fig)

    # ------------------------------------------------
    # RFM TABLE — FILTERABLE
    # ------------------------------------------------

    st.divider()
    st.subheader("Customer RFM Data")

    filter_cluster = st.selectbox(
        "Filter by Cluster",
        options=["All"] + sorted(rfm["Cluster"].unique().tolist())
    )
    display_rfm = rfm if filter_cluster == "All" else rfm[rfm["Cluster"] == filter_cluster]
    st.dataframe(display_rfm.head(200), use_container_width=True)
    st.caption(
        f"Showing {min(200, len(display_rfm))} of {len(display_rfm):,} customers "
        f"({'all clusters' if filter_cluster == 'All' else f'Cluster {filter_cluster}'})"
    )