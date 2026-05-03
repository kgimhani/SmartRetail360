import os, sys
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT, "data", "smartretail.db")
sys.path.insert(0, PROJECT)
from build_db import ensure_db
ensure_db()

import os, sys



import os, sys, sqlite3, json, pickle, warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import sys

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SmartRetail360 | Restaurant Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        color: #E87040;
        margin-bottom: 0;
    }
    .sub-title {
        color: #888;
        font-size: 1rem;
        margin-top: -0.3rem;
    }
    .metric-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #E87040;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { color: #E87040; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #aaa; font-size: 0.85rem; margin: 0; }
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        color: #E87040;
        border-bottom: 2px solid #E87040;
        padding-bottom: 0.3rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    [data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    [data-testid="stSidebar"] * { color: #eee !important; }
</style>
""", unsafe_allow_html=True)


# ── Load data (cached) ────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df       = pd.read_sql("SELECT * FROM transactions", conn)
    rfm      = pd.read_sql("SELECT * FROM rfm", conn)
    products = pd.read_sql("SELECT * FROM products", conn)
    daily    = pd.read_sql("SELECT * FROM daily_timeseries", conn)
    cat_rev  = pd.read_sql("SELECT * FROM category_revenue", conn)
    ot_rev   = pd.read_sql("SELECT * FROM ordertype_revenue", conn)
    conn.close()

    # Fix date columns — use the actual column names in your DB
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    daily["date"]     = pd.to_datetime(daily["date"], errors="coerce")

    return df, rfm, products, daily, cat_rev, ot_rev


# ── Check DB exists ───────────────────────────────────────────


# ── Load ──────────────────────────────────────────────────────
try:
    df, rfm, products, daily, cat_rev, ot_rev = load_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.exception(e)
    st.stop()


# ── Sidebar filters ───────────────────────────────────────────
st.sidebar.markdown("## 🍽️ SmartRetail360")
st.sidebar.markdown("---")

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()

date_range = st.sidebar.date_input(
    "📅 Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Handle single vs range date selection
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0] if date_range else min_date

order_type = st.sidebar.multiselect(
    "🛵 Order Type",
    options=df["OrderType"].dropna().unique().tolist(),
    default=df["OrderType"].dropna().unique().tolist(),
)

categories = st.sidebar.multiselect(
    "🍲 Category",
    options=df["Category"].dropna().unique().tolist(),
    default=df["Category"].dropna().unique().tolist(),
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Navigation")
st.sidebar.page_link("Home.py",              label="🏠 Home",             icon="🏠")
st.sidebar.page_link("pages/1_Sales.py",     label="📈 Sales Analytics",  icon="📈")
st.sidebar.page_link("pages/2_Products.py",  label="🍕 Product Analysis", icon="🍕")
st.sidebar.page_link("pages/3_Segments.py",  label="👥 RFM Segments",     icon="👥")
st.sidebar.page_link("pages/4_Forecast.py",  label="🔮 Forecast",         icon="🔮")
st.sidebar.page_link("pages/5_Insights.py",  label="💡 ML Insights",      icon="💡")


# ── Filter dataframe ──────────────────────────────────────────
mask = (
    (df["InvoiceDate"].dt.date >= start_date) &
    (df["InvoiceDate"].dt.date <= end_date)
)
if order_type:
    mask &= df["OrderType"].isin(order_type)
if categories:
    mask &= df["Category"].isin(categories)

dff = df[mask].copy()

if len(dff) == 0:
    st.warning("⚠️ No data for selected filters. Adjust filters in sidebar.")
    st.stop()


# ── Header ────────────────────────────────────────────────────
st.markdown('<p class="main-title">🍽️ SmartRetail360</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Restaurant Analytics Dashboard — All metrics update with sidebar filters</p>',
            unsafe_allow_html=True)
st.markdown("---")


# ── KPI Metrics ───────────────────────────────────────────────
total_revenue  = dff["line_revenue"].sum()
total_profit   = dff["profit"].sum()
total_orders   = dff["OrderID"].nunique()
avg_order_val  = total_revenue / total_orders if total_orders > 0 else 0
profit_margin  = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
total_items    = dff["Quantity"].sum()

col1, col2, col3, col4, col5, col6 = st.columns(6)

metrics = [
    (col1, "£{:,.0f}".format(total_revenue),   "Total Revenue"),
    (col2, "£{:,.0f}".format(total_profit),    "Total Profit"),
    (col3, "{:,}".format(total_orders),         "Total Orders"),
    (col4, "£{:.2f}".format(avg_order_val),    "Avg Order Value"),
    (col5, "{:.1f}%".format(profit_margin),     "Profit Margin"),
    (col6, "{:,}".format(int(total_items)),     "Items Sold"),
]

for col, val, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <p>{label}</p>
            <h3>{val}</h3>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")


# ── Revenue over time ─────────────────────────────────────────
st.markdown('<p class="section-header">📈 Revenue Over Time</p>', unsafe_allow_html=True)

daily_filtered = dff.groupby(dff["InvoiceDate"].dt.date).agg(
    revenue=("line_revenue", "sum"),
    profit =("profit",       "sum"),
    orders =("OrderID",      "nunique"),
).reset_index().rename(columns={"InvoiceDate": "Date"})
daily_filtered["Date"] = pd.to_datetime(daily_filtered["Date"])

fig_rev = px.area(
    daily_filtered, x="date", y="revenue",
    labels={"date": "Date", "revenue": "Revenue (£)"},
    title="Daily Revenue (£)",
    color_discrete_sequence=["#E87040"],
    template="plotly_dark",
)
fig_rev.update_layout(
    plot_bgcolor="#0e0e1a",
    paper_bgcolor="#0e0e1a",
    font_color="#ccc",
    showlegend=False,
    margin=dict(l=10, r=10, t=40, b=10),
    height=300,
)
st.plotly_chart(fig_rev, use_container_width=True)


# ── Two charts side by side ───────────────────────────────────
col_left, col_right = st.columns(2)

# Category revenue
with col_left:
    st.markdown('<p class="section-header">🍲 Revenue by Category</p>', unsafe_allow_html=True)
    cat_dff = dff.groupby("Category")["line_revenue"].sum().reset_index().sort_values("line_revenue", ascending=False)
    fig_cat = px.bar(
        cat_dff, x="Category", y="line_revenue",
        color="Category",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template="plotly_dark",
        labels={"line_revenue": "Revenue (£)", "Category": ""},
    )
    fig_cat.update_layout(
        plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
        font_color="#ccc", showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10), height=320,
    )
    fig_cat.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_cat, use_container_width=True)

# Order type donut
with col_right:
    st.markdown('<p class="section-header">🛵 Order Type Split</p>', unsafe_allow_html=True)
    ot_dff = dff.groupby("OrderType")["line_revenue"].sum().reset_index()
    fig_ot = px.pie(
        ot_dff, names="OrderType", values="line_revenue",
        hole=0.55,
        color_discrete_sequence=["#E87040", "#378ADD"],
        template="plotly_dark",
    )
    fig_ot.update_layout(
        plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
        font_color="#ccc",
        margin=dict(l=10, r=10, t=10, b=10), height=320,
    )
    st.plotly_chart(fig_ot, use_container_width=True)


# ── Day of week heatmap ───────────────────────────────────────
st.markdown('<p class="section-header">📅 Revenue Heatmap (Day × Hour)</p>', unsafe_allow_html=True)

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
heatmap_data = dff.groupby(["day_of_week", "hour"])["line_revenue"].sum().reset_index()
heatmap_pivot = heatmap_data.pivot(index="day_of_week", columns="hour", values="line_revenue").fillna(0)
heatmap_pivot.index = [days[i] for i in heatmap_pivot.index if i < 7]

fig_heat = px.imshow(
    heatmap_pivot,
    color_continuous_scale="Oranges",
    aspect="auto",
    labels=dict(x="Hour of Day", y="Day of Week", color="Revenue (£)"),
    template="plotly_dark",
)
fig_heat.update_layout(
    plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
    font_color="#ccc",
    margin=dict(l=10, r=10, t=10, b=10), height=280,
)
st.plotly_chart(fig_heat, use_container_width=True)


# ── Top 10 items ──────────────────────────────────────────────
st.markdown('<p class="section-header">🏆 Top 10 Menu Items</p>', unsafe_allow_html=True)

top10 = dff.groupby("Description").agg(
    Revenue=("line_revenue", "sum"),
    Profit =("profit",       "sum"),
    Orders =("OrderID",      "nunique"),
    Qty    =("Quantity",     "sum"),
).reset_index().nlargest(10, "Revenue")

fig_top = px.bar(
    top10, x="Revenue", y="Description",
    orientation="h",
    color="Profit",
    color_continuous_scale="Oranges",
    template="plotly_dark",
    labels={"Description": "", "Revenue": "Revenue (£)"},
)
fig_top.update_layout(
    plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
    font_color="#ccc", yaxis=dict(autorange="reversed"),
    margin=dict(l=10, r=10, t=10, b=10), height=380,
)
st.plotly_chart(fig_top, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>SmartRetail360 | Restaurant Analytics Dashboard | "
    f"Data: {min_date} → {max_date}</small></center>",
    unsafe_allow_html=True
)
