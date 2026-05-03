import os, sys
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT, "data", "smartretail.db")
sys.path.insert(0, PROJECT)
from build_db import ensure_db
ensure_db()

import os, sys



import os, sys, sqlite3, warnings
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")


st.set_page_config(page_title="Product Analysis | SmartRetail360",
                   page_icon="🍕", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem; color: #E87040;
        border-bottom: 2px solid #E87040;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem; margin-bottom: 1rem;
    }
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #eee !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_data():
    conn      = sqlite3.connect(DB_PATH)
    products  = pd.read_sql("SELECT * FROM products", conn)
    cat_rev   = pd.read_sql("SELECT * FROM category_revenue", conn)
    sentiment = pd.read_sql("SELECT * FROM product_sentiment", conn)
    conn.close()
    return products, cat_rev, sentiment


if not os.path.exists(DB_PATH):
    st.error("❌ Database not found. Run the pipeline notebook first.")
    st.stop()

products, cat_rev, sentiment = load_data()

st.sidebar.markdown("## 🍕 Product Analysis")
cat_filter = st.sidebar.multiselect("🍲 Category",
                                     products["Category"].dropna().unique().tolist(),
                                     default=products["Category"].dropna().unique().tolist())

prods = products[products["Category"].isin(cat_filter)] if cat_filter else products

# Title
st.markdown("# 🍕 Product Analysis")
st.markdown("*Menu item performance, category breakdown, and profitability*")
st.markdown("---")

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Menu Items",     f"{len(prods):,}")
c2.metric("Total Revenue",  f"£{prods['total_revenue'].sum():,.0f}")
c3.metric("Total Profit",   f"£{prods['total_profit'].sum():,.0f}")
c4.metric("Items Sold",     f"{prods['total_qty'].sum():,.0f}")

st.markdown("")

# Category treemap
st.markdown('<p class="section-header">🗺️ Category Revenue Treemap</p>', unsafe_allow_html=True)
fig_tree = px.treemap(
    prods, path=["Category", "Description"],
    values="total_revenue",
    color="Total Profit (£)",
    color_continuous_scale="Oranges",
    template="plotly_dark",
)
fig_tree.update_layout(
    plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
    font_color="#ccc", height=420,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_tree, use_container_width=True)

# Top and bottom performers
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-header">🏆 Top 15 Items by Revenue</p>', unsafe_allow_html=True)
    top15 = prods.nlargest(15, "total_revenue")
    fig_top = px.bar(top15, x="total_revenue", y="Description",
                      orientation="h", color="Total Profit (£)",
                      color_continuous_scale="Oranges",
                      template="plotly_dark",
                      labels={"total_revenue": "Revenue (£)", "Description": ""})
    fig_top.update_layout(
        plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
        font_color="#ccc", yaxis=dict(autorange="reversed"),
        margin=dict(l=10,r=10,t=10,b=10), height=450)
    st.plotly_chart(fig_top, use_container_width=True)

with col2:
    st.markdown('<p class="section-header">📦 Top 15 Items by Quantity Sold</p>', unsafe_allow_html=True)
    top15q = prods.nlargest(15, "Qty Sold")
    fig_qty = px.bar(top15q, x="Qty Sold", y="Description",
                      orientation="h", color="Qty Sold",
                      color_continuous_scale="Blues",
                      template="plotly_dark",
                      labels={"Qty Sold": "Quantity Sold", "Description": ""})
    fig_qty.update_layout(
        plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
        font_color="#ccc", yaxis=dict(autorange="reversed"),
        margin=dict(l=10,r=10,t=10,b=10), height=450)
    st.plotly_chart(fig_qty, use_container_width=True)

# Profitability scatter
st.markdown('<p class="section-header">💰 Revenue vs Profit (All Items)</p>', unsafe_allow_html=True)
fig_scatter = px.scatter(
    prods, x="total_revenue", y="Total Profit (£)",
    color="Category", size="Qty Sold",
    hover_data=["Description"],
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Vivid,
    labels={"total_revenue": "Revenue (£)", "Total Profit (£)": "Profit (£)"},
)
fig_scatter.update_layout(
    plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
    font_color="#ccc", height=400,
    margin=dict(l=10,r=10,t=10,b=10),
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Category performance table
st.markdown('<p class="section-header">📋 Category Performance Table</p>', unsafe_allow_html=True)
cat_tbl = cat_rev.copy()
cat_tbl["Margin"] = (cat_tbl["profit"] / cat_tbl["revenue"] * 100).round(1)
cat_tbl = cat_tbl.sort_values("revenue", ascending=False)
cat_tbl = cat_tbl.rename(columns={
    "Category": "Category", "revenue": "Revenue",
    "profit": "Profit", "orders": "Orders", "qty": "Qty Sold"
})
cat_tbl["Revenue"] = cat_tbl["Revenue"].map("£{:,.0f}".format)
cat_tbl["Profit"]  = cat_tbl["Profit"].map("£{:,.0f}".format)
cat_tbl["Margin"]  = cat_tbl["Margin"].map("{:.1f}%".format)
st.dataframe(cat_tbl[["Category","Revenue","Profit","Margin","Orders","Qty Sold"]],
             use_container_width=True, hide_index=True)

# Sentiment
st.markdown('<p class="section-header">😊 Menu Item Sentiment Analysis</p>', unsafe_allow_html=True)
st.caption("*Sentiment is based on VADER NLP analysis of menu item names*")
sent_counts = sentiment["Sentiment"].value_counts().reset_index()
sent_counts.columns = ["Sentiment", "Count"]
col_a, col_b = st.columns([1, 2])
with col_a:
    fig_sent = px.pie(sent_counts, names="Sentiment", values="Count",
                       color="Sentiment",
                       color_discrete_map={"Positive":"#1D9E75","Neutral":"#F0C27F","Negative":"#E24B4A"},
                       hole=0.5, template="plotly_dark")
    fig_sent.update_layout(plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                            font_color="#ccc", margin=dict(l=10,r=10,t=10,b=10), height=280)
    st.plotly_chart(fig_sent, use_container_width=True)
with col_b:
    top_pos = sentiment[sentiment["Sentiment"] == "Positive"].nlargest(10, "total_revenue")
    st.markdown("**Top Positively Perceived High-Revenue Items:**")
    if len(top_pos) > 0:
        st.dataframe(top_pos[["Description","Category","total_revenue","total_sold"]].rename(
            columns={"total_revenue":"Revenue","total_sold":"Sold"}),
            use_container_width=True, hide_index=True)
    else:
        st.info("No positive sentiment items in current filter.")

# Full product table
st.markdown('<p class="section-header">📋 All Products Table</p>', unsafe_allow_html=True)
prods_display = prods.copy().sort_values("total_revenue", ascending=False)
prods_display["total_revenue"] = prods_display["total_revenue"].map("£{:,.2f}".format)
prods_display["Total Profit (£)"]  = prods_display["Total Profit (£)"].map("£{:,.2f}".format)
prods_display["avg_price"]     = prods_display["avg_price"].map("£{:.2f}".format)
prods_display = prods_display.rename(columns={
    "Description":"Item","Category":"Category",
    "total_revenue":"Revenue","Total Profit (£)":"Profit",
    "Qty Sold":"Qty","n_orders":"Orders","avg_price":"Avg Price"
})
st.dataframe(prods_display[["Item","Category","Revenue","Profit","Qty","Orders","Avg Price"]],
             use_container_width=True, hide_index=True)
