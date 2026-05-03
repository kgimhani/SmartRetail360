"""
SmartRetail360 — RFM Segments Page
File: app/pages/3_Segments.py
"""

import os, sys, sqlite3, warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT, "data", "smartretail.db")

st.set_page_config(page_title="RFM Segments | SmartRetail360",
                   page_icon="👥", layout="wide")

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
    .info-box {
        background: #1a1a2e; border-radius: 10px;
        padding: 1rem 1.5rem; border-left: 4px solid #E87040;
        margin-bottom: 1rem;
    }
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #eee !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    rfm  = pd.read_sql("SELECT * FROM rfm", conn)
    conn.close()
    rfm["OrderDate"] = pd.to_datetime(rfm["OrderDate"], errors="coerce")
    return rfm


if not os.path.exists(DB_PATH):
    st.error("❌ Database not found. Run the pipeline notebook first.")
    st.stop()

rfm = load_data()

st.markdown("# 👥 RFM Order Segments")
st.markdown("*Order-level RFM analysis: Recency, Frequency, Monetary value*")
st.markdown("""
<div class="info-box">
    <b>ℹ️ Note:</b> Since this dataset has no Customer IDs, RFM is computed per Order.
    <b>Recency</b> = days since order, <b>Frequency</b> = items in that order, <b>Monetary</b> = order total.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Orders",       f"{len(rfm):,}")
c2.metric("Avg Order Value",    f"£{rfm['Monetary'].mean():,.2f}")
c3.metric("Avg Items/Order",    f"{rfm['Frequency'].mean():.1f}")
c4.metric("Avg Recency (days)", f"{rfm['Recency'].mean():.0f}")

st.markdown("")

# Segment bar chart
st.markdown('<p class="section-header">📊 Order Segment Distribution</p>', unsafe_allow_html=True)
seg_counts = rfm["Segment"].value_counts().reset_index()
seg_counts.columns = ["Segment", "Count"]
colors = {"High Value":"#1D9E75","Good Value":"#378ADD",
          "Average":"#F0C27F","Low Value":"#F0A500","Lapsed":"#E24B4A"}
fig_seg = px.bar(seg_counts, x="Segment", y="Count",
                  color="Segment",
                  color_discrete_map=colors,
                  template="plotly_dark",
                  labels={"Count": "Number of Orders", "Segment": ""})
fig_seg.update_layout(plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                       font_color="#ccc", showlegend=False,
                       margin=dict(l=10,r=10,t=10,b=10), height=320)
st.plotly_chart(fig_seg, use_container_width=True)

# Segment stats
st.markdown('<p class="section-header">📋 Segment Summary</p>', unsafe_allow_html=True)
seg_stats = rfm.groupby("Segment").agg(
    Orders     = ("OrderID",   "count"),
    Avg_Value  = ("Monetary",  "mean"),
    Total_Rev  = ("Monetary",  "sum"),
    Avg_Items  = ("Frequency", "mean"),
    Avg_Recency= ("Recency",   "mean"),
).reset_index().sort_values("Total_Rev", ascending=False)

seg_stats["Avg_Value"]   = seg_stats["Avg_Value"].map("£{:.2f}".format)
seg_stats["Total_Rev"]   = seg_stats["Total_Rev"].map("£{:,.0f}".format)
seg_stats["Avg_Items"]   = seg_stats["Avg_Items"].map("{:.1f}".format)
seg_stats["Avg_Recency"] = seg_stats["Avg_Recency"].map("{:.0f} days".format)
seg_stats = seg_stats.rename(columns={
    "Segment":"Segment","Orders":"Orders",
    "Avg_Value":"Avg Order Value","Total_Rev":"Total Revenue",
    "Avg_Items":"Avg Items","Avg_Recency":"Avg Recency"
})
st.dataframe(seg_stats, use_container_width=True, hide_index=True)

# 3D RFM scatter
st.markdown('<p class="section-header">🔵 RFM 3D Scatter (K-Means Clusters)</p>', unsafe_allow_html=True)

if "Cluster" in rfm.columns:
    rfm_sample = rfm.sample(min(1000, len(rfm)), random_state=42)
    fig_3d = px.scatter_3d(
        rfm_sample,
        x="Recency", y="Frequency", z="Monetary",
        color=rfm_sample["Cluster"].astype(str),
        hover_data=["Segment"],
        template="plotly_dark",
        labels={"Recency":"Recency (days)","Frequency":"Items","Monetary":"Revenue (£)"},
        title="K-Means Clusters in RFM Space",
    )
    fig_3d.update_layout(
        plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
        font_color="#ccc", height=480,
        margin=dict(l=10,r=10,t=40,b=10),
    )
    st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.info("Cluster data not found. Re-run pipeline Cell 6.")

# RFM Score distribution
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-header">📈 RFM Total Score Distribution</p>', unsafe_allow_html=True)
    fig_hist = px.histogram(rfm, x="RFM_Total", nbins=13,
                             color_discrete_sequence=["#E87040"],
                             template="plotly_dark",
                             labels={"RFM_Total":"RFM Total Score","count":"Orders"})
    fig_hist.update_layout(plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                            font_color="#ccc", margin=dict(l=10,r=10,t=10,b=10), height=300)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown('<p class="section-header">💰 Revenue per Segment</p>', unsafe_allow_html=True)
    seg_rev = rfm.groupby("Segment")["Monetary"].sum().reset_index().sort_values("Monetary", ascending=False)
    fig_rev = px.pie(seg_rev, names="Segment", values="Monetary",
                      hole=0.5, template="plotly_dark",
                      color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_rev.update_layout(plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                           font_color="#ccc", margin=dict(l=10,r=10,t=10,b=10), height=300)
    st.plotly_chart(fig_rev, use_container_width=True)

# Recency histogram
st.markdown('<p class="section-header">📅 Order Recency Distribution</p>', unsafe_allow_html=True)
fig_rec = px.histogram(rfm, x="Recency", nbins=50,
                        color_discrete_sequence=["#378ADD"],
                        template="plotly_dark",
                        labels={"Recency":"Days Since Order","count":"Orders"})
fig_rec.update_layout(plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                       font_color="#ccc", margin=dict(l=10,r=10,t=10,b=10), height=300)
st.plotly_chart(fig_rec, use_container_width=True)
