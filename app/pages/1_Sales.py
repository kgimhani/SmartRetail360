"""
SmartRetail360 — Sales Analytics Page
File: app/pages/1_Sales.py
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

st.set_page_config(page_title="Sales Analytics | SmartRetail360",
                   page_icon="📈", layout="wide")

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
    conn  = sqlite3.connect(DB_PATH)
    df    = pd.read_sql("SELECT * FROM transactions", conn)
    daily = pd.read_sql("SELECT * FROM daily_timeseries", conn)
    conn.close()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    daily["date"]     = pd.to_datetime(daily["date"], errors="coerce")
    return df, daily


if not os.path.exists(DB_PATH):
    st.error("❌ Database not found. Run the pipeline notebook first.")
    st.stop()

df, daily = load_data()

# Sidebar filters
st.sidebar.markdown("## 📈 Sales Analytics")
min_d = df["InvoiceDate"].min().date()
max_d = df["InvoiceDate"].max().date()
date_range = st.sidebar.date_input("📅 Date Range", value=(min_d, max_d),
                                    min_value=min_d, max_value=max_d)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    s, e = date_range
else:
    s = e = date_range[0] if date_range else min_d

ot = st.sidebar.multiselect("🛵 Order Type",
                              df["OrderType"].dropna().unique().tolist(),
                              default=df["OrderType"].dropna().unique().tolist())

mask = ((df["InvoiceDate"].dt.date >= s) & (df["InvoiceDate"].dt.date <= e))
if ot:
    mask &= df["OrderType"].isin(ot)
dff = df[mask].copy()

# Page title
st.markdown("# 📈 Sales Analytics")
st.markdown(f"*Showing {len(dff):,} transactions from {s} to {e}*")
st.markdown("---")

if len(dff) == 0:
    st.warning("No data for selected filters.")
    st.stop()

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue",  f"£{dff['line_revenue'].sum():,.0f}")
c2.metric("Total Profit",   f"£{dff['profit'].sum():,.0f}")
c3.metric("Total Orders",   f"{dff['OrderID'].nunique():,}")
c4.metric("Profit Margin",  f"{dff['profit'].sum()/dff['line_revenue'].sum()*100:.1f}%" if dff['line_revenue'].sum() > 0 else "N/A")

st.markdown("")

# Monthly revenue trend
st.markdown('<p class="section-header">📅 Monthly Revenue & Profit</p>', unsafe_allow_html=True)
monthly = dff.groupby([dff["InvoiceDate"].dt.to_period("M")]).agg(
    Revenue=("line_revenue", "sum"),
    Profit =("profit",       "sum"),
    Orders =("OrderID",      "nunique"),
).reset_index()
monthly["Month"] = monthly["InvoiceDate"].astype(str)

fig = go.Figure()
fig.add_trace(go.Bar(x=monthly["Month"], y=monthly["Revenue"],
                      name="Revenue", marker_color="#E87040", opacity=0.85))
fig.add_trace(go.Bar(x=monthly["Month"], y=monthly["Profit"],
                      name="Profit", marker_color="#378ADD", opacity=0.85))
fig.update_layout(
    barmode="group", template="plotly_dark",
    plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
    font_color="#ccc", height=360,
    legend=dict(orientation="h", y=1.1),
    margin=dict(l=10, r=10, t=20, b=10),
)
fig.update_xaxes(tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Day of week analysis
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-header">📆 Revenue by Day of Week</p>', unsafe_allow_html=True)
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    dow  = dff.groupby("day_of_week")["line_revenue"].sum().reset_index()
    dow["Day"] = dow["day_of_week"].apply(lambda x: days[int(x)] if int(x) < 7 else str(x))
    dow = dow.sort_values("day_of_week")
    fig2 = px.bar(dow, x="Day", y="line_revenue",
                   color="line_revenue", color_continuous_scale="Oranges",
                   template="plotly_dark",
                   labels={"line_revenue": "Revenue (£)", "Day": ""})
    fig2.update_layout(plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                        font_color="#ccc", showlegend=False,
                        margin=dict(l=10,r=10,t=10,b=10), height=300)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.markdown('<p class="section-header">⏰ Revenue by Hour</p>', unsafe_allow_html=True)
    hourly = dff.groupby("hour")["line_revenue"].sum().reset_index()
    fig3 = px.line(hourly, x="hour", y="line_revenue",
                    markers=True, template="plotly_dark",
                    color_discrete_sequence=["#E87040"],
                    labels={"line_revenue": "Revenue (£)", "hour": "Hour of Day"})
    fig3.update_layout(plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                        font_color="#ccc",
                        margin=dict(l=10,r=10,t=10,b=10), height=300)
    st.plotly_chart(fig3, use_container_width=True)

# Order type over time
st.markdown('<p class="section-header">🛵 Delivery vs Collection Over Time</p>', unsafe_allow_html=True)
ot_monthly = dff.groupby([dff["InvoiceDate"].dt.to_period("M"), "OrderType"])["line_revenue"].sum().reset_index()
ot_monthly["Month"] = ot_monthly["InvoiceDate"].astype(str)
fig4 = px.line(ot_monthly, x="Month", y="line_revenue", color="OrderType",
                markers=True, template="plotly_dark",
                color_discrete_sequence=["#E87040", "#378ADD"],
                labels={"line_revenue": "Revenue (£)", "Month": ""})
fig4.update_layout(plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                    font_color="#ccc", height=320,
                    margin=dict(l=10,r=10,t=10,b=10))
fig4.update_xaxes(tickangle=-45)
st.plotly_chart(fig4, use_container_width=True)

# Summary table
st.markdown('<p class="section-header">📋 Monthly Summary Table</p>', unsafe_allow_html=True)
monthly_tbl = dff.groupby([dff["InvoiceDate"].dt.to_period("M")]).agg(
    Revenue   = ("line_revenue", "sum"),
    Profit    = ("profit",       "sum"),
    Cost      = ("cost",         "sum"),
    Orders    = ("OrderID",      "nunique"),
    ItemsSold = ("Quantity",     "sum"),
).reset_index()
monthly_tbl["Month"]  = monthly_tbl["InvoiceDate"].astype(str)
monthly_tbl["Margin"] = (monthly_tbl["Profit"] / monthly_tbl["Revenue"] * 100).round(1)
monthly_tbl = monthly_tbl[["Month","Revenue","Profit","Cost","Margin","Orders","ItemsSold"]]
monthly_tbl["Revenue"] = monthly_tbl["Revenue"].map("£{:,.0f}".format)
monthly_tbl["Profit"]  = monthly_tbl["Profit"].map("£{:,.0f}".format)
monthly_tbl["Cost"]    = monthly_tbl["Cost"].map("£{:,.0f}".format)
monthly_tbl["Margin"]  = monthly_tbl["Margin"].map("{:.1f}%".format)
st.dataframe(monthly_tbl, use_container_width=True, hide_index=True)
