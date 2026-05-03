import os, sys
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT, "data", "smartretail.db")
sys.path.insert(0, PROJECT)
from build_db import ensure_db
ensure_db()

"""
SmartRetail360 — ML Insights Page
File: app/pages/5_Insights.py
"""

import os, sys, sqlite3, json, pickle, warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


st.set_page_config(page_title="ML Insights | SmartRetail360",
                   page_icon="💡", layout="wide")

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
    .insight-card {
        background: #1a1a2e; border-radius: 10px;
        padding: 1rem 1.5rem; border-left: 4px solid #E87040;
        margin-bottom: 0.8rem;
    }
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #eee !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_data():
    conn     = sqlite3.connect(DB_PATH)
    products = pd.read_sql("SELECT * FROM products", conn)
    cat_rev  = pd.read_sql("SELECT * FROM category_revenue", conn)
    daily    = pd.read_sql("SELECT * FROM daily_timeseries", conn)
    conn.close()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    return products, cat_rev, daily

@st.cache_resource
def load_models():
    results = {}
    # Cancellation model
    cm_path = os.path.join(PROJECT, "models", "cancel_model.pkl")
    cf_path = os.path.join(PROJECT, "models", "cancel_features.json")
    meta_path = os.path.join(PROJECT, "models", "cancel_meta.json")
    if os.path.exists(cm_path):
        with open(cm_path, "rb") as f:
            results["cancel_model"] = pickle.load(f)
        with open(cf_path) as f:
            results["cancel_feats"] = json.load(f)
        with open(meta_path) as f:
            results["cancel_meta"] = json.load(f)
    # KMeans
    km_path = os.path.join(PROJECT, "models", "kmeans_model.pkl")
    if os.path.exists(km_path):
        with open(km_path, "rb") as f:
            results["kmeans"] = pickle.load(f)
    return results


if not os.path.exists(DB_PATH):
    st.error("❌ Database not found. Run the pipeline notebook first.")
    st.stop()

products, cat_rev, daily = load_data()
models = load_models()

st.markdown("# 💡 ML Insights")
st.markdown("*Machine learning model results and business intelligence*")
st.markdown("---")

# ── Section 1: Cancellation Model ────────────────────────────
st.markdown('<p class="section-header">❌ Cancellation Risk Analysis</p>', unsafe_allow_html=True)

if "cancel_model" in models:
    meta = models["cancel_meta"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Model AUC-ROC",    f"{meta.get('auc', 0):.4f}")
    col2.metric("Cancellation Rate", f"{meta.get('cancel_rate', 0):.1%}")
    col3.metric("Model Type",       "Random Forest")

    st.markdown("")

    # Feature importance
    feat_names = models["cancel_feats"]
    importances = models["cancel_model"].feature_importances_
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(12)

    fig_fi = px.bar(fi_df, x="Importance", y="Feature",
                     orientation="h",
                     color="Importance", color_continuous_scale="Oranges",
                     template="plotly_dark",
                     labels={"Importance":"Importance Score","Feature":""},
                     title="Top Feature Importances — Cancellation Predictor")
    fig_fi.update_layout(
        plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
        font_color="#ccc", yaxis=dict(autorange="reversed"),
        margin=dict(l=10,r=10,t=40,b=10), height=380
    )
    st.plotly_chart(fig_fi, use_container_width=True)
else:
    st.warning("ML model is loading. Please refresh in a moment.")

# ── Section 2: Business Insights ─────────────────────────────
st.markdown('<p class="section-header">📊 Key Business Insights</p>', unsafe_allow_html=True)

# Best day of week
if "day_of_week" in daily.columns and "revenue" in daily.columns:
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow  = daily.groupby("day_of_week")["revenue"].mean().reset_index()
    best_day_idx = int(dow.loc[dow["revenue"].idxmax(), "day_of_week"])
    best_day     = days[best_day_idx] if best_day_idx < 7 else str(best_day_idx)
    best_day_rev = dow["revenue"].max()
    worst_day_idx= int(dow.loc[dow["revenue"].idxmin(), "day_of_week"])
    worst_day    = days[worst_day_idx] if worst_day_idx < 7 else str(worst_day_idx)
else:
    best_day = "N/A"; best_day_rev = 0; worst_day = "N/A"

# Best category
if len(cat_rev) > 0:
    best_cat      = cat_rev.loc[cat_rev["revenue"].idxmax(), "Category"]
    best_cat_rev  = cat_rev["revenue"].max()
    best_cat_marg = (cat_rev.loc[cat_rev["revenue"].idxmax(), "profit"] /
                     cat_rev.loc[cat_rev["revenue"].idxmax(), "revenue"] * 100)
else:
    best_cat = "N/A"; best_cat_rev = 0; best_cat_marg = 0

# Best item
if len(products) > 0:
    best_item     = products.loc[products["total_revenue"].idxmax(), "Description"]
    best_item_rev = products["total_revenue"].max()
    most_sold     = products.loc[products["total_qty"].idxmax(), "Description"]
else:
    best_item = "N/A"; best_item_rev = 0; most_sold = "N/A"

insights = [
    (f"🗓️ <b>Best Day:</b> {best_day}",
     f"Highest average daily revenue: £{best_day_rev:,.0f}. Consider promotions on {worst_day} (lowest traffic)."),
    (f"🍲 <b>Top Category:</b> {best_cat}",
     f"Revenue: £{best_cat_rev:,.0f} | Profit margin: {best_cat_marg:.1f}%"),
    (f"⭐ <b>Star Menu Item:</b> {best_item}",
     f"Highest revenue item: £{best_item_rev:,.0f}. Feature prominently in menus."),
    (f"📦 <b>Best Seller:</b> {most_sold}",
     "Highest quantity sold. Ensure consistent stock and quality."),
]

for title, body in insights:
    st.markdown(f"""
    <div class="insight-card">
        <p style="margin:0;font-size:1rem">{title}</p>
        <p style="margin:0;color:#aaa;font-size:0.85rem;margin-top:0.3rem">{body}</p>
    </div>
    """, unsafe_allow_html=True)

# ── Section 3: Profitability Analysis ────────────────────────
st.markdown('<p class="section-header">💰 Profitability Deep-Dive</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    cat_marg = cat_rev.copy()
    cat_marg["Margin"] = (cat_marg["profit"] / cat_marg["revenue"] * 100).round(1)
    cat_marg = cat_marg.sort_values("Margin", ascending=False)
    fig_marg = px.bar(cat_marg, x="Category", y="Margin",
                       color="Margin", color_continuous_scale="Greens",
                       template="plotly_dark",
                       labels={"Margin":"Profit Margin (%)","Category":""},
                       title="Profit Margin by Category")
    fig_marg.update_layout(plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                            font_color="#ccc", showlegend=False,
                            margin=dict(l=10,r=10,t=40,b=10), height=320)
    fig_marg.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_marg, use_container_width=True)

with col2:
    if len(products) > 0:
        top_profit = products.nlargest(10, "total_profit")
        fig_prof   = px.bar(top_profit, x="total_profit", y="Description",
                             orientation="h",
                             color="total_profit", color_continuous_scale="Greens",
                             template="plotly_dark",
                             labels={"total_profit":"Total Profit (£)","Description":""},
                             title="Top 10 Items by Profit")
        fig_prof.update_layout(plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                                font_color="#ccc", yaxis=dict(autorange="reversed"),
                                margin=dict(l=10,r=10,t=40,b=10), height=320)
        st.plotly_chart(fig_prof, use_container_width=True)

# ── Section 4: Revenue Trend ──────────────────────────────────
st.markdown('<p class="section-header">📅 Revenue Rolling Averages</p>', unsafe_allow_html=True)
daily_plot = daily.dropna(subset=["date","revenue"]).sort_values("date")
daily_plot["7-day MA"]  = daily_plot["revenue"].rolling(7).mean()
daily_plot["28-day MA"] = daily_plot["revenue"].rolling(28).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=daily_plot["date"], y=daily_plot["revenue"],
                              name="Daily Revenue", mode="lines",
                              line=dict(color="#555", width=1), opacity=0.5))
fig_ma.add_trace(go.Scatter(x=daily_plot["date"], y=daily_plot["7-day MA"],
                              name="7-day MA", mode="lines",
                              line=dict(color="#E87040", width=2)))
fig_ma.add_trace(go.Scatter(x=daily_plot["date"], y=daily_plot["28-day MA"],
                              name="28-day MA", mode="lines",
                              line=dict(color="#378ADD", width=2)))
fig_ma.update_layout(template="plotly_dark",
                      plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                      font_color="#ccc", height=350,
                      legend=dict(orientation="h", y=1.1),
                      margin=dict(l=10,r=10,t=20,b=10))
st.plotly_chart(fig_ma, use_container_width=True)

# ── Section 5: KMeans Summary ─────────────────────────────────
if "kmeans" in models:
    st.markdown('<p class="section-header">🔵 K-Means Cluster Summary</p>', unsafe_allow_html=True)
    km     = models["kmeans"]
    n_clus = km.n_clusters
    st.markdown(f"""
    <div class="insight-card">
        <b>✅ K-Means Model:</b> {n_clus} clusters identified in RFM space.<br>
        <small style="color:#aaa">The model grouped orders by Recency, Frequency (items), and Monetary value.
        Clusters reveal natural groupings of order patterns — useful for identifying
        peak-value order times, popular bundles, and low-value periods.</small>
    </div>
    """, unsafe_allow_html=True)
