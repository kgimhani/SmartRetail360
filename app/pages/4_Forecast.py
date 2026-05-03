import os, sys
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT, "data", "smartretail.db")
sys.path.insert(0, PROJECT)
from build_db import ensure_db
ensure_db()

"""
SmartRetail360 — Forecast Page
File: app/pages/4_Forecast.py
"""

import os, sys, sqlite3, json, pickle, warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

warnings.filterwarnings("ignore")


st.set_page_config(page_title="Revenue Forecast | SmartRetail360",
                   page_icon="🔮", layout="wide")

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
    daily = pd.read_sql("SELECT * FROM daily_timeseries", conn)
    conn.close()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    return daily

@st.cache_resource
def load_model():
    model_path = os.path.join(PROJECT, "models", "forecast_model.pkl")
    feats_path = os.path.join(PROJECT, "models", "forecast_features.json")
    meta_path  = os.path.join(PROJECT, "models", "forecast_meta.json")
    if not os.path.exists(model_path):
        return None, None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feats_path)  as f:
        feats = json.load(f)
    with open(meta_path)   as f:
        meta  = json.load(f)
    return model, feats, meta


if not os.path.exists(DB_PATH):
    st.error("❌ Database not found. Run the pipeline notebook first.")
    st.stop()

daily          = load_data()
model, feats, meta = load_model()

st.markdown("# 🔮 Revenue Forecast")
st.markdown("*GradientBoosting model trained on historical daily revenue*")
st.markdown("---")

if model is None:
    st.error("❌ Forecast model not found. Run pipeline Cell 7 first.")
    st.stop()

# Model stats
c1, c2, c3 = st.columns(3)
avg_rev = daily["revenue"].mean()
c1.metric("Avg Daily Revenue", f"£{avg_rev:,.0f}", help="Average daily revenue")
c2.metric("Forecast Horizon", "30 Days", help="Days forecasted ahead")
c3.metric("Training Days", f"{len(daily):,}", help="Days used to train model")

st.markdown("")

# Historical vs future forecast
st.markdown('<p class="section-header">📈 Historical Revenue + 30-Day Forecast</p>', unsafe_allow_html=True)
st.sidebar.markdown("## 🔮 Forecast Settings")
forecast_days = st.sidebar.slider("Forecast horizon (days)", 7, 90, 30)

# Build lag features for forecast
daily_clean = daily.copy().dropna(subset=["date","revenue"])
daily_clean = daily_clean.sort_values("date").reset_index(drop=True)

# Reconstruct lag features
for lag in [1, 7, 14, 28]:
    daily_clean[f"lag_{lag}"] = daily_clean["revenue"].shift(lag)
daily_clean["rolling_7"]  = daily_clean["revenue"].shift(1).rolling(7).mean()
daily_clean["rolling_28"] = daily_clean["revenue"].shift(1).rolling(28).mean()

feat_cols = ["day_of_week","month","is_weekend","trend","lag_1","lag_7","lag_14","lag_28","rolling_7","rolling_28"]
available_feats = [f for f in feat_cols if f in daily_clean.columns]

train_d   = daily_clean.dropna(subset=available_feats)
y_pred_in = model.predict(train_d[available_feats])

# Future forecast
last_row   = daily_clean.iloc[-1]
last_date  = last_row["date"]
rev_buffer = daily_clean["revenue"].tolist()

future_dates   = []
future_revenue = []

for i in range(1, forecast_days + 1):
    fd   = last_date + timedelta(days=i)
    fidx = len(daily_clean) + i - 1

    row_feats = {
        "day_of_week" : fd.dayofweek,
        "month"       : fd.month,
        "is_weekend"  : int(fd.dayofweek >= 5),
        "trend"       : fidx,
        "lag_1"       : rev_buffer[-1],
        "lag_7"       : rev_buffer[-7]  if len(rev_buffer) >= 7  else np.mean(rev_buffer),
        "lag_14"      : rev_buffer[-14] if len(rev_buffer) >= 14 else np.mean(rev_buffer),
        "lag_28"      : rev_buffer[-28] if len(rev_buffer) >= 28 else np.mean(rev_buffer),
        "rolling_7"   : np.mean(rev_buffer[-7:])  if len(rev_buffer) >= 7  else np.mean(rev_buffer),
        "rolling_28"  : np.mean(rev_buffer[-28:]) if len(rev_buffer) >= 28 else np.mean(rev_buffer),
    }

    pred = max(model.predict(pd.DataFrame([row_feats])[available_feats])[0], 0)
    future_dates.append(fd)
    future_revenue.append(pred)
    rev_buffer.append(pred)

# Plot
hist_recent = daily_clean.tail(90)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_recent["date"], y=hist_recent["revenue"],
    name="Historical", mode="lines",
    line=dict(color="#E87040", width=2),
))
fig.add_trace(go.Scatter(
    x=future_dates, y=future_revenue,
    name="Forecast", mode="lines+markers",
    line=dict(color="#378ADD", width=2, dash="dot"),
    marker=dict(size=5),
))
fig.add_vrect(
    x0=last_date, x1=future_dates[-1],
    fillcolor="#378ADD", opacity=0.05,
    layer="below", line_width=0,
)
fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
    font_color="#ccc", height=400,
    legend=dict(orientation="h", y=1.1),
    margin=dict(l=10,r=10,t=20,b=10),
    xaxis_title="Date", yaxis_title="Revenue (£)",
)
st.plotly_chart(fig, use_container_width=True)

# Forecast table
st.markdown('<p class="section-header">📋 Forecast Table</p>', unsafe_allow_html=True)
forecast_df = pd.DataFrame({
    "Date"           : future_dates,
    "Day"            : [d.strftime("%A") for d in future_dates],
    "Forecast Rev(£)": [f"£{v:,.0f}" for v in future_revenue],
})
st.dataframe(forecast_df, use_container_width=True, hide_index=True)

# Day-of-week pattern
st.markdown('<p class="section-header">📅 Historical Revenue by Day of Week</p>', unsafe_allow_html=True)
days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
dow_avg = daily_clean.groupby("day_of_week")["revenue"].mean().reset_index()
dow_avg["Day"] = dow_avg["day_of_week"].apply(lambda x: days[int(x)] if int(x) < 7 else str(x))
dow_avg = dow_avg.sort_values("day_of_week")
fig_dow = px.bar(dow_avg, x="Day", y="revenue",
                  color="revenue", color_continuous_scale="Oranges",
                  template="plotly_dark",
                  labels={"revenue":"Avg Daily Revenue (£)","Day":""})
fig_dow.update_layout(plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                       font_color="#ccc", showlegend=False,
                       margin=dict(l=10,r=10,t=10,b=10), height=300)
st.plotly_chart(fig_dow, use_container_width=True)

# Monthly seasonality
st.markdown('<p class="section-header">📅 Average Revenue by Month</p>', unsafe_allow_html=True)
month_avg = daily_clean.groupby("month")["revenue"].mean().reset_index()
month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
month_avg["Month"] = month_avg["month"].map(month_names)
fig_mon = px.bar(month_avg, x="Month", y="revenue",
                  color="revenue", color_continuous_scale="Oranges",
                  template="plotly_dark",
                  labels={"revenue":"Avg Daily Revenue (£)","Month":""})
fig_mon.update_layout(plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                       font_color="#ccc", showlegend=False,
                       margin=dict(l=10,r=10,t=10,b=10), height=300)
st.plotly_chart(fig_mon, use_container_width=True)
