
import os, sqlite3, pickle, json, warnings
import pandas as pd
import numpy as np
from datetime import timedelta
warnings.filterwarnings("ignore")

PROJECT   = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(PROJECT, "data", "raw", "Anonymized_Restaurant_Sales_Data.csv")
DB_PATH   = os.path.join(PROJECT, "data", "smartretail.db")
MODELS    = os.path.join(PROJECT, "models")

def db_exists():
    return os.path.exists(DB_PATH)

def build():
    print("Building database...")
    os.makedirs(os.path.join(PROJECT,"data","processed"), exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    df_raw = pd.read_csv(DATA_FILE)
    df = df_raw[df_raw["Cancelled"]=="No"].copy()
    df["InvoiceDate"] = pd.to_datetime(df["Date"]+" "+df["Time"], format="%d/%m/%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["OrderID"] = df["OrderID"].astype(str).str.replace("RES_ORD_","",regex=False).str.replace(".0","",regex=False).str.strip()
    df = df.rename(columns={"Line item name":"Description","Price Per Item":"Price","Gross Sales":"line_revenue","Est. Profit":"profit","Est. Cost":"cost"})
    df = df.drop(columns=["Date","Time"], errors="ignore")
    df["date"] = df["InvoiceDate"].dt.date
    df["year"] = df["InvoiceDate"].dt.year
    df["month"] = df["InvoiceDate"].dt.month
    df["day_of_week"] = df["InvoiceDate"].dt.dayofweek
    df["hour"] = df["InvoiceDate"].dt.hour
    df = df.reset_index(drop=True)
    snapshot = df["InvoiceDate"].max() + timedelta(days=1)
    rfm = df.groupby("OrderID").agg(Recency=("InvoiceDate",lambda x:(snapshot-x.max()).days),Frequency=("Description","count"),Monetary=("line_revenue","sum"),OrderDate=("InvoiceDate","max")).reset_index()
    rfm["R_Score"] = pd.qcut(rfm["Recency"],5,labels=[5,4,3,2,1],duplicates="drop").astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"),5,labels=[1,2,3,4,5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),5,labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_Total"] = rfm["R_Score"]+rfm["F_Score"]+rfm["M_Score"]
    def seg(row):
        t=row["R_Score"]+row["F_Score"]+row["M_Score"]
        if t>=12: return "High Value"
        elif t>=9: return "Good Value"
        elif t>=6: return "Average"
        elif row["R_Score"]<=2: return "Lapsed"
        else: return "Low Value"
    rfm["Segment"] = rfm.apply(seg,axis=1)
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    feat = rfm[["Recency","Frequency","Monetary"]].copy()
    feat["Frequency"] = np.log1p(feat["Frequency"])
    feat["Monetary"] = np.log1p(feat["Monetary"])
    sc = StandardScaler()
    X = sc.fit_transform(feat)
    sil=[]
    for k in range(2,7):
        km_t=KMeans(n_clusters=k,random_state=42,n_init=10)
        sil.append(silhouette_score(X,km_t.fit_predict(X)))
    best_k=max(3,min(list(range(2,7))[np.argmax(sil)],5))
    km=KMeans(n_clusters=best_k,random_state=42,n_init=20)
    rfm["Cluster"]=km.fit_predict(X)
    with open(os.path.join(MODELS,"kmeans_model.pkl"),"wb") as f: pickle.dump(km,f)
    with open(os.path.join(MODELS,"rfm_scaler.pkl"),"wb") as f: pickle.dump(sc,f)
    from sklearn.ensemble import GradientBoostingRegressor,RandomForestClassifier
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    daily=df.groupby("date").agg(revenue=("line_revenue","sum"),profit=("profit","sum"),quantity=("Quantity","sum"),n_orders=("OrderID","nunique")).reset_index()
    daily["date"]=pd.to_datetime(daily["date"])
    daily=daily.sort_values("date").reset_index(drop=True)
    daily["day_of_week"]=daily["date"].dt.dayofweek
    daily["month"]=daily["date"].dt.month
    daily["is_weekend"]=(daily["day_of_week"]>=5).astype(int)
    daily["trend"]=np.arange(len(daily))
    for lag in [1,7,14,28]: daily[f"lag_{lag}"]=daily["revenue"].shift(lag)
    daily["rolling_7"]=daily["revenue"].shift(1).rolling(7).mean()
    daily["rolling_28"]=daily["revenue"].shift(1).rolling(28).mean()
    dc=daily.dropna().reset_index(drop=True)
    fc=["day_of_week","month","is_weekend","trend","lag_1","lag_7","lag_14","lag_28","rolling_7","rolling_28"]
    sp=int(len(dc)*0.85)
    fm=GradientBoostingRegressor(n_estimators=300,learning_rate=0.05,max_depth=4,random_state=42)
    fm.fit(dc.iloc[:sp][fc],dc.iloc[:sp]["revenue"])
    preds=np.maximum(fm.predict(dc.iloc[sp:][fc]),0)
    mae=mean_absolute_error(dc.iloc[sp:]["revenue"],preds)
    mape=np.mean(np.abs((dc.iloc[sp:]["revenue"].values-preds)/(dc.iloc[sp:]["revenue"].values+1)))*100
    with open(os.path.join(MODELS,"forecast_model.pkl"),"wb") as f: pickle.dump(fm,f)
    with open(os.path.join(MODELS,"forecast_features.json"),"w") as f: json.dump(fc,f)
    with open(os.path.join(MODELS,"forecast_meta.json"),"w") as f: json.dump({"mae":round(mae,2),"mape":round(mape,2)},f)
    df_all=df_raw.copy()
    df_all["InvoiceDate"]=pd.to_datetime(df_all["Date"]+" "+df_all["Time"],format="%d/%m/%Y %H:%M",errors="coerce")
    df_all=df_all.dropna(subset=["InvoiceDate"])
    df_all["hour"]=df_all["InvoiceDate"].dt.hour
    df_all["day_of_week"]=df_all["InvoiceDate"].dt.dayofweek
    df_all["month"]=df_all["InvoiceDate"].dt.month
    df_all["is_weekend"]=(df_all["day_of_week"]>=5).astype(int)
    df_all["is_cancelled"]=(df_all["Cancelled"]=="Yes").astype(int)
    df_all["OrderType_enc"]=(df_all["OrderType"]=="Delivery").astype(int)
    cd=pd.get_dummies(df_all["Category"],prefix="cat")
    df_all=pd.concat([df_all,cd],axis=1)
    cc=[c for c in df_all.columns if c.startswith("cat_")]
    fcc=["hour","day_of_week","month","is_weekend","OrderType_enc","Quantity","Price Per Item"]+cc
    Xc=df_all[fcc].fillna(0)
    yc=df_all["is_cancelled"]
    Xtr,Xte,ytr,yte=train_test_split(Xc,yc,test_size=0.2,stratify=yc,random_state=42)
    sc2=StandardScaler()
    Xtr_s=sc2.fit_transform(Xtr)
    Xte_s=sc2.transform(Xte)
    cm2=RandomForestClassifier(n_estimators=200,max_depth=6,random_state=42,class_weight="balanced",n_jobs=-1)
    cm2.fit(Xtr_s,ytr)
    try: auc=roc_auc_score(yte,cm2.predict_proba(Xte_s)[:,1])
    except: auc=0.5
    with open(os.path.join(MODELS,"cancel_model.pkl"),"wb") as f: pickle.dump(cm2,f)
    with open(os.path.join(MODELS,"cancel_scaler.pkl"),"wb") as f: pickle.dump(sc2,f)
    with open(os.path.join(MODELS,"cancel_features.json"),"w") as f: json.dump(fcc,f)
    with open(os.path.join(MODELS,"cancel_meta.json"),"w") as f: json.dump({"auc":round(auc,4),"cancel_rate":round(float(yc.mean()),4)},f)
    import nltk
    nltk.download("vader_lexicon",quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia=SentimentIntensityAnalyzer()
    def gs(text):
        if pd.isna(text) or str(text).strip()=="": return "Neutral"
        s=sia.polarity_scores(str(text))["compound"]
        return "Positive" if s>=0.05 else ("Negative" if s<=-0.05 else "Neutral")
    ps=df.groupby(["Description","Category"]).agg(total_revenue=("line_revenue","sum"),total_sold=("Quantity","sum"),n_orders=("OrderID","nunique"),avg_price=("Price","mean"),total_profit=("profit","sum")).reset_index()
    ps["Sentiment"]=ps["Description"].apply(gs)
    ps["SentimentScore"]=ps["Description"].apply(lambda x:sia.polarity_scores(str(x))["compound"] if pd.notna(x) else 0)
    pt=df.groupby(["Description","Category"]).agg(total_revenue=("line_revenue","sum"),total_qty=("Quantity","sum"),n_orders=("OrderID","nunique"),avg_price=("Price","mean"),total_profit=("profit","sum")).reset_index()
    cr=df.groupby("Category").agg(revenue=("line_revenue","sum"),profit=("profit","sum"),orders=("OrderID","nunique"),qty=("Quantity","sum")).reset_index()
    or_=df.groupby("OrderType").agg(revenue=("line_revenue","sum"),profit=("profit","sum"),orders=("OrderID","nunique")).reset_index()
    conn=sqlite3.connect(DB_PATH)
    df.to_sql("transactions",conn,if_exists="replace",index=False)
    rfm.to_sql("rfm",conn,if_exists="replace",index=False)
    daily.to_sql("daily_timeseries",conn,if_exists="replace",index=False)
    pt.to_sql("products",conn,if_exists="replace",index=False)
    cr.to_sql("category_revenue",conn,if_exists="replace",index=False)
    or_.to_sql("ordertype_revenue",conn,if_exists="replace",index=False)
    ps.to_sql("product_sentiment",conn,if_exists="replace",index=False)
    conn.close()
    print("Database built!")

def ensure_db():
    if not db_exists():
        build()
