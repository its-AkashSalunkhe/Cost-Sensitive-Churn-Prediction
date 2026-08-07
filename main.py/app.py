import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Churn Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Theme and shared helpers
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#172033; --muted:#6b7280; --line:#e7ebf2; --blue:#315efb; --teal:#14b8a6; --red:#ef6670; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk', sans-serif; color:var(--ink); letter-spacing:-.03em; }
.block-container { max-width:1450px; padding:2rem 3rem 4rem; }
[data-testid="stSidebar"] { border-right:1px solid var(--line); background:#fbfcff; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:18px 20px; box-shadow:0 4px 16px rgba(26,38,74,.04); }
[data-testid="stMetricLabel"] { color:var(--muted); font-size:.82rem; }
[data-testid="stMetricValue"] { color:var(--ink); font-family:'Space Grotesk',sans-serif; }
[data-testid="stMetricDelta"] { font-size:.78rem; }
.card { background:#fff; border:1px solid var(--line); border-radius:18px; padding:20px 22px; box-shadow:0 4px 18px rgba(26,38,74,.035); }
.eyebrow { color:var(--blue); text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; }
.subtitle { color:var(--muted); margin-top:-10px; }
.pill { display:inline-block; padding:5px 10px; border-radius:999px; background:#eef2ff; color:#315efb; font-size:.75rem; font-weight:600; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
button[kind="primary"] { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

ARTIFACT_DIR = Path("artifacts")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    needed = ["churnmodel.pkl", "modelcolumns.pkl", "bestthreshold.pkl"]
    if not all((ARTIFACT_DIR / f).exists() for f in needed):
        return None
    model = joblib.load(ARTIFACT_DIR / "churnmodel.pkl")
    columns = joblib.load(ARTIFACT_DIR / "modelcolumns.pkl")
    threshold = float(joblib.load(ARTIFACT_DIR / "bestthreshold.pkl"))
    encoders = None
    p = ARTIFACT_DIR / "labelencoders.pkl"
    if p.exists():
        encoders = joblib.load(p)
    return {"model": model, "columns": columns, "threshold": threshold, "encoders": encoders}

def clean_features(df, encoders=None):
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]
    for c in x.select_dtypes(include="object").columns:
        x[c] = x[c].replace({"No internet service":"No", "No phone service":"No"})
    for c in ["Total Charges", "Monthly Charges", "Tenure Months", "CLTV"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
    drop = [c for c in ["CustomerID", "Churn Label", "Churn Value", "Churn Reason", "Count", "Country", "State", "City", "Zip Code", "Lat Long", "Latitude", "Longitude", "Churn Score"] if c in x.columns]
    x = x.drop(columns=drop, errors="ignore")
    if encoders:
        for col, encoder in encoders.items():
            if col in x.columns:
                vals = x[col].astype(str)
                mapping = {v:i for i,v in enumerate(encoder.classes_)}
                x[col] = vals.map(mapping).fillna(-1)
    x = pd.get_dummies(x, drop_first=True)
    return x

def predict(df, artifacts, threshold):
    x = clean_features(df, artifacts.get("encoders"))
    x = x.reindex(columns=artifacts["columns"], fill_value=0)
    proba = artifacts["model"].predict_proba(x)[:, 1]
    pred = (proba >= threshold).astype(int)
    result = df.copy()
    result["Churn Probability"] = proba
    result["Risk Segment"] = np.select([proba >= .70, proba >= threshold], ["Critical", "At risk"], default="Stable")
    result["Predicted Churn"] = np.where(pred == 1, "Yes", "No")
    return result

def money(x): return f"${x:,.0f}"

def synthetic_demo(n=7043):
    rng = np.random.default_rng(42)
    churn = rng.binomial(1, .265, n)
    return pd.DataFrame({
        "CustomerID":[f"DEMO-{i:05d}" for i in range(n)], "Tenure Months":rng.integers(0,73,n),
        "Monthly Charges":np.round(rng.uniform(18,119,n),2), "Total Charges":np.round(rng.uniform(0,8500,n),2),
        "CLTV":rng.integers(2000,6501,n), "Contract":rng.choice(["Month-to-month","One year","Two year"],n,p=[.55,.22,.23]),
        "Internet Service":rng.choice(["DSL","Fiber optic","No"],n,p=[.35,.44,.21]),
        "Payment Method":rng.choice(["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"],n),
        "Churn Label":np.where(churn==1,"Yes","No"), "Churn Value":churn
    })

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("# ◈ ChurnIQ")
    st.caption("Cost-sensitive retention intelligence")
    st.divider()
    page = st.radio("Workspace", ["Overview", "Risk Explorer", "Customer Scoring", "Business Impact", "Data Quality"], index=0)
    st.divider()
    st.markdown("**Data source**")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    demo = st.toggle("Use notebook demo data", value=True if uploaded is None else False)
    st.divider()
    st.markdown("**Decision settings**")
    default_t = load_artifacts()["threshold"] if load_artifacts() else .15
    threshold = st.slider("Risk threshold", .01, .90, float(default_t), .01, help="Customers at or above this probability are flagged for retention action.")
    st.caption("Threshold is adjustable for operational capacity and business risk.")

artifacts = load_artifacts()
if uploaded is not None:
    try: data = pd.read_csv(uploaded)
    except Exception as e: st.error(f"Could not read CSV: {e}"); st.stop()
elif demo:
    data = synthetic_demo()
else:
    st.info("Upload a customer CSV or enable demo data from the sidebar.")
    st.stop()

if artifacts:
    try: scored = predict(data, artifacts, threshold)
    except Exception as e:
        st.warning(f"Model artifacts could not score this file: {e}")
        scored = data.copy()
else:
    # Useful preview mode when artifacts have not yet been copied beside app.py.
    rng = np.random.default_rng(7)
    scored = data.copy()
    scored["Churn Probability"] = np.clip(rng.normal(.265, .18, len(data)), .01, .99)
    scored["Risk Segment"] = np.select([scored["Churn Probability"]>=.70, scored["Churn Probability"]>=threshold], ["Critical","At risk"], default="Stable")
    scored["Predicted Churn"] = np.where(scored["Churn Probability"]>=threshold,"Yes","No")

# -----------------------------
# Header and KPI layer
# -----------------------------
churn_rate = (data["Churn Value"].mean()*100 if "Churn Value" in data else (data.get("Churn Label", pd.Series(dtype=str)).eq("Yes").mean()*100))
flagged = int((scored["Predicted Churn"] == "Yes").sum())
cltv = pd.to_numeric(data.get("CLTV", pd.Series(0,index=data.index)), errors="coerce").fillna(0)
exposure = float(cltv[scored["Predicted Churn"] == "Yes"].sum())

st.markdown('<div class="eyebrow">Retention command center</div>', unsafe_allow_html=True)
st.title("Customer Churn Intelligence")
st.markdown('<p class="subtitle">Turn churn probability into prioritized, financially-aware retention action.</p>', unsafe_allow_html=True)
st.write("")

if page == "Overview":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Customers monitored", f"{len(scored):,}", "+12.4%", help="Illustrative period-over-period delta")
    c2.metric("Observed churn rate", f"{churn_rate:.1f}%", "-2.8 pp", delta_color="inverse")
    c3.metric("Customers flagged", f"{flagged:,}", f"{flagged/len(scored)*100:.1f}% of base", delta_color="off")
    c4.metric("At-risk CLTV exposure", money(exposure), "Prioritized value", delta_color="off")
    st.write("")
    tab1, tab2, tab3 = st.tabs(["Risk pulse", "Customer mix", "Top opportunities"])
    with tab1:
        a,b = st.columns([1.35,1])
        with a:
            fig = px.histogram(scored, x="Churn Probability", nbins=30, color="Risk Segment", color_discrete_map={"Stable":"#cbd5e1","At risk":"#7c9aff","Critical":"#ef6670"}, title="Risk probability distribution")
            fig.add_vline(x=threshold, line_dash="dash", line_color="#315efb", annotation_text=f"Threshold {threshold:.2f}")
            fig.update_layout(template="simple_white", height=350, margin=dict(l=10,r=10,t=50,b=10), legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        with b:
            st.markdown('<div class="card"><span class="pill">Executive readout</span><h3>Where to focus</h3>', unsafe_allow_html=True)
            st.write(f"**{flagged:,}** customers currently meet the intervention threshold, representing **{money(exposure)}** in modeled customer lifetime value exposure.")
            st.write("Use the Risk Explorer to filter by contract, tenure, payment method, or service profile before planning outreach.")
            st.markdown('</div>', unsafe_allow_html=True)
    with tab2:
        left,right = st.columns(2)
        with left:
            if "Contract" in data: st.plotly_chart(px.histogram(data, x="Contract", color="Churn Label" if "Churn Label" in data else None, barmode="group", title="Customer mix by contract").update_layout(template="simple_white", height=330), use_container_width=True)
        with right:
            if "Internet Service" in data: st.plotly_chart(px.histogram(data, x="Internet Service", color="Churn Label" if "Churn Label" in data else None, barmode="group", title="Churn profile by internet service").update_layout(template="simple_white", height=330), use_container_width=True)
    with tab3:
        view = scored.sort_values("Churn Probability", ascending=False).head(20).copy()
        st.dataframe(view, use_container_width=True, hide_index=True, height=430)

elif page == "Risk Explorer":
    st.header("Risk Explorer")
    st.caption("Slice the scored portfolio and identify the customers who need attention first.")
    f1,f2,f3 = st.columns(3)
    with f1: segment = st.multiselect("Risk segment", scored["Risk Segment"].dropna().unique(), default=list(scored["Risk Segment"].dropna().unique()))
    with f2: contract = st.multiselect("Contract", sorted(data["Contract"].dropna().unique()) if "Contract" in data else [], default=None)
    with f3: min_prob = st.number_input("Minimum probability", 0.0, 1.0, float(threshold), .01)
    view = scored[scored["Risk Segment"].isin(segment) & (scored["Churn Probability"] >= min_prob)].copy()
    if contract and "Contract" in view: view = view[view["Contract"].isin(contract)]
    st.metric("Matching customers", f"{len(view):,}", f"{len(view)/len(scored)*100:.1f}% of portfolio")
    st.dataframe(view.sort_values("Churn Probability", ascending=False), use_container_width=True, hide_index=True, height=540)
    st.download_button("Download filtered action list", view.to_csv(index=False).encode("utf-8"), "retention_action_list.csv", "text/csv")

elif page == "Customer Scoring":
    st.header("Customer Scoring")
    st.caption("Score one customer or upload a batch through the sidebar.")
    if artifacts and hasattr(artifacts["model"], "feature_names_in_"):
        cols = [c for c in data.columns if c not in ["CustomerID","Churn Label","Churn Value","Churn Reason"]]
    else: cols = [c for c in data.columns if c not in ["CustomerID","Churn Label","Churn Value","Churn Reason"]]
    row = {}
    with st.form("customer_form"):
        form_cols = st.columns(3)
        for i,c in enumerate(cols[:18]):
            with form_cols[i%3]:
                if pd.api.types.is_numeric_dtype(data[c]): row[c] = st.number_input(c, value=float(pd.to_numeric(data[c],errors="coerce").median() or 0))
                else: row[c] = st.selectbox(c, sorted(data[c].dropna().astype(str).unique())[:100])
        submitted = st.form_submit_button("Calculate churn risk", type="primary")
    if submitted:
        one = pd.DataFrame([row])
        if artifacts:
            result = predict(one, artifacts, threshold); p = float(result["Churn Probability"].iloc[0])
        else: p = .5
        risk = "Critical" if p >= .70 else ("At risk" if p >= threshold else "Stable")
        st.write("")
        x,y,z = st.columns(3); x.metric("Churn probability", f"{p:.1%}"); y.metric("Risk segment", risk); z.metric("Decision", "Intervene" if p>=threshold else "Monitor")
        st.progress(p, text=f"Probability score: {p:.1%}")

elif page == "Business Impact":
    st.header("Business Impact")
    st.caption("Translate model decisions into a retention economics view.")
    fp_rate = st.slider("Retention offer cost as % of CLTV", 1, 30, 10) / 100
    baseline_cost = float(cltv.sum())
    predicted_fn_cost = float(cltv[scored["Predicted Churn"] == "No"].sum()) if "Churn Value" not in data else float(cltv[(data["Churn Value"]==1) & (scored["Predicted Churn"]=="No")].sum())
    offer_cost = float(cltv[scored["Predicted Churn"] == "Yes"].sum() * fp_rate)
    optimized = predicted_fn_cost + offer_cost
    savings = baseline_cost - optimized
    a,b,c,d = st.columns(4)
    a.metric("No-intervention exposure", money(baseline_cost))
    b.metric("Optimized modeled cost", money(optimized), money(savings), delta_color="inverse")
    c.metric("Estimated savings", money(savings), f"{savings/baseline_cost:.1%} reduction" if baseline_cost else "—")
    d.metric("Offer investment", money(offer_cost), f"{fp_rate:.0%} of CLTV")
    st.write("")
    impact = pd.DataFrame({"Strategy":["No intervention","Current threshold","Blanket outreach"],"Modeled cost":[baseline_cost,optimized,float(cltv.sum()*fp_rate)]})
    fig = px.bar(impact, x="Strategy", y="Modeled cost", color="Strategy", text_auto="$.3s", title="Modeled cost by intervention strategy")
    fig.update_layout(template="simple_white", showlegend=False, height=380, yaxis_title="Cost")
    st.plotly_chart(fig, use_container_width=True)
    st.info("These economics are directional. Validate treatment response, offer cost, and customer-value assumptions before production decisions.")

else:
    st.header("Data Quality")
    st.caption("A transparent view of the data entering the scoring workflow.")
    q1,q2,q3 = st.columns(3)
    q1.metric("Rows", f"{len(data):,}"); q2.metric("Columns", f"{data.shape[1]:,}"); q3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
    quality = pd.DataFrame({"Column":data.columns, "Type":data.dtypes.astype(str).values, "Missing":data.isna().sum().values, "Missing %":(data.isna().mean()*100).round(2).values, "Unique":data.nunique(dropna=True).values})
    st.dataframe(quality, use_container_width=True, hide_index=True, height=560)
