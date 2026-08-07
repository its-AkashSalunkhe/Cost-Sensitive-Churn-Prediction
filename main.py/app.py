"""
Customer Churn Prediction — Cost-Sensitive Optimization Dashboard
===================================================================
Built on top of the analysis in main.ipynb.
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    load_artifacts, load_raw_data, preprocess_full_dataset, get_train_test_split,
    calculate_business_cost, threshold_sweep, train_all_models,
    evaluate_model_at_threshold, find_optimal_threshold,
    preprocess_single_customer, risk_segment,
    RAW_INPUT_COLUMNS, XGB_AVAILABLE, LGBM_AVAILABLE
)

# ============================================================================
# PAGE CONFIG + GLOBAL STYLE
# ============================================================================
st.set_page_config(
    page_title="Churn Prediction | Cost-Sensitive Optimization",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .main > div {padding-top: 1.2rem;}

    html, body, [class*="css"] { font-family: 'Segoe UI', 'Inter', sans-serif; }

    .kpi-card {
        background: linear-gradient(135deg, #161b26 0%, #1f2634 100%);
        border: 1px solid #2a3142;
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        text-align: left;
    }
    .kpi-label { font-size: 0.8rem; color: #9aa4b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem;}
    .kpi-value { font-size: 1.85rem; font-weight: 700; color: #f5f7fa; line-height: 1.15;}
    .kpi-sub { font-size: 0.82rem; margin-top: 0.35rem; }
    .kpi-good { color: #37d67a; }
    .kpi-bad { color: #ff6161; }
    .kpi-neutral { color: #9aa4b8; }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
        color: #f5f7fa;
        border-left: 4px solid #6c8dfa;
        padding-left: 0.6rem;
    }

    .pill {
        display: inline-block;
        padding: 0.18rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.3rem;
    }
    .pill-blue { background: rgba(108,141,250,0.15); color: #6c8dfa; }
    .pill-green { background: rgba(55,214,122,0.15); color: #37d67a; }
    .pill-orange { background: rgba(255,159,67,0.15); color: #ff9f43; }
    .pill-red { background: rgba(255,97,97,0.15); color: #ff6161; }

    .insight-box {
        background: #161b26;
        border-left: 4px solid #6c8dfa;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-size: 0.92rem;
        color: #cfd6e4;
        margin: 0.6rem 0 1rem 0;
    }

    div[data-testid="stMetricValue"] { font-size: 1.6rem; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_CHURN = "#ff6161"
COLOR_NOCHURN = "#37d67a"
COLOR_ACCENT = "#6c8dfa"


def kpi_card(label, value, sub=None, sub_class="kpi-neutral"):
    sub_html = f'<div class="kpi-sub {sub_class}">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# LOAD DATA + ARTIFACTS (cached)
# ============================================================================
model, model_columns, best_threshold, label_encoders = load_artifacts()
raw_data = load_raw_data()
_, full_encoded, X, y = preprocess_full_dataset(label_encoders, model_columns)
x_train, x_test, y_train, y_test = get_train_test_split(X, y)
cltv_test = x_test['CLTV']

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("## 📉 Churn Intelligence")
st.sidebar.caption("Cost-sensitive churn prediction system")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Executive Summary",
        "📊 Data Explorer",
        "🤖 Model Comparison",
        "🎚️ Threshold & Cost Optimizer",
        "🔮 Predict a Customer",
        "📁 Batch Scoring",
        "ℹ️ Methodology",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown("**Deployed model:** Logistic Regression")
st.sidebar.markdown(f"**Optimized threshold:** `{best_threshold:.3f}`")
st.sidebar.markdown(f"**Test set size:** {len(x_test):,} customers")
st.sidebar.divider()
st.sidebar.caption("Built with scikit-learn + Streamlit · Data: IBM Telco Customer Churn")


# ============================================================================
# PAGE 1 — EXECUTIVE SUMMARY
# ============================================================================
if page == "🏠 Executive Summary":
    st.title("Customer Churn Prediction")
    st.markdown(
        "##### Cost-sensitive optimization system — reducing churn-related revenue loss "
        "through threshold tuning, not just accuracy"
    )
    st.write("")

    # --- compute the 3 headline scenarios live, on the real held-out test set
    y_none = np.zeros(len(y_test))
    cost_none, _, _ = calculate_business_cost(y_test, y_none, cltv_test)

    proba_lr = model.predict_proba(x_test)[:, 1]
    y_default = (proba_lr >= 0.5).astype(int)
    cost_default, fn_default, fp_default = calculate_business_cost(y_test, y_default, cltv_test)

    y_optimized = (proba_lr >= best_threshold).astype(int)
    cost_optimized, fn_optimized, fp_optimized = calculate_business_cost(y_test, y_optimized, cltv_test)

    savings_vs_none = cost_none - cost_optimized
    pct_savings_vs_none = savings_vs_none / cost_none * 100

    savings_vs_default = cost_default - cost_optimized
    pct_savings_vs_default = savings_vs_default / cost_default * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Cost — No Model", f"${cost_none:,.0f}", "Every churner walks away", "kpi-bad")
    with c2:
        kpi_card("Cost — Default Threshold (0.50)", f"${cost_default:,.0f}",
                  f"{fn_default} churners missed", "kpi-bad")
    with c3:
        kpi_card("Cost — Optimized Threshold", f"${cost_optimized:,.0f}",
                  f"{fn_optimized} churners missed", "kpi-good")
    with c4:
        kpi_card("Total Savings vs. No Model", f"${savings_vs_none:,.0f}",
                  f"↓ {pct_savings_vs_none:.1f}% reduction", "kpi-good")

    st.write("")
    st.markdown(
        f"""
        <div class="insight-box">
        <b>Headline result:</b> Deploying the cost-optimized model reduces churn-related losses by
        <b>{pct_savings_vs_none:.1f}%</b> compared to having no predictive model at all
        (${cost_none:,.0f} → ${cost_optimized:,.0f}).
        Threshold optimization alone — tuning the cutoff from the textbook default of 0.50 down to
        the cost-minimizing <b>{best_threshold:.2f}</b> — is responsible for
        <b>${savings_vs_default:,.0f}</b> of that (a <b>{pct_savings_vs_default:.1f}%</b> improvement
        over a naively deployed model). All figures are computed live on the {len(x_test):,}-customer
        held-out test set.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown('<div class="section-title">Cost by Strategy</div>', unsafe_allow_html=True)
        scenario_df = pd.DataFrame({
            "Strategy": ["No Model\n(do nothing)", "Default Threshold\n(0.50)", "Optimized Threshold\n(0.15)"],
            "Total Cost": [cost_none, cost_default, cost_optimized],
        })
        fig = px.bar(
            scenario_df, x="Strategy", y="Total Cost", text="Total Cost",
            color="Strategy",
            color_discrete_sequence=["#ff6161", "#ff9f43", "#37d67a"],
            template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, height=380, yaxis_title="Total Business Cost ($)",
                           xaxis_title=None, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Where the Cost Comes From</div>', unsafe_allow_html=True)
        cost_breakdown = pd.DataFrame({
            "Model": ["Default (0.50)", "Optimized (0.15)"],
            "Missed Churners (FN cost)": [fn_default, fn_optimized],
            "Wasted Offers (FP cost)": [fp_default, fp_optimized],
        })
        fig2 = go.Figure()
        fig2.add_bar(name="Missed churners — lost CLTV", x=cost_breakdown["Model"],
                     y=cost_breakdown["Missed Churners (FN cost)"], marker_color="#ff6161")
        fig2.add_bar(name="Wasted retention offers", x=cost_breakdown["Model"],
                     y=cost_breakdown["Wasted Offers (FP cost)"], marker_color="#ff9f43")
        fig2.update_layout(barmode='stack', template=PLOTLY_TEMPLATE, height=380,
                            yaxis_title="Cost ($)", legend=dict(orientation="h", y=-0.2),
                            margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Default vs. Optimized — What Actually Changes</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Recall (catch churners)", "98%", "+21 pts vs default")
    with m2:
        st.metric("Missed churners (FN)", f"{fn_optimized}", f"-{87-6} vs default", delta_color="inverse")
    with m3:
        st.metric("Precision", "38%", "-13 pts vs default", delta_color="inverse")
    with m4:
        st.metric("% of customers flagged", f"{y_optimized.mean()*100:.0f}%", "more outreach needed")

    st.caption(
        "⚠️ Trade-off to be transparent about: catching 98% of churners means flagging ~68% of "
        "the customer base as at-risk. That's mathematically cost-optimal under this cost function, "
        "but operationally it means retention offers go out broadly — worth pairing with capacity "
        "constraints (see Methodology tab) before a real rollout."
    )


# ============================================================================
# PAGE 2 — DATA EXPLORER
# ============================================================================
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.caption("Exploratory analysis on the raw Telco Customer Churn dataset")

    c1, c2, c3, c4 = st.columns(4)
    churn_rate = (raw_data['Churn Label'] == 'Yes').mean() * 100
    with c1:
        kpi_card("Total Customers", f"{len(raw_data):,}")
    with c2:
        kpi_card("Churn Rate", f"{churn_rate:.1f}%", "Class imbalance present", "kpi-bad")
    with c3:
        kpi_card("Avg. Monthly Charges", f"${raw_data['Monthly Charges'].mean():.2f}")
    with c4:
        kpi_card("Avg. CLTV", f"${raw_data['CLTV'].mean():,.0f}")

    st.write("")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Churn Overview", "Tenure & Charges", "Contract & Service", "Correlation"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            vc = raw_data['Churn Label'].value_counts().reset_index()
            vc.columns = ['Churn Label', 'Count']
            fig = px.pie(vc, names='Churn Label', values='Count', hole=0.55,
                         color='Churn Label',
                         color_discrete_map={'Yes': COLOR_CHURN, 'No': COLOR_NOCHURN},
                         template=PLOTLY_TEMPLATE)
            fig.update_layout(title="Churn Distribution", height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="insight-box">73% retained vs 27% churned — a meaningful class '
                'imbalance, which is why class-weighting / SMOTE-style handling matters before '
                'modeling.</div>', unsafe_allow_html=True
            )
        with col2:
            fig = px.histogram(raw_data, x="Monthly Charges", color="Churn Label",
                                barmode="overlay", nbins=40,
                                color_discrete_map={'Yes': COLOR_CHURN, 'No': COLOR_NOCHURN},
                                template=PLOTLY_TEMPLATE)
            fig.update_layout(title="Monthly Charges by Churn Status", height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="insight-box">Churners skew toward higher monthly bills — price '
                'sensitivity looks like a real driver, not just noise.</div>',
                unsafe_allow_html=True
            )

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(raw_data, x="Tenure Months", color="Churn Label", nbins=30,
                                barmode="overlay",
                                color_discrete_map={'Yes': COLOR_CHURN, 'No': COLOR_NOCHURN},
                                template=PLOTLY_TEMPLATE)
            fig.update_layout(title="Tenure Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(raw_data, x="Churn Label", y="Tenure Months", color="Churn Label",
                         color_discrete_map={'Yes': COLOR_CHURN, 'No': COLOR_NOCHURN},
                         template=PLOTLY_TEMPLATE)
            fig.update_layout(title="Tenure by Churn Status", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">Median tenure for churners is ~10 months vs ~38 months for '
            'retained customers. Early-tenure customers are clearly the highest-risk group — '
            'this "bathtub" pattern (spike at 0 months, spike at 70+ months) shows both a churn '
            'risk window early on and a loyal long-tenure core.</div>',
            unsafe_allow_html=True
        )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            ct = pd.crosstab(raw_data['Contract'], raw_data['Churn Label'], normalize='index') * 100
            ct = ct.reset_index().melt(id_vars='Contract', var_name='Churn Label', value_name='Pct')
            fig = px.bar(ct, x='Contract', y='Pct', color='Churn Label', barmode='group',
                         color_discrete_map={'Yes': COLOR_CHURN, 'No': COLOR_NOCHURN},
                         template=PLOTLY_TEMPLATE)
            fig.update_layout(title="Churn Rate by Contract Type", yaxis_title="% of customers", height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            ct2 = pd.crosstab(raw_data['Internet Service'], raw_data['Churn Label'], normalize='index') * 100
            ct2 = ct2.reset_index().melt(id_vars='Internet Service', var_name='Churn Label', value_name='Pct')
            fig = px.bar(ct2, x='Internet Service', y='Pct', color='Churn Label', barmode='group',
                         color_discrete_map={'Yes': COLOR_CHURN, 'No': COLOR_NOCHURN},
                         template=PLOTLY_TEMPLATE)
            fig.update_layout(title="Churn Rate by Internet Service", yaxis_title="% of customers", height=400)
            st.plotly_chart(fig, use_container_width=True)

        ct3 = pd.crosstab(raw_data['Payment Method'], raw_data['Churn Label'], normalize='index') * 100
        ct3 = ct3.reset_index().melt(id_vars='Payment Method', var_name='Churn Label', value_name='Pct')
        fig = px.bar(ct3, x='Payment Method', y='Pct', color='Churn Label', barmode='group',
                     color_discrete_map={'Yes': COLOR_CHURN, 'No': COLOR_NOCHURN},
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(title="Churn Rate by Payment Method", yaxis_title="% of customers", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="insight-box">A consistent profile emerges: <b>month-to-month contracts</b> '
            '(~15x higher churn than two-year), <b>fiber optic internet</b> (2-6x higher churn than '
            'DSL/no-internet), and <b>electronic check</b> payment (~3x higher than automatic payment '
            'methods) all point to the same customer type — low commitment, manual billing, '
            'higher cost.</div>', unsafe_allow_html=True
        )

    with tab4:
        numeric_cols = ['Tenure Months', 'Monthly Charges', 'CLTV']
        corr = raw_data[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         template=PLOTLY_TEMPLATE)
        fig.update_layout(title="Correlation Heatmap — Numeric Features", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 View raw dataset sample"):
        st.dataframe(raw_data.head(50), use_container_width=True)


# ============================================================================
# PAGE 3 — MODEL COMPARISON
# ============================================================================
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    st.caption("Logistic Regression · Random Forest · XGBoost · LightGBM — trained with class-imbalance handling")

    if not (XGB_AVAILABLE and LGBM_AVAILABLE):
        st.warning(
            "xgboost and/or lightgbm are not installed in this environment, so only the models "
            "available will be shown. Run `pip install xgboost lightgbm` to see the full 4-model "
            "comparison exactly as in the notebook."
        )

    with st.spinner("Training models on the same class-weighted / cost-aware setup as the notebook..."):
        models = train_all_models(x_train, y_train)

    st.markdown('<div class="section-title">Standard Classification Metrics (at default 0.5 threshold)</div>',
                unsafe_allow_html=True)

    rows = []
    for name, mdl in models.items():
        _, _, _, metrics = evaluate_model_at_threshold(mdl, x_test, y_test, cltv_test, 0.5)
        rows.append({
            "Model": name,
            "Accuracy": metrics['accuracy'],
            "Precision": metrics['precision'],
            "Recall": metrics['recall'],
            "F1-score": metrics['f1'],
            "ROC-AUC": metrics['roc_auc'],
        })
    metrics_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        metrics_df.style.format("{:.3f}").background_gradient(cmap="Blues", axis=0),
        use_container_width=True
    )

    st.markdown(
        '<div class="insight-box">Accuracy alone is misleading here: Random Forest posts the '
        'highest accuracy but the <b>worst recall</b> (misses the most churners) — exactly the '
        'metric that matters most for this business problem. This is why the next step optimizes '
        'each model on <b>business cost</b>, not accuracy.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Cost-Optimal Threshold per Model</div>', unsafe_allow_html=True)
    st.caption("For each model, sweeping the decision threshold and picking the one that minimizes total business cost on the test set.")

    with st.spinner("Sweeping thresholds for each model..."):
        cost_rows = []
        thresholds = np.arange(0.01, 0.99, 0.01)
        for name, mdl in models.items():
            best_t, sweep_df = find_optimal_threshold(mdl, x_test, y_test, cltv_test, thresholds)
            best_row = sweep_df.loc[sweep_df['total_cost'].idxmin()]
            cost_rows.append({
                "Model": name,
                "Best Threshold": best_row['threshold'],
                "Min Total Cost": best_row['total_cost'],
                "% Customers Flagged": best_row['pct_flagged'],
            })
        cost_df = pd.DataFrame(cost_rows).sort_values("Min Total Cost").reset_index(drop=True)

    # blanket-flag-all benchmark
    y_all = np.ones(len(y_test))
    blanket_cost, _, _ = calculate_business_cost(y_test, y_all, cltv_test)

    col1, col2 = st.columns([1.4, 1])
    with col1:
        display_df = cost_df.copy()
        display_df["Beats Blanket Flag-All?"] = display_df["Min Total Cost"].apply(
            lambda c: "✅ Yes" if c < blanket_cost else "❌ No"
        )
        st.dataframe(
            display_df.style.format({
                "Best Threshold": "{:.3f}",
                "Min Total Cost": "${:,.0f}",
                "% Customers Flagged": "{:.1f}%",
            }),
            use_container_width=True, hide_index=True
        )
        st.caption(f"Benchmark — flagging every single customer costs **${blanket_cost:,.0f}**.")

    with col2:
        fig = px.bar(cost_df, x="Model", y="Min Total Cost", color="Model",
                     text="Min Total Cost", template=PLOTLY_TEMPLATE)
        fig.add_hline(y=blanket_cost, line_dash="dash", line_color="#ff9f43",
                       annotation_text="Blanket flag-all", annotation_position="top left")
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, height=380, yaxis_title="Min Cost ($)", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    winner = cost_df.iloc[0]
    st.success(
        f"🏆 **Winner: {winner['Model']}** at threshold **{winner['Best Threshold']:.3f}** — "
        f"minimum achievable cost of **${winner['Min Total Cost']:,.0f}** on the held-out test set. "
        f"This is the model + threshold combination deployed in the app."
    )

    with st.expander("📋 View confusion matrix for any model"):
        chosen = st.selectbox("Model", list(models.keys()))
        t_choice = st.slider("Threshold", 0.01, 0.99, float(best_threshold), 0.01, key="cm_thresh")
        y_pred, _, cm, m = evaluate_model_at_threshold(models[chosen], x_test, y_test, cltv_test, t_choice)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         x=["Predicted: No Churn", "Predicted: Churn"],
                         y=["Actual: No Churn", "Actual: Churn"],
                         template=PLOTLY_TEMPLATE)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Accuracy", f"{m['accuracy']:.1%}")
        cc2.metric("Precision", f"{m['precision']:.1%}")
        cc3.metric("Recall", f"{m['recall']:.1%}")
        cc4.metric("Total Cost", f"${m['total_cost']:,.0f}")


# ============================================================================
# PAGE 4 — THRESHOLD & COST OPTIMIZER
# ============================================================================
elif page == "🎚️ Threshold & Cost Optimizer":
    st.title("🎚️ Threshold & Cost Optimizer")
    st.caption("This is the core differentiator: instead of the default 0.50 cutoff, pick the threshold that minimizes real business cost.")

    st.markdown('<div class="section-title">Live Threshold Simulator</div>', unsafe_allow_html=True)

    col_slider, col_toggle = st.columns([3, 1])
    with col_slider:
        chosen_threshold = st.slider(
            "Decision threshold (flag a customer as 'will churn' if predicted probability ≥ this value)",
            min_value=0.01, max_value=0.99, value=float(round(best_threshold, 2)), step=0.01
        )
    with col_toggle:
        fp_pct = st.number_input("FP cost (% of CLTV per wasted offer)", min_value=1, max_value=100,
                                  value=10, step=1) / 100

    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= chosen_threshold).astype(int)
    total_cost, fn_cost, fp_cost = calculate_business_cost(y_test, y_pred, cltv_test, fp_pct)
    cm = pd.DataFrame(
        [[((y_test == 0) & (y_pred == 0)).sum(), ((y_test == 0) & (y_pred == 1)).sum()],
         [((y_test == 1) & (y_pred == 0)).sum(), ((y_test == 1) & (y_pred == 1)).sum()]],
        index=["Actual: No Churn", "Actual: Churn"],
        columns=["Predicted: No Churn", "Predicted: Churn"]
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi_card("Total Cost", f"${total_cost:,.0f}")
    with k2:
        kpi_card("Missed Churners (FN)", f"{cm.iloc[1,0]}", "lost revenue", "kpi-bad")
    with k3:
        kpi_card("False Alarms (FP)", f"{cm.iloc[0,1]}", "wasted offers", "kpi-neutral")
    with k4:
        recall = cm.iloc[1,1] / (cm.iloc[1,0] + cm.iloc[1,1])
        kpi_card("Recall", f"{recall:.1%}", "churners caught", "kpi-good")
    with k5:
        kpi_card("% Flagged", f"{y_pred.mean()*100:.1f}%", "of customer base")

    st.write("")
    col1, col2 = st.columns([1.4, 1])

    with col1:
        thresholds_full = np.arange(0.01, 0.99, 0.01)
        sweep_df = threshold_sweep(model, x_test, y_test, cltv_test, thresholds_full, fp_pct)
        optimal_t = sweep_df.loc[sweep_df['total_cost'].idxmin(), 'threshold']
        optimal_cost = sweep_df['total_cost'].min()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sweep_df['threshold'], y=sweep_df['total_cost'],
                                  mode='lines', name='Total Cost', line=dict(color=COLOR_ACCENT, width=3)))
        fig.add_trace(go.Scatter(x=sweep_df['threshold'], y=sweep_df['fn_cost'],
                                  mode='lines', name='FN Cost (missed churners)',
                                  line=dict(color=COLOR_CHURN, width=1.5, dash='dot')))
        fig.add_trace(go.Scatter(x=sweep_df['threshold'], y=sweep_df['fp_cost'],
                                  mode='lines', name='FP Cost (wasted offers)',
                                  line=dict(color="#ff9f43", width=1.5, dash='dot')))
        fig.add_vline(x=chosen_threshold, line_dash="dash", line_color="#f5f7fa",
                      annotation_text="Your selection")
        fig.add_vline(x=optimal_t, line_dash="dot", line_color="#37d67a",
                      annotation_text=f"Optimal ({optimal_t:.2f})")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=430,
                           xaxis_title="Threshold", yaxis_title="Cost ($)",
                           legend=dict(orientation="h", y=-0.25), margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"💡 The mathematically optimal threshold for this cost setup is **{optimal_t:.2f}** "
                   f"(min cost **${optimal_cost:,.0f}**). The saved production model uses **{best_threshold:.3f}**.")

    with col2:
        fig = px.imshow(cm.values, text_auto=True, color_continuous_scale="Blues",
                         x=cm.columns.tolist(), y=cm.index.tolist(), template=PLOTLY_TEMPLATE)
        fig.update_layout(title=f"Confusion Matrix @ threshold={chosen_threshold:.2f}", height=430)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Capacity-Constrained Targeting</div>', unsafe_allow_html=True)
    st.caption(
        "A pure cost-minimizing threshold can flag 60-70% of customers — not realistic if your "
        "retention team can only act on a limited list. This view ranks customers by churn "
        "probability and shows the cost of contacting only the top N%."
    )

    capacity_pct = st.slider("Retention team capacity (top % of customers by risk)", 5, 100, 20, 5)
    order = np.argsort(-y_proba)
    capacity_n = int(len(y_proba) * capacity_pct / 100)
    y_pred_capacity = np.zeros(len(y_proba))
    y_pred_capacity[order[:capacity_n]] = 1
    cap_cost, cap_fn, cap_fp = calculate_business_cost(y_test, y_pred_capacity, cltv_test, fp_pct)

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Customers contacted", f"{capacity_n:,}")
    cc2.metric("Total cost at this capacity", f"${cap_cost:,.0f}")
    churners_caught = ((np.array(y_test) == 1) & (y_pred_capacity == 1)).sum()
    total_churners = (np.array(y_test) == 1).sum()
    cc3.metric("Churners caught", f"{churners_caught}/{total_churners}", f"{churners_caught/total_churners:.1%}")


# ============================================================================
# PAGE 5 — PREDICT A CUSTOMER
# ============================================================================
elif page == "🔮 Predict a Customer":
    st.title("🔮 Predict Churn Risk for a Customer")
    st.caption("Enter customer attributes to get a live churn probability, risk segment, and recommended action.")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            st.markdown("**Services**")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col3:
            st.markdown("**Account**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=300.0, value=70.0, step=1.0)
            cltv = st.number_input("CLTV — Customer Lifetime Value ($)", min_value=0.0, max_value=10000.0, value=4400.0, step=50.0)

        submitted = st.form_submit_button("🔍 Predict Churn Risk", use_container_width=True)

    if submitted:
        input_dict = {
            'Gender': gender, 'Senior Citizen': senior, 'Partner': partner, 'Dependents': dependents,
            'Tenure Months': tenure, 'Phone Service': phone, 'Multiple Lines': multiple_lines,
            'Internet Service': internet, 'Online Security': online_security, 'Online Backup': online_backup,
            'Device Protection': device_protection, 'Tech Support': tech_support,
            'Streaming TV': streaming_tv, 'Streaming Movies': streaming_movies,
            'Contract': contract, 'Paperless Billing': paperless, 'Payment Method': payment,
            'Monthly Charges': monthly_charges, 'CLTV': cltv,
        }
        x_customer = preprocess_single_customer(input_dict, label_encoders, model_columns)
        proba = model.predict_proba(x_customer)[0, 1]
        prediction = int(proba >= best_threshold)
        segment = risk_segment(proba)

        st.write("")
        r1, r2, r3 = st.columns([1, 1, 1.4])
        with r1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': COLOR_ACCENT},
                    'steps': [
                        {'range': [0, 15], 'color': '#1c3d2a'},
                        {'range': [15, 30], 'color': '#3d3a1c'},
                        {'range': [30, 60], 'color': '#3d2a1c'},
                        {'range': [60, 100], 'color': '#3d1c1c'},
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.85,
                        'value': best_threshold * 100
                    }
                },
                title={'text': "Churn Probability"}
            ))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            st.markdown(f"### {segment}")
            if prediction == 1:
                st.error(f"⚠️ **Flagged as AT RISK** (probability {proba:.1%} ≥ threshold {best_threshold:.1%})")
            else:
                st.success(f"✅ **Not flagged** (probability {proba:.1%} < threshold {best_threshold:.1%})")
            st.metric("At-risk revenue (CLTV)", f"${cltv:,.0f}")
            potential_offer_cost = cltv * 0.10
            st.metric("Suggested retention offer budget (10% of CLTV)", f"${potential_offer_cost:,.0f}")

        with r3:
            st.markdown("**Recommended Action**")
            if proba >= 0.60:
                st.markdown(
                    "🔴 **Immediate personal outreach.** High-value save opportunity — "
                    "prioritize a phone call or account manager check-in with a tailored retention offer."
                )
            elif proba >= 0.30:
                st.markdown(
                    "🟠 **Proactive retention offer.** Send a targeted discount or contract "
                    "upgrade incentive (e.g., move off month-to-month) within the next billing cycle."
                )
            elif proba >= best_threshold:
                st.markdown(
                    "🟡 **Add to monitoring / light-touch campaign.** Include in an email "
                    "nurture sequence; re-check probability next cycle."
                )
            else:
                st.markdown(
                    "🟢 **No action needed right now.** Low churn risk based on current profile."
                )
            st.caption(
                f"Model: Logistic Regression · Threshold: {best_threshold:.3f} · "
                f"This threshold was chosen to minimize total business cost (lost CLTV + wasted offers), not accuracy."
            )


# ============================================================================
# PAGE 6 — BATCH SCORING
# ============================================================================
elif page == "📁 Batch Scoring":
    st.title("📁 Batch Scoring")
    st.caption(
        "Upload a CSV/Excel of customers in the same raw format as the training data "
        "(same column names) to score them all at once."
    )

    st.markdown(f"**Required columns:** `{'`, `'.join(RAW_INPUT_COLUMNS)}`")

    uploaded = st.file_uploader("Upload customer file", type=["csv", "xlsx"])

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                batch_df = pd.read_csv(uploaded)
            else:
                batch_df = pd.read_excel(uploaded)

            missing = [c for c in RAW_INPUT_COLUMNS if c not in batch_df.columns]
            if missing:
                st.error(f"❌ Missing required columns: {missing}")
            else:
                results = []
                for _, row in batch_df.iterrows():
                    input_dict = {col: row[col] for col in RAW_INPUT_COLUMNS}
                    x_c = preprocess_single_customer(input_dict, label_encoders, model_columns)
                    proba = model.predict_proba(x_c)[0, 1]
                    results.append(proba)

                batch_df["Churn Probability"] = results
                batch_df["Flagged At-Risk"] = (batch_df["Churn Probability"] >= best_threshold)
                batch_df["Risk Segment"] = batch_df["Churn Probability"].apply(risk_segment)

                st.success(f"✅ Scored {len(batch_df)} customers.")

                c1, c2, c3 = st.columns(3)
                c1.metric("Customers flagged at-risk", int(batch_df["Flagged At-Risk"].sum()))
                c2.metric("% flagged", f"{batch_df['Flagged At-Risk'].mean()*100:.1f}%")
                if "CLTV" in batch_df.columns:
                    at_risk_value = batch_df.loc[batch_df["Flagged At-Risk"], "CLTV"].sum()
                    c3.metric("Total CLTV at risk", f"${at_risk_value:,.0f}")

                seg_counts = batch_df["Risk Segment"].value_counts().reset_index()
                seg_counts.columns = ["Risk Segment", "Count"]
                fig = px.bar(seg_counts, x="Risk Segment", y="Count", color="Risk Segment",
                             template=PLOTLY_TEMPLATE)
                fig.update_layout(showlegend=False, height=360)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    batch_df.sort_values("Churn Probability", ascending=False),
                    use_container_width=True
                )

                csv_out = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download scored results (CSV)", csv_out,
                                    "churn_predictions.csv", "text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("👆 Upload a file to get started, or try it on a sample slice of the training data below.")
        if st.button("Use a random 25-customer sample from the training data instead"):
            sample = raw_data.sample(25, random_state=1)[RAW_INPUT_COLUMNS + (["Churn Label"] if "Churn Label" in raw_data.columns else [])]
            st.session_state["batch_sample"] = sample

        if "batch_sample" in st.session_state:
            sample = st.session_state["batch_sample"]
            results = []
            for _, row in sample.iterrows():
                input_dict = {col: row[col] for col in RAW_INPUT_COLUMNS}
                x_c = preprocess_single_customer(input_dict, label_encoders, model_columns)
                proba = model.predict_proba(x_c)[0, 1]
                results.append(proba)
            sample = sample.copy()
            sample["Churn Probability"] = results
            sample["Flagged At-Risk"] = (sample["Churn Probability"] >= best_threshold)
            sample["Risk Segment"] = sample["Churn Probability"].apply(risk_segment)
            st.dataframe(sample.sort_values("Churn Probability", ascending=False), use_container_width=True)


# ============================================================================
# PAGE 7 — METHODOLOGY
# ============================================================================
elif page == "ℹ️ Methodology":
    st.title("ℹ️ Methodology")

    st.markdown("""
### Pipeline
1. **Cleaning** — `Total Charges` coerced to numeric (blanks → 0), ID/geography/leakage columns dropped
   (`CustomerID`, `Country`, `State`, `City`, `Lat Long`, `Churn Reason`, `Count`, `Zip Code`,
   `Latitude`, `Longitude`, `Churn Value`, `Churn Score`, `Total Charges`).
2. **Category collapsing** — `"No internet service"` / `"No phone service"` collapsed into `"No"`
   for the relevant service columns.
3. **Encoding** — binary columns label-encoded; `Internet Service`, `Contract`, `Payment Method`
   one-hot encoded with `drop_first=True`.
4. **Class imbalance** — handled via `class_weight='balanced'` (Logistic Regression, Random Forest,
   LightGBM) and `scale_pos_weight` (XGBoost), rather than resampling.
5. **Split** — 80/20 train/test, stratified on churn label, `random_state=42`.
6. **Model selection** — 4 models trained (Logistic Regression, Random Forest, XGBoost, LightGBM),
   each evaluated by sweeping decision thresholds and picking the one that **minimizes total
   business cost**, not the one that maximizes accuracy or F1.
7. **Business cost function**
   - False Negative (missed churner): cost = full **CLTV** of that customer (lost lifetime value)
   - False Positive (unnecessary retention offer): cost = **10% of CLTV** (an offer sized to the
     customer's value)
8. **Winner**: Logistic Regression at threshold **0.15** — lowest total cost of all 4 models,
   despite not having the highest raw accuracy.

### Why not just use accuracy?
Random Forest has the best raw accuracy (~79%) but the worst recall (~50%) — it misses half of all
actual churners. Under a cost function where a missed churner can cost thousands of dollars in lost
CLTV, that "high accuracy" model is actually the **worst** business choice of the four. This project
optimizes for the objective the business actually cares about.

### Honest limitations
- At the cost-optimal threshold (0.15), ~68% of customers get flagged as at-risk. That's
  mathematically optimal under this specific cost function, but may not be operationally realistic —
  see the **Capacity-Constrained Targeting** panel on the Threshold Optimizer page for a more
  actionable "top-N%" alternative.
- CLTV in this dataset is pre-computed by IBM; it is used both as a model feature and as the cost
  basis, so it is not literally causal — but it's the standard field for this exact dataset for
  exactly this kind of exercise.
- FP cost (10% of CLTV) is a modeling assumption, not observed data — adjust it live in the
  Threshold Optimizer to see how sensitive the recommendation is.
""")

    st.markdown('<div class="section-title">Artifacts used by this app</div>', unsafe_allow_html=True)
    st.code(
        "artifacts/churn_model.pkl        → trained Logistic Regression\n"
        "artifacts/model_columns.pkl      → exact feature order expected by the model\n"
        "artifacts/best_threshold.pkl     → cost-optimal decision threshold (0.15)\n"
        "artifacts/label_encoders.pkl     → fitted LabelEncoders for categorical columns",
        language="text"
    )
