import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main container */
    .block-container {padding-top: 2rem;}

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        color: white;
    }
    div[data-testid="stMetric"] label {color: rgba(255,255,255,.75) !important;}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {color: white !important;}

    /* Sidebar */
    section[data-testid="stSidebar"] {background: #0e1117;}
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown p {color: #fafafa;}

    /* Prediction result boxes */
    .result-safe {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 1.5rem; border-radius: 12px; text-align: center; color: white;
    }
    .result-fraud {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        padding: 1.5rem; border-radius: 12px; text-align: center; color: white;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load assets (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/best_model_Random_Forest_smote.joblib")


@st.cache_data
def load_metadata():
    with open("models/model_metadata.json") as f:
        return json.load(f)


@st.cache_data
def load_results():
    return pd.read_csv("results/model_comparison_results.csv")


@st.cache_data
def load_dataset():
    return pd.read_csv("data/processed/creditcard.csv")


model = load_model()
metadata = load_metadata()
results_df = load_results()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Dashboard", "Predict", "Data Explorer"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** {metadata['model_name']}  \n"
    f"**Strategy:** {metadata['sampling_strategy']}  \n"
    f"**Threshold:** {metadata['threshold']}  \n"
    f"**Trained:** {metadata['training_date'][:10]}"
)


# ===========================================================================
# PAGE 0 — HOME
# ===========================================================================
if page == "Home":
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1rem 0;">
        <h1 style="font-size:3rem; margin-bottom:0;">
            <span style="background: linear-gradient(135deg, #667eea, #764ba2);
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Credit Card Fraud Detection
            </span>
        </h1>
        <p style="font-size:1.25rem; color:#888; margin-top:0.5rem;">
            Machine Learning Pipeline &mdash; from EDA to Real-Time Prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Key figures ---
    m = metadata["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions Analyzed", "283,726")
    c2.metric("Fraud Rate", "0.17%")
    c3.metric("Best Model F1", f"{m['F1']:.2%}")
    c4.metric("Best Model AUPRC", f"{m['AUPRC']:.4f}")

    st.markdown("")

    # --- Project overview ---
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("### About the Project")
        st.markdown("""
        This project builds an **end-to-end fraud detection system** using the
        [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
        (284,807 transactions, September 2013, European cardholders).

        The pipeline covers:
        - **Exploratory Data Analysis** — class imbalance, distributions, correlations
        - **Model Training** — Logistic Regression, Random Forest, Naive Bayes
        - **Resampling Strategies** — Baseline, SMOTE, Undersampling
        - **Anomaly Detection** — Isolation Forest (unsupervised)
        - **REST API** — Flask with structured logging & IP tracking
        - **Interactive Dashboard** — this Streamlit app
        """)

    with col_right:
        st.markdown("### Tech Stack")
        tech = {
            "Machine Learning": "scikit-learn, imbalanced-learn",
            "Data Processing": "pandas, numpy",
            "Visualization": "Plotly, Streamlit",
            "API": "Flask, Gunicorn",
            "Deployment": "Docker (multi-stage, non-root)",
            "Logging": "Daily rotation, IP tracking",
        }
        for category, tools in tech.items():
            st.markdown(f"**{category}**  \n`{tools}`")

    st.markdown("---")

    # --- Page navigation cards ---
    st.markdown("### Explore the App")
    card1, card2, card3 = st.columns(3, gap="medium")

    with card1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2);
                    padding: 1.5rem; border-radius: 12px; color: white; height: 180px;">
            <h3 style="margin-top:0; color: white;">Dashboard</h3>
            <p style="color: rgba(255,255,255,.85);">
                Compare model performance across all strategies.
                Visualize AUPRC, F1, Precision-Recall trade-offs.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with card2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e, #38ef7d);
                    padding: 1.5rem; border-radius: 12px; color: white; height: 180px;">
            <h3 style="margin-top:0; color: white;">Predict</h3>
            <p style="color: rgba(255,255,255,.85);">
                Submit a transaction and get a real-time fraud probability
                with an interactive gauge visualization.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with card3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #eb3349, #f45c43);
                    padding: 1.5rem; border-radius: 12px; color: white; height: 180px;">
            <h3 style="margin-top:0; color: white;">Data Explorer</h3>
            <p style="color: rgba(255,255,255,.85);">
                Dive into the dataset — class distribution, amount analysis,
                feature correlations with fraud.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.caption(
        f"Model: {metadata['model_name']} + {metadata['sampling_strategy'].upper()} "
        f"| Trained: {metadata['training_date'][:10]} "
        f"| Threshold: {metadata['threshold']}"
    )


# ===========================================================================
# PAGE 1 — DASHBOARD
# ===========================================================================
elif page == "Dashboard":
    st.title("Model Performance Dashboard")
    st.caption("Comparison of all trained models and strategies on the test set.")

    # --- Best model metrics row ---
    st.subheader("Best Model Metrics")
    m = metadata["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUPRC", f"{m['AUPRC']:.4f}")
    c2.metric("ROC AUC", f"{m['ROC_AUC']:.4f}")
    c3.metric("Precision", f"{m['Precision']:.4f}")
    c4.metric("Recall", f"{m['Recall']:.4f}")
    c5.metric("F1 Score", f"{m['F1']:.4f}")

    st.markdown("---")

    # --- Model comparison charts ---
    st.subheader("Model Comparison")

    # Prepare display label
    df_plot = results_df.copy()
    df_plot["Label"] = df_plot["Model"] + " / " + df_plot["Strategy"]

    col_left, col_right = st.columns(2)

    with col_left:
        fig_auprc = px.bar(
            df_plot.sort_values("AUPRC", ascending=True),
            x="AUPRC", y="Label", orientation="h",
            color="AUPRC",
            color_continuous_scale="Viridis",
            title="AUPRC by Model & Strategy",
        )
        fig_auprc.update_layout(
            yaxis_title="", xaxis_title="AUPRC",
            coloraxis_showscale=False, height=400,
        )
        st.plotly_chart(fig_auprc, use_container_width=True)

    with col_right:
        fig_f1 = px.bar(
            df_plot.sort_values("F1", ascending=True),
            x="F1", y="Label", orientation="h",
            color="F1",
            color_continuous_scale="Magma",
            title="F1 Score by Model & Strategy",
        )
        fig_f1.update_layout(
            yaxis_title="", xaxis_title="F1 Score",
            coloraxis_showscale=False, height=400,
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    # --- Precision / Recall scatter ---
    st.subheader("Precision vs Recall Trade-off")
    fig_pr = px.scatter(
        df_plot, x="Recall", y="Precision",
        size="F1", color="Model", symbol="Strategy",
        hover_data=["AUPRC", "F1"],
        title="Each point is a model + strategy combination",
    )
    fig_pr.update_layout(height=450)
    st.plotly_chart(fig_pr, use_container_width=True)

    # --- Full results table ---
    st.subheader("Full Results Table")
    display_cols = ["Model", "Strategy", "AUPRC", "ROC_AUC", "Precision", "Recall", "F1", "Threshold"]
    st.dataframe(
        df_plot[display_cols].style.format({
            "AUPRC": "{:.4f}", "ROC_AUC": "{:.4f}",
            "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}",
        }),
        use_container_width=True, hide_index=True,
    )


# ===========================================================================
# PAGE 2 — PREDICT
# ===========================================================================
elif page == "Predict":
    st.title("Fraud Prediction")
    st.caption("Enter transaction features to get a real-time fraud prediction.")

    features = metadata["features"]

    # --- Input mode selector ---
    input_mode = st.radio(
        "Input method", ["Manual form", "JSON paste"], horizontal=True
    )

    input_data = {}

    if input_mode == "Manual form":
        st.markdown("#### Transaction Features")
        cols = st.columns(4)
        for i, feat in enumerate(features):
            with cols[i % 4]:
                default_val = 0.0 if feat != "Amount" else 100.0
                input_data[feat] = st.number_input(
                    feat, value=default_val, format="%.6f", key=feat
                )
    else:
        st.markdown("#### Paste JSON")
        sample = json.dumps({f: 0.0 for f in features}, indent=2)
        json_input = st.text_area("Transaction JSON", value=sample, height=350)
        try:
            input_data = json.loads(json_input)
        except json.JSONDecodeError:
            st.error("Invalid JSON format.")
            input_data = {}

    # --- Predict button ---
    st.markdown("---")
    if st.button("Run Prediction", type="primary", use_container_width=True):
        if not input_data:
            st.error("Please provide valid input data.")
        else:
            try:
                df = pd.DataFrame([input_data])
                df = df[features]

                proba = float(model.predict_proba(df)[0, 1])
                prediction = int(proba >= metadata["threshold"])

                st.markdown("### Result")

                col_prob, col_pred = st.columns(2)

                with col_prob:
                    # Gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba * 100,
                        number={"suffix": "%"},
                        title={"text": "Fraud Probability"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#eb3349" if prediction else "#11998e"},
                            "steps": [
                                {"range": [0, metadata["threshold"] * 100], "color": "#e8f5e9"},
                                {"range": [metadata["threshold"] * 100, 100], "color": "#ffebee"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 3},
                                "thickness": 0.8,
                                "value": metadata["threshold"] * 100,
                            },
                        },
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col_pred:
                    if prediction == 1:
                        st.markdown(
                            '<div class="result-fraud">'
                            '<h2>FRAUD DETECTED</h2>'
                            f'<p style="font-size:1.2rem">Probability: {proba:.4%}</p>'
                            f'<p>Threshold: {metadata["threshold"]}</p>'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="result-safe">'
                            '<h2>LEGITIMATE</h2>'
                            f'<p style="font-size:1.2rem">Probability: {proba:.4%}</p>'
                            f'<p>Threshold: {metadata["threshold"]}</p>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

            except KeyError as e:
                st.error(f"Missing feature: {e}")
            except Exception as e:
                st.error(f"Prediction error: {e}")


# ===========================================================================
# PAGE 3 — DATA EXPLORER
# ===========================================================================
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.caption("Explore the processed credit card transactions dataset.")

    df = load_dataset()

    # --- Dataset overview ---
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1] - 1}")
    fraud_count = int(df["Class"].sum())
    c3.metric("Fraud Cases", f"{fraud_count:,}")
    c4.metric("Fraud Rate", f"{df['Class'].mean():.3%}")

    st.markdown("---")

    # --- Class distribution ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Class Distribution")
        class_counts = df["Class"].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]
        class_counts["Label"] = class_counts["Class"].map({0: "Legitimate", 1: "Fraud"})
        fig_class = px.pie(
            class_counts, values="Count", names="Label",
            color="Label",
            color_discrete_map={"Legitimate": "#11998e", "Fraud": "#eb3349"},
            hole=0.5,
        )
        fig_class.update_layout(height=350)
        st.plotly_chart(fig_class, use_container_width=True)

    with col_right:
        st.subheader("Transaction Amount Distribution")
        fig_amount = px.histogram(
            df, x="Amount", nbins=100,
            color_discrete_sequence=["#667eea"],
            title="All Transactions",
        )
        fig_amount.update_layout(height=350, xaxis_title="Amount", yaxis_title="Count")
        st.plotly_chart(fig_amount, use_container_width=True)

    # --- Fraud vs legit amount comparison ---
    st.subheader("Amount: Fraud vs Legitimate")
    df_label = df.copy()
    df_label["Type"] = df_label["Class"].map({0: "Legitimate", 1: "Fraud"})
    fig_box = px.box(
        df_label, x="Type", y="Amount",
        color="Type",
        color_discrete_map={"Legitimate": "#11998e", "Fraud": "#eb3349"},
        title="Transaction Amount by Class",
    )
    fig_box.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # --- Feature correlation with fraud ---
    st.subheader("Feature Correlation with Fraud")
    corr_with_class = df.corr(method="spearman")["Class"].drop("Class").sort_values()
    fig_corr = px.bar(
        x=corr_with_class.values, y=corr_with_class.index,
        orientation="h",
        color=corr_with_class.values,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        labels={"x": "Spearman Correlation", "y": "Feature"},
    )
    fig_corr.update_layout(height=600, coloraxis_showscale=False)
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- Raw data preview ---
    st.subheader("Raw Data Preview")
    n_rows = st.slider("Number of rows", 5, 100, 10)
    st.dataframe(df.head(n_rows), use_container_width=True, hide_index=True)
