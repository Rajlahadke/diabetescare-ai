"""DiabetesCare AI — a polished, educational Streamlit screening demo.

Machine-learning classifications are not diagnoses or validated clinical risk
estimates. Do not enter identifiable patient information into a public demo.
"""

from dataclasses import asdict, replace
from datetime import datetime, timezone
from html import escape
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_utils import (
    GENDERS,
    SMOKING_STATUSES,
    PatientProfile,
    grouped_importances,
    load_artifacts,
    predict_profile,
)

st.set_page_config(
    page_title="DiabetesCare AI | Health Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "Educational machine-learning demo. Not a diagnostic device."},
)

DEEP = "#103E46"
TEAL = "#0D8C83"
MINT = "#E8F7F3"
MUTED = "#688289"
ORANGE = "#BD7044"
FONT = "'DM Sans', sans-serif"

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --deep:#103e46; --teal:#0d8c83; --muted:#688289; --bg:#f5faf9; }
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
.stApp {
 background:radial-gradient(circle at 99% 2%,rgba(176,231,224,.58),transparent 23%),
 radial-gradient(circle at 2% 34%,rgba(223,244,242,.48),transparent 26%),#f5faf9;
 color:#163c43;
}
.block-container {max-width:1240px;padding-top:1.45rem;padding-bottom:2.5rem;}
[data-testid="stHeader"] {background:transparent;}
h1,h2,h3 {font-family:'Manrope',sans-serif;letter-spacing:-.04em;color:#143d44;}
p {line-height:1.64;}
.topnav {display:flex;align-items:center;justify-content:space-between;gap:10px;padding:6px 0 22px;}
.nav-brand {font-family:'Manrope',sans-serif;color:#103e46;font-size:19px;font-weight:800;letter-spacing:-.7px;}
.brandmark {display:inline-grid;place-items:center;width:36px;height:36px;border-radius:12px;background:#0d8c83;color:#fff;margin-right:9px;}
.nav-aside {font-size:11px;font-weight:800;letter-spacing:1.1px;color:#0b7c76;background:#e5f5f1;
 border:1px solid #c9e9e2;border-radius:40px;padding:9px 14px;white-space:nowrap;}
.hero {display:flex;justify-content:space-between;align-items:center;gap:30px;
 padding:48px 52px;border-radius:26px;position:relative;overflow:hidden;
 background:linear-gradient(116deg,#103e46 0%,#155960 60%,#127e7a 100%);
 box-shadow:0 20px 60px rgba(18,65,70,.11);margin-bottom:20px;}
.hero:after {content:'';position:absolute;width:400px;height:400px;right:-150px;top:-185px;
 background:radial-gradient(circle,rgba(166,247,230,.19),transparent 72%);border-radius:50%;}
.hero-main {position:relative;z-index:2;max-width:680px;}
.eyebrow {color:#83dac9;font-size:10px;font-weight:800;letter-spacing:2.25px;margin-bottom:13px;}
.hero h1 {font-size:clamp(34px,4vw,53px);line-height:1.16;font-weight:800;margin:0 0 16px;letter-spacing:-2.5px;color:#fff;}
.hero h1 span {color:#9ce5d4;}
.hero p {font-size:14px;color:#d1e5e4;max-width:560px;margin:0 0 21px;line-height:1.85;}
.hero-pill {display:inline-block;border:1px solid rgba(201,248,237,.28);background:rgba(255,255,255,.08);
 color:#e0f9f3;padding:9px 13px;border-radius:30px;font-size:11px;font-weight:700;margin-right:8px;margin-top:7px;}
.hero-art {width:215px;min-width:190px;height:175px;display:flex;align-items:center;justify-content:center;
 background:rgba(255,255,255,.055);border-radius:28px;border:1px solid rgba(240,255,255,.14);
 transform:rotate(-5deg);position:relative;z-index:2;box-shadow:0 18px 36px rgba(0,0,0,.1);}
.section-overline {color:#08877e;font-size:11px;letter-spacing:1.9px;font-weight:800;margin:22px 0 4px;}
.section-heading {font-size:28px;font-family:'Manrope',sans-serif;font-weight:800;letter-spacing:-1.2px;color:#103e46;margin:0 0 6px;}
.section-subtitle {font-size:13px;color:#6a858b;margin-bottom:17px;}
[data-testid="stVerticalBlockBorderWrapper"] {border:1px solid #e1eceb!important;background:rgba(255,255,255,.96);
 border-radius:19px!important;box-shadow:0 10px 32px rgba(28,77,80,.035);}
div[data-testid="stMetric"] {border:1px solid #deedeb;padding:15px 18px;background:white;border-radius:16px;}
[data-testid="stMetricLabel"] {color:#637f83!important;font-weight:650;}
[data-testid="stMetricValue"] {color:#134950!important;font-family:'Manrope',sans-serif;font-size:25px!important;}
div.stButton > button[kind="primary"],div.stFormSubmitButton > button {background:#0d8c83;color:white;
 border-radius:12px;border:1px solid #0d8c83;font-weight:800;min-height:46px;}
div.stButton > button[kind="primary"]:hover,div.stFormSubmitButton > button:hover {
 background:#096e68;color:#fff;border-color:#096e68;}
div.stButton > button[kind="secondary"] {background:white;border:1px solid #d5e8e5;border-radius:12px;
 color:#335d61;font-weight:700;min-height:42px;}
div.stButton > button[kind="secondary"]:hover {color:#0d8c83;border-color:#0d8c83;}
input,[data-baseweb="select"] > div {border-radius:10px!important;}
[data-testid="stAlert"] {border-radius:13px;}
.kpi-label {font-size:11px;color:#758d90;letter-spacing:1px;font-weight:800;}
.kpi-num {font-size:24px;color:#153f48;letter-spacing:-1px;font-weight:800;font-family:'Manrope',sans-serif;}
.mini {font-size:12px;color:#71898d;}
.result-card {border-radius:20px;border:1px solid #cae8e1;padding:26px 30px;
 background:linear-gradient(115deg,#e6f6f2,#fbffff);}
.result-card.alert {background:linear-gradient(110deg,#fff2e9,#fffaf5);border-color:#f3d3be;}
.result-kicker {font-size:10px;font-weight:900;letter-spacing:1.6px;color:#5f7b7b;margin-bottom:9px;}
.result-title {font-size:26px;font-family:'Manrope',sans-serif;font-weight:800;letter-spacing:-1px;color:#11464b;}
.result-card.alert .result-title {color:#8d4c26;}
.result-text {font-size:13px;color:#5d777d;margin-top:7px;line-height:1.75;}
.hint {color:#658088;font-size:12px;margin:0;line-height:1.75;}
.process-card {border:1px solid #dfebe9;border-radius:17px;padding:21px;background:white;min-height:155px;}
.process-step {color:#0b8d82;font-size:11px;font-weight:800;letter-spacing:1px;}
.process-title {font-size:16px;font-weight:800;color:#143e44;margin:9px 0 7px;}
.process-text {font-size:12px;line-height:1.7;color:#68848a;}
.disclaimer {padding:18px 22px;border-radius:14px;border:1px solid #f0ddbd;background:#fff9ef;
 color:#815d30;font-size:12px;line-height:1.75;}
.footer {text-align:center;color:#779093;font-size:11px;letter-spacing:.2px;
 padding:25px 0 5px;border-top:1px solid #deebe8;margin-top:30px;}
@media(max-width:770px) {.hero {padding:29px 24px;border-radius:20px;}
 .hero-art {display:none;} .nav-aside {font-size:9px;letter-spacing:.3px;padding:8px 10px;}
 .hero h1 {letter-spacing:-1.4px;} .section-heading {font-size:24px;}}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing the prediction model…")
def get_model():
    return load_artifacts()


try:
    model, scaler, feature_columns = get_model()
except Exception:
    st.error("Model files could not be opened. Verify the three .pkl files are in the repository root and that dependencies match requirements.txt.")
    st.stop()


DEMO_BASELINE = PatientProfile()
DEMO_ALTERNATE = PatientProfile(
    gender="Male", age=58.0, hypertension=True, heart_disease=False,
    smoking_history="former", bmi=32.1, hba1c=8.1, glucose=215.0,
)

DEFAULTS = {
    "gender": DEMO_BASELINE.gender,
    "age": DEMO_BASELINE.age,
    "hypertension": "Yes" if DEMO_BASELINE.hypertension else "No",
    "heart_disease": "Yes" if DEMO_BASELINE.heart_disease else "No",
    "smoking_history": DEMO_BASELINE.smoking_history,
    "bmi": DEMO_BASELINE.bmi,
    "hba1c": DEMO_BASELINE.hba1c,
    "glucose": DEMO_BASELINE.glucose,
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def select_demo(profile: PatientProfile):
    st.session_state.update({
        "gender": profile.gender, "age": profile.age,
        "hypertension": "Yes" if profile.hypertension else "No",
        "heart_disease": "Yes" if profile.heart_disease else "No",
        "smoking_history": profile.smoking_history,
        "bmi": profile.bmi, "hba1c": profile.hba1c,
        "glucose": profile.glucose,
    })
    st.session_state.pop("assessment", None)


def title_block(number: str, heading: str, subtitle: str):
    st.markdown(
        f'<div class="section-overline">{escape(number)}</div>'
        f'<div class="section-heading">{escape(heading)}</div>'
        f'<div class="section-subtitle">{escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def plotly_style(fig, height=280):
    fig.update_layout(
        height=height, margin=dict(l=15, r=15, t=12, b=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=DEEP, size=12),
        showlegend=False, hoverlabel=dict(bgcolor="#ffffff", font_size=12),
    )
    return fig


def score_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(score * 100, 1),
        number=dict(suffix="%", font=dict(size=32, color=DEEP)),
        title=dict(text="Positive-class model score", font=dict(size=13, color=MUTED)),
        gauge=dict(
            axis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100], tickfont=dict(size=10)),
            bar=dict(color=TEAL, thickness=0.25),
            bgcolor="#e7f1ef", borderwidth=0,
            steps=[dict(range=[0, 100], color="#e7f1ef")],
        ),
    ))
    plotly_style(fig, height=250)
    fig.update_layout(margin=dict(l=20, r=20, t=45, b=4))
    return fig


def marker_chart(profile: PatientProfile):
    fig = go.Figure()
    for row, name, value, upper, units, color in [
        (1, "HbA1c", profile.hba1c, 20, "%", "#0D8C83"),
        (0, "Blood glucose", profile.glucose, 400, "mg/dL*", "#4C83A1"),
    ]:
        fig.add_trace(go.Bar(
            x=[value], y=[name], orientation="h", name=name,
            marker=dict(color=color, line=dict(width=0)), width=0.33,
            customdata=[[value, units]],
            hovertemplate="%{y}: %{customdata[0]:.1f} %{customdata[1]}<extra></extra>",
        ))
        fig.add_annotation(x=1.02, xref="paper", y=name, yref="y",
            text=f"<b>{value:.1f}</b> {units}", showarrow=False,
            font=dict(size=11, color=DEEP), xanchor="left")
    # Separate scales are essential: HbA1c and glucose units cannot be compared.
    # Provide each as percent of its own input range, not one shared numeric axis.
    fig.data[0].x = [profile.hba1c / 20 * 100]
    fig.data[1].x = [profile.glucose / 400 * 100]
    fig.update_xaxes(range=[0, 100], visible=False)
    fig.update_yaxes(showgrid=False, showline=False, tickfont=dict(size=12))
    plotly_style(fig, height=175)
    fig.update_layout(margin=dict(l=5, r=105, t=15, b=15), bargap=.67)
    return fig


def importance_chart():
    data = grouped_importances(model, feature_columns)
    if data.empty:
        return None
    fig = go.Figure(go.Bar(
        x=data["Importance"] * 100,
        y=data["Indicator"], orientation="h",
        marker=dict(color=["#a9d8d3" if val < 0.15 else TEAL for val in data["Importance"]],
                    line=dict(width=0)),
        text=[f"{val:.1f}%" for val in data["Importance"] * 100],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}% of total importance<extra></extra>",
    ))
    fig.update_xaxes(title="Overall importance (%)", range=[0, max(70, data["Importance"].max() * 116)],
        gridcolor="#e5f0ee", zeroline=False, ticksuffix="%")
    fig.update_yaxes(title="", automargin=True)
    plotly_style(fig, height=320)
    fig.update_layout(margin=dict(l=0, r=35, t=12, b=30))
    return fig


def sensitivity_chart(profile: PatientProfile, x_axis: str):
    if x_axis == "Blood glucose":
        x = np.linspace(70, 300, 35)
        profiles = [replace(profile, glucose=float(v)) for v in x]
        label = "Blood glucose (mg/dL*)"
        current = profile.glucose
    else:
        x = np.linspace(4.0, 11.0, 35)
        profiles = [replace(profile, hba1c=float(v)) for v in x]
        label = "HbA1c (%)"
        current = profile.hba1c
    scores = [predict_profile(p, model, scaler, feature_columns)[1] for p in profiles]
    if any(s is None for s in scores):
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=np.array(scores) * 100, mode="lines", name="Model score",
        line=dict(color=TEAL, width=3, shape="hv"),
        fill="tozeroy", fillcolor="rgba(13,140,131,.075)",
        hovertemplate="%{x:.1f}: %{y:.1f}% model score<extra></extra>",
    ))
    fig.add_vline(x=current, line=dict(color=ORANGE, width=2, dash="dash"))
    fig.update_xaxes(title=label, gridcolor="#e9f1f0", zeroline=False)
    fig.update_yaxes(title="Positive-class score (%)", range=[0, 102],
                     ticksuffix="%", gridcolor="#e9f1f0", zeroline=False)
    plotly_style(fig, height=315)
    fig.update_layout(margin=dict(l=8, r=14, t=12, b=30))
    return fig


def make_csv(profile: PatientProfile, prediction: int, score):
    details = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "gender": profile.gender, "age": profile.age,
        "hypertension": int(profile.hypertension),
        "heart_disease": int(profile.heart_disease),
        "smoking_history": profile.smoking_history,
        "bmi": profile.bmi, "hba1c_level": profile.hba1c,
        "blood_glucose_level": profile.glucose,
        "predicted_class": prediction,
        "positive_class_model_score": "" if score is None else round(score, 5),
        "notice": "Educational ML output; not medical diagnosis or calibrated clinical risk",
    }
    return pd.DataFrame([details]).to_csv(index=False).encode("utf-8")


# Top bar / hero
st.markdown("""
<div class="topnav"><div class="nav-brand"><span class="brandmark">✚</span>DiabetesCare AI</div>
<div class="nav-aside">● &nbsp; EDUCATIONAL ML DEMO</div></div>
<div class="hero">
<div class="hero-main">
<div class="eyebrow">INTELLIGENT HEALTHCARE ANALYTICS</div>
<h1>Clearer insights.<br><span>More informed conversations.</span></h1>
<p>Explore a diabetes classification model through eight patient indicators, interactive visualizations, and explainable model behavior — presented in one thoughtful workspace.</p>
<span class="hero-pill">✦ 8 patient inputs</span><span class="hero-pill">◉ Interactive analysis</span><span class="hero-pill">✓ No database storage</span>
</div>
<div class="hero-art">
<svg width="185" height="135" viewBox="0 0 185 135" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="11" y="13" width="163" height="110" rx="17" fill="#FFFFFF" fill-opacity="0.09" stroke="#B5E4DF" stroke-opacity="0.28"/>
<path d="M20 79H46L58 53L78 99L97 30L113 78L124 65L137 79H165" stroke="#B9FFF0" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="146" cy="31" r="16" fill="#A7EEE1" fill-opacity="0.2"/>
<path d="M146 24V38M139 31H153" stroke="#DFFFF7" stroke-width="3.5" stroke-linecap="round"/>
</svg></div></div>
""", unsafe_allow_html=True)

st.markdown('<div class="disclaimer"><b>Important:</b> This is an educational model demonstration, not a clinical screening or diagnostic tool. The model score is not a validated probability of developing diabetes. Consult a qualified clinician for interpretation of health measurements. Avoid entering identifiable patient information.</div>', unsafe_allow_html=True)

st.write("")
a, b, c = st.columns(3, gap="medium")
with a:
    st.metric("Prediction model", "Decision Tree", help="Verified from the uploaded model file")
with b:
    st.metric("Patient indicators", "8", help="Patient characteristics and measurements entered below")
with c:
    st.metric("Encoded features", str(len(feature_columns)), help="Features after applying the training-time one-hot encoding")

# Demo presets are placed BEFORE the form widgets so their state can update safely.
title_block("01  /  ASSESSMENT", "Start with a patient profile", "Enter measurements or try a synthetic sample to explore how the application works.")

left_demo, right_demo, demo_hint = st.columns([1.1, 1.1, 2.2], gap="small", vertical_alignment="center")
with left_demo:
    st.button("↺  Example · baseline", on_click=select_demo, args=(DEMO_BASELINE,), use_container_width=True)
with right_demo:
    st.button("↗  Example · elevated", on_click=select_demo, args=(DEMO_ALTERNATE,), use_container_width=True)
with demo_hint:
    st.caption("Examples are fictional and are not real patient records.")

with st.container(border=True):
    with st.form("patient_assessment", border=False):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### 👤  Demographics & history")
            gender = st.selectbox("Gender", GENDERS, key="gender")
            age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, step=1.0, key="age")
            smoking_history = st.selectbox("Smoking history", SMOKING_STATUSES, key="smoking_history")
            hc1, hc2 = st.columns(2)
            with hc1:
                hypertension = st.selectbox("Hypertension", ["No", "Yes"], key="hypertension")
            with hc2:
                heart_disease = st.selectbox("Heart disease", ["No", "Yes"], key="heart_disease")
        with right:
            st.markdown("#### 🧬  Health measurements")
            bmi = st.number_input("Body mass index (BMI)", min_value=5.0, max_value=80.0,
                                  step=0.1, format="%.1f", key="bmi")
            hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=20.0,
                                    step=0.1, format="%.1f", key="hba1c",
                                    help="Enter a measured HbA1c level; the input is not a diagnosis.")
            glucose = st.number_input("Blood glucose (mg/dL*)", min_value=40.0, max_value=500.0,
                                      step=1.0, format="%.0f", key="glucose",
                                      help="*mg/dL is the assumed unit for the model's blood glucose data; confirm with your original dataset.")
            st.caption("Inputs are processed in memory for prediction. No patient history is saved by this application.")
        st.write("")
        submitted = st.form_submit_button("✦  Generate assessment", type="primary", use_container_width=True)

if submitted:
    profile = PatientProfile(
        gender=gender, age=age, hypertension=(hypertension == "Yes"),
        heart_disease=(heart_disease == "Yes"), smoking_history=smoking_history,
        bmi=bmi, hba1c=hba1c, glucose=glucose,
    )
    try:
        predicted_class, score = predict_profile(profile, model, scaler, feature_columns)
        st.session_state["assessment"] = (profile, predicted_class, score)
    except Exception:
        st.error("The prediction failed. Check that your feature list, scaler and model were exported together from the same training run.")
        st.stop()

if "assessment" in st.session_state:
    profile, predicted_class, score = st.session_state["assessment"]
    title_block("02  /  MODEL RESULT", "Your assessment overview", "The result reflects the last profile you submitted, not a medical diagnosis.")

    result_col, gauge_col = st.columns([1.18, 1], gap="large", vertical_alignment="center")
    with result_col:
        positive = (predicted_class == 1)
        result_name = "Positive model classification" if positive else "Negative model classification"
        result_text = (
            "For these inputs, the trained decision tree assigned class 1. This is not confirmation of diabetes; a clinician must assess clinical history and appropriate tests."
            if positive else
            "For these inputs, the trained decision tree assigned class 0. A negative prediction cannot rule out diabetes or replace appropriate clinical testing."
        )
        st.markdown(
            f'<div class="result-card {"alert" if positive else ""}">'
            f'<div class="result-kicker">MODEL CLASSIFICATION • CLASS {predicted_class}</div>'
            f'<div class="result-title">{"◈" if positive else "✓"} &nbsp; {escape(result_name)}</div>'
            f'<div class="result-text">{escape(result_text)}</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Class predicted", str(predicted_class), help="Class 1 = positive; class 0 = negative")
        with m2:
            if score is not None:
                st.metric("Positive-class score", f"{score * 100:.1f}%", help="Model's predict_proba output. Not medically validated or calibrated.")
            else:
                st.metric("Model score", "N/A")
        st.caption("The model's probability output is a machine-learning score, not your medical likelihood of having or developing diabetes.")
    with gauge_col:
        with st.container(border=True):
            if score is not None:
                st.plotly_chart(score_gauge(score), use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("This classifier does not provide a positive-class score.")

    title_block("03  /  DATA EXPLORER", "Your health indicators", "Explore the inputs used by the model; different measurements use different units and scales.")
    indicator_col, snapshot_col = st.columns([1.2, 1], gap="medium")
    with indicator_col:
        with st.container(border=True):
            st.markdown("**Key laboratory measurements**")
            st.plotly_chart(marker_chart(profile), use_container_width=True, config={"displayModeBar": False})
            st.caption("Bars show values relative to each measurement's input range, not diagnostic thresholds. *Blood glucose unit is assumed from the dataset conventions.")
    with snapshot_col:
        with st.container(border=True):
            st.markdown("**Submitted profile**")
            snapshot = pd.DataFrame({
                "Indicator": ["Age", "Gender", "BMI", "Hypertension", "Heart disease", "Smoking history", "HbA1c", "Glucose"],
                "Value": [f"{profile.age:.0f} years", profile.gender, f"{profile.bmi:.1f}",
                          "Yes" if profile.hypertension else "No", "Yes" if profile.heart_disease else "No",
                          profile.smoking_history, f"{profile.hba1c:.1f}%", f"{profile.glucose:.0f} mg/dL*"],
            })
            st.dataframe(snapshot, use_container_width=True, hide_index=True, height=316)

    title_block("04  /  WHAT-IF EXPLORER", "See how the model responds", "Change one laboratory indicator at a time while holding the remaining profile fixed.")
    with st.container(border=True):
        c_sel, c_note = st.columns([1, 2], vertical_alignment="bottom")
        with c_sel:
            axis = st.radio("Explore indicator", ["Blood glucose", "HbA1c"], horizontal=True)
        with c_note:
            st.caption("The dashed line represents the submitted value. Stepwise jumps reflect a decision tree's prediction structure.")
        sensitivity = sensitivity_chart(profile, axis)
        if sensitivity is not None:
            st.plotly_chart(sensitivity, use_container_width=True, config={"displayModeBar": False})
        st.caption("Model sensitivity is not evidence that changing a measurement would cause the score to change in real life. Not a treatment recommendation.")

    title_block("05  /  INTERPRETABILITY", "What drives this model?", "Global feature importance for the trained decision tree, summed across one-hot columns.")
    imp_col, meaning_col = st.columns([1.45, 1], gap="large")
    with imp_col:
        with st.container(border=True):
            fig = importance_chart()
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("This model does not expose built-in feature importance.")
    with meaning_col:
        with st.container(border=True):
            st.markdown("#### How to interpret this chart")
            st.markdown("""
- **Global, not personal.** Importance summarizes the trained tree overall, not the reasons for this individual result.
- **Correlations, not causes.** Important features are useful to the model, but their weights do not prove medical causation.
- **Not model accuracy.** Feature importance cannot establish clinical validity or predictive performance.
            """)
            st.info("Clinical usefulness must be evaluated with independent validation and healthcare professionals.", icon="ℹ️")

    with st.container(border=True):
        dl_col, dl_text = st.columns([1, 2], vertical_alignment="center")
        with dl_col:
            st.download_button("↓  Download assessment CSV", data=make_csv(profile, predicted_class, score),
                               file_name="diabetescare_assessment.csv", mime="text/csv",
                               use_container_width=True)
        with dl_text:
            st.caption("Includes only the inputs and prediction from this assessment, plus an educational-use notice. Download locally if you wish to retain it.")

else:
    with st.container(border=True):
        st.markdown("### ✨ Your dashboard is ready")
        st.write("Submit the form above to reveal the interactive model result, visual health indicators, a what-if explorer, and feature-importance insights.")
        st.caption("Tip: choose a fictional example profile to test the experience without using personal health information.")

# Always visible explainer cards
st.write("")
title_block("HOW IT WORKS", "A transparent three-step workflow", "Built on your original saved model, scaler and encoded feature columns.")
steps = [
    ("01 / INPUT", "Patient characteristics", "Eight demographic, history and laboratory measurements are entered via the assessment form."),
    ("02 / PREPROCESS", "Training-aligned features", "Gender and smoking history are one-hot encoded; 15 features are ordered and scaled using your saved artifacts."),
    ("03 / INFERENCE", "Decision tree prediction", "Your saved DecisionTreeClassifier returns a class and a positive-class score for visual analysis."),
]
cols = st.columns(3, gap="medium")
for col, (step_no, title, desc) in zip(cols, steps):
    with col:
        st.markdown(
            f'<div class="process-card"><div class="process-step">{escape(step_no)}</div>'
            f'<div class="process-title">{escape(title)}</div>'
            f'<div class="process-text">{escape(desc)}</div></div>', unsafe_allow_html=True
        )

st.markdown('<div class="footer"><b>DiabetesCare AI</b> · Built with Python, scikit-learn, Plotly & Streamlit<br>Educational ML research demonstration · Not a medical device · No diagnostic or treatment recommendations</div>', unsafe_allow_html=True)