# DiabetesCare AI 🩺

A modern, interactive **educational ML demo** for exploring a saved diabetes classification model using Streamlit and Plotly.

> **Medical disclaimer:** Not a medical device, diagnostic service, or validated clinical risk tool. The positive-class score is a decision tree's machine-learning output, **not** a calibrated probability of having or developing diabetes. Don't use it to make medical decisions or upload identifiable patient data.

## Features

- Responsive, mint/teal medical-style interface with custom cards, clinical-themed hero and clean typography.
- Eight profile fields with fictional demo presets.
- Class prediction and labeled **positive-class model score**.
- Separate-scale health-marker visualization and downloadable CSV assessment.
- Interactive what-if chart: HbA1c or glucose changed one at a time (not causal).
- **Global** feature importance chart from the trained model (not individual explanation).
- No database or application-level patient-history persistence.

## Model artifacts included

| File | Purpose |
| --- | --- |
| `best_diabetes_model.pkl` | Fitted `DecisionTreeClassifier(max_depth=8, random_state=42)` |
| `scaler_diabetes.pkl` | Fitted `StandardScaler` |
| `diabetes_feature_columns.pkl` | Exact ordered list of 15 model features |

These artifacts were generated using **scikit-learn 1.6.1**; the requirements pin that version to preserve compatibility. Only load trusted pickle files. Dataset provenance, evaluation metrics, calibration and clinical validation were not supplied and cannot be inferred from these artifacts.

## Run locally

Python 3.11 or 3.12 recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run inference tests:

```bash
python -m unittest discover -s tests -v
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository (suggested: `diabetescare-ai`). Upload **all** files in this folder, including the three `.pkl` files and `.streamlit/config.toml`. Don't upload a virtual environment.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and choose **Create app**.
3. Select your repository and branch (usually `main`), with entrypoint **`app.py`**.
4. Under **Advanced settings**, select Python **3.12**.
5. Choose a custom app subdomain if available, then deploy.

**Troubleshooting:** If you see `InconsistentVersionWarning`, check that `scikit-learn==1.6.1` actually installed. If files are missing, confirm the three `.pkl` artifacts are in the same repository folder as `model_utils.py`.

## Project structure

```text
DiabetesCare-AI/
├── .streamlit/
│   └── config.toml
├── app.py
├── model_utils.py
├── requirements.txt
├── best_diabetes_model.pkl
├── scaler_diabetes.pkl
├── diabetes_feature_columns.pkl
├── tests/
│   └── test_inference.py
├── .gitignore
└── README.md
```

## Notes on medical interpretation

- Positive class (`1`) and negative class (`0`) are **model labels only**.
- Sensitivity plots alter one model input, but do not establish causation, benefit or treatment effects.
- Feature importance is global tree impurity-based importance, not a patient-specific explanation.
- Blood glucose is displayed in **mg/dL** as an assumed dataset convention; verify this against the original training data. HbA1c is assumed to be in percent.
- Streamlit processes submitted values server-side. The app itself does not store patient profiles, but this is not a substitute for healthcare-grade data privacy controls.