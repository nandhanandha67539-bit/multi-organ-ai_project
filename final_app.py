"""
Multi-Organ AI Screening - Phase 3 (real models connected)

Folder layout (both must be in the SAME folder):
    final_app.py
    models/   (the 9 files saved from Google Colab)
Run with:  streamlit run final_app.py
"""
import os
import warnings
from datetime import date

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

# ---------------------------------------------------------------------------
# IMPORTANT: set this after running the "label check" cell in Colab (Step 1).
# True  -> in heart.csv, target = 1 means heart disease
# False -> in heart.csv, target = 1 means NO heart disease (healthier)
# ---------------------------------------------------------------------------
HEART_CLASS1_MEANS_DISEASE = False

DISEASE = {"heart": "Heart disease", "kidney": "Chronic kidney disease", "lung": "Lung cancer"}
CLASS1_IS_DISEASE = {"heart": HEART_CLASS1_MEANS_DISEASE, "kidney": True, "lung": True}

FILES = ["heart_classical", "heart_qsvm", "kidney_classical", "kidney_qsvm",
         "lung_classical", "lung_qsvm"]


# =========================== LOGIC (no Streamlit) ===========================
def load_models():
    """Load every .pkl that exists. Returns (models dict, error messages)."""
    models, errors = {}, {}
    for name in FILES:
        path = os.path.join(MODEL_DIR, name + ".pkl")
        if not os.path.exists(path):
            errors[name] = "file not found"
            continue
        try:
            models[name] = joblib.load(path)
        except Exception as e:  # version mismatch etc.
            errors[name] = str(e)[:200]
    return models, errors


def qiskit_available():
    try:
        import qiskit  # noqa: F401
        import qiskit_machine_learning  # noqa: F401
        return True
    except Exception:
        return False


def build_feature_map(d):
    from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
    name = d["feature_map"]
    reps = d.get("reps", 2 if "reps=2" in name else 1)
    if name.startswith("ZZ"):
        return ZZFeatureMap(4, reps=reps, entanglement="linear")
    return ZFeatureMap(4, reps=reps)


def compute_kernel(d, x_scaled):
    """Quantum kernel between the new point and the saved training points."""
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    qk = FidelityQuantumKernel(feature_map=build_feature_map(d))
    return qk.evaluate(x_vec=x_scaled, y_vec=d["X_train_q"])


def qsvm_predict(d, qx):
    """qx: the 4 selected features in the same space the q_scaler was fitted on."""
    qs = d["q_scaler"]
    smax = qs.feature_range[1]
    if hasattr(qs, "feature_names_in_") and isinstance(qx, pd.DataFrame):
        qx = qx[list(qs.feature_names_in_)]
    x = np.clip(qs.transform(qx), 0, smax)
    K = compute_kernel(d, x)
    return int(d["svc"].predict(K)[0])


def prepare(organ, inp, M):
    """Turn the entered values into the model-ready vector for each organ."""
    c, q = M[organ + "_classical"], M.get(organ + "_qsvm")
    if organ == "heart":
        df_in = pd.DataFrame([inp])[c["features"]]
        xs = c["scaler"].transform(df_in)
        names = list(c["features"])
        qx = df_in[q["features"]] if q else None
    elif organ == "kidney":
        df_in = pd.DataFrame([inp])[list(c["num_cols"]) + list(c["bin_cols"])]
        xs = c["preprocess"].transform(df_in)
        names = list(c["feature_names"])
        qx = xs[:, q["top4_idx"]] if q else None
    else:
        df_in = pd.DataFrame([inp])[list(c["feature_names"])]
        xs = c["scaler"].transform(df_in)
        names = list(c["feature_names"])
        qx = xs[:, q["top4_idx"]] if q else None
    return xs, names, qx


def predict_organ(organ, inp, M, use_qsvm=True):
    c, q = M[organ + "_classical"], M.get(organ + "_qsvm")
    xs, names, qx = prepare(organ, inp, M)
    model = c["model"]
    flag = CLASS1_IS_DISEASE[organ]

    p1 = float(model.predict_proba(xs)[0, 1])
    cls = int(model.predict(xs)[0])
    out = {
        "organ": organ,
        "classical_disease": (cls == 1) == flag,
        "classical_score": p1 if flag else 1 - p1,   # score for the DISEASE outcome
        "qsvm_disease": None,
        "qsvm_error": None,
    }

    # simple, exact explanation for a linear model: coefficient x scaled value
    contrib = pd.Series(model.coef_[0] * xs[0], index=names)
    contrib = contrib if flag else -contrib            # positive = pushes toward disease
    out["contrib"] = contrib.reindex(contrib.abs().sort_values(ascending=False).index)

    if use_qsvm and q is not None:
        try:
            qcls = qsvm_predict(q, qx)
            out["qsvm_disease"] = (qcls == 1) == flag
        except Exception as e:
            out["qsvm_error"] = str(e)[:200]
    return out


def make_report(organ, shown, res):
    lines = [f"MULTI-ORGAN AI SCREENING REPORT ({DISEASE[organ]})",
             f"Date: {date.today()}", "", "ENTERED VALUES"]
    lines += [f"  {k}: {v}" for k, v in shown.items()]
    lines += ["", "MODEL RESULTS",
              f"  Classical model (Logistic Regression): "
              f"{'disease pattern' if res['classical_disease'] else 'no disease pattern'}"
              f" (model score {res['classical_score']*100:.0f}%)"]
    if res["qsvm_disease"] is not None:
        lines.append(f"  QSVM (4 features): "
                     f"{'disease pattern' if res['qsvm_disease'] else 'no disease pattern'}")
    elif res["qsvm_error"]:
        lines.append("  QSVM: not available on this computer")
    lines += ["", "FEATURES THAT MOST INFLUENCED THE CLASSICAL MODEL (+ raises, - lowers the score)"]
    lines += [f"  {n}: {v:+.2f}" for n, v in res["contrib"].head(6).items()]
    lines += ["", "LIMITATIONS",
              "  - Educational screening prototype, NOT a medical diagnosis.",
              "  - Trained on small public datasets; results may not generalise.",
              "  - The model score is not a calibrated clinical probability.",
              "  - Feature influence shows what drove the model, not the medical cause.",
              "  - Consult a qualified doctor for any health concern."]
    return "\n".join(lines)


# ============================== UI (Streamlit) ==============================
try:
    import streamlit as st
except ImportError:  # lets the logic above be tested without Streamlit
    st = None

if st is not None:
    st.set_page_config(page_title="Multi-Organ AI Screening", page_icon="🩺", layout="wide")

    @st.cache_resource
    def get_models():
        return load_models()

    @st.cache_data
    def get_results(organ):
        p = os.path.join(MODEL_DIR, f"{organ}_results.csv")
        return pd.read_csv(p) if os.path.exists(p) else None

    M, ERR = get_models()
    QISKIT = qiskit_available()

    class Form:
        """Collects entered values (codes for the model, labels for the report)."""
        def __init__(self):
            self.inp, self.shown = {}, {}

        def num(self, col, key, label, lo, hi, default, step, fmt=None):
            v = col.number_input(label, lo, hi, default, step, format=fmt)
            self.inp[key], self.shown[label] = v, v

        def cat(self, col, key, label, options, index=0):
            lab = col.selectbox(label, list(options.keys()), index=index)
            self.inp[key], self.shown[label] = options[lab], lab

    YN = {"No": 0, "Yes": 1}

    # ----------------------------- forms ---------------------------------
    def heart_form(f):
        c = st.columns(3)
        f.num(c[0], "age", "Age (years)", 1, 120, 50, 1)
        f.cat(c[1], "sex", "Sex", {"Female": 0, "Male": 1}, 1)
        f.cat(c[2], "cp", "Chest pain type",
              {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3})
        f.num(c[0], "trestbps", "Resting blood pressure (mm Hg)", 80, 220, 125, 1)
        f.num(c[1], "chol", "Cholesterol (mg/dl)", 100, 600, 240, 1)
        f.cat(c[2], "fbs", "Fasting blood sugar > 120 mg/dl", YN)
        f.cat(c[0], "restecg", "Resting ECG",
              {"Normal": 0, "ST-T abnormality": 1, "LV hypertrophy": 2})
        f.num(c[1], "thalach", "Max heart rate achieved", 60, 220, 150, 1)
        f.cat(c[2], "exang", "Exercise-induced angina", YN)
        f.num(c[0], "oldpeak", "ST depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
        f.cat(c[1], "slope", "ST slope", {"Upsloping": 0, "Flat": 1, "Downsloping": 2}, 1)
        f.num(c[2], "ca", "Major vessels coloured (0-4)", 0, 4, 0, 1)
        f.cat(c[0], "thal", "Thalassemia",
              {"Normal": 2, "Fixed defect": 1, "Reversible defect": 3})

    KIDNEY_NUM = [
        ("age", "Age (years)", 1, 100, 45, 1, None),
        ("bp", "Blood pressure (mm Hg)", 40, 200, 80, 1, None),
        ("sg", "Urine specific gravity", 1.000, 1.050, 1.020, 0.005, "%.3f"),
        ("al", "Albumin (0-5)", 0, 5, 0, 1, None),
        ("su", "Sugar (0-5)", 0, 5, 0, 1, None),
        ("bgr", "Blood glucose random (mg/dl)", 20, 500, 110, 1, None),
        ("bu", "Blood urea (mg/dl)", 1.0, 400.0, 30.0, 1.0, None),
        ("sc", "Serum creatinine (mg/dl)", 0.1, 80.0, 1.0, 0.1, None),
        ("sod", "Sodium (mEq/L)", 100.0, 170.0, 140.0, 1.0, None),
        ("pot", "Potassium (mEq/L)", 2.0, 50.0, 4.4, 0.1, None),
        ("hemo", "Haemoglobin (g/dl)", 3.0, 20.0, 14.5, 0.1, None),
        ("pcv", "Packed cell volume (%)", 9.0, 60.0, 44.0, 1.0, None),
        ("wc", "White blood cell count (cells/cumm)", 2000, 30000, 7800, 100, None),
        ("rc", "Red blood cell count (millions/cmm)", 2.0, 8.0, 5.2, 0.1, None),
    ]
    KIDNEY_CAT = [
        ("rbc", "Red blood cells", {"Normal": 0, "Abnormal": 1}),
        ("pc", "Pus cells", {"Normal": 0, "Abnormal": 1}),
        ("pcc", "Pus cell clumps", {"Not present": 0, "Present": 1}),
        ("ba", "Bacteria", {"Not present": 0, "Present": 1}),
        ("htn", "Hypertension", YN),
        ("dm", "Diabetes mellitus", YN),
        ("cad", "Coronary artery disease", YN),
        ("appet", "Appetite", {"Good": 0, "Poor": 1}),
        ("pe", "Pedal edema", YN),
        ("ane", "Anaemia", YN),
    ]

    def kidney_form(f):
        st.caption("Default values are typical normal values. Replace them with the patient's real lab values.")
        c = st.columns(4)
        for i, (k, lab, lo, hi, de, stp, fmt) in enumerate(KIDNEY_NUM):
            f.num(c[i % 4], k, lab, lo, hi, de, stp, fmt)
        for i, (k, lab, opts) in enumerate(KIDNEY_CAT):
            f.cat(c[i % 4], k, lab, opts)

    LUNG_SYMPTOMS = [("SMOKING", "Smoking"), ("YELLOW_FINGERS", "Yellow fingers"),
                     ("ANXIETY", "Anxiety"), ("PEER_PRESSURE", "Peer pressure"),
                     ("CHRONIC_DISEASE", "Chronic disease"), ("FATIGUE", "Fatigue"),
                     ("ALLERGY", "Allergy"), ("WHEEZING", "Wheezing"),
                     ("ALCOHOL_CONSUMING", "Alcohol consumption"), ("COUGHING", "Coughing"),
                     ("SHORTNESS_OF_BREATH", "Shortness of breath"),
                     ("SWALLOWING_DIFFICULTY", "Swallowing difficulty"),
                     ("CHEST_PAIN", "Chest pain")]

    def lung_form(f):
        st.caption("This model uses a symptom survey (yes/no answers), not lab tests.")
        c = st.columns(4)
        f.num(c[0], "AGE", "Age (years)", 1, 120, 50, 1)
        f.cat(c[1], "GENDER", "Gender", {"Female": 0, "Male": 1}, 1)
        for i, (k, lab) in enumerate(LUNG_SYMPTOMS):
            f.cat(c[(i + 2) % 4], k, lab, YN)

    FORMS = {"heart": heart_form, "kidney": kidney_form, "lung": lung_form}

    NOTES = {
        "heart": "Our test set had 61 patients. QSVM and the classical models performed comparably, "
                 "and no model was clearly better.",
        "kidney": "Our test set had 80 patients. All models scored very high because the lab values separate "
                  "the two groups strongly. This may not hold on real hospital data.",
        "lung": "Our test set had only 8 healthy patients (the data is imbalanced), so results are unstable. "
                "QSVM scored lower than the classical models here.",
    }

    # ------------------------------ pages --------------------------------
    def organ_page(organ, title):
        st.title(title)
        if organ + "_classical" not in M:
            st.error(f"{organ}_classical.pkl could not be loaded: {ERR.get(organ + '_classical')}")
            st.stop()
        if organ == "heart":
            st.caption("Dataset: UCI Heart Disease (Cleveland), duplicates removed. "
                       "Entries use the dataset's own numeric coding.")

        f = Form()
        with st.form(organ):
            FORMS[organ](f)
            submitted = st.form_submit_button("Predict")

        key = "res_" + organ
        if submitted:
            with st.spinner("Running the models (QSVM may take a few seconds)..."):
                res = predict_organ(organ, f.inp, M, use_qsvm=QISKIT)
            st.session_state[key] = (res, dict(f.shown))

        if key not in st.session_state:
            return
        res, shown = st.session_state[key]
        name = DISEASE[organ]

        st.subheader("Model output")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Classical model (Logistic Regression)**")
            if res["classical_disease"]:
                st.error(f"Pattern consistent with: {name}")
            else:
                st.success(f"No {name.lower()} pattern found")
            st.metric("Model score for this outcome", f"{res['classical_score'] * 100:.0f}%")
            st.caption("This is an uncalibrated model score, not a clinical probability.")
        with c2:
            st.markdown("**Quantum model (QSVM, 4 features)**")
            if res["qsvm_disease"] is None:
                if not QISKIT:
                    st.info("QSVM needs Qiskit on this computer (see setup steps). Classical result shown only.")
                else:
                    st.warning(f"QSVM could not run: {res['qsvm_error'] or 'model file missing'}")
            elif res["qsvm_disease"]:
                st.error(f"Pattern consistent with: {name}")
            else:
                st.success(f"No {name.lower()} pattern found")
            st.caption("QSVM gives a class only (no probability).")

        if res["qsvm_disease"] is not None:
            if res["qsvm_disease"] == res["classical_disease"]:
                st.write("Both models agree.")
            else:
                st.warning("The two models disagree. Treat this result with extra caution.")
        st.warning("Educational screening prototype. NOT a medical diagnosis. Please consult a doctor.")

        t1, t2, t3 = st.tabs(["📊 Model comparison", "🔍 Explainable AI", "📄 Report"])
        with t1:
            r = get_results(organ)
            if r is None:
                st.info(f"{organ}_results.csv not found in the models folder.")
            else:
                st.dataframe(r, hide_index=True)
                cols = [c for c in ["Accuracy", "F1-score"] if c in r.columns]
                st.bar_chart(r.set_index("Model")[cols])
                st.caption("Results on the held-out test set. " + NOTES[organ])
        with t2:
            top = res["contrib"].head(8)
            st.write("Features that moved the **classical model's** score the most for this input "
                     "(positive = pushes toward the disease outcome, negative = pushes away).")
            st.bar_chart(top)
            st.caption("This shows what drove the model, not the medical cause. "
                       "Explanation for QSVM is not provided.")
        with t3:
            st.text(make_report(organ, shown, res))
            st.download_button("Download report (.txt)", make_report(organ, shown, res),
                               file_name=f"{organ}_report.txt")

    # --------------------------- navigation ------------------------------
    st.sidebar.title("🩺 Multi-Organ AI")
    page = st.sidebar.radio("Select an organ", ["Home", "❤️ Heart", "🫘 Kidney", "🫁 Lung"])
    st.sidebar.info("Real trained models (Phase 2) connected to this website.")

    if page == "Home":
        st.title("Hybrid Quantum–Classical AI-Based Multi-Organ Disease Prediction")
        st.caption("With Explainable AI")
        st.markdown("""
**How it works:** enter clinical data → preprocessing → classical ML + Quantum SVM (QSVM) →
explanation → downloadable report.

> Educational screening tool. **Not a medical diagnosis.**
""")
        st.subheader("System status")
        ok = [n for n in FILES if n in M]
        st.write(f"Model files loaded: **{len(ok)} / {len(FILES)}**")
        st.write("QSVM engine (Qiskit): " + ("✅ available" if QISKIT else "❌ not installed (classical models still work)"))
        for n, e in ERR.items():
            st.warning(f"{n}: {e}")
        if not HEART_CLASS1_MEANS_DISEASE:
            st.info("Heart label setting: class 1 is treated as 'no heart disease'. "
                    "Verify this with the label check in Colab.")
    elif page == "❤️ Heart":
        organ_page("heart", "❤️ Heart Disease Prediction")
    elif page == "🫘 Kidney":
        organ_page("kidney", "🫘 Kidney Disease (CKD) Prediction")
    else:
        organ_page("lung", "🫁 Lung Cancer Prediction (symptom survey)")
        st.caption("Trained on a small, imbalanced symptom survey. Not a clinical test.")
