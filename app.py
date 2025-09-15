import os
import io
import uuid
import json
import asyncio
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import joblib

# Headless plotting for servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from anyio import Semaphore

# Load .env early so os.getenv works everywhere
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)

# Gemini SDK
import google.generativeai as genai


# ================= Configuration =================
MODEL_PATH = os.getenv("MODEL_PATH", "heart_stack_calibrated.pkl")
STATIC_DIR = os.getenv("STATIC_DIR", "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Concurrency guard for heavy work (SHAP, image IO, Gemini)
MAX_CONCURRENT_THREADS = int(os.getenv("MAX_CONCURRENT_THREADS", "4"))
THREAD_GUARD = Semaphore(MAX_CONCURRENT_THREADS)

# Expected CSV columns (keep in sync with training schema)
REQUIRED_COLUMNS = [
    "Age",
    "sex",
    "chest pain type",
    "Trestbps",
    "cholesteral",
    "fasting blood sugar",
    "resting ecg",
    "max heart rate",
    "exercise induced angina",
    "oldpeak",
    "slope",
    "number of vessels colored",
    "thal",
]

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Heart Risk API")

# CORS for Next.js dev
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# Serve generated images at /static/*
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ================= Model load =================
try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

_feature_names = getattr(MODEL, "feature_names_in_", None)
FEATURE_ORDER = list(_feature_names) if _feature_names is not None else list(REQUIRED_COLUMNS)


# ================= Utilities =================
def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    ordered = [c for c in FEATURE_ORDER if c in df.columns]
    return df[ordered]


def _predict(model, X: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        try:
            class_labels = getattr(model, "classes_", None)
            idxs = np.argmax(proba, axis=1)
            preds = class_labels[idxs].tolist() if class_labels is not None else idxs.tolist()
        except Exception:
            preds = np.argmax(proba, axis=1).tolist()
        out["predictions"] = preds
        out["probabilities"] = proba.tolist()
    else:
        preds = model.predict(X)
        out["predictions"] = np.asarray(preds).tolist()
    return out


def _save_current_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _predict_proba_wrapper(model):
    def fn(X_):
        X_np = np.asarray(X_)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        cols = FEATURE_ORDER[: X_np.shape[1]]
        X_df = pd.DataFrame(X_np, columns=cols)
        X_df = _ensure_columns(X_df)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_df)
        preds = model.predict(X_df)
        return np.asarray(preds).reshape(-1, 1)
    return fn


def _compute_shap_values(model, X: pd.DataFrame, max_rows: int = 256, nsamples: int = 50):
    X = _ensure_columns(X)
    bg = X.sample(min(50, len(X)), random_state=0) if len(X) > 50 else X
    X_small = X.iloc[:max_rows].copy()

    # Use TreeExplainer if possible (fast)
    if any(s in type(model).__name__ for s in ["Tree", "Forest", "XGB", "LGBM"]):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_small)
    else:
        wrapper = _predict_proba_wrapper(model)
        explainer = shap.KernelExplainer(wrapper, bg)
        sv = explainer.shap_values(X_small, nsamples=nsamples)

    return sv, X_small


def _plot_summary(sv, X_small, save_path: str):
    try:
        shap.summary_plot(sv, X_small, show=False)
    except Exception:
        shap.summary_plot(sv, X_small, show=False)
    _save_current_fig(save_path)


def _plot_bar(sv, X_small, save_path: str):
    try:
        shap.plots.bar(sv, show=False, max_display=20)
    except Exception:
        shap.summary_plot(sv, X_small, plot_type="bar", show=False)
    _save_current_fig(save_path)


def _extract_top_features(sv, X_small, k: int = 10) -> List[str]:
    cols = list(X_small.columns)

    try:
        vals = getattr(sv, "values", None)
        if vals is not None:
            arr = np.asarray(vals)
        elif isinstance(sv, list):
            stacked = np.stack([np.asarray(s) for s in sv], axis=0)
            arr = stacked.mean(axis=0)
        else:
            arr = np.asarray(sv)
            if arr.ndim == 3:
                arr = arr.mean(axis=0)
    except Exception:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            arr = arr.mean(axis=0)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    imp = np.abs(arr).mean(axis=0)
    n_features = arr.shape[1]
    if n_features != len(cols):
        if n_features < len(cols):
            cols = cols[:n_features]
        else:
            cols = cols + [f"f{i}" for i in range(len(cols), n_features)]

    sorted_indices = np.argsort(-imp)
    top_k = sorted_indices[:min(k, len(cols))]
    return [cols[int(i)] for i in top_k]


async def run_blocking(func, *args, **kwargs):
    async with THREAD_GUARD:
        return await run_in_threadpool(func, *args, **kwargs)


def _gemini_summary(pred_json: Dict[str, Any], top_features: List[str]) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API key not configured."
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = (
        "You are a clinical assistant generating a brief, user-friendly summary of model outputs.\n"
        f"Predictions JSON: {json.dumps(pred_json)[:6000]}\n"
        f"Top features (approx): {', '.join(top_features[:10])}\n"
        "Explain what the predictions indicate and caution about clinical validation.\n"
        "Keep it under 150 words, neutral tone."
    )
    try:
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "").strip() or "No summary generated."
    except Exception as e:
        return f"Gemini error: {e}"


# ================= Endpoint =================
@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    try:
        df = _ensure_columns(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    pred_task = asyncio.create_task(run_blocking(_predict, MODEL, df))
    sv, X_small = await run_blocking(_compute_shap_values, MODEL, df, 256)

    summary_name = f"shap_summary_{uuid.uuid4().hex}.png"
    summary_path = os.path.join(STATIC_DIR, summary_name)
    bar_name = f"shap_bar_{uuid.uuid4().hex}.png"
    bar_path = os.path.join(STATIC_DIR, bar_name)

    plot_summary_task = asyncio.create_task(run_blocking(_plot_summary, sv, X_small, summary_path))
    plot_bar_task = asyncio.create_task(run_blocking(_plot_bar, sv, X_small, bar_path))

    pred_out = await pred_task
    await asyncio.gather(plot_summary_task, plot_bar_task)

    shap_images = [f"/static/{summary_name}", f"/static/{bar_name}"]

    top_features = await run_blocking(_extract_top_features, sv, X_small, 10)
    gemini_text = await run_blocking(_gemini_summary, pred_out, top_features)

    return {
        "ok": True,
        "predictions": pred_out.get("predictions", []),
        "probabilities": pred_out.get("probabilities", []),
        "feature_order": FEATURE_ORDER,
        "top_features": top_features,
        "shap_images": shap_images,
        "gemini_summary": gemini_text,
    }
