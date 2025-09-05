# handler.py
import os

# ===== 1a) Limitar threads das libs nativas (antes de qualquer import pesado) =====
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import sys
import json
import pickle
from pathlib import Path
from flask import Flask, request, Response

# ========= Descobrir raiz do repo =========
def find_repo_root(start: Path) -> Path:
    candidates = [start, start.parent, start.parent.parent]
    for c in candidates:
        if (c / "requirements.txt").exists() or (c / "model").exists():
            return c
    return start.parent

CURRENT_DIR = Path(__file__).resolve().parent  # .../src
REPO_ROOT = find_repo_root(CURRENT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ========= Import do pipeline =========
from rossmann.Rossmann import Rossmann  # noqa: E402

# ========= Caminho do modelo =========
MODEL_PATH = REPO_ROOT / "model" / "model_rossman.pkl"
_model = None  # cache em memória


def _download_model_if_needed(path: Path) -> None:
    """Baixa o modelo de MODEL_URL se não existir localmente."""
    if path.exists():
        return
    url = os.getenv("MODEL_URL")
    if not url:
        raise FileNotFoundError(
            f"Modelo não encontrado em: {path} e a variável MODEL_URL não foi definida."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import requests
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    except Exception:
        import urllib.request
        with urllib.request.urlopen(url, timeout=60) as resp:
            path.write_bytes(resp.read())


def get_model():
    """Carrega o modelo apenas na primeira chamada (economiza RAM no boot)."""
    global _model
    if _model is None:
        _download_model_if_needed(MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


# ========= App =========
app = Flask(__name__)


# rota raiz
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "API Rossmann no ar 🚀"
    }, 200


# rota de ping
@app.get("/ping")
def ping():
    return {"ping": "pong"}, 200


# ===== 1b) /health com modo "deep" usando o cache do get_model() =====
@app.get("/health")
def health():
    exists = MODEL_PATH.exists()
    deep = request.args.get("deep") == "1"

    load_ok = None
    err = None

    if deep and exists:
        try:
            # usa o cache: só carrega uma vez e reaproveita
            _ = get_model()
            load_ok = True
        except Exception as e:
            load_ok = False
            err = str(e)

    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_exists": exists,
        "model_load_ok": load_ok,  # None = não checou
        "model_error": err
    }, 200


# ===== 1c) /rossmann/predict com logs de marcos + JSON tolerante =====
@app.post("/rossmann/predict")
def rossmann_predict():
    try:
        app.logger.info("predict:start")
        import pandas as pd

        # Tolerar ausência/erro de Content-Type usando force=True
        payload = request.get_json(silent=True, force=True)
        if payload is None:
            app.logger.info("predict:payload_none")
            return Response("[]", status=200, mimetype="application/json")

        # Normaliza em DataFrame
        if isinstance(payload, dict):
            df_in = pd.DataFrame(payload, index=[0])
        elif isinstance(payload, list) and payload:
            # DataFrame direto da lista de dicts
            df_in = pd.DataFrame(payload)
        else:
            app.logger.info("predict:payload_empty_list")
            return Response("[]", status=200, mimetype="application/json")

        app.logger.info("predict:payload_ok rows=%d cols=%d", len(df_in), len(df_in.columns))

        # Pipeline
        pipeline = Rossmann()
        df1 = pipeline.data_cleaning(df_in.copy())
        app.logger.info("predict:after_cleaning rows=%d cols=%d", len(df1), len(df1.columns))
        df2 = pipeline.feature_engineering(df1)
        app.logger.info("predict:after_fe rows=%d cols=%d", len(df2), len(df2.columns))
        df3 = pipeline.data_preparation(df2)
        app.logger.info("predict:after_prep rows=%d cols=%d", len(df3), len(df3.columns))

        model = get_model()
        app.logger.info("predict:model_loaded")

        df_response = pipeline.get_prediction(model, df_in, df3)
        app.logger.info("predict:after_predict")

        # Se já vier string JSON do pipeline, devolve direto
        if isinstance(df_response, str):
            body = df_response
        else:
            body = json.dumps(df_response, ensure_ascii=False)

        return Response(body, status=200, mimetype="application/json")

    except Exception as e:
        import traceback
        app.logger.exception("predict:error")
        body = {"error": str(e), "type": e.__class__.__name__}
        # Opcional: exporte DEBUG_ERRORS=1 no Render para receber traceback no JSON
        if os.getenv("DEBUG_ERRORS", "0") == "1":
            body["trace"] = traceback.format_exc()
        return Response(json.dumps(body, ensure_ascii=False), status=500, mimetype="application/json")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
