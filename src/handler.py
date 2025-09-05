# handler.py
import os
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


# rota de healthcheck
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



# rota principal de predição
@app.post("/rossmann/predict")
def rossmann_predict():
    try:
        import pandas as pd

        if "application/json" not in (request.headers.get("Content-Type") or ""):
            return Response(
                json.dumps({"error": "Content-Type deve ser application/json"}),
                status=400,
                mimetype="application/json",
            )

        payload = request.get_json(silent=True)
        if payload is None:
            return Response("[]", status=200, mimetype="application/json")

        # Normaliza em DataFrame
        if isinstance(payload, dict):
            df_in = pd.DataFrame(payload, index=[0])
        elif isinstance(payload, list) and payload:
            cols = sorted(set().union(*(d.keys() for d in payload)))
            df_in = pd.DataFrame(payload, columns=cols)
        else:
            return Response("[]", status=200, mimetype="application/json")

        # Pipeline
        pipeline = Rossmann()
        df1 = pipeline.data_cleaning(df_in.copy())
        df2 = pipeline.feature_engineering(df1)
        df3 = pipeline.data_preparation(df2)

        model = get_model()
        df_response = pipeline.get_prediction(model, df_in, df3)

        # Se já vier string JSON do pipeline, devolve direto
        if isinstance(df_response, str):
            body = df_response
        else:
            body = json.dumps(df_response, ensure_ascii=False)

        return Response(body, status=200, mimetype="application/json")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(
            json.dumps({"error": str(e)}),
            status=500,
            mimetype="application/json",
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
