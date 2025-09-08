# handler.py
import os

# ===== 1) Limitar threads das libs nativas (antes de qualquer import pesado) =====
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import sys
import json
from pathlib import Path
from flask import Flask, request, Response
from flask_cors import CORS  # CORS

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

# ========= Caminhos do modelo e features =========
MODEL_PATH = REPO_ROOT / "model" / "model_rossman.ubj"          # modelo em UBJ
FEATURE_NAMES_PATH = REPO_ROOT / "model" / "feature_names.json"  # nomes das features
_model = None  # cache em memória


def _download_file_if_needed(path: Path, url_env: str) -> None:
    """Baixa um arquivo a partir de uma env var (ex.: MODEL_URL ou FEATURE_NAMES_URL) se não existir localmente."""
    if path.exists():
        return
    url = os.getenv(url_env)
    if not url:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    except Exception:
        import urllib.request
        with urllib.request.urlopen(url, timeout=60) as resp:
            path.write_bytes(resp.read())


def get_model():
    """
    Carrega o modelo XGBoost (Booster) salvo em UBJ e devolve um wrapper
    com a mesma interface usada no pipeline (get_booster() e predict()).
    """
    global _model
    if _model is not None:
        return _model

    # opcional: baixar de URLs se configuradas
    _download_file_if_needed(MODEL_PATH, "MODEL_URL")
    _download_file_if_needed(FEATURE_NAMES_PATH, "FEATURE_NAMES_URL")

    import xgboost as xgb

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {MODEL_PATH}\n"
            f"Dica: confirme se 'model/model_rossman.ubj' foi commitado ou defina MODEL_URL."
        )

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))

    # garantir feature_names no booster
    feat_names = getattr(booster, "feature_names", None)
    if not feat_names:
        if FEATURE_NAMES_PATH.exists():
            try:
                feat_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
                booster.feature_names = feat_names
            except Exception as e:
                raise RuntimeError(
                    f"Falha ao ler feature_names em {FEATURE_NAMES_PATH}: {e}"
                ) from e

    class _XGBWrapped:
        def __init__(self, booster_obj: "xgb.Booster"):
            self._booster = booster_obj

        def get_booster(self):
            return self._booster

        def predict(self, X):
            # X é um DataFrame já reordenado pelo pipeline para bater com feature_names
            import xgboost as xgb
            dtest = xgb.DMatrix(X)
            return self._booster.predict(dtest)

    _model = _XGBWrapped(booster)
    return _model


# ========= App =========
app = Flask(__name__)

# ===== CORS: habilitar chamadas do navegador (ajuste origins se quiser restringir) =====
CORS(app, resources={r"/rossmann/*": {"origins": "*"}})

# rota raiz
@app.get("/")
def root():
    return {"status": "running", "message": "API Rossmann no ar 🚀"}, 200


# rota de ping
@app.get("/ping")
def ping():
    return {"ping": "pong"}, 200


# ===== /health com modo "leve" e opção deep=1 =====
@app.get("/health")
def health():
    """
    Sem deep (padrão): não carrega o modelo em RAM (checagem leve).
    Com deep=1: carrega o modelo via get_model() e reporta sucesso/falha.
    """
    try:
        deep = request.args.get("deep") == "1"
        app.logger.info(f"/health called deep={deep}")

        exists = MODEL_PATH.exists()
        body = {
            "status": "ok",
            "model_path": str(MODEL_PATH),
            "model_exists": exists,
            "model_load_ok": None,
            "model_error": None,
        }

        if deep and exists:
            try:
                m = get_model()
                booster = m.get_booster()
                body.update(
                    {
                        "model_load_ok": True,
                        "feature_count": len(getattr(booster, "feature_names", []) or []),
                        "note": "deep check: booster carregado em memória",
                    }
                )
            except Exception as e:
                body.update({"model_load_ok": False, "model_error": str(e)})

        # checagem leve de tamanho/hash
        if exists:
            try:
                import hashlib
                size = MODEL_PATH.stat().st_size
                h = hashlib.sha256()
                with open(MODEL_PATH, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                body.update({"model_size_bytes": size, "model_sha256": h.hexdigest()})
            except Exception:
                pass

        return Response(json.dumps(body, ensure_ascii=False), 200, mimetype="application/json")
    except Exception as e:
        app.logger.exception("/health error")
        body = {"status": "error", "message": str(e)}
        return Response(json.dumps(body, ensure_ascii=False), 500, mimetype="application/json")


# ===== /rossmann/predict com logs + JSON tolerante =====
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

        # Predição
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
        if os.getenv("DEBUG_ERRORS", "0") == "1":
            body["trace"] = traceback.format_exc()
        return Response(json.dumps(body, ensure_ascii=False), status=500, mimetype="application/json")


# ===== Forecast próximas semanas por loja =====
from datetime import date as _date, timedelta as _td

@app.get("/rossmann/forecast/<int:store_id>")
def forecast_store(store_id: int):
    """
    Gera linhas futuras (diárias) para a loja e retorna:
      - daily: previsões diárias do modelo
      - weekly: agregação por semana ISO (ano/semana + início/fim)
    Parâmetros:
      weeks (int, default 6)
      start (YYYY-MM-DD, default amanhã)
    """
    try:
        import pandas as pd

        # parâmetros: ?weeks=6&start=YYYY-MM-DD
        weeks = int(request.args.get("weeks", "6"))
        start_str = request.args.get("start")
        start = pd.to_datetime(start_str).date() if start_str else (_date.today() + _td(days=1))

        # gera datas futuras (weeks*7)
        dates = [start + _td(days=i) for i in range(weeks * 7)]

        # regra simples: domingo fechado; demais dias abertos; sem feriado/escola; sem promo
        rows = []
        for d in dates:
            dow = d.isoweekday()  # 1=seg ... 7=dom
            open_ = 0 if dow == 7 else 1
            rows.append({
                # nomes originais, iguais aos CSVs, porque seu pipeline renomeia
                "Store": store_id,
                "DayOfWeek": dow,
                "Date": d.isoformat(),
                "Open": open_,
                "Promo": 0,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
                # metadados padrão (ajuste se quiser por loja)
                "StoreType": "a",
                "Assortment": "a",
                "CompetitionDistance": None,
                "CompetitionOpenSinceMonth": None,
                "CompetitionOpenSinceYear": None,
                "Promo2": 0,
                "Promo2SinceWeek": None,
                "Promo2SinceYear": None,
                "PromoInterval": ""
            })

        df_in = pd.DataFrame(rows)

        # pipeline + modelo
        pipeline = Rossmann()
        df1 = pipeline.data_cleaning(df_in.copy())
        df2 = pipeline.feature_engineering(df1)
        df3 = pipeline.data_preparation(df2)
        model = get_model()
        json_resp = pipeline.get_prediction(model, df_in, df3)  # string JSON

        # pós-processamento: diário + semanal
        df_out = pd.read_json(json_resp)
        df_out["Date"] = pd.to_datetime(df_out["Date"])
        iso = df_out["Date"].dt.isocalendar()
        df_out["iso_year"] = iso.year.astype(int)
        df_out["iso_week"] = iso.week.astype(int)

        weekly = (
            df_out.groupby(["iso_year", "iso_week"], as_index=False)["predictions"].sum()
                .rename(columns={"iso_year": "year", "iso_week": "week", "predictions": "pred_week"})
        )

        # início/fim da semana ISO (segunda a domingo)
        def _wk_bounds(y, w):
            start_wk = pd.to_datetime(f"{int(y)}-W{int(w):02d}-1").date()
            return start_wk, start_wk + _td(days=6)

        weekly["week_start"], weekly["week_end"] = zip(*weekly.apply(
            lambda r: _wk_bounds(r["year"], r["week"]), axis=1
        ))

        body = {
            "store": store_id,
            "weeks": weeks,
            "start": start.isoformat(),
            "assumptions": {"open_sunday": False, "promo_default": 0, "state_holiday": "0", "school_holiday": 0},
            "daily": df_out[["Date", "predictions"]]
                        .assign(Date=lambda d: d["Date"].dt.date.astype(str))
                        .to_dict(orient="records"),
            "weekly": weekly[["year","week","week_start","week_end","pred_week"]]
                        .assign(week_start=lambda d: d["week_start"].astype(str),
                                week_end=lambda d: d["week_end"].astype(str))
                        .to_dict(orient="records"),
        }
        return Response(json.dumps(body, ensure_ascii=False), 200, mimetype="application/json")

    except Exception as e:
        import traceback
        app.logger.exception("forecast:error")
        payload = {"error": str(e), "type": e.__class__.__name__}
        if os.getenv("DEBUG_ERRORS","0") == "1":
            payload["trace"] = traceback.format_exc()
        return Response(json.dumps(payload, ensure_ascii=False), 500, mimetype="application/json")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
