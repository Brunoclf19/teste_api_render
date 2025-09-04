# api/handler.py
import os, sys, pickle
import pandas as pd
from flask import Flask, request, Response

# --- forçar raiz no sys.path ---
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rossmann.Rossmann import Rossmann  # ← agora deve funcionar

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model_rossman.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# api/handler.py (apenas o endpoint)
@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    try:
        test_json = request.get_json()
        if not test_json:
            return Response('[]', status=200, mimetype='application/json')

        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        pipeline = Rossmann()
        df1 = pipeline.data_cleaning(test_raw)
        df2 = pipeline.feature_engineering(df1)
        df3 = pipeline.data_preparation(df2)
        df_response = pipeline.get_prediction(model, test_raw, df3)
        return Response(df_response, status=200, mimetype='application/json')

    except Exception as e:
        import traceback, json
        traceback.print_exc()  # <-- imprime stacktrace no terminal
        msg = {"error": str(e)}
        return Response(json.dumps(msg), status=500, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    # escuta em todas as interfaces, mas CHAME pelo 127.0.0.1
    app.run(host='0.0.0.0', port=port, debug=True)
