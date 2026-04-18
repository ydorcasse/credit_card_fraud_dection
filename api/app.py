import json
import os
import traceback
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import Flask, request, jsonify

from logger import setup_logger

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logger = setup_logger()

app = Flask(__name__)

# Load model and metadata at startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model_Random_Forest_smote.joblib")
METADATA_PATH = os.getenv("METADATA_PATH", "models/model_metadata.json")

model = joblib.load(MODEL_PATH)
logger.info(f"Model loaded from {MODEL_PATH}")

with open(METADATA_PATH) as f:
    metadata = json.load(f)

THRESHOLD = metadata["threshold"]
FEATURES = metadata["features"]

logger.info(f"Metadata loaded | threshold={THRESHOLD} | features_count={len(FEATURES)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_client_ip():
    """Extract real client IP, respecting proxy headers."""
    if request.headers.get("X-Forwarded-For"):
        return request.headers["X-Forwarded-For"].split(",")[0].strip()
    if request.headers.get("X-Real-Ip"):
        return request.headers["X-Real-Ip"]
    return request.remote_addr


# ---------------------------------------------------------------------------
# Middleware — log every request
# ---------------------------------------------------------------------------
@app.before_request
def log_request_info():
    client_ip = get_client_ip()
    logger.info(
        f"REQUEST | ip={client_ip} | method={request.method} | path={request.path}"
    )


@app.after_request
def log_response_info(response):
    client_ip = get_client_ip()
    logger.info(
        f"RESPONSE | ip={client_ip} | method={request.method} | path={request.path} | status={response.status_code}"
    )
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": metadata.get("model_name"),
        "threshold": THRESHOLD,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    client_ip = get_client_ip()
    try:
        data = request.get_json(force=True)

        df = pd.DataFrame([data])
        df = df[FEATURES]

        proba = float(model.predict_proba(df)[0, 1])
        prediction = int(proba >= THRESHOLD)

        logger.info(
            f"PREDICTION | ip={client_ip} | proba={proba:.6f} | prediction={prediction}"
        )

        return jsonify({
            "fraud_probability": proba,
            "fraud_prediction": prediction,
            "threshold_used": THRESHOLD,
        })

    except KeyError as e:
        logger.error(f"MISSING_FEATURE | ip={client_ip} | feature={e}")
        return jsonify({"error": f"Missing feature: {e}"}), 422

    except Exception as e:
        logger.error(
            f"PREDICTION_ERROR | ip={client_ip} | error={e} | trace={traceback.format_exc()}"
        )
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
