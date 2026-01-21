#!/usr/bin/env python

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = None


def load_model(model_path: Path) -> None:
    global model
    with model_path.open("rb") as f:
        model = pickle.load(f)


@app.route("/health", methods=["GET"])
def health() -> tuple[dict, int]:
    if model is None:
        return {"status": "model_not_loaded"}, 500
    return {"status": "ok"}, 200


@app.route("/predict", methods=["POST"])
def predict() -> tuple[dict, int]:
    if model is None:
        return {"error": "model not loaded"}, 500

    payload = request.get_json(silent=True)
    if not payload:
        return {"error": "Invalid or empty JSON payload"}, 400

    if "record" in payload:
        records = [payload["record"]]
    elif "records" in payload:
        records = payload["records"]
    else:
        return {"error": "Provide 'record' or 'records' in JSON payload"}, 400

    try:
        data = pd.DataFrame.from_records(records)
        preds = model.predict(data)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}, 400

    return {"predictions": preds.tolist()}, 200


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve student performance model")
    parser.add_argument(
        "--model-path",
        default="model.pkl",
        help="Path to the trained model pipeline",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9696)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    load_model(model_path)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
