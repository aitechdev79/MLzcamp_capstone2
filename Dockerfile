FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY predict.py /app/predict.py
COPY model.pkl /app/model.pkl

ENV MODEL_PATH=/app/model.pkl

EXPOSE 9696

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
