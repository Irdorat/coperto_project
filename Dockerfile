FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-runtime.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-runtime.txt

COPY src/ src/
COPY data/processed/daily.csv data/processed/daily.csv
COPY train.py predict.py ./

RUN python train.py

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "predict.py", "--date", "2026-10-01", "--restaurant", "1"]