FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    xgboost \
    pandas \
    numpy \
    python-dotenv \
    pyyaml \
    loguru

COPY configs/ ./configs/
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]