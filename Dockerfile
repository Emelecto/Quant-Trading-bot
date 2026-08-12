# Dockerfile para DeepFin Trading Bot (API + Dashboard)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Descarga datos de ejemplo al arrancar (paper trading, $0)
RUN python -m data.fetch_data || true

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"]
