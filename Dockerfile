# Dockerfile
FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional, but handy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy app and model (place your best_model_for_insurance.pkl next to Dockerfile)
COPY app.py ./
COPY best_model_for_insurance.pkl ./

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config for headless
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "app.py"]