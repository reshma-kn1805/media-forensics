# ============================================================
# STAGE 1 — Build React Frontend
# ============================================================

FROM node:22-slim AS frontend-build

WORKDIR /frontend

COPY frontend/package*.json ./

RUN npm install

COPY frontend ./

RUN npm run build


# ============================================================
# STAGE 2 — Run FastAPI + ML + React
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------

COPY ml-service/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt


# ------------------------------------------------------------
# Backend + ML service
# ------------------------------------------------------------

COPY ml-service /app/ml-service
COPY backend /app/backend


# ------------------------------------------------------------
# Built React frontend
# ------------------------------------------------------------

COPY --from=frontend-build /frontend/dist /app/frontend/dist


# ------------------------------------------------------------
# Python module paths
# ------------------------------------------------------------

ENV PYTHONPATH=/app/ml-service:/app/backend


# ------------------------------------------------------------
# Railway listens on port 8000
# ------------------------------------------------------------

EXPOSE 8000


# ------------------------------------------------------------
# Start FastAPI
# ------------------------------------------------------------

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]