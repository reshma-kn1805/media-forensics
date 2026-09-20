FROM python:3.11-slim

WORKDIR /app

COPY ml-service/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY ml-service /app/ml-service
COPY backend /app/backend
COPY frontend/dist /app/frontend/dist

ENV PYTHONPATH=/app/ml-service:/app/backend

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]