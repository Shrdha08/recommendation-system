# Build context is the repository root so the image can include the model
# weights and processed data that live outside backend/.
FROM python:3.12-slim

WORKDIR /app/backend

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data/processed \
    NCF_WEIGHTS_PATH=/app/models/ncf_weights.npz

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY backend/seed.py backend/entrypoint.sh ./
COPY data/processed/final_movies.csv data/processed/final_ratings.csv /app/data/processed/
COPY ["models/neural collaborative filtering/saved model/ncf_weights.npz", "/app/models/"]

RUN chmod +x entrypoint.sh

EXPOSE 8000
CMD ["./entrypoint.sh"]
