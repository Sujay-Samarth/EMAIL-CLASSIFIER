# ── Build ──
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time
RUN python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Copy project
COPY . .

# Train the model during build (ensures saved_model.pkl is present)
RUN python model/train.py

EXPOSE 5000

# Run with gunicorn in production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
