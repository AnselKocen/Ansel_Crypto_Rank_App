FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for numpy, scikit-learn, wordcloud, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY requirements.txt requirements_web.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements_web.txt

# Download NLTK data required by the app
RUN python -c "import nltk; [nltk.download(package, raise_on_error=True) for package in ['vader_lexicon', 'punkt', 'punkt_tab', 'stopwords', 'wordnet']]"

# Copy application code
COPY web/ ./web/
COPY game/ ./game/
COPY skills/ ./skills/
COPY text/ ./text/
COPY figures/ ./figures/
COPY *.py ./
COPY *.csv ./

EXPOSE 8700

CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8700"]
