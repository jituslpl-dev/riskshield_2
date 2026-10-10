FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --no-deps . && useradd --uid 10001 --create-home appuser && mkdir registry runtime && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/live')"
CMD ["uvicorn", "riskshield.api:app", "--host", "0.0.0.0", "--port", "8000"]
