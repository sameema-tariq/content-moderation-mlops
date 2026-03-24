# content-moderation-mlops

Production-grade content moderation API with a full MLOps pipeline. Built with FastAPI, scikit-learn, MLflow, Prometheus, and Docker.

---

## What it does

Classifies text as **spam** or **ham** using a TF-IDF + Logistic Regression model, served via a REST API with experiment tracking and monitoring.

---

## Project Structure

```
content-moderation-mlops/
├── app/                        # FastAPI application
│   ├── main.py                 # API endpoints: /health, /predict, /metrics
│   ├── predictor.py            # Loads model bundle and serves predictions
│   ├── schemas.py              # Pydantic request/response validation
│   └── logger.py               # Shared logger
├── pipeline/                   # ML pipeline
│   ├── extract.py              # Reads SMS zip, returns DataFrame
│   ├── preprocessing.py        # Text cleaning (URLs, phone numbers, stopwords)
│   ├── train.py                # Trains TF-IDF + LogisticRegression pipeline
│   ├── test_evaluate.py        # Evaluation metrics + MLflow logging
│   ├── save_model.py           # Saves model as pickle bundle
│   └── utils.py                # Preview samples, save preprocessed CSV
├── config/
│   └── settings.py             # Paths, hyperparameters, constants
├── tests/
│   └── test_api.py             # API unit tests (pytest)
├── monitoring/
│   └── prometheus.yml          # Prometheus scrape config
├── Scripts/
│   └── run.sh                  # One command to build, train, test, and serve
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Stack

| Tool | Purpose |
|------|---------|
| FastAPI | REST API |
| scikit-learn | TF-IDF + Logistic Regression |
| MLflow | Experiment tracking |
| Prometheus | Metrics scraping |
| Docker / Docker Compose | Containerisation |
| slowapi | Rate limiting |
| pytest | Unit tests |

---

## Services

| Service | Port | URL |
|---------|------|-----|
| API | 8000 | http://localhost:8000 |
| MLflow UI | 5000 | http://localhost:5000 |
| Prometheus | 9090 | http://localhost:9090 |

---

## Quickstart

```bash
git clone <repo-url>
cd content-moderation-mlops

# Place dataset in data/
# sms+spam+collection.zip → data/sms+spam+collection.zip

./Scripts/run.sh
```

This will:
1. Build the Docker image
2. Start all services (API, MLflow, Prometheus)
3. Run the full pipeline: extract → preprocess → train → save model
4. Run tests
5. Serve the API at http://localhost:8000

---

## API Endpoints

### `GET /health`
```json
{"status": "ok", "model_version": "v1"}
```

### `POST /predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won a free prize. Call now!"}'
```
```json
{"label": "spam", "confidence": 0.9876}
```

### `GET /metrics`
Prometheus metrics endpoint — scraped every 15 seconds.

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.9892 |
| F1 (spam) | 0.9592 |
| Precision | 0.9724 |
| Recall | 0.9463 |
| CV F1 (5-fold) | 0.9587 ± 0.0105 |

---

## Dataset

UCI SMS Spam Collection — 5,572 messages (4,825 ham / 747 spam).
