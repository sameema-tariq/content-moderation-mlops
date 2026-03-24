"""FastAPI app — serves the spam classification model via REST endpoints."""

from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
from contextlib import asynccontextmanager

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.predictor import Predictor
from app.schemas import PredictionRequest, PredictionResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup and release resources on shutdown."""
    app.state.predictor = Predictor()
    yield


app = FastAPI(title="Content Moderation API", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

Instrumentator().instrument(app).expose(app)
spam_counter = Counter("spam_prediction_total", "Total spam predictions made")
ham_counter = Counter("ham_prediction_total", "Total ham predictions made")


@app.get("/health")
def health(request: Request):
    """Return API status and the currently loaded model version."""
    return {"status": "ok", "model_version": request.app.state.predictor.version}


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
def predict(request: Request, body: PredictionRequest) -> PredictionResponse:
    """Classify input text as spam or ham and return a confidence score."""
    result = request.app.state.predictor.predict(body.text)
    if result["label"] == "spam":
        spam_counter.inc()
    else:
        ham_counter.inc()
    return PredictionResponse(**result)