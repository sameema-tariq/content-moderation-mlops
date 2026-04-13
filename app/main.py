"""FastAPI app — serves the spam classification model via REST endpoints (API + minimal UI)."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.logger import get_logger
from app.predictor import Predictor
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    PredictionRequest,
    PredictionResponse,
)
from config.settings import TRUSTED_PROXY_IPS, TRUST_PROXY_HEADERS

logger = get_logger(__name__)

UI_DIR = Path(__file__).resolve().parent / "ui"
templates = Jinja2Templates(directory=str(UI_DIR / "templates"))


def _wants_html(request: Request) -> bool:
    if request.headers.get("hx-request", "").lower() == "true":
        return True
    accept = request.headers.get("accept", "")
    return "text/html" in accept


def _json_error(code: str, message: str, details=None) -> dict:
    return ErrorResponse(error={"code": code, "message": message, "details": details}).model_dump()


def _client_ip_for_rate_limit(request: Request) -> str:
    """
    Rate-limit key based on client IP, with safe proxy support.

    Only trusts X-Forwarded-For/X-Real-IP if enabled and the immediate peer is a trusted proxy.
    """

    peer_ip = getattr(getattr(request, "client", None), "host", None) or ""

    if TRUST_PROXY_HEADERS and peer_ip in TRUSTED_PROXY_IPS:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            forwarded = xff.split(",")[0].strip()
            if forwarded:
                return forwarded
        xri = request.headers.get("x-real-ip", "").strip()
        if xri:
            return xri

    return peer_ip or "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup and release resources on shutdown."""
    try:
        app.state.predictor = Predictor()
    except FileNotFoundError as exc:
        app.state.predictor = None
        logger.warning(f"Model not loaded on startup: {exc}")
    yield


app = FastAPI(title="Content Moderation API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")

limiter = Limiter(key_func=_client_ip_for_rate_limit if TRUST_PROXY_HEADERS else get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    if _wants_html(request):
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "title": "Request failed", "detail": str(exc.detail)},
            status_code=exc.status_code,
        )
    return JSONResponse(
        status_code=exc.status_code,
        content=_json_error("http_error", str(exc.detail)),
    )


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    if _wants_html(request):
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "title": "Validation error", "detail": "Please check your input."},
            status_code=422,
        )
    return JSONResponse(
        status_code=422,
        content=_json_error("validation_error", "Validation error", details=exc.errors()),
    )


@app.exception_handler(RateLimitExceeded)
def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    if _wants_html(request):
        return templates.TemplateResponse(
            "partials/error.html",
            {
                "request": request,
                "title": "Rate limited",
                "detail": "Too many requests. Please wait and try again.",
            },
            status_code=429,
        )
    return JSONResponse(status_code=429, content=_json_error("rate_limited", "Rate limit exceeded"))


SAMPLE_TEXT = "Congratulations! You won a free prize. Call now!"


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Minimal UI (server-rendered) for trying predictions."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Text Moderation Console", "sample_text": SAMPLE_TEXT},
    )


@app.get("/ui/status", response_class=HTMLResponse)
def ui_status(request: Request):
    """HTMX fragment: show current API/model status."""
    predictor = request.app.state.predictor
    return templates.TemplateResponse(
        "partials/status.html",
        {
            "request": request,
            "model_loaded": predictor is not None,
            "model_version": getattr(predictor, "version", None),
        },
    )


@app.post("/ui/predict", response_class=HTMLResponse)
@limiter.limit("10/minute")
def ui_predict(request: Request, text: str = Form(...)) -> HTMLResponse:
    """HTMX endpoint: runs prediction and returns an HTML fragment."""
    predictor = request.app.state.predictor
    if predictor is None:
        return templates.TemplateResponse(
            "partials/error.html",
            {
                "request": request,
                "title": "Service unavailable",
                "detail": "Model not loaded. Train and save the model first.",
            },
            status_code=503,
        )

    cleaned = (text or "").strip()
    if not cleaned:
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "title": "Invalid input", "detail": "Please enter some text."},
            status_code=422,
        )

    result = predictor.predict(cleaned)
    return templates.TemplateResponse(
        "partials/prediction.html",
        {
            "request": request,
            "label": result["label"],
            "confidence": result["confidence"],
            "model_version": getattr(predictor, "version", None),
        },
    )


@app.get("/health")
def health(request: Request):
    """Liveness probe: API process is up."""
    predictor = request.app.state.predictor
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "model_version": getattr(predictor, "version", None),
    }


@app.get("/ready")
def ready(request: Request):
    """Readiness probe: model is loaded and /predict can serve traffic."""
    if request.app.state.predictor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Train and save the model first."
        )
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
def predict(request: Request, body: PredictionRequest) -> PredictionResponse:
    """Classify input text as spam or ham and return a confidence score."""
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Train and save the model first."
        )

    result = predictor.predict(body.text)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
@limiter.limit("10/minute")
def predict_batch(request: Request, body: BatchPredictionRequest) -> BatchPredictionResponse:
    """Classify multiple inputs in one request. Returns predictions in input order."""
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Train and save the model first."
        )

    predictions: list[PredictionResponse] = []
    for text in body.texts:
        result = predictor.predict(text)
        predictions.append(PredictionResponse(**result))

    return BatchPredictionResponse(count=len(predictions), predictions=predictions)

