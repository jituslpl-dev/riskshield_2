"""Application factory with injected services and explicit HTTP contracts."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
import hmac
import json
import logging
import os
import threading
import time
import uuid
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt
from riskshield.domain.policy import CostPolicy
from riskshield.infrastructure.registry import FileModelRegistry, ModelIntegrityError
from riskshield.infrastructure.audit import AuditStore
from riskshield.services.prediction import PredictionService
from riskshield.schemas import Transaction, Loan

logger = logging.getLogger("riskshield")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


@dataclass(frozen=True)
class Settings:
    registry_path: str = "registry"
    audit_path: str = "runtime/audit.sqlite"
    api_key: str = ""
    environment: str = "local"

    @classmethod
    def from_environment(cls):
        return cls(
            os.getenv("RISKSHIELD_REGISTRY", "registry"),
            os.getenv("RISKSHIELD_AUDIT", "runtime/audit.sqlite"),
            os.getenv("RISKSHIELD_API_KEY", ""),
            os.getenv("RISKSHIELD_ENV", "local"),
        )


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[dict[str, StrictFloat | StrictInt]] = Field(min_length=1, max_length=256)
    policy_mode: str = "expected_cost"


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, StrictFloat | StrictInt]


def create_app(settings=None, repository=None, audit=None, policy=None):
    settings = settings or Settings.from_environment()
    if settings.environment == "production" and len(settings.api_key) < 24:
        raise ValueError("Production requires an API key of at least 24 characters.")
    repository = repository or FileModelRegistry(settings.registry_path)
    owns_audit = audit is None
    audit = audit or AuditStore(settings.audit_path)
    policy_path = os.getenv("RISKSHIELD_POLICY", "")
    if policy is None and policy_path:
        from pathlib import Path

        policy = CostPolicy(**json.loads(Path(policy_path).read_text()))
    service = PredictionService(repository, audit, policy or CostPolicy())
    counters = {"requests": 0, "errors": 0, "seconds": 0.0}
    counter_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(application):
        yield
        if owns_audit:
            audit.close()

    app = FastAPI(
        title="RiskShield 2",
        version="2.0.0",
        lifespan=lifespan,
        docs_url=None if settings.environment == "production" else "/docs",
        redoc_url=None if settings.environment == "production" else "/redoc",
    )
    app.state.service = service

    def authorize(x_api_key: str | None = Header(default=None)):
        if settings.api_key and (x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key)):
            raise HTTPException(401, "Invalid API key.")

    @app.middleware("http")
    async def telemetry(request: Request, call_next):
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        response.headers["X-Request-ID"] = request_id
        with counter_lock:
            counters["requests"] += 1
            counters["errors"] += int(response.status_code >= 400)
            counters["seconds"] += duration
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "status": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                }
            )
        )
        return response

    def invoke(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except FileNotFoundError:
            raise HTTPException(503, "Model unavailable; register and promote a trained model.")
        except ModelIntegrityError:
            raise HTTPException(503, "Model integrity verification failed.")
        except ValueError as error:
            raise HTTPException(409 if "Idempotency" in str(error) else 422, str(error))

    @app.get("/live")
    def live():
        return {"status": "alive"}

    @app.get("/ready")
    @app.get("/health")
    def ready():
        names = repository.names()
        if not names:
            raise HTTPException(503, "No active models.")
        for name in names:
            invoke(repository.load, name)
        return {"status": "ready", "models": names}

    @app.get("/metrics", dependencies=[Depends(authorize)], response_class=PlainTextResponse)
    def metrics_endpoint():
        with counter_lock:
            return (
                "\n".join(
                    [
                        f"riskshield_http_requests_total {counters['requests']}",
                        f"riskshield_http_errors_total {counters['errors']}",
                        f"riskshield_http_duration_seconds_sum {counters['seconds']}",
                    ]
                )
                + "\n"
            )

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def models():
        return [invoke(repository.load, name)[1] for name in repository.names()]

    @app.post("/v1/models/{name}/predict", dependencies=[Depends(authorize)])
    def predict(
        name: str, body: ScoreRequest, idempotency_key: str | None = Header(default=None, max_length=128)
    ):
        return invoke(service.score, name, body.rows, idempotency_key, body.policy_mode)

    @app.post("/v1/models/{name}/explain", dependencies=[Depends(authorize)])
    def explain(name: str, body: ExplainRequest):
        return invoke(service.explain, name, body.features)

    @app.post("/predict", dependencies=[Depends(authorize)])
    def fraud_alias(body: Transaction):
        result = invoke(service.score, "fraud_demo", [body.model_dump()], policy_mode="legacy_threshold")
        prediction = result["predictions"][0]
        bundle, _ = invoke(repository.load, "fraud_demo")
        return {
            **prediction,
            "fraud_probability": prediction["probability"],
            "model_version": result["model_version"],
            "decline_threshold": float(bundle["threshold"]),
            "review_threshold": float(bundle.get("review_threshold", bundle["threshold"] / 2)),
        }

    @app.post("/credit/predict", dependencies=[Depends(authorize)])
    def credit_alias(body: Loan):
        result = invoke(service.score, "credit_demo", [body.model_dump()], policy_mode="legacy_threshold")
        prediction = result["predictions"][0]
        bundle, _ = invoke(repository.load, "credit_demo")
        return {
            **prediction,
            "default_probability": prediction["probability"],
            "model_version": result["model_version"],
            "decline_threshold": float(bundle["threshold"]),
            "review_threshold": float(bundle.get("review_threshold", bundle["threshold"] / 2)),
        }

    @app.post("/explain", dependencies=[Depends(authorize)])
    def fraud_explain(body: Transaction):
        return {
            **fraud_alias(body),
            **invoke(service.explain, "fraud_demo", body.model_dump()),
            "explanation_scale": "uncalibrated model probability",
        }

    @app.post("/credit/explain", dependencies=[Depends(authorize)])
    def credit_explain(body: Loan):
        return {
            **credit_alias(body),
            **invoke(service.explain, "credit_demo", body.model_dump()),
            "explanation_scale": "uncalibrated model probability",
        }

    return app


app = create_app()
