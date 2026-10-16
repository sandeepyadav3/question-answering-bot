import asyncio
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import router
from app.core.body_limit import BodyLimitMiddleware
from app.core.config import Settings
from app.core.exceptions import error_response, register_error_handlers
from app.core.logging import RequestMetrics, configure_logging, emit, request_metrics, summary


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    settings = settings if settings is not None else Settings()
    application = FastAPI(title=settings.service_name, version="0.1.0")
    application.state.settings = settings
    application.state.generation_semaphore = asyncio.Semaphore(settings.max_concurrent_generations)
    application.add_middleware(
        BodyLimitMiddleware,
        max_bytes=(
            settings.max_document_bytes
            + settings.max_questions_file_bytes
            + settings.multipart_overhead_bytes
        ),
    )

    @application.middleware("http")
    async def assign_request_id(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        try:
            request_id = UUID(supplied) if len(supplied) <= 36 else uuid4()
        except ValueError:
            request_id = uuid4()
        request.state.request_id = request_id
        token = request_metrics.set(RequestMetrics(str(request_id)))
        started = perf_counter()
        status = 500
        path = request.url.path
        route = (
            path
            if path in {"/health", "/api/v1/qa"} or path.startswith("/api/v1/documents")
            else "other"
        )
        try:
            try:
                response = await call_next(request)
            except Exception:
                emit("request_failed", error_code="INTERNAL_ERROR")
                response = error_response(
                    request, "INTERNAL_ERROR", "Request processing failed.", 500
                )
            status = response.status_code
            response.headers["X-Request-ID"] = str(request_id)
            response.headers["X-Content-Type-Options"] = "nosniff"
            if request.url.path == "/" or request.url.path.startswith("/static/"):
                response.headers["Content-Security-Policy"] = (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                    "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
                )
            return response
        except asyncio.CancelledError:
            status = 499
            raise
        finally:
            emit(
                "request_completed",
                **summary(),
                route=route,
                http_status=status,
                latency_ms=round((perf_counter() - started) * 1000, 3),
            )
            request_metrics.reset(token)

    register_error_handlers(application)
    application.include_router(router)
    static_directory = Path(__file__).resolve().parent / "static"
    application.mount("/static", StaticFiles(directory=static_directory), name="static")

    @application.get("/", include_in_schema=False)
    async def homepage():
        return FileResponse(static_directory / "index.html")

    return application


app = create_app()
