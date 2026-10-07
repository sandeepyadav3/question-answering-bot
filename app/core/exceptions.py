from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.schemas.responses import ErrorResponse, RequestError


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def error_response(request: Request, code: str, message: str, status_code: int) -> JSONResponse:
    body = ErrorResponse(
        error=RequestError(code=code, message=message, request_id=request.state.request_id)
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
        return error_response(request, exc.code, exc.message, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Avoid echoing request contents in validation errors.
        return error_response(request, "INVALID_REQUEST", "Request validation failed.", 422)

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        response = error_response(request, "HTTP_ERROR", str(exc.detail), exc.status_code)
        if exc.headers:
            response.headers.update(exc.headers)
        return response
