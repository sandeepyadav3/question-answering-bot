import asyncio

from fastapi.routing import APIRoute
from python_multipart.exceptions import MultipartParseError
from starlette.exceptions import HTTPException

from app.core.exceptions import ServiceError


class UploadRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def validate_form(request):
            media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if media_type != "multipart/form-data":
                raise ServiceError(
                    "INVALID_REQUEST", "Use multipart/form-data with two files.", 422
                )
            try:
                async with request.form(max_files=2, max_fields=0) as form:
                    keys = [key for key, _ in form.multi_items()]
                    if sorted(keys) != ["document_file", "questions_file"]:
                        raise ServiceError(
                            "INVALID_REQUEST",
                            "Supply exactly questions_file and document_file.",
                            422,
                        )
                    return await handler(request)
            except (HTTPException, MultipartParseError):
                raise ServiceError("INVALID_REQUEST", "Invalid multipart upload.", 422)

        async def with_deadline(request):
            try:
                async with asyncio.timeout(request.app.state.settings.request_timeout_seconds):
                    return await validate_form(request)
            except TimeoutError:
                raise ServiceError("REQUEST_TIMEOUT", "Request processing deadline exceeded.", 504)

        return with_deadline
