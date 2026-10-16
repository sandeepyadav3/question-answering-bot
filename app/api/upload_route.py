import asyncio

from fastapi.routing import APIRoute
from python_multipart.exceptions import MultipartParseError
from starlette.exceptions import HTTPException

from app.core.exceptions import ServiceError
from app.dependencies import require_tenant


class UploadRoute(APIRoute):
    expected_files = ("document_file", "questions_file")
    media_message = "Use multipart/form-data with two files."
    file_message = "Supply exactly questions_file and document_file."
    requires_tenant = False

    def get_route_handler(self):
        handler = super().get_route_handler()
        expected = tuple(self.expected_files)

        async def validate_form(request):
            media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if media_type != "multipart/form-data":
                raise ServiceError("INVALID_REQUEST", self.media_message, 422)
            try:
                async with request.form(max_files=len(expected), max_fields=0) as form:
                    keys = [key for key, _ in form.multi_items()]
                    if sorted(keys) != sorted(expected):
                        raise ServiceError("INVALID_REQUEST", self.file_message, 422)
                    return await handler(request)
            except (HTTPException, MultipartParseError):
                raise ServiceError("INVALID_REQUEST", "Invalid multipart upload.", 422)

        async def with_deadline(request):
            try:
                async with asyncio.timeout(request.app.state.settings.request_timeout_seconds):
                    if self.requires_tenant:
                        require_tenant(request)
                    return await validate_form(request)
            except TimeoutError:
                raise ServiceError("REQUEST_TIMEOUT", "Request processing deadline exceeded.", 504)

        return with_deadline


class DocumentUploadRoute(UploadRoute):
    expected_files = ("document_file",)
    media_message = "Use multipart/form-data for the document upload."
    file_message = "Supply exactly document_file."
    requires_tenant = True


class QuestionsUploadRoute(UploadRoute):
    expected_files = ("questions_file",)
    media_message = "Use multipart/form-data for the questions upload."
    file_message = "Supply exactly questions_file."
    requires_tenant = True
