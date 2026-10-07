from starlette.requests import Request

from app.core.exceptions import error_response


class BodyLimitMiddleware:
    """Bound total upload bytes before multipart parsing creates temporary files.

    Buffers at most the configured request limit in memory; replay uses 64 KiB
    frames. Deployment-level admission limits are still needed for many clients.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/api/v1/qa":
            return await self.app(scope, receive, send)
        request = Request(scope)
        length = request.headers.get("content-length")
        if length is not None:
            if not length.isascii() or not length.isdigit():
                response = error_response(
                    request, "INVALID_REQUEST", "Invalid Content-Length.", 400
                )
                return await response(scope, receive, send)
            # Avoid integer conversion of an arbitrarily large header value.
            normalized = length.lstrip("0") or "0"
            if len(normalized) > len(str(self.max_bytes)) or int(normalized) > self.max_bytes:
                return await self.reject(request, scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            data = message.get("body", b"")
            if len(body) + len(data) > self.max_bytes:
                return await self.reject(request, scope, receive, send)
            body.extend(data)
            if not message.get("more_body", False):
                break
        offset = 0

        async def replay():
            nonlocal offset
            if offset > len(body):
                return await receive()
            chunk = bytes(body[offset : offset + 65536])
            offset += len(chunk)
            more = offset < len(body)
            if not more:
                offset = len(body) + 1
            return {"type": "http.request", "body": chunk, "more_body": more}

        await self.app(scope, replay, send)

    async def reject(self, request, scope, receive, send):
        response = error_response(
            request, "REQUEST_TOO_LARGE", f"Request exceeds {self.max_bytes} bytes.", 413
        )
        await response(scope, receive, send)
