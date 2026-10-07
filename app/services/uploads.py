from pathlib import PurePath

from starlette.datastructures import UploadFile

from app.core.exceptions import ServiceError


def validate_file_type(upload: UploadFile, *, questions: bool = False) -> str:
    extension = PurePath(upload.filename or "").suffix.lower()
    allowed = {".json"} if questions else {".json", ".pdf"}
    if extension not in allowed:
        raise ServiceError("UNSUPPORTED_FILE_TYPE", "Upload a supported PDF or JSON file.", 415)
    mime = (upload.content_type or "").split(";", 1)[0].strip().lower()
    expected = "application/json" if extension == ".json" else "application/pdf"
    # Browsers and command-line clients may omit a MIME type or use a generic one.
    if mime not in {"", "application/octet-stream", expected}:
        raise ServiceError(
            "UNSUPPORTED_FILE_TYPE", "File extension and content type disagree.", 415
        )
    return extension[1:]


async def read_bounded(upload: UploadFile, limit: int) -> bytes:
    contents = bytearray()
    while True:
        chunk = await upload.read(min(65536, limit + 1 - len(contents)))
        if not chunk:
            break
        contents.extend(chunk)
        if len(contents) > limit:
            raise ServiceError("FILE_TOO_LARGE", f"File exceeds the {limit}-byte limit.", 413)
    if not contents:
        raise ServiceError("EMPTY_FILE", "Uploaded files must not be empty.", 422)
    return bytes(contents)
