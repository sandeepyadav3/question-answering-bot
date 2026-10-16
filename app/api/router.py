from fastapi import APIRouter

from app.api.routes import health, qa
from app.api.routes.documents import questions_router, upload_router

router = APIRouter()
router.include_router(health.router)
router.include_router(qa.router)
router.include_router(upload_router)
router.include_router(questions_router)
