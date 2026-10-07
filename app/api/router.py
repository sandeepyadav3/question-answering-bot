from fastapi import APIRouter

from app.api.routes import health, qa

router = APIRouter()
router.include_router(health.router)
router.include_router(qa.router)
