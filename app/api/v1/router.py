"""v1 API router."""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, exams, files, plans, profile, results, scan_jobs, subscriptions, wallet

router = APIRouter(prefix="/v1")
router.include_router(auth.router, prefix="/auth", tags=["v1-auth"])
router.include_router(profile.router, prefix="/profile", tags=["v1-profile"])
router.include_router(plans.router, prefix="/plans", tags=["v1-plans"])
router.include_router(subscriptions.router, prefix="/subscriptions", tags=["v1-subscriptions"])
router.include_router(wallet.router, prefix="/wallet", tags=["v1-wallet"])
router.include_router(exams.router, prefix="", tags=["v1-exams"])
router.include_router(scan_jobs.router, prefix="", tags=["v1-scan"])
router.include_router(results.router, prefix="", tags=["v1-results"])
router.include_router(files.router, prefix="/files", tags=["v1-files"])
