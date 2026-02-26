"""OptiMark - FastAPI main application."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import engine, Base
from app.routers import auth, exams, scan, subscription, admin
from app.api.v1.router import router as v1_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: create tables on startup."""
    logger.info("Starting OptiMark API")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Compatibility guard for older DBs that were created before v1 columns.
        # This keeps legacy/manual auth working even before full Alembic migration.
        await conn.exec_driver_sql(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS google_sub VARCHAR(255)"
        )
        await conn.exec_driver_sql(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE"
        )
        await conn.exec_driver_sql(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP"
        )
        await conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_google_sub ON users (google_sub)"
        )
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.STORAGE_ROOT).mkdir(parents=True, exist_ok=True)
    yield
    await engine.dispose()
    logger.info("OptiMark API shutdown complete")


app = FastAPI(
    title="OptiMark",
    description="Automated OMR Grading System API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - required for frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return stable 500 response."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    """Preserve explicit HTTP exceptions with their original status/detail."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    """Return consistent 422 payload or 404 for invalid path IDs."""
    errors = exc.errors()
    first_error = errors[0] if errors else {}
    loc = first_error.get("loc", [])
    
    # Special case: if someone visits /v1/exams/<uuid> (legacy URL),
    # FastAPI path parser fails because it expects an int. Return 404.
    if len(loc) >= 2 and loc[0] == "path" and loc[1] == "exam_id" and first_error.get("type") == "int_parsing":
        return JSONResponse(status_code=404, content={"detail": "Exam not found"})

    loc_path = ".".join(str(p) for p in loc if p not in {"body", "query", "path"})
    msg = first_error.get("msg", "Validation failed")
    detail = f"{loc_path}: {msg}" if loc_path else msg
    return JSONResponse(
        status_code=422,
        content={
            "detail": detail,
            "errors": errors,
        },
    )

# Mount uploads for serving scanned images (optional)
upload_path = Path(settings.UPLOAD_DIR)
upload_path.mkdir(parents=True, exist_ok=True)
app.mount(f"/{settings.UPLOAD_DIR}", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Include routers (mounted at /api for frontend compatibility)
app.include_router(auth.router, prefix="/api")
app.include_router(exams.router, prefix="/api")
app.include_router(scan.router, prefix="/api")
app.include_router(subscription.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(v1_router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "OptiMark OMR Grading API", "status": "ok"}
