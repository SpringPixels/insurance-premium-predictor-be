from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
import traceback

from config.database import Base, engine
from routers import health, auth, predictions, admin, payments, activity, segmentation, contact

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This securely runs your table creation asynchronously on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

# Pass the lifespan function to your FastAPI instance
app = FastAPI(lifespan=lifespan)

# Define the origins that are allowed to make requests to your API
origins = [
    "http://localhost:4200",  # Default Angular dev server port
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Global exception handlers ---

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Return all HTTP errors as { detail: '...' } JSON consistently."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail or "An error occurred."},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert Pydantic validation errors into readable messages."""
    messages = []
    for error in exc.errors():
        field = " → ".join(str(loc) for loc in error.get("loc", [])[1:])  # skip 'body'
        msg = error.get("msg", "Invalid value")
        messages.append(f"{field}: {msg}" if field else msg)
    detail = "; ".join(messages) if messages else "Invalid request data."
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": detail},
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions, log them, and return a safe 500 message."""
    print(f"[UNHANDLED ERROR] {request.method} {request.url}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected server error occurred. Please try again later."},
    )

# Include routers
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(admin.router)
app.include_router(payments.router)
app.include_router(activity.router)
app.include_router(segmentation.router)
app.include_router(contact.router)