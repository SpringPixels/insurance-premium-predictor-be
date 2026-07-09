from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

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

# Include routers
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(admin.router)
app.include_router(payments.router)
app.include_router(activity.router)
app.include_router(segmentation.router)
app.include_router(contact.router)