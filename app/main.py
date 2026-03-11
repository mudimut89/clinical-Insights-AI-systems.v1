from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .db import Base, engine, SessionLocal
from .models import User, UserRole
from .auth import hash_password
from .routers import auth as auth_router
from .routers import predict as predict_router

app = FastAPI(title=settings.app_name)

origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"] ,
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)

    # Bootstrap initial admin (idempotent)
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == settings.bootstrap_admin_email).first()
        if not existing:
            admin = User(
                email=settings.bootstrap_admin_email,
                hashed_password=hash_password(settings.bootstrap_admin_password),
                role=UserRole.admin,
                is_active=True,
            )
            db.add(admin)
            db.commit()
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(auth_router.router)
app.include_router(predict_router.router)
