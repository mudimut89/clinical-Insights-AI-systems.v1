from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, EmailStr, Field


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    refresh_token: str


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    role: str = "clinician"


class UserOut(BaseModel):
    id: int
    email: EmailStr
    role: str
    is_active: bool


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class PredictRequest(BaseModel):
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    label: str
    prediction: int
    proba_asd: float
    confidence: float
    probabilities: Dict[str, float] = {}
    details: Optional[Dict[str, Any]] = None
