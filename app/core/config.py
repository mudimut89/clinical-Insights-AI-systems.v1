from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Autism Detection System"
    database_url: str = Field(default="sqlite:///./autism.db", validation_alias="DATABASE_URL")

    jwt_secret_key: str = Field(default="CHANGE_ME", validation_alias="JWT_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    access_token_exp_minutes: int = 30
    refresh_token_exp_days: int = 7

    # Initial admin bootstrap
    bootstrap_admin_email: str = Field(default="admin@example.com", validation_alias="BOOTSTRAP_ADMIN_EMAIL")
    bootstrap_admin_password: str = Field(default="admin123", validation_alias="BOOTSTRAP_ADMIN_PASSWORD")

    cors_allow_origins: str = Field(default="http://localhost:5173", validation_alias="CORS_ALLOW_ORIGINS")

    # Paths are resolved relative to the project root by backend code
    model_path: str = "ml/artifacts_tabular/model.joblib"
    schema_path: str = "ml/artifacts_tabular/schema.json"


settings = Settings()
