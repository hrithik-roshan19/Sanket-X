from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    data_dir: Path = Path("data")
    db_path: Path = Path("metadata.db")
    model_dir: Path = Path("data/models")
    raw_upload_dir: Path = Path("data/raw_uploads")
    canonical_dir: Path = Path("data/canonical")
    geo_dir: Path = Path("data/geo")

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    live_ingest_enabled: bool = False
    live_gefs_cycles: str = "00,06,12,18"
    live_gefs_publish_lag_hours: float = 5.5
    # Operational GEFSv12 has 31 members. Historical reforecast training remains 5/11 members.
    live_gefs_members: str = "gec00,gep01,gep02,gep03,gep04"
    # The operational feed supports 31 members, but the current historical models were
    # trained on five. Do not silently change ensemble size: spread is a model feature.
    operational_gefs_member_catalog: str = "gec00," + ",".join(f"gep{i:02d}" for i in range(1, 31))
    live_download_workers: int = 20
    live_obs_provisional_days: int = 14
    live_obs_final_days: int = 21
    live_retrain_min_new_rows: int = 500

    allow_local_retrain: bool = False
    upload_enabled: bool = True
    admin_api_key: str = ""
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = 30
    metrics_enabled: bool = True

    warm_caches_on_startup: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def gefs_cycle_list(self) -> list[str]:
        return [c.strip().zfill(2) for c in self.live_gefs_cycles.split(",") if c.strip()]

    @property
    def gefs_member_list(self) -> list[str]:
        return [m.strip() for m in self.live_gefs_members.split(",") if m.strip()]


settings = Settings()
