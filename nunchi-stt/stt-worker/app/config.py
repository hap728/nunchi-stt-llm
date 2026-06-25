from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_name: str
    db_username: str
    db_password: str
    ncp_endpoint: str
    ncp_region: str
    ncp_bucket: str
    ncp_access_key: str
    ncp_secret_key: str
    whisper_model_size: str
    whisper_model_dir: str
    whisper_device: str
    whisper_compute_type: str
    whisper_batch_size: int
    ffmpeg_enable_loudnorm: bool
    text_normalization_enabled: bool
    text_normalization_terms_path: str
    worker_poll_interval_sec: int
    worker_batch_size: int
    worker_enable_polling: bool
    llm_schedule_enabled: bool
    ollama_host: str
    llm_model: str
    llm_reference_date: str
    llm_request_timeout_sec: int
    schedule_results_table: str


def load_settings() -> Settings:
    return Settings(
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=int(os.getenv("DB_PORT", "3306")),
        db_name=os.getenv("DB_NAME", ""),
        db_username=os.getenv("DB_USERNAME", ""),
        db_password=os.getenv("DB_PASSWORD", ""),
        ncp_endpoint=os.getenv("NCP_ENDPOINT", "https://kr.object.ncloudstorage.com"),
        ncp_region=os.getenv("NCP_REGION", "kr-standard"),
        ncp_bucket=os.getenv("NCP_BUCKET", "voicebucket"),
        ncp_access_key=os.getenv("NCP_ACCESS_KEY", ""),
        ncp_secret_key=os.getenv("NCP_SECRET_KEY", ""),
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "medium"),
        whisper_model_dir=os.getenv("WHISPER_MODEL_DIR", "/models/faster-whisper"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_batch_size=int(os.getenv("WHISPER_BATCH_SIZE", "2")),
        ffmpeg_enable_loudnorm=os.getenv("FFMPEG_ENABLE_LOUDNORM", "false").lower() == "true",
        text_normalization_enabled=os.getenv("TEXT_NORMALIZATION_ENABLED", "true").lower() == "true",
        text_normalization_terms_path=os.getenv(
            "TEXT_NORMALIZATION_TERMS_PATH",
            str(ROOT_DIR / "app" / "domain_terms.tsv"),
        ),
        worker_poll_interval_sec=int(os.getenv("WORKER_POLL_INTERVAL_SEC", "10")),
        worker_batch_size=int(os.getenv("WORKER_BATCH_SIZE", "3")),
        worker_enable_polling=os.getenv("WORKER_ENABLE_POLLING", "true").lower() == "true",
        llm_schedule_enabled=os.getenv("LLM_SCHEDULE_ENABLED", "true").lower() == "true",
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        llm_model=os.getenv("LLM_MODEL", "exaone3.5:7.8b"),
        llm_reference_date=os.getenv("LLM_REFERENCE_DATE", os.getenv("REFERENCE_DATE", "")),
        llm_request_timeout_sec=int(os.getenv("LLM_REQUEST_TIMEOUT_SEC", "300")),
        schedule_results_table=os.getenv("SCHEDULE_RESULTS_TABLE", "schedule_results"),
    )
