from __future__ import annotations

import os
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

ReasoningMode = Literal["fast", "balanced", "deep"]
DatabaseDriver = Literal["sqlite", "postgres"]
AuthMode = Literal["api_key", "jwt"]
ModelBackend = Literal["heuristic", "api_llm", "local_llama", "hybrid"]


class ReasoningModeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reflection_pass: bool
    reasoning_depth: int = Field(ge=1, le=12)
    max_tokens: int = Field(ge=64, le=8192)
    temperature: float = Field(ge=0.0, le=2.0)


MODE_CONFIGS: dict[ReasoningMode, ReasoningModeConfig] = {
    "fast": ReasoningModeConfig(
        reflection_pass=False,
        reasoning_depth=1,
        max_tokens=256,
        temperature=0.2,
    ),
    "balanced": ReasoningModeConfig(
        reflection_pass=True,
        reasoning_depth=3,
        max_tokens=512,
        temperature=0.4,
    ),
    "deep": ReasoningModeConfig(
        reflection_pass=True,
        reasoning_depth=6,
        max_tokens=1024,
        temperature=0.7,
    ),
}


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    app_name: str = "Humoniod AI"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    reasoning_mode: ReasoningMode = "balanced"
    model_timeout_seconds: float = Field(default=4.0, gt=0.0, le=30.0)
    model_max_retries: int = Field(default=2, ge=0, le=5)
    model_backend: ModelBackend = "heuristic"
    local_model_path: str = ""
    local_model_threads: int = Field(default=4, ge=1, le=64)
    local_model_context_size: int = Field(default=2048, ge=256, le=32768)
    local_model_max_tokens: int = Field(default=512, ge=32, le=8192)
    internet_lookup_enabled: bool = False
    internet_lookup_timeout_seconds: float = Field(default=3.0, gt=0.2, le=15.0)
    internet_lookup_max_results: int = Field(default=3, ge=1, le=8)
    internet_lookup_max_chars: int = Field(default=420, ge=120, le=2000)
    memory_top_k: int = Field(default=3, ge=1, le=20)
    memory_max_summary_tokens: int = Field(default=120, ge=20, le=2000)
    memory_context_max_tokens: int = Field(default=220, ge=40, le=4000)
    memory_long_term_every_n_messages: int = Field(default=5, ge=2, le=100)
    privacy_redaction_enabled: bool = True
    privacy_remove_sensitive_context_keys: bool = True
    relationship_memory_text_enabled: bool = False
    database_driver: DatabaseDriver = "sqlite"
    memory_db_path: str = "data/humoniod_memory.db"
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/humoniod_ai"
    embedding_provider: str = "sentence_transformers"
    embedding_dim: int = Field(default=384, ge=8, le=4096)
    embedding_enabled: bool = True
    model_routing_enabled: bool = True
    self_evaluation_enabled: bool = True
    tool_execution_enabled: bool = True
    tool_max_timeout_seconds: float = Field(default=2.0, gt=0.05, le=15.0)
    humanoid_mode_enabled: bool = True
    strict_response_mode: bool = True
    default_answer_style: str = "factual"
    expose_full_prompt_in_response: bool = False
    response_match_model_enabled: bool = False
    response_match_model_name: str = "fast_model"
    response_match_model_timeout_seconds: float = Field(default=1.2, gt=0.05, le=10.0)
    response_match_model_max_retries: int = Field(default=0, ge=0, le=2)
    response_match_model_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    telemetry_enabled: bool = True
    analyzer_enabled: bool = True
    analyzer_file_path: str = "data/runtime_analyzer.jsonl"
    analyzer_poll_seconds: float = Field(default=2.0, ge=0.25, le=60.0)
    analyzer_request_delta_threshold: int = Field(default=5, ge=1, le=1000)
    analyzer_latency_delta_ms: int = Field(default=40, ge=1, le=5000)
    analyzer_max_file_bytes: int = Field(default=524288, ge=10240, le=10485760)
    analyzer_max_records: int = Field(default=800, ge=20, le=100000)
    auth_enabled: bool = False
    auth_mode: AuthMode = "api_key"
    auth_api_key_header: str = "x-api-key"
    auth_api_key: str = "dev-api-key"
    rate_limit_enabled: bool = False
    requests_per_minute: int = Field(default=60, ge=1, le=5000)

    @property
    def reasoning_profile(self) -> ReasoningModeConfig:
        return MODE_CONFIGS[self.reasoning_mode]


def get_settings() -> Settings:
    raw_mode = os.getenv("REASONING_MODE", "balanced").lower()
    reasoning_mode: ReasoningMode = "balanced"
    if raw_mode in MODE_CONFIGS:
        reasoning_mode = cast(ReasoningMode, raw_mode)
    raw_db_driver = os.getenv("DATABASE_DRIVER", "sqlite").lower()
    database_driver: DatabaseDriver = "sqlite"
    if raw_db_driver in {"sqlite", "postgres"}:
        database_driver = cast(DatabaseDriver, raw_db_driver)
    raw_auth_mode = os.getenv("AUTH_MODE", "api_key").lower()
    auth_mode: AuthMode = "api_key"
    if raw_auth_mode in {"api_key", "jwt"}:
        auth_mode = cast(AuthMode, raw_auth_mode)
    raw_model_backend = os.getenv("MODEL_BACKEND", "heuristic").lower()
    model_backend: ModelBackend = "heuristic"
    if raw_model_backend in {"heuristic", "api_llm", "local_llama", "hybrid"}:
        model_backend = cast(ModelBackend, raw_model_backend)

    return Settings(
        app_name=os.getenv("APP_NAME", "Humoniod AI"),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        reasoning_mode=reasoning_mode,
        model_timeout_seconds=_parse_float("MODEL_TIMEOUT_SECONDS", 4.0),
        model_max_retries=_parse_int("MODEL_MAX_RETRIES", 2),
        model_backend=model_backend,
        local_model_path=os.getenv("LOCAL_MODEL_PATH", ""),
        local_model_threads=_parse_int("LOCAL_MODEL_THREADS", 4),
        local_model_context_size=_parse_int("LOCAL_MODEL_CONTEXT_SIZE", 2048),
        local_model_max_tokens=_parse_int("LOCAL_MODEL_MAX_TOKENS", 512),
        internet_lookup_enabled=_parse_bool("INTERNET_LOOKUP_ENABLED", False),
        internet_lookup_timeout_seconds=_parse_float("INTERNET_LOOKUP_TIMEOUT_SECONDS", 3.0),
        internet_lookup_max_results=_parse_int("INTERNET_LOOKUP_MAX_RESULTS", 3),
        internet_lookup_max_chars=_parse_int("INTERNET_LOOKUP_MAX_CHARS", 420),
        memory_top_k=_parse_int("MEMORY_TOP_K", 3),
        memory_max_summary_tokens=_parse_int("MEMORY_MAX_SUMMARY_TOKENS", 120),
        memory_context_max_tokens=_parse_int("MEMORY_CONTEXT_MAX_TOKENS", 220),
        memory_long_term_every_n_messages=_parse_int("MEMORY_LONG_TERM_EVERY_N_MESSAGES", 5),
        privacy_redaction_enabled=_parse_bool("PRIVACY_REDACTION_ENABLED", True),
        privacy_remove_sensitive_context_keys=_parse_bool("PRIVACY_REMOVE_SENSITIVE_CONTEXT_KEYS", True),
        relationship_memory_text_enabled=_parse_bool("RELATIONSHIP_MEMORY_TEXT_ENABLED", False),
        database_driver=database_driver,
        memory_db_path=os.getenv("MEMORY_DB_PATH", "data/humoniod_memory.db"),
        postgres_dsn=os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/humoniod_ai"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"),
        embedding_dim=_parse_int("EMBEDDING_DIM", 384),
        embedding_enabled=_parse_bool("EMBEDDING_ENABLED", True),
        model_routing_enabled=_parse_bool("MODEL_ROUTING_ENABLED", True),
        self_evaluation_enabled=_parse_bool("SELF_EVALUATION_ENABLED", True),
        tool_execution_enabled=_parse_bool("TOOL_EXECUTION_ENABLED", True),
        tool_max_timeout_seconds=_parse_float("TOOL_MAX_TIMEOUT_SECONDS", 2.0),
        humanoid_mode_enabled=_parse_bool("HUMANOID_MODE_ENABLED", True),
        strict_response_mode=_parse_bool("STRICT_RESPONSE_MODE", True),
        default_answer_style=os.getenv("DEFAULT_ANSWER_STYLE", "factual"),
        expose_full_prompt_in_response=_parse_bool("EXPOSE_FULL_PROMPT_IN_RESPONSE", False),
        response_match_model_enabled=_parse_bool("RESPONSE_MATCH_MODEL_ENABLED", False),
        response_match_model_name=os.getenv("RESPONSE_MATCH_MODEL_NAME", "fast_model"),
        response_match_model_timeout_seconds=_parse_float("RESPONSE_MATCH_MODEL_TIMEOUT_SECONDS", 1.2),
        response_match_model_max_retries=_parse_int("RESPONSE_MATCH_MODEL_MAX_RETRIES", 0),
        response_match_model_threshold=_parse_float("RESPONSE_MATCH_MODEL_THRESHOLD", 0.45),
        telemetry_enabled=_parse_bool("TELEMETRY_ENABLED", True),
        analyzer_enabled=_parse_bool("ANALYZER_ENABLED", True),
        analyzer_file_path=os.getenv("ANALYZER_FILE_PATH", "data/runtime_analyzer.jsonl"),
        analyzer_poll_seconds=_parse_float("ANALYZER_POLL_SECONDS", 2.0),
        analyzer_request_delta_threshold=_parse_int("ANALYZER_REQUEST_DELTA_THRESHOLD", 5),
        analyzer_latency_delta_ms=_parse_int("ANALYZER_LATENCY_DELTA_MS", 40),
        analyzer_max_file_bytes=_parse_int("ANALYZER_MAX_FILE_BYTES", 524288),
        analyzer_max_records=_parse_int("ANALYZER_MAX_RECORDS", 800),
        auth_enabled=_parse_bool("AUTH_ENABLED", False),
        auth_mode=auth_mode,
        auth_api_key_header=os.getenv("AUTH_API_KEY_HEADER", "x-api-key"),
        auth_api_key=os.getenv("AUTH_API_KEY", "dev-api-key"),
        rate_limit_enabled=_parse_bool("RATE_LIMIT_ENABLED", False),
        requests_per_minute=_parse_int("REQUESTS_PER_MINUTE", 60),
    )


def _parse_float(env_name: str, default: float) -> float:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_int(env_name: str, default: int) -> int:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_bool(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
