import os


DEFAULT_LOCAL_QWEN3_API_BASE = "http://127.0.0.1:8000/v1"
LOCAL_QWEN3_API_BASE_ENV_VARS = (
    "GEPA_LOCAL_QWEN3_API_BASE",
    "LOCAL_QWEN3_API_BASE",
)
OPENAI_COMPATIBLE_API_BASE_ENV_VARS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
)
LOCAL_QWEN3_ENABLE_THINKING_ENV_VARS = (
    "GEPA_LOCAL_QWEN3_ENABLE_THINKING",
    "LOCAL_QWEN3_ENABLE_THINKING",
)


def _first_non_empty_env(var_names: tuple[str, ...]) -> str | None:
    for env_var in var_names:
        value = os.environ.get(env_var)
        if value and value.strip():
            return value.strip()
    return None


def get_local_qwen3_api_base() -> str:
    return _first_non_empty_env(LOCAL_QWEN3_API_BASE_ENV_VARS) or DEFAULT_LOCAL_QWEN3_API_BASE


def get_default_openai_compatible_api_base() -> str | None:
    return _first_non_empty_env(OPENAI_COMPATIBLE_API_BASE_ENV_VARS)


def get_local_qwen3_enable_thinking(default: bool = False) -> bool:
    value = _first_non_empty_env(LOCAL_QWEN3_ENABLE_THINKING_ENV_VARS)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_api_base_for_model(model_name: str, api_base: str | None = None) -> str:
    if api_base and api_base.strip():
        return api_base.strip()

    normalized = model_name.strip().lower()
    if normalized in {"qwen3-8b", "openai/qwen3-8b"}:
        return get_local_qwen3_api_base()

    return get_default_openai_compatible_api_base() or get_local_qwen3_api_base()


def resolve_extra_body_for_model(model_name: str) -> dict:
    normalized = model_name.strip().lower()
    if normalized in {"qwen3-8b", "openai/qwen3-8b"}:
        return {"chat_template_kwargs": {"enable_thinking": get_local_qwen3_enable_thinking(default=False)}}
    return {}
