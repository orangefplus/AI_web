# 讯飞星辰 MaaS 平台配置
#
# 真实凭据从环境变量读取,默认值留空,方便在 CI / 容器里
# 注入 .env 或 secret manager 的值;本地开发请把 .env.example
# 复制为 .env 并填入自己的 key。
import os


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


# 对话模型 (LLM)
xf_api_key: str = _env("XF_API_KEY", "")
chat_model_name: str = _env("CHAT_MODEL_NAME", "xopqwen36v35b")
xf_chat_base_url: str = _env(
    "XF_CHAT_BASE_URL", "https://maas-api.cn-huabei-1.xf-yun.com/v2"
)

# Embedding 模型
embedding_model_name: str = _env("EMBEDDING_MODEL_NAME", "xop3qwen8bembedding")
xf_embedding_base_url: str = _env(
    "XF_EMBEDDING_BASE_URL", "https://maas-api.cn-huabei-1.xf-yun.com/v2"
)
