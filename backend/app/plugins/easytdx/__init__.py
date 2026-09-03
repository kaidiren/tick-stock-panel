"""easy-tdx 内置数据源插件(通达信协议直连, 免费行情, tickflow 的平替)。"""
from app.plugins.easytdx.provider import EasyTdxProvider, availability

PROVIDER_NAME = "easytdx"

__all__ = [
    "PROVIDER_NAME",
    "EasyTdxProvider",
    "availability",
]
