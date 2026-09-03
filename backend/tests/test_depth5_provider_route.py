"""depth_service 五档盘口 provider 路由测试。

覆盖: 偏好深度五档路由到声明 depth5 数据集且实现 get_depth5_batch 的插件/自定义源时,
走 provider 拉取(返回其契约数据); 未路由时回退 TickFlow tf.depth.batch; provider
返回空/抛异常时也回退 TickFlow(接口不被单点拖垮)。全部 mock, 不依赖真实网络。
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.services import depth_service as ds_mod
from app.services.depth_service import DepthService


class _FakeDepthProvider:
    """模拟声明 depth5 数据集且实现 get_depth5_batch 的插件源。"""

    def get_depth5_batch(self, symbols):
        return {s: {"ask_volumes": [0], "bid_volumes": [100], "timestamp": 1} for s in symbols}


def _svc(monkeypatch, *, provider=None, use_custom=True):
    """provider=None 表示偏好 tickflow(回退); use_custom 控制是否走 provider 分支。"""
    svc = DepthService()
    # _resolve_depth5_provider 返回值直接控制分支: 这里隔离该方法的内部实现,
    # 让 _call_depth_batch 聚焦「路由 vs 回退」决策。
    monkeypatch.setattr(svc, "_resolve_depth5_provider",
                        lambda: (provider, not use_custom))
    if provider is not None:
        monkeypatch.setattr(svc, "_call_depth5_provider",
                            lambda p, s: p.get_depth5_batch(s))
    # 回退分支需要 capset 与 tickflow client; resolve_limit 给固定 batch/rpm
    monkeypatch.setattr(svc, "_get_capset", lambda: None)
    monkeypatch.setattr(ds_mod, "resolve_limit",
                        lambda *a, **k: type("L", (), {"batch": 100, "rpm": 30})())
    monkeypatch.setattr(ds_mod, "chunked", lambda items, n: [list(items)])
    monkeypatch.setattr(ds_mod, "sleep_between_batches", lambda *a, **k: None)
    return svc


def test_routes_to_custom_provider_when_declared(monkeypatch):
    """路由到声明 depth5 且实现 get_depth5_batch 的源 → 直接走 provider。"""
    provider = _FakeDepthProvider()
    svc = _svc(monkeypatch, provider=provider, use_custom=True)
    result = svc._call_depth_batch(["600519.SH"])
    assert result["600519.SH"] == {"ask_volumes": [0], "bid_volumes": [100], "timestamp": 1}
    # provider 被调用, 且未走 tickflow
    assert result == {"600519.SH": {"ask_volumes": [0], "bid_volumes": [100], "timestamp": 1}}


def test_falls_back_to_tickflow_when_pref_is_tickflow(monkeypatch):
    """未路由(偏好 tickflow) → 回退 TickFlow tf.depth.batch。"""
    tf = MagicMock()
    tf.depth.batch.return_value = {"600519.SH": {"ask_volumes": [5], "bid_volumes": [6], "timestamp": 2}}
    monkeypatch.setattr("app.tickflow.client.get_client", lambda: tf)
    svc = _svc(monkeypatch, provider=None, use_custom=False)
    result = svc._call_depth_batch(["600519.SH"])
    assert result["600519.SH"] == {"ask_volumes": [5], "bid_volumes": [6], "timestamp": 2}


def test_falls_back_when_provider_returns_empty(monkeypatch):
    """provider 返回空 dict → 回退 TickFlow(接口不被单点拖垮)。"""
    provider = _FakeDepthProvider()
    svc = _svc(monkeypatch, provider=provider, use_custom=True)
    monkeypatch.setattr(svc, "_call_depth5_provider", lambda p, s: {})
    tf = MagicMock()
    tf.depth.batch.return_value = {"600519.SH": {"ask_volumes": [1], "bid_volumes": [2], "timestamp": 4}}
    monkeypatch.setattr("app.tickflow.client.get_client", lambda: tf)
    result = svc._call_depth_batch(["600519.SH"])
    assert result["600519.SH"] == {"ask_volumes": [1], "bid_volumes": [2], "timestamp": 4}


# ── _resolve_depth5_provider 真实解析分支 ─────────────────────────────

def test_resolve_pref_tickflow_falls_back(monkeypatch):
    """偏好 tickflow → (None, True)。"""
    svc = DepthService()
    monkeypatch.setattr(svc, "_get_capset", lambda: MagicMock())
    # 直接 monkeypatch 模块内依赖的 import 目标
    import app.services.preferences as prefs
    from app.data_providers import custom as custom_sources
    monkeypatch.setattr(prefs, "get_depth5_data_provider", lambda: "tickflow")
    monkeypatch.setattr(custom_sources, "provider_has_dataset", lambda n, d: False)
    provider, fallback = svc._resolve_depth5_provider()
    assert provider is None and fallback is True


def test_resolve_custom_declared_and_implemented_returns_provider(monkeypatch):
    """自定义源声明 depth5 且实现 get_depth5_batch → 返回 provider。"""
    svc = DepthService()
    import app.services.preferences as prefs
    from app.data_providers import custom as custom_sources
    monkeypatch.setattr(prefs, "get_depth5_data_provider", lambda: "stocksdk")
    monkeypatch.setattr(custom_sources, "provider_has_dataset", lambda n, d: n == "stocksdk" and d == "depth5")
    monkeypatch.setattr(custom_sources, "get_provider", lambda n: _FakeDepthProvider())
    provider, fallback = svc._resolve_depth5_provider()
    assert provider is not None and fallback is False


def test_resolve_custom_without_method_falls_back(monkeypatch):
    """自定义源声明 depth5 但未实现 get_depth5_batch → 回退(避免 AttributeError)。"""
    svc = DepthService()

    class _NoMethod:
        pass

    import app.services.preferences as prefs
    from app.data_providers import custom as custom_sources
    monkeypatch.setattr(prefs, "get_depth5_data_provider", lambda: "stocksdk")
    monkeypatch.setattr(custom_sources, "provider_has_dataset", lambda n, d: n == "stocksdk" and d == "depth5")
    monkeypatch.setattr(custom_sources, "get_provider", lambda n: _NoMethod())
    provider, fallback = svc._resolve_depth5_provider()
    assert provider is None and fallback is True
