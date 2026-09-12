"""easy-tdx provider 契约测试。

不依赖真实通达信连接: mock EasyTdxProvider._open_client/_open_mac 返回 pandas
DataFrame 样例, 只验证 Python 侧归一化(符号映射/单位/契约列/空值/异常降级)与数据集声明。
"""
from __future__ import annotations

import contextlib
import datetime as dt

import polars as pl

from app.plugins.easytdx.provider import EasyTdxProvider


def _ctx(obj):
    """把 fake client 对象包装成 contextmanager, 供 _open_client/_open_mac(with 用法) mock。"""
    @contextlib.contextmanager
    def _cm():
        yield obj
    return _cm()


class _Thrower:
    """任意方法调用即抛指定异常, 模拟 client 连接/请求失败。"""
    def __init__(self, msg):
        self._msg = msg

    def __getattr__(self, name):
        def _f(*a, **k):
            raise RuntimeError(self._msg)
        return _f


# ---------- 符号映射 ----------

def test_market_and_code_of_symbol():
    from app.plugins.easytdx.provider import _code_of_symbol, _market_of_symbol, _to_app_symbol
    assert _market_of_symbol("600519.SH") == 1
    assert _market_of_symbol("000001.SZ") == 0
    assert _market_of_symbol("832000.BJ") == 2
    assert _code_of_symbol("600519.SH") == "600519"
    assert _to_app_symbol("600519", 1) == "600519.SH"
    assert _to_app_symbol("000001", 0) == "000001.SZ"


def test_period_from_freq():
    from app.plugins.easytdx.provider import _period_from_freq
    assert _period_from_freq("1m") == 7          # MIN_1
    assert _period_from_freq("5m") == 0          # MIN_5
    assert _period_from_freq("30m") == 2         # MIN_30


# ---------- minute 归一化 ----------

def test_minute_frame_canonical_columns():
    df = pl.DataFrame({
        "datetime": pl.Series([dt.datetime(2026, 9, 2, 9, 30), dt.datetime(2026, 9, 2, 9, 31)]),
        "open": [10.0, 10.1],
        "high": [10.2, 10.2],
        "low": [9.9, 10.0],
        "close": [10.1, 10.15],
        "vol": [1000, 2000],
        "amount": [10100.0, 20300.0],
    })
    from app.plugins.easytdx.provider import _minute_frame
    out = _minute_frame(df, "600519.SH")
    assert out.columns == ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]
    assert out["symbol"][0] == "600519.SH"
    assert out.schema["datetime"] == pl.Datetime("us")
    assert out.schema["volume"] == pl.Float64


def test_minute_frame_vol_renamed_to_volume():
    df = pl.DataFrame({
        "datetime": pl.Series([dt.datetime(2026, 9, 2, 9, 30)]),
        "open": [10.0], "high": [10.2], "low": [9.9], "close": [10.1],
        "vol": [1000], "amount": [10100.0],
    })
    from app.plugins.easytdx.provider import _minute_frame
    out = _minute_frame(df, "000001.SZ")
    assert "volume" in out.columns
    assert "vol" not in out.columns
    assert out["volume"][0] == 1000.0


def test_minute_frame_empty_returns_empty():
    from app.plugins.easytdx.provider import _minute_frame
    assert _minute_frame(pl.DataFrame(), "600519.SH").is_empty()


def test_get_minute_accepts_tz_aware_window(monkeypatch):
    """回归: get_minute 收到 tz-aware 窗口(如 fetch_minute_single 传 CN_TZ 北京时间)
    不得抛 SchemaError(naive datetime vs tz-aware 比较), 且按北京墙钟正确过滤。

    曾因 start/end 是 tz-aware 而 frame.datetime 是 naive, polars is_between 抛
    SchemaError → _try_custom_minute 捕获 → 回退 TickFlow → 免费服务报"不支持K线"。
    """
    p = EasyTdxProvider()

    class FakeMac:
        def get_stock_kline(self, mkt, code, period, start=0, count=800, times=1, adjust=None):
            import pandas as pd
            # 返回北京墙钟 naive 分钟(09:31 交易时段 + 11:40 午休)
            return pd.DataFrame({
                "datetime": pd.to_datetime([
                    "2026-09-03 09:31:00", "2026-09-03 09:32:00", "2026-09-03 11:40:00",
                ]),
                "open": [3.6, 3.61, 3.6], "high": [3.62, 3.62, 3.61],
                "low": [3.59, 3.6, 3.59], "close": [3.61, 3.6, 3.6],
                "vol": [100, 200, 300], "amount": [360.0, 720.0, 1080.0],
            })

    monkeypatch.setattr(p, "_open_mac", lambda: _ctx(FakeMac()))
    # CN_TZ 北京时间 tz-aware 窗口 (09:25 ~ 11:30)
    from app.market_time import CN_TZ
    start = dt.datetime(2026, 9, 3, 9, 25, 0, tzinfo=CN_TZ)
    end = dt.datetime(2026, 9, 3, 11, 30, 0, tzinfo=CN_TZ)
    df = p.get_minute(["601778.SH"], start, end, "stock", "1m")
    # 09:31/09:32 在窗口内, 11:40 午休在窗口外
    assert df.height == 2
    times = [t.strftime("%H:%M") for t in df["datetime"].to_list()]
    assert times == ["09:31:00", "09:32:00"] or times == ["09:31", "09:32"]


# ---------- depth5 归一化 ----------

def test_get_depth_batch_uses_mac_level1(monkeypatch):
    """MAC get_stock_quotes 的买一/卖一价量 → prices/volumes 数组契约。

    五档改走 MAC 一档(标准协议报价命令恒空); 数量单位为手, 且涨停卖一 0 /
    跌停买一 0 必须原样透传 — depth_service 靠 volumes[0]==0 判真封。
    """
    p = EasyTdxProvider()
    seen: dict = {}

    class FakeClient:
        def get_stock_quotes(self, pairs, fields=None):
            import pandas as pd
            seen["pairs"] = pairs
            seen["fields"] = fields
            return pd.DataFrame([
                {"market": 0, "code": "002161", "name": "远望谷",
                 "bid_price": 8.04, "ask_price": 0.0,
                 "bid_volume": 338507, "ask_volume": 0},
                {"market": 1, "code": "600519", "name": "贵州茅台",
                 "bid_price": 1275.16, "ask_price": 1276.0,
                 "bid_volume": 9, "ask_volume": 14},
            ])

    monkeypatch.setattr(p, "_open_mac", lambda: _ctx(FakeClient()))
    out = p.get_depth_batch(["002161.SZ", "600519.SH"])

    from easy_tdx.codec.bitmap import PresetField
    assert seen["pairs"] == [(0, "002161"), (1, "600519")]
    assert seen["fields"] == PresetField.QUOTE
    # 涨停真封: 卖一量 0 透传, 封单量为买一量
    assert out["002161.SZ"]["ask_volumes"] == [0.0]
    assert out["002161.SZ"]["bid_volumes"] == [338507.0]
    assert out["002161.SZ"]["ask_prices"] == [0.0]
    assert out["002161.SZ"]["bid_prices"] == [8.04]
    assert out["002161.SZ"]["timestamp"] is None
    # 普通盘口
    assert out["600519.SH"]["bid_volumes"] == [9.0]
    assert out["600519.SH"]["ask_volumes"] == [14.0]
    assert set(out) == {"002161.SZ", "600519.SH"}


def test_get_depth_batch_chunks_batches(monkeypatch):
    """超过单批上限(80)按批请求并合并结果。"""
    p = EasyTdxProvider()
    calls: list[int] = []

    class FakeClient:
        def get_stock_quotes(self, pairs, fields=None):
            import pandas as pd
            calls.append(len(pairs))
            return pd.DataFrame([
                {"market": mkt, "code": code, "name": "x",
                 "bid_price": 1.0, "ask_price": 1.01, "bid_volume": 1, "ask_volume": 2}
                for mkt, code in pairs
            ])

    monkeypatch.setattr(p, "_open_mac", lambda: _ctx(FakeClient()))
    out = p.get_depth_batch([f"{600000 + i}.SH" for i in range(81)])
    assert calls == [80, 1]
    assert len(out) == 81


def test_get_depth_batch_empty_symbols(monkeypatch):
    p = EasyTdxProvider()
    assert p.get_depth_batch([]) == {}


def test_get_depth_batch_exception_degrades_to_empty(monkeypatch):
    p = EasyTdxProvider()
    monkeypatch.setattr(p, "_open_mac", lambda: _ctx(_Thrower("boom")))
    assert p.get_depth_batch(["600519.SH"]) == {}


def test_get_depth_batch_empty_response(monkeypatch):
    """MAC 返回空表(非交易时段服务器回空)时返回空 dict, 不伪造盘口。"""
    p = EasyTdxProvider()

    class FakeClient:
        def get_stock_quotes(self, pairs, fields=None):
            import pandas as pd
            return pd.DataFrame()

    monkeypatch.setattr(p, "_open_mac", lambda: _ctx(FakeClient()))
    assert p.get_depth_batch(["600519.SH"]) == {}


# ---------- realtime 归一化 ----------

def test_get_realtime_normalizes_units(monkeypatch):
    p = EasyTdxProvider()

    class FakeClient:
        def get_security_quotes(self, pairs):
            import pandas as pd
            return pd.DataFrame([{
                "market": 1, "code": "600519", "name": "贵州茅台",
                "price": 1297.5, "pre_close": 1299.56, "open": 1302.8,
                "high": 1303.0, "low": 1291.2, "vol": 20308, "amount": 2634084352.0,
            }])

    monkeypatch.setattr(p, "_open_client", lambda: _ctx(FakeClient()))
    out = p.get_realtime(symbols=["600519.SH"])
    assert out[0]["symbol"] == "600519.SH"
    assert out[0]["last_price"] == 1297.5
    # change_pct = (price-pre_close)/pre_close, 小数制
    assert abs(out[0]["change_pct"] - (1297.5 - 1299.56) / 1299.56) < 1e-9
    assert out[0]["volume"] == 20308.0


def test_get_realtime_empty_and_exception(monkeypatch):
    p = EasyTdxProvider()
    monkeypatch.setattr(p, "_open_client", lambda: _ctx(_Thrower("boom")))
    assert p.get_realtime(symbols=["600519.SH"]) == []


# ---------- daily 归一化 ----------

def test_get_daily_normalizes(monkeypatch):
    from datetime import date
    p = EasyTdxProvider()

    class FakeClient:
        def get_security_bars(self, market, code, cat, start, count):
            import pandas as pd
            return pd.DataFrame({
                "date": pd.to_datetime(["2026-09-01", "2026-09-02"]),
                "open": [1295.0, 1302.8], "high": [1307.99, 1303.0],
                "low": [1286.1, 1291.2], "close": [1299.56, 1297.5],
                "vol": [3266402, 2030845], "amount": [4242440960.0, 2634084352.0],
            })

    monkeypatch.setattr(p, "_open_client", lambda: _ctx(FakeClient()))
    df = p.get_daily(["600519.SH"], None, None)
    assert df.columns == ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
    assert df.height == 2
    assert df["symbol"].unique().to_list() == ["600519.SH"]
    assert df.schema["date"] == pl.Date
    assert df.schema["close"] == pl.Float64
    assert df["date"][0] == date(2026, 9, 1)


# ---------- adj_factor 归一化 (hfq/none 比值) ----------

def test_get_adj_factors_ratio(monkeypatch):
    p = EasyTdxProvider()

    class FakeMac:
        def get_stock_kline(self, mkt, code, period, start, count, times, adjust):
            import pandas as pd
            dates = ["2025-01-02", "2025-01-03"]
            # adjust=0(NONE), 2(HFQ)
            close = [10.0, 10.0] if adjust == 0 else [10.0, 20.0]
            return pd.DataFrame({"datetime": pd.to_datetime(dates), "close": close})

    monkeypatch.setattr(p, "_open_mac", lambda: _ctx(FakeMac()))
    df = p.get_adj_factors(["600519.SH"], None, None)
    assert df.columns == ["symbol", "trade_date", "ex_factor"]
    assert df.height == 2
    # hfq/none: 2025-01-03 -> 20/10=2.0
    rows = df.sort("trade_date").to_dicts()
    assert abs(rows[1]["ex_factor"] - 2.0) < 1e-9


def test_get_adj_factors_empty_symbols():
    p = EasyTdxProvider()
    assert p.get_adj_factors([], None, None).is_empty()


# ---------- 数据集声明 ----------

def test_datasets_declared():
    p = EasyTdxProvider()
    assert "depth5" in p.config.datasets
    assert "full_minute" in p.config.datasets
    assert "minute" in p.config.datasets
    assert "realtime" in p.config.datasets
    assert "daily" in p.config.datasets
    assert "adj_factor" in p.config.datasets
    assert "financial" not in p.config.datasets


def test_plugin_discovered_in_loader():
    from app.data_providers import custom as cs
    plugins = {p["name"]: p for p in cs.list_plugins()}
    assert "easytdx" in plugins
    assert plugins["easytdx"]["runtime"] == "python"
    assert "daily" in plugins["easytdx"]["datasets"]
    assert "depth5" in plugins["easytdx"]["datasets"]
    assert "full_minute" in plugins["easytdx"]["datasets"]
    assert cs.is_builtin("easytdx")
