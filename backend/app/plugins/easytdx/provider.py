"""easy-tdx 内置数据源 provider。

通过通达信行情服务器协议直连(免费、免Key、免注册)抓取 A 股数据, 归一化到项目
内部 schema。方法签名对齐 custom.GenericHTTPProvider 与 StockSDKProvider
(service 分流点按这套签名调用), 注入 custom loader 注册表后各 service 无需改动
即可路由到本 provider。

数据契约(与本项目标准化一致):
  - daily  -> normalize_daily (symbol/date/open/high/low/close/volume/amount)
  - adj_factor -> normalize_adj_factors (symbol/trade_date/ex_factor)
  - minute -> [symbol, datetime(北京墙钟 naive), open, high, low, close, volume, amount]
  - realtime -> list[dict] record (last_price/prev_close/open/high/low/volume/amount/change_pct/timestamp)
  - depth5 -> {appSymbol: {ask_volumes, bid_volumes, timestamp}} (depth_service sealed 契约)
  - full_minute -> get_intraday_batch 当日窗口批量分钟 (与 minute 同形)

合规提示: 通达信协议直连第三方行情服务器, 未经正式授权, 存在行情版权与反爬风险。
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime

import polars as pl

from app.data_providers.normalizer import normalize_adj_factors, normalize_daily

logger = logging.getLogger(__name__)

# easy-tdx 支持的数据集(financial 不支持 → 不声明, 自动回退 tickflow)
_DATASETS = ("daily", "adj_factor", "realtime", "minute", "full_minute", "depth5")

# 分钟 canonical 列(与 stock-sdk _MINUTE_CANONICAL 一致)
_MINUTE_CANONICAL = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]

# 全市场标的列表分页大小(通达信每页上限)
_LIST_PAGE = 1000

# 单次桥接的符号批大小(标准协议批量报价上限 80)
_QUOTE_BATCH = 80

# 日K单页条数上限(easy-tdx get_security_bars count>800 返回空)与深度上限
_DAILY_PAGE = 800
_DAILY_MAX_BARS = 20000


@dataclass
class _EasyTdxConfig:
    """轻量 config shim, 让 custom loader 的 list_sources/provider_has_dataset 能识别本 provider。"""

    name: str = "easytdx"
    display_name: str = "easy-tdx (通达信协议直连)"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


def _market_of_symbol(symbol: str) -> int:
    """app symbol(600519.SH/000001.SZ) → easy-tdx market 枚举值(SH=1/SZ=0/BJ=2)。"""
    suffix = (symbol.split(".")[-1] if "." in symbol else "").upper()
    if suffix.startswith("SH"):
        return 1
    if suffix.startswith("BJ"):
        return 2
    return 0  # 深证/其他


def _code_of_symbol(symbol: str) -> str:
    """app symbol → 纯代码(600519.SH→600519)。"""
    return symbol.split(".")[0]


def _to_app_symbol(code: str, market: int) -> str:
    """code + market → app symbol(600519.SH)。"""
    suffix = {1: "SH", 0: "SZ", 2: "BJ"}.get(int(market), "")
    return f"{code}.{suffix}" if suffix else code


class EasyTdxProvider:
    """内置 easy-tdx 数据源。"""

    name = "easytdx"
    builtin = True

    def __init__(self) -> None:
        self.config = _EasyTdxConfig()
        self._client = None

    def close(self) -> None:  # loader.load_all 会对每个 provider 调 close
        import contextlib
        c, self._client = self._client, None
        if c is not None:
            with contextlib.suppress(Exception):
                c.close()

    # ---- 客户端惰性初始化 ----
    def _get_client(self):
        """返回唯一的 TdxClient(标准协议: 五档/日K/除权/标的列表)。懒加载。"""
        if self._client is None:
            from easy_tdx import TdxClient
            self._client = TdxClient.from_best_host()
        return self._client

    def _get_mac_client(self):
        """返回 MacClient(MAC 协议: 分钟K/分时/全市场报价)。懒加载(独立连接)。"""
        from easy_tdx.mac.client import MacClient
        return MacClient.from_best_host()

    # ---- availability(探活) ----
    def _ping(self) -> bool:
        from easy_tdx import TdxClient
        try:
            with TdxClient.from_best_host() as c:
                c.get_security_count(1)  # SH 证券数
                return True
        except Exception:
            return False

    # ---- daily ----
    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame()
        from easy_tdx import KlineCategory

        frames: list[pl.DataFrame] = []
        total = len(symbols)
        for i, sym in enumerate(symbols):
            try:
                c = self._get_client()
                # get_security_bars 单次 count 上限 800 (更大返回空), 超出按 800 分批拉取
                for start in range(0, _DAILY_MAX_BARS, _DAILY_PAGE):
                    bars = c.get_security_bars(
                        _market_of_symbol(sym), _code_of_symbol(sym),
                        KlineCategory.DAY, start, _DAILY_PAGE,
                    )
                    if bars is None or len(bars) == 0:
                        break
                    df = normalize_daily(bars.to_dict(orient="records"), default_symbol=sym, source=self.name)
                    if not df.is_empty():
                        frames.append(df)
                    if len(bars) < _DAILY_PAGE:
                        break
            except Exception as e:
                logger.debug("easy-tdx daily 拉取失败(%s): %s", sym, e)
                continue
            if on_chunk_done:
                on_chunk_done(i + 1, total)
        if not frames:
            return pl.DataFrame()
        out = pl.concat(frames, how="diagonal_relaxed")
        if (start_time is not None and end_time is not None):
            out = out.filter(pl.col("date").is_between(start_time.date(), end_time.date()))
        return out

    # ---- adj_factor ----
    def get_adj_factors(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        """推导除权因子 ex_factor = close_hfq / close_none(个股级, 非累积)。

        与 stock-sdk 同策略: 拉同周期后复权(hfq)与不复权(none)日K, 按日期对齐取比值。
        后复权价 = 永不调整的"含权累计价", 除权事件使 hfq/none 比值跳变, 比值即
        pre/post 复权因子, 与 _apply_adj_factor 的 cum_prod 语义一致, 无单位猜测风险。
        """
        if not symbols:
            return pl.DataFrame()
        from easy_tdx import Period
        from easy_tdx.mac.enums import Adjust

        flat: list[dict] = []
        total = len(symbols)
        mac = self._get_mac_client()
        for i, sym in enumerate(symbols):
            try:
                none_df = mac.get_stock_kline(
                    _market_of_symbol(sym), _code_of_symbol(sym), Period.DAILY,
                    0, _DAILY_PAGE, 1, Adjust.NONE,
                )
                hfq_df = mac.get_stock_kline(
                    _market_of_symbol(sym), _code_of_symbol(sym), Period.DAILY,
                    0, _DAILY_PAGE, 1, Adjust.HFQ,
                )
                if (none_df is None or len(none_df) == 0) or (hfq_df is None or len(hfq_df) == 0):
                    continue
                # MAC 协议日K时间列名为 datetime(Timestamp), 取日期部分对齐
                none_by = dict(zip(none_df["datetime"].astype(str).str[:10], none_df["close"], strict=False))
                hfq_by = dict(zip(hfq_df["datetime"].astype(str).str[:10], hfq_df["close"], strict=False))
                for d, hfq_c in hfq_by.items():
                    none_c = none_by.get(d)
                    if none_c and hfq_c:
                        flat.append({
                            "symbol": sym,
                            "trade_date": date.fromisoformat(d),
                            "ex_factor": float(hfq_c) / float(none_c),
                        })
            except Exception as e:
                logger.debug("easy-tdx adj 拉取失败(%s): %s", sym, e)
                continue
            if on_chunk_done:
                on_chunk_done(i + 1, total)
        if not flat:
            return pl.DataFrame()
        df = normalize_adj_factors(flat, source=self.name)
        if start_time is not None and end_time is not None and not df.is_empty():
            df = df.filter(pl.col("trade_date").is_between(start_time.date(), end_time.date()))
        return df

    # ---- realtime (批量/全市场) ----
    def get_realtime(self, universes: list[str] | None = None, symbols: list[str] | None = None) -> list[dict]:
        """实时快照 → list[dict] record。通配符: 无 symbols 时取全市场(分页)。"""
        try:
            c = self._get_client()
            if symbols:
                pairs = [(_market_of_symbol(s), _code_of_symbol(s)) for s in symbols]
                df = c.get_security_quotes(pairs)
            else:
                # 全市场: 标志位不传会自动取全市场? easy-tdx 需要显式, 这里走分页报价列表兜底
                df = self._get_mac_client().get_stock_quotes_list(__import__("easy_tdx").Category.A, count=6000)
            rows = _normalize_realtime(df)
            return rows
        except Exception as e:
            logger.warning("easy-tdx realtime 拉取失败: %s", e)
            return []

    # ---- minute ----
    def get_minute(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",
        freq: str = "1m",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame()
        period = _period_from_freq(freq)
        frames: list[pl.DataFrame] = []
        mac = self._get_mac_client()
        total = len(symbols)
        for i, sym in enumerate(symbols):
            try:
                df = mac.get_stock_kline(_market_of_symbol(sym), _code_of_symbol(sym), period, count=300)
            except Exception as e:
                logger.debug("easy-tdx minute 拉取失败(%s): %s", sym, e)
                continue
            if df is None or len(df) == 0:
                continue
            frame = _minute_frame(df, sym)
            if (start_time is not None and end_time is not None):
                frame = frame.filter(pl.col("datetime").is_between(start_time, end_time))
            if not frame.is_empty():
                frames.append(frame)
            if on_chunk_done:
                on_chunk_done(i + 1, total)
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    # ---- full_minute (修复轮: 当日窗口批量) ----
    def get_intraday_batch(self, symbols: list[str], count: int = 300, asset_type: str = "stock") -> pl.DataFrame:
        """全量分钟修复轮: 批量拉 1 分钟K(当日窗口, 不在此强过滤)。

        返回最近 count 根分钟K(跨自然日); 由 minute_refresh 的 _write_minute_partition
        按各自 datetime 日期落盘当日分区。若上游能拉到当日数据即满足"当日窗口";
        不能时(如测试/休市)返回最近可得分钟, 不因窗口过滤而整轮落空。
        """
        if not symbols:
            return pl.DataFrame()
        return self.get_minute(symbols, None, None, asset_type, "1m", None)

    # ---- depth5 (五档盘口) ----
    def get_depth5_batch(self, symbols: list[str]) -> dict:
        """按标的批量拉五档, 转成 depth_service sealed 契约。

        标准协议 get_security_quotes 返回 bid1/bid_vol1...bid5/ask1/ask_vol1...ask5,
        vol 单位为股, 与 tickflow depth.batch 语义一致; 抽成 ask_volumes/bid_volumes
        数组供 depth_service 取 ask_volumes[0]/bid_volumes[0] 判断真假封。
        """
        if not symbols:
            return {}
        try:
            c = self._get_client()
            out: dict[str, dict | None] = {}
            for i in range(0, len(symbols), _QUOTE_BATCH):
                chunk = symbols[i:i + _QUOTE_BATCH]
                pairs = [(_market_of_symbol(s), _code_of_symbol(s)) for s in chunk]
                df = c.get_security_quotes(pairs)
                if df is None or len(df) == 0:
                    continue
                for _, row in df.iterrows():
                    sym = _to_app_symbol(str(row["code"]), int(row["market"]))
                    bid_vols = [float(row.get(f"bid_vol{j}", 0) or 0) for j in range(1, 6)]
                    ask_vols = [float(row.get(f"ask_vol{j}", 0) or 0) for j in range(1, 6)]
                    out[sym] = {
                        "ask_volumes": ask_vols,
                        "bid_volumes": bid_vols,
                        "timestamp": None,
                    }
            return out
        except Exception as e:
            logger.warning("easy-tdx depth5 拉取失败(%d symbols): %s", len(symbols), e)
            return {}

    # ---- instruments (标的维表) ----
    def get_instruments(self, asset_type: str = "stock") -> list[dict]:
        """返回 tickflow Instrument 形状的行(symbol/name/code/exchange/region/type + ext)。

        用 MAC 协议全市场报价列表(get_stock_quotes_list)一次拉全市场(~5565 只, ~2s),
        复用 instrument_sync 的 flatten 路径。仅股票(asset_type=stock 时), 其他返回空。
        """
        if asset_type != "stock":
            return []
        from easy_tdx import Category

        try:
            mac = self._get_mac_client()
            df = mac.get_stock_quotes_list(Category.A, count=6000)
            if df is None or len(df) == 0:
                return []
            rows: list[dict] = []
            for _, r in df.iterrows():
                code = str(r["code"])
                market = int(r["market"])
                suffix = _to_app_symbol(code, market).split(".")[-1]
                rows.append({
                    "symbol": _to_app_symbol(code, market),
                    "name": r.get("name"),
                    "code": code,
                    "exchange": suffix,
                    "region": "CN",
                    "type": "stock",
                    "ext": {
                        "pre_close": float(r["pre_close"]) if r.get("pre_close") else None,
                        "limit_up": float(r.get("limit_up")) if r.get("limit_up") else None,
                        "limit_down": float(r.get("limit_down")) if r.get("limit_down") else None,
                        "float_shares": float(r.get("float_shares")) if r.get("float_shares") else None,
                        "total_shares": float(r.get("total_shares")) if r.get("total_shares") else None,
                    },
                })
            return rows
        except Exception as e:
            logger.warning("easy-tdx instruments 拉取失败: %s", e)
            return []

    # ---- 测试(设置页试拉) ----
    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        symbols = symbols or ["600519.SH"]
        if dataset == "daily":
            df = self.get_daily(symbols, None, None)
            return _preview("daily", df)
        if dataset == "adj_factor":
            df = self.get_adj_factors(symbols, None, None)
            return _preview("adj_factor", df)
        if dataset == "minute":
            df = self.get_minute(symbols, None, None)
            return _preview("minute", df)
        if dataset == "full_minute":
            df = self.get_intraday_batch(symbols)
            return _preview("full_minute", df)
        if dataset == "realtime":
            rows = self.get_realtime(symbols=symbols)
            head = rows[:5]
            return {
                "provider": self.name,
                "dataset": "realtime",
                "rows": len(rows),
                "columns": list(head[0].keys()) if head else [],
                "preview": head,
            }
        if dataset == "depth5":
            device = self.get_depth5_batch(symbols)
            preview = [{"symbol": s, **(d or {})} for s, d in list(device.items())[:5]]
            return {
                "provider": self.name,
                "dataset": "depth5",
                "rows": sum(1 for d in device.values() if d),
                "columns": ["symbol", "ask_volumes", "bid_volumes", "timestamp"],
                "preview": preview,
            }
        raise ValueError(f"easy-tdx 不支持数据集: {dataset}")


def _preview(dataset: str, df: pl.DataFrame) -> dict:
    return {
        "provider": "easytdx",
        "dataset": dataset,
        "rows": df.height,
        "columns": df.columns,
        "preview": df.head(5).to_dicts() if not df.is_empty() else [],
    }


def _period_from_freq(freq: str) -> int:
    """项目 freq('1m'/'5m'/'1d'...) → easy-tdx Period 枚举值(MIN_1=7/MIN_5=0/MIN_15=1/MIN_30=2/MIN_60=3)。"""
    from easy_tdx import Period
    n = "".join(ch for ch in str(freq) if ch.isdigit()) or "1"
    return {
        "1": Period.MIN_1, "5": Period.MIN_5, "15": Period.MIN_15,
        "30": Period.MIN_30, "60": Period.MIN_60,
    }.get(n, Period.MIN_1)


def _minute_frame(df, symbol: str) -> pl.DataFrame:
    """easy-tdx 分钟 DataFrame → canonical 列(北京墙钟 datetime + symbol)。"""
    if df is None or len(df) == 0:
        return pl.DataFrame()
    out = {
        "symbol": pl.Series([symbol] * len(df)),
        "datetime": pl.Series(df["datetime"]),
        "open": pl.Series(df["open"], dtype=pl.Float64),
        "high": pl.Series(df["high"], dtype=pl.Float64),
        "low": pl.Series(df["low"], dtype=pl.Float64),
        "close": pl.Series(df["close"], dtype=pl.Float64),
        "volume": pl.Series(df["vol"], dtype=pl.Float64),
        "amount": pl.Series(df["amount"] if "amount" in df.columns else [0.0] * len(df), dtype=pl.Float64),
    }
    frame = pl.DataFrame(out)
    # datetime 归一为北京墙钟 naive
    if frame.schema["datetime"] != pl.Datetime("us"):
        frame = frame.with_columns(pl.col("datetime").cast(pl.Datetime("us"), strict=False))
    return frame.select([c for c in _MINUTE_CANONICAL if c in frame.columns])


def _normalize_realtime(df) -> list[dict]:
    """easy-tdx 报价 DataFrame → realtime record(list[dict])。

    使用标准协议字段(price/pre_close/open/high/low/vol/amount)映射到项目 realtime 契约。
    """
    if df is None or len(df) == 0:
        return []
    rows: list[dict] = []
    for _, r in df.iterrows():
        sym = _to_app_symbol(str(r["code"]), int(r["market"]))
        change_pct = None
        pre_close = r.get("pre_close")
        price = r.get("price")
        if pre_close and price:
            change_pct = (float(price) - float(pre_close)) / float(pre_close)
        rows.append({
            "symbol": sym,
            "name": r.get("name"),
            "last_price": float(r["price"]) if r.get("price") else None,
            "prev_close": float(pre_close) if pre_close else None,
            "open": float(r["open"]) if r.get("open") else None,
            "high": float(r["high"]) if r.get("high") else None,
            "low": float(r["low"]) if r.get("low") else None,
            "volume": float(r["vol"]) if r.get("vol") else None,
            "amount": float(r["amount"]) if r.get("amount") else None,
            "change_pct": change_pct,
            "change_amount": (float(price) - float(pre_close)) if price and pre_close else None,
            "timestamp": None,
        })
    return rows


def availability() -> tuple[bool, str]:
    """依赖探活: 能连通通达信服务器即返回 (True, 'ok')。"""
    try:
        p = EasyTdxProvider()
        if p._ping():
            p.close()
            return True, "ok"
        p.close()
        return False, "无法连通通达信行情服务器"
    except Exception as e:
        return False, str(e)
