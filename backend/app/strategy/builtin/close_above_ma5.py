"""站上MA5 — 收盘价上穿MA5 (昨日收于下方, 今日站上)"""

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    matrix_feature,
)
from app.backtest.matrix import (
    valid_shift as shift,
)

META = {
    "id": "close_above_ma5",
    "name": "站上MA5",
    "description": "收盘价上穿MA5: 昨日收于MA5下方, 今日收盘站上MA5 (短线动量修复)",
    "tags": ["均线", "突破"],
    "asset_types": ["stock", "etf"],
    "timeframes": ["1d"],
    "params": [
        {
            "id": "require_volume_confirm",
            "label": "要求量比>1 配合",
            "type": "bool",
            "default": False,
        },
    ],
    "scoring": {"momentum_20d": 0.4, "turnover_rate": 0.35, "amount": 0.25},
    "order_by": "score",
    "descending": True,
    "limit": 100,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = []
EXIT_SIGNALS = ["signal_ma20_breakdown"]
STOP_LOSS = -0.05
MAX_HOLD_DAYS = 10


class CloseAboveMa5MatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({"close"})

    def required_warmup_bars(self, params: dict) -> int:
        del params
        return 20

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        ma5 = matrix_feature(market, "ma5")
        close = market.close
        prev_close = shift(close, 1)
        prev_ma5 = shift(ma5, 1)
        # 上穿: 昨日收于 MA5 下方, 今日收盘站上 MA5
        entry = (close > ma5) & (prev_close <= prev_ma5)
        if params.get("require_volume_confirm", False):
            entry &= matrix_feature(market, "vol_ratio_5d") > 1.0
        # 出场: 收盘跌回 MA5 下方 (昨日尚在上方)
        exit_ = (close < ma5) & (shift(close, 1) >= shift(ma5, 1))
        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            exit=exit_.astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(exit_, 0, -1).astype(np.int16),
        )


MATRIX_STRATEGY = CloseAboveMa5MatrixStrategy()
