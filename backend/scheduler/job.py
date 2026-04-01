"""
Background scheduler: runs every minute per symbol.

Flow per tick:
  1. Fetch latest OHLCV bars from yfinance
  2. Upsert into PostgreSQL
  3. Compute / store technical features
  4. Run TimesFM prediction
  5. Run decision engine
  6. Persist signal to DB
  7. Cache latest signal + metrics in Redis (for fast API reads)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler

from backend.config import CONTEXT_LEN, INTERVAL, SYMBOLS
from backend.db.models import Feature, Prediction, Price, SessionLocal, Signal
from backend.engine.decision import decide
from backend.features.indicators import compute_all_features
from backend.models.timesfm_model import predictor
from backend.utils.redis_client import redis_client

logger = logging.getLogger(__name__)

# How many bars to request from yfinance to cover context + indicator warm-up
_FETCH_BARS = max(CONTEXT_LEN + 200, 500)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch_ohlcv(symbol: str) -> pd.DataFrame | None:
    """Return last _FETCH_BARS 1-minute bars for *symbol*, or None on error."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d", interval=INTERVAL)
        if df.empty:
            logger.warning("yfinance returned empty data for %s", symbol)
            return None
        df = df.tail(_FETCH_BARS)
        # Strip timezone so SQLAlchemy / PostgreSQL stays happy
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df
    except Exception as exc:
        logger.error("yfinance fetch failed for %s: %s", symbol, exc)
        return None


def _upsert_prices(session, symbol: str, df: pd.DataFrame) -> None:
    """Bulk-upsert OHLCV rows (merge = INSERT OR UPDATE via SQLAlchemy)."""
    for ts, row in df.iterrows():
        obj = Price(
            symbol=symbol,
            timestamp=ts.to_pydatetime(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=int(row["Volume"]),
        )
        session.merge(obj)
    session.commit()


def _upsert_features(session, symbol: str, df_feat: pd.DataFrame) -> None:
    """Persist computed feature values (skips OHLCV columns)."""
    ohlcv_cols = {"open", "high", "low", "close", "volume"}
    feature_cols = [c for c in df_feat.columns if c not in ohlcv_cols]
    for ts, row in df_feat.iterrows():
        for col in feature_cols:
            obj = Feature(
                symbol=symbol,
                timestamp=ts.to_pydatetime(),
                feature_name=col,
                value=float(row[col]),
            )
            session.merge(obj)
    session.commit()


# ---------------------------------------------------------------------------
# Indicator lookup helpers
# ---------------------------------------------------------------------------

def _get_latest_feature(session, symbol: str, name: str, default: float) -> float:
    row = (
        session.query(Feature)
        .filter(Feature.symbol == symbol, Feature.feature_name == name)
        .order_by(Feature.timestamp.desc())
        .first()
    )
    return float(row.value) if row else default


# ---------------------------------------------------------------------------
# Core per-symbol processing
# ---------------------------------------------------------------------------

def process_symbol(symbol: str) -> None:
    session = SessionLocal()
    try:
        # 1. Fetch
        df = _fetch_ohlcv(symbol)
        if df is None or df.empty:
            return

        # 2. Persist raw prices
        _upsert_prices(session, symbol, df)

        # 3. Compute & persist features
        df_feat = compute_all_features(df)
        if df_feat.empty:
            logger.warning("Not enough data to compute features for %s", symbol)
            return
        _upsert_features(session, symbol, df_feat)

        # 4. TimesFM prediction
        close_prices = df_feat["close"].tolist()
        if len(close_prices) < CONTEXT_LEN:
            logger.info("Not enough bars for TimesFM on %s (%d)", symbol, len(close_prices))
            return

        delta, confidence, q_low, q_high = predictor.predict(close_prices)

        latest_ts = df_feat.index[-1].to_pydatetime()
        pred = Prediction(
            symbol=symbol,
            timestamp=latest_ts,
            forecast_time=latest_ts + pd.Timedelta(minutes=1),
            predicted_delta=float(delta),
            confidence=float(confidence),
            quantile_low=float(q_low),
            quantile_high=float(q_high),
        )
        session.add(pred)
        session.commit()

        # 5. Read latest indicator values
        rsi       = _get_latest_feature(session, symbol, "rsi_14",        50.0)
        vol_spike = _get_latest_feature(session, symbol, "volume_spike",    1.0)
        macd_hist = _get_latest_feature(session, symbol, "macd_hist",       0.0)
        stoch_k   = _get_latest_feature(session, symbol, "stoch_k",        50.0)
        adx       = _get_latest_feature(session, symbol, "adx_14",         20.0)
        vs_sma20  = _get_latest_feature(session, symbol, "close_vs_sma20",  0.0)

        # 6. Decision
        result = decide(
            pred_delta=delta,
            confidence=confidence,
            rsi=rsi,
            volume_spike=vol_spike,
            macd_hist=macd_hist,
            stoch_k=stoch_k,
            adx=adx,
            close_vs_sma20=vs_sma20,
        )

        sig = Signal(
            symbol=symbol,
            timestamp=latest_ts,
            signal=result.signal,
            confidence=result.confidence,
            reason="; ".join(result.reasons)[:200],
        )
        session.add(sig)
        session.commit()

        # 7. Cache in Redis for low-latency API reads
        pipe = redis_client.pipeline()
        pipe.set(f"signal:{symbol}",      result.signal)
        pipe.set(f"confidence:{symbol}",  result.confidence)
        pipe.set(f"pred_delta:{symbol}",  delta)
        pipe.set(f"score:{symbol}",       result.score)
        pipe.set(f"last_update:{symbol}", datetime.now(timezone.utc).isoformat())
        pipe.set(f"rsi:{symbol}",         rsi)
        pipe.set(f"volume_spike:{symbol}", vol_spike)
        pipe.execute()

        logger.info(
            "[%s] %s | Δ=%.4f | conf=%.2f | score=%d | rsi=%.1f",
            symbol, result.signal, delta, confidence, result.score, rsi,
        )

    except Exception as exc:
        logger.exception("Unhandled error processing %s: %s", symbol, exc)
        session.rollback()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Scheduler entry point
# ---------------------------------------------------------------------------

def run_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        func=lambda: [process_symbol(s) for s in SYMBOLS],
        trigger="interval",
        minutes=1,
        id="market_loop",
        max_instances=1,          # prevent overlap if a tick runs long
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started – watching %s", SYMBOLS)
    return scheduler
