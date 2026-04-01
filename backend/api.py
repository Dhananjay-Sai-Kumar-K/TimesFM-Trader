"""
FastAPI backend – serves price history, signals, predictions, and
live/cached data from Redis.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.config import SYMBOLS
from backend.db.models import Feature, Prediction, Price, SessionLocal, Signal
from backend.utils.redis_client import redis_client

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Market Predictor API",
    description="Real-time market prediction powered by TimesFM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

@app.get("/symbols", tags=["Market"])
def get_symbols():
    return {"symbols": SYMBOLS}


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

@app.get("/prices/{symbol}", tags=["Market"])
def get_prices(
    symbol: str,
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol!r} not tracked.")

    rows = (
        db.query(Price)
        .filter(Price.symbol == symbol)
        .order_by(Price.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "timestamp": p.timestamp.isoformat(),
            "open":   p.open,
            "high":   p.high,
            "low":    p.low,
            "close":  p.close,
            "volume": p.volume,
        }
        for p in reversed(rows)
    ]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

@app.get("/signals/{symbol}", tags=["Signals"])
def get_signals(
    symbol: str,
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol!r} not tracked.")

    rows = (
        db.query(Signal)
        .filter(Signal.symbol == symbol)
        .order_by(Signal.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "timestamp":  s.timestamp.isoformat(),
            "signal":     s.signal,
            "confidence": s.confidence,
            "reason":     s.reason,
        }
        for s in reversed(rows)
    ]


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

@app.get("/predictions/{symbol}", tags=["Signals"])
def get_predictions(
    symbol: str,
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol!r} not tracked.")

    rows = (
        db.query(Prediction)
        .filter(Prediction.symbol == symbol)
        .order_by(Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "timestamp":       p.timestamp.isoformat(),
            "forecast_time":   p.forecast_time.isoformat(),
            "predicted_delta": p.predicted_delta,
            "confidence":      p.confidence,
            "quantile_low":    p.quantile_low,
            "quantile_high":   p.quantile_high,
        }
        for p in reversed(rows)
    ]


# ---------------------------------------------------------------------------
# Latest (served from Redis cache – microsecond latency)
# ---------------------------------------------------------------------------

def _redis_get(key: str, cast=str):
    val = redis_client.get(key)
    if val is None:
        return None
    try:
        return cast(val.decode())
    except Exception:
        return None


@app.get("/latest", tags=["Signals"])
def get_latest() -> dict[str, Any]:
    """Return the most recent signal, confidence, and delta for all symbols."""
    result: dict[str, Any] = {}
    for symbol in SYMBOLS:
        result[symbol] = {
            "signal":       _redis_get(f"signal:{symbol}")        or "UNKNOWN",
            "confidence":   _redis_get(f"confidence:{symbol}",    float),
            "pred_delta":   _redis_get(f"pred_delta:{symbol}",    float),
            "score":        _redis_get(f"score:{symbol}",         int),
            "rsi":          _redis_get(f"rsi:{symbol}",           float),
            "volume_spike": _redis_get(f"volume_spike:{symbol}",  float),
            "last_update":  _redis_get(f"last_update:{symbol}"),
        }
    return result


# ---------------------------------------------------------------------------
# Features snapshot
# ---------------------------------------------------------------------------

@app.get("/features/{symbol}", tags=["Market"])
def get_features(
    symbol: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return latest value of every feature for a symbol."""
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol!r} not tracked.")

    # Get the latest timestamp
    latest_price = (
        db.query(Price)
        .filter(Price.symbol == symbol)
        .order_by(Price.timestamp.desc())
        .first()
    )
    if not latest_price:
        return {"symbol": symbol, "features": {}}

    # Get all features at that timestamp (within 2-minute window)
    cutoff = latest_price.timestamp - timedelta(minutes=2)
    rows = (
        db.query(Feature)
        .filter(
            Feature.symbol == symbol,
            Feature.timestamp >= cutoff,
        )
        .all()
    )
    features = {r.feature_name: r.value for r in rows}
    return {
        "symbol":    symbol,
        "timestamp": latest_price.timestamp.isoformat(),
        "features":  features,
    }
