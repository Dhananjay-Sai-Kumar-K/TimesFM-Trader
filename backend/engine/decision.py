"""
Decision engine: combines TimesFM predictions with technical indicators
to emit BUY / SELL / HOLD signals.

Signal scoring
--------------
Each condition contributes +1 (bullish) or -1 (bearish) to a tally.
A positive tally above the BUY_THRESHOLD emits BUY; below SELL_THRESHOLD
emits SELL; otherwise HOLD.  This makes the engine more robust than a
single hard AND condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SignalType = Literal["BUY", "SELL", "HOLD"]

BUY_THRESHOLD  = 2   # need ≥2 bullish factors
SELL_THRESHOLD = -2  # need ≥2 bearish factors

DELTA_THRESHOLD = 0.003   # 0.3 % minimum predicted move to count
RSI_OVERSOLD   = 35
RSI_OVERBOUGHT = 65
VOL_SPIKE_MIN  = 1.3      # 30 % above average volume


@dataclass
class DecisionResult:
    signal: SignalType
    confidence: float        # model confidence [0, 1]
    score: int               # raw scoring tally
    reasons: list[str]


def decide(
    pred_delta: float,
    confidence: float,
    rsi: float,
    volume_spike: float,
    macd_hist: float = 0.0,
    stoch_k: float = 50.0,
    adx: float = 20.0,
    close_vs_sma20: float = 0.0,
) -> DecisionResult:
    """
    Multi-factor rule-based decision engine.

    Parameters
    ----------
    pred_delta      : TimesFM predicted price Δ (absolute $)
    confidence      : TimesFM confidence [0, 1]
    rsi             : RSI-14
    volume_spike    : volume / 20-period average volume
    macd_hist       : MACD histogram value
    stoch_k         : Stochastic %K
    adx             : ADX-14 (trend strength)
    close_vs_sma20  : (close - SMA20) / SMA20

    Returns
    -------
    DecisionResult
    """
    score = 0
    reasons: list[str] = []

    # ── TimesFM signal ──────────────────────────────────────────────
    if pred_delta > DELTA_THRESHOLD * 100:          # rough absolute threshold
        score += 1
        reasons.append(f"TimesFM: +{pred_delta:.4f} predicted Δ (bullish)")
    elif pred_delta < -DELTA_THRESHOLD * 100:
        score -= 1
        reasons.append(f"TimesFM: {pred_delta:.4f} predicted Δ (bearish)")

    # ── RSI ──────────────────────────────────────────────────────────
    if rsi < RSI_OVERSOLD:
        score += 1
        reasons.append(f"RSI {rsi:.1f} oversold")
    elif rsi > RSI_OVERBOUGHT:
        score -= 1
        reasons.append(f"RSI {rsi:.1f} overbought")

    # ── Volume ───────────────────────────────────────────────────────
    if volume_spike > VOL_SPIKE_MIN:
        if score > 0:
            score += 1
            reasons.append(f"Volume spike {volume_spike:.2f}x (confirming bullish)")
        elif score < 0:
            score -= 1
            reasons.append(f"Volume spike {volume_spike:.2f}x (confirming bearish)")
        else:
            reasons.append(f"Volume spike {volume_spike:.2f}x (no clear direction)")

    # ── MACD histogram ───────────────────────────────────────────────
    if macd_hist > 0:
        score += 1
        reasons.append(f"MACD hist {macd_hist:.4f} positive (bullish momentum)")
    elif macd_hist < 0:
        score -= 1
        reasons.append(f"MACD hist {macd_hist:.4f} negative (bearish momentum)")

    # ── Stochastic ───────────────────────────────────────────────────
    if stoch_k < 20:
        score += 1
        reasons.append(f"Stoch %K {stoch_k:.1f} oversold")
    elif stoch_k > 80:
        score -= 1
        reasons.append(f"Stoch %K {stoch_k:.1f} overbought")

    # ── ADX (only acts as a multiplier – strong trend strengthens) ───
    if adx > 25:
        if score > 0:
            score += 1
            reasons.append(f"ADX {adx:.1f} strong trend (confirms bullish)")
        elif score < 0:
            score -= 1
            reasons.append(f"ADX {adx:.1f} strong trend (confirms bearish)")

    # ── Price vs SMA-20 ──────────────────────────────────────────────
    if close_vs_sma20 < -0.02:
        score += 1
        reasons.append(f"Price {close_vs_sma20:.2%} below SMA20 (potential bounce)")
    elif close_vs_sma20 > 0.02:
        score -= 1
        reasons.append(f"Price {close_vs_sma20:.2%} above SMA20 (potential pullback)")

    # ── Final verdict ────────────────────────────────────────────────
    if score >= BUY_THRESHOLD:
        signal: SignalType = "BUY"
    elif score <= SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    return DecisionResult(
        signal=signal,
        confidence=confidence,
        score=score,
        reasons=reasons,
    )
