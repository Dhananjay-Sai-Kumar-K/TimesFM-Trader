"""
TimesFM predictor wrapper.

TimesFM (google-research/timesfm) is a pre-trained time-series foundation model.
We wrap it so the rest of the codebase only sees a simple .predict() interface.

Installation:
    pip install timesfm

Model checkpoint is downloaded automatically on first use via HuggingFace Hub.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class TimesFMPredictor:
    """Thin wrapper around the TimesFM model."""

    def __init__(self, context_len: int = 512, horizon: int = 10) -> None:
        self.context_len = context_len
        self.horizon = horizon
        self._model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazily load the TimesFM checkpoint."""
        try:
            import timesfm  # noqa: PLC0415

            # timesfm ≥ 1.3 uses TimesFm (capital F lowercase m)
            self._model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="cpu",          # switch to "gpu" if CUDA is available
                    per_core_batch_size=32,
                    horizon_len=self.horizon,
                    context_len=self.context_len,
                    num_layers=20,          # 200-m variant
                    model_dims=1280,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
                ),
            )
            logger.info("TimesFM model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load TimesFM model: %s", exc)
            self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self, series: Sequence[float]
    ) -> tuple[float, float, float, float]:
        """
        Run a forecast on the supplied price series.

        Parameters
        ----------
        series : sequence of floats
            Historical close prices, oldest-first. Must have at least
            ``context_len`` data-points.

        Returns
        -------
        (predicted_delta, confidence, quantile_low, quantile_high)
            predicted_delta : predicted price change from last known price
            confidence      : rough confidence score in [0, 1]
            quantile_low    : lower bound of the 10th percentile forecast
            quantile_high   : upper bound of the 90th percentile forecast
        """
        series = list(series)
        if len(series) < self.context_len:
            raise ValueError(
                f"Need at least {self.context_len} data points, got {len(series)}."
            )

        last_price: float = series[-1]
        input_arr = np.array(series[-self.context_len :], dtype=np.float32)

        if self._model is None:
            # Graceful fallback: return a naïve random-walk prediction
            logger.warning("TimesFM model unavailable – using fallback predictor.")
            return self._fallback_predict(input_arr, last_price)

        try:
            # TimesFM ≥ 1.3 API:
            #   forecast_on_df  /  forecast
            # We use the numpy path for simplicity.
            point_forecast, quantile_forecast = self._model.forecast(
                inputs=[input_arr.tolist()],
                freq=[0],               # 0 = high-frequency / minute-level
            )
            # point_forecast  : (batch, horizon)
            # quantile_forecast : (batch, horizon, n_quantiles)
            pf = np.array(point_forecast[0])    # shape (horizon,)
            qf = np.array(quantile_forecast[0]) # shape (horizon, n_quantiles)

            next_price = float(pf[0])
            delta = next_price - last_price

            # Quantiles are typically [0.1, 0.2, … 0.9] – pick 10th and 90th
            q_low  = float(qf[0, 0])  - last_price
            q_high = float(qf[0, -1]) - last_price

            # Use inter-quantile range relative to price as an uncertainty proxy
            iqr = abs(float(qf[0, -1]) - float(qf[0, 0]))
            confidence = float(np.clip(1.0 - iqr / (abs(last_price) + 1e-9), 0.0, 1.0))

            return delta, confidence, q_low, q_high

        except Exception as exc:
            logger.error("TimesFM inference error: %s", exc)
            return self._fallback_predict(input_arr, last_price)

    # ------------------------------------------------------------------
    # Fallback (used when model fails to load or inference errors out)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_predict(
        series: np.ndarray, last_price: float
    ) -> tuple[float, float, float, float]:
        """
        Naïve predictor: assumes next price ≈ last price + exponentially
        weighted mean of recent returns.  Used only when TimesFM is not
        available.
        """
        returns = np.diff(series) / (series[:-1] + 1e-9)
        weights = np.exp(np.linspace(0, 1, len(returns)))
        weights /= weights.sum()
        expected_return = float(np.dot(weights, returns))
        delta = expected_return * last_price
        std = float(np.std(returns)) * last_price
        return delta, 0.5, delta - std, delta + std


# ---------------------------------------------------------------------------
# Module-level singleton – imported by scheduler and API
# ---------------------------------------------------------------------------
predictor = TimesFMPredictor()
