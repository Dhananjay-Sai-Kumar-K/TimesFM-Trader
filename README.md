# Market Predictor 📈🤖

Production-ready market prediction system using Google's **TimesFM** foundation model, FastAPI, PostgreSQL, Redis, and a beautiful React + Tailwind dashboard.

## 🌟 Features
- **TimesFM Integration**: Zero-shot forecasting of financial time-series using state-of-the-art transformer models.
- **Advanced Technical Indicators**: Computes RSI, MACD, Bollinger Bands, ATR, Stochastics, ADX, and VWAP on the fly.
- **Multi-Factor Decision Engine**: Combines AI predictions + technical signals for robust BUY/SELL/HOLD recommendations.
- **Live Trading Dashboard**: Glassmorphism UI, interactive charts via lightweight-charts, and real-time polling.
- **Microsecond API**: Caches predictions and data in Redis for hyper-fast frontend load times.
- **Automated Pipeline**: Background APScheduler fetching data & running predictions every minute.

## 🛠️ Stack
- **Backend**: Python, FastAPI, SQLAlchemy, APScheduler, Pandas, yfinance
- **AI**: TimesFM (google-research/timesfm), PyTorch
- **Frontend**: React (Vite), TypeScript, Lightweight Charts, Tailwind CSS (via pure CSS implementation)
- **Data**: PostgreSQL (Persistence), Redis (Caching)
- **Deployment**: Docker, Docker Compose

## 🚀 Quickstart

### 1. Requirements
- Docker and Docker Compose

### 2. Start Services
Run everything natively using Docker Compose:

```bash
docker-compose up -d --build
```

### 3. Access the Apps
- **Frontend Dashboard**: [http://localhost:3000](http://localhost:3000)
- **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

*Note: The system needs at least 1 minute after startup to fetch initial data and run the first TimesFM prediction batch.*

## 📁 Repository Structure

- `backend/`
  - `api.py`: FastAPI endpoints.
  - `config.py`: Core configurations (symbols to track, prediction horizon).
  - `main.py`: Application entry point.
  - `db/`: SQLAlchemy schemas for prices, features, and signals.
  - `engine/`: Decision logic combining AI and indicators.
  - `features/`: Technical indicator calculation pipeline.
  - `models/`: TimesFM wrapper instance.
  - `scheduler/`: Background runner.
- `frontend/`: Real-time UI built with React + Vite.

## ⚠️ Disclaimer
This software is for **educational and research purposes only**. Do not use it to trade real money without extensive backtesting and risk management.
