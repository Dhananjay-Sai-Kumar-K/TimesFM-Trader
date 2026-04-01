from sqlalchemy import Column, Integer, Float, String, DateTime, Index, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

class Price(Base):
    __tablename__ = 'prices'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)

    __table_args__ = (Index('idx_symbol_timestamp', 'symbol', 'timestamp'),)

class Feature(Base):
    __tablename__ = 'features'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    timestamp = Column(DateTime, index=True)
    feature_name = Column(String(50))
    value = Column(Float)

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    timestamp = Column(DateTime, index=True)
    forecast_time = Column(DateTime)
    predicted_delta = Column(Float)
    confidence = Column(Float)
    quantile_low = Column(Float)
    quantile_high = Column(Float)

class Signal(Base):
    __tablename__ = 'signals'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    timestamp = Column(DateTime, index=True)
    signal = Column(String(10))  # BUY, SELL, HOLD
    confidence = Column(Float)
    reason = Column(String(200))

# Create tables (run once)
engine = create_engine(os.getenv('DATABASE_URL'))
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)