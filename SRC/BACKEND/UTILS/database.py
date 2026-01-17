import os
from logging import Logger
from contextlib import asynccontextmanager
from fastapi import FastAPI,Request
from sqlalchemy.ext.asyncio import AsyncEngine,create_async_engine,async_sessionmaker

def get_engine(logger:Logger)->AsyncEngine:

    """
    Return an instance of databse engine.
    """

    port = os.getenv("DB_PORT")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")
    host_name = os.getenv("DB_HOST","db")
    url = f"postgresql+asyncpg://{user}:{password}@{host_name}:{port}/{db_name}"

    engine = create_async_engine(url)
    return engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application to manage database engine lifecycle.
    """

    engine = get_engine(app.logger)
    
    app.state.db_engine = engine
    app.state.SessionLocal = async_sessionmaker(bind=engine)

    yield 

    engine.dispose()
