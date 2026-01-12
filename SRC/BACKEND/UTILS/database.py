import os
from sqlalchemy import create_engine,Engine
from sqlalchemy.orm import sessionmaker
from logging import Logger
from contextlib import asynccontextmanager
from fastapi import FastAPI,Request


def get_engine(logger:Logger)->Engine:

    """
    Return an instance of databse engine.
    """

    port = os.getenv("DB_PORT")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")
    host_name = os.getenv("DB_HOST","db")
    url = f"postgresql+psycopg2://{user}:{password}@{host_name}:{port}/{db_name}"

    engine = create_engine(url)
    return engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application to manage database engine lifecycle.
    """

    engine = get_engine(app.logger)
    
    app.state.db_engine = engine
    app.state.SessionLocal = sessionmaker(bind=engine)

    yield 

    engine.dispose()

def get_db(request: Request):
    """
    Dependency that provides a database session for each request.
    """

    session = request.app.state.SessionLocal()
    try:
        yield session
    finally:
        session.close()

