FROM python:3.11

WORKDIR /app

COPY setup_db.py dataset.json ./

RUN pip install psycopg2-binary SQLAlchemy

CMD ["python", "setup_db.py"]