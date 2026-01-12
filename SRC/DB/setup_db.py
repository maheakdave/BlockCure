from sqlalchemy import create_engine,Column,String,MetaData,Table,Connection
from sqlalchemy.dialects.postgresql import insert
import json
import os
import hashlib
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def row_id(row:dict[str])->str:

    """
    Creates an ID by combining row values
    """

    key = f"{row['diagnosis']}|{row['symptoms']}|{row['treatment']}"
    return hashlib.sha256(key.encode()).hexdigest()

def create_and_conn_database()->tuple[Connection,Table]:

    """
    Creates and connects to a database
    """

    port = os.getenv("DB_PORT")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")
    host_name = os.getenv("DB_HOST","db")
    url = f"postgresql+psycopg2://{user}:{password}@{host_name}:{port}/{db_name}"

    logger.info("Connecting to database at %s", url)
    engine = create_engine(url)

    metadata_obj = MetaData()
    table = Table(
        "transactions",
        metadata_obj,
        Column('id',String,primary_key=True),
        Column('diagnosis',String),
        Column('treatment',String),
        Column('symptoms',String)
    )
    logger.info("Ensuring tables exist")
    metadata_obj.create_all(engine)
    conn = engine.connect()
    logger.info("Database connection established")
    return conn,table

def populate_db()->None:

    """
    THis function is meant to populate the database with the contents of dataset.json
    """

    logger.info("Starting database population")
    conn,transactions_table = create_and_conn_database()

    logger.info("Loading dataset.json")
    with open('dataset.json',"r") as f:
        dataset = json.load(f)

    logger.info("Preparing %d rows", len(dataset))
    dataset = [{"id":row_id(row),"diagnosis":row['diagnosis'],"symptoms":row['symptoms'],"treatment":row['treatment']} for row in dataset]
    stmt = insert(transactions_table).values(dataset)
    stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
    
    result = conn.execute(stmt)
    conn.commit()

    logger.info("Inserted %d rows", result.rowcount)
    logger.info("Database setup completed successfully")

if __name__ == "__main__":
    populate_db()