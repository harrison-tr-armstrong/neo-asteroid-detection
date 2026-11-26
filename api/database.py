from sqlmodel import create_engine, SQLModel, Session
from api import models


# Define the connection URL for the SQLite database.
# The database file is created in the project root directory if it does not exist.
DATABASE_URL = "sqlite:///./neo_database.db"


# Create the database engine, which manages connections between the application and SQLite.
# check_same_thread=False allows database connections to be used across different threads,
# which is useful when working with FastAPI.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


# Create all database tables defined by the SQLModel models.
# Existing tables are left unchanged if they already exist.
def create_db_and_tables():
   SQLModel.metadata.create_all(engine)


# Provide a database session to API endpoints when requested through FastAPI dependencies.
def get_session():
   # Open a session for the duration of the request and close it afterwards.
   with Session(engine) as session:
      yield session
