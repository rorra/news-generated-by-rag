from config import engine
from models.db_models import Base

# Create the tables
Base.metadata.create_all(engine)
