from database import Base
from sqlalchemy import Column, Integer, String, Boolean, Float

class FactChecker(Base):
    __tablename__ = 'factchecker'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(String)
    url = Column(String)
    truth = Column(Boolean)
   
    