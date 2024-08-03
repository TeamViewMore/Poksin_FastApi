import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, ForeignKey, Text, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, scoped_session

from datetime import datetime

# Load .env file
load_dotenv()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
BUCKET_NAME = os.getenv("BUCKET_NAME")

if not DATABASE_URL or not BUCKET_NAME:
    raise RuntimeError("DATABASE_URL and BUCKET_NAME must be set in the environment variables.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = scoped_session(
    sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
)
Base = declarative_base()
Base.query = session.query_property()

class EvidenceEntity(Base):
    __tablename__ = 'EvidenceEntity'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime)
    last_modified_at = Column(DateTime)
    description = Column(String(255))
    done = Column(Boolean, default=False)
    fileUrls = Column(Text)
    title = Column(String(255))
    category_id = Column(Integer, ForeignKey('category.id'))
    user_id = Column(Integer, ForeignKey('user.id'))

    segments = relationship("ViolenceSegment", back_populates="evidence", order_by="ViolenceSegment.id")

    class Config:
        orm_mode = True

class ViolenceSegment(Base):
    __tablename__ = "violence_segment"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    evidence_id = Column(Integer, ForeignKey('EvidenceEntity.id'))
    s3_url = Column(String(255))
    duration = Column(Float)
    
    evidence = relationship("EvidenceEntity", back_populates="segments")
    class Config:
        orm_mode = True
