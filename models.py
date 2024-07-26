from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os


Base = declarative_base()

class EvidenceEntity(Base):
    __tablename__ = "evidenceentity"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_modified_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    description = Column(String(255))
    fileUrls = Column(Text)
    title = Column(String(255))
    category_id = Column(Integer)
    user_id = Column(Integer)

class ViolenceSegment(Base):
    __tablename__ = "violence_segment"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    evidence_id = Column(Integer, ForeignKey('evidenceentity.id'))
    s3_url = Column(String(255))
    duration = Column(Float)
    evidence = relationship("EvidenceEntity", back_populates="segments")

EvidenceEntity.segments = relationship("ViolenceSegment", order_by=ViolenceSegment.id, back_populates="evidence")

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)
