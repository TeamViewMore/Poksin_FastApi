from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Category(Base):
    __tablename__ = 'CategoryEntity'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))

class User(Base):
    __tablename__ = 'UserEntity'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255))

class EvidenceEntity(Base):
    __tablename__ = 'EvidenceEntity'
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime)
    last_modified_at = Column(DateTime)
    description = Column(String(255))
    done = Column(Boolean, default=False)
    fileUrls = Column(Text)
    title = Column(String(255))
    category_id = Column(Integer, ForeignKey('CategoryEntity.id'))
    user_id = Column(Integer, ForeignKey('UserEntity.id'))
    category = relationship('Category', back_populates="evidences")
    user = relationship('User', back_populates="evidences")

Category.evidences = relationship('EvidenceEntity', order_by=EvidenceEntity.id, back_populates="category")
User.evidences = relationship('EvidenceEntity', order_by=EvidenceEntity.id, back_populates="user")

class ViolenceSegment(Base):
    __tablename__ = 'violence_segment'
    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(Integer, ForeignKey('EvidenceEntity.id'))
    s3_url = Column(String(255))
    duration = Column(Float)
    evidence = relationship("EvidenceEntity", back_populates="segments")

EvidenceEntity.segments = relationship("ViolenceSegment", order_by=ViolenceSegment.id, back_populates="evidence")
