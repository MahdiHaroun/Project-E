from .database import Base 
from sqlalchemy import Column, Integer, String, Boolean , ForeignKey 
from sqlalchemy.sql.sqltypes import TIMESTAMP
from sqlalchemy.sql import text
from sqlalchemy.orm import relationship


class image_Captioning(Base):
    __tablename__ = "image_captioning"

    id = Column(Integer, primary_key=True, nullable=False)
    image_path = Column(String, nullable=False)
    caption = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), nullable=False)



class Chatbot(Base):
    __tablename__ = "chatbot"

    id = Column(Integer, primary_key=True, nullable=False)
    message = Column(String, nullable=False)
    response = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), nullable=False)
    model_used = Column(String, nullable=False, default="LLaMA-2-7b-chat-hf")



class Summarization(Base):
    __tablename__ = "summarization"

    id = Column(Integer, primary_key=True, nullable=False)
    voice_path = Column(String, nullable=False)
    encoded_text = Column(String, nullable=False)
    summary = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'), nullable=False)