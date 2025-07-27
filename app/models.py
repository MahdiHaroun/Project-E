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
