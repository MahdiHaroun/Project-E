from pydantic import BaseModel
from datetime import datetime


class ImageCaptioningRequest(BaseModel):
    image_path: str
    
    

class ImageCaptioningResponse(BaseModel):
    id: int
    image_path: str
    caption: str
    created_at: datetime

    class Config:
        from_attributes = True
