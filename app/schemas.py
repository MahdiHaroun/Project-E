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


class ChatbotResponse(BaseModel):
    id: int
    message: str
    response: str
    created_at: datetime
    model_used: str

    class Config:
        from_attributes = True
        protected_namespaces = ()

class ChatbotRequest(BaseModel):
    message: str
    model_used: str

    class Config:
        from_attributes = True
        protected_namespaces = ()