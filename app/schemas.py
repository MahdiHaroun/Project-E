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


class SummarizationRequest(BaseModel):
    voice_path: str 



class SummarizationResponse(BaseModel):
    id: int
    voice_path: str
    encoded_text: str
    summary: str
    created_at: datetime

    class Config:
        from_attributes = True
        protected_namespaces = ()





class PDFPathRequest(BaseModel):
    pdf_path: str

class QuestionRequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    id: int
    pdf_path: str
    question: str
    answer: str
    created_at: datetime

    class Config:
        orm_mode = True


