from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from . import models
from .database import engine
from .routers import image_cap, chatbot , summriz , chatbot_pdf


models.Base.metadata.create_all(bind=engine)


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(image_cap.router)
app.include_router(chatbot.router)
app.include_router(summriz.router)
app.include_router(chatbot_pdf.router)








@app.get("/")
async def root():
    return {"message": "Welcome to E-project!"}


