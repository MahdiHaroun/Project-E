from fastapi import FastAPI
from . import models
from .database import engine
from .routers import image_cap


models.Base.metadata.create_all(bind=engine)


app = FastAPI()


app.include_router(image_cap.router)






@app.get("/")
async def root():
    return {"message": "Welcome to E-project!"}


