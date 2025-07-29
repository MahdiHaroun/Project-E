from .. import models , schemas 
from ..database import get_db
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from functools import lru_cache



router = APIRouter(
    prefix="/summriz",
    tags=["Summarization"]
)


class Summarization_pipline:
    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",  # Use the multilingual version
            chunk_length_s=60,
        )

@lru_cache()
def get_summarization_pipeline():
    return Summarization_pipline()



class SummarizationModel:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        model_id = "philschmid/bart-large-cnn-samsum"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cpu")
        self.summarize = {
            "pipeline": pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=-1)
        }
@lru_cache()
def get_summarization_model():
    return SummarizationModel()




    

@router.post("/" , response_model =schemas.SummarizationResponse , status_code=status.HTTP_201_CREATED)
async def create_summarization(new_summarization : schemas.SummarizationRequest , db: Session = Depends(get_db)):
    voice_path = new_summarization.voice_path
    summarization_pipeline = get_summarization_pipeline()
    summarization_model = get_summarization_model()

    
    transcribed_text = summarization_pipeline.pipe(voice_path , batch_size=8)["text"]

    summary = summarization_model.summarize["pipeline"](transcribed_text, max_length=150, min_length=60, do_sample=False)[0]['summary_text']

    new_summarization = models.Summarization(
        voice_path=voice_path,
        encoded_text=transcribed_text,
        summary=summary,
    )

    db.add(new_summarization)
    db.commit()
    db.refresh(new_summarization)

    return new_summarization
