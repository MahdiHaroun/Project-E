from .. import schemas, models
from ..database import get_db
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, status, HTTPException
from functools import lru_cache
from typing import Optional
from .. config import settings

router = APIRouter(
    prefix="/chatbot",
    tags=["Chatbot"]
)





class ChatbotModel_FACEBOOK:
    def __init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import os
        
        # Temporarily clear any HF token to avoid authentication issues
        original_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
        if 'HUGGINGFACE_HUB_TOKEN' in os.environ:
            del os.environ['HUGGINGFACE_HUB_TOKEN']
        
        model_name = "facebook/blenderbot-400M-distill"
        print("Loading Facebook Blenderbot model... (this happens only once)")
        
        try:
            # Use use_auth_token=False to explicitly disable authentication
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=False)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=False)
        finally:
            # Restore the original token if it existed
            if original_token:
                os.environ['HUGGINGFACE_HUB_TOKEN'] = original_token
                
        print("Facebook Blenderbot model loaded successfully!")

@lru_cache()
def get_facebook_model():
    return ChatbotModel_FACEBOOK()

class ChatbotModel_LLAMA:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "meta-llama/Llama-2-7b-chat-hf"
        access_token = settings.chatbot_model_token  
        print("Loading chatbot model... (this happens only once)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
        print("Chatbot model loaded successfully!")

@lru_cache()
def get_llama_model():
    return ChatbotModel_LLAMA()

@router.post("/", response_model=schemas.ChatbotResponse, status_code=status.HTTP_201_CREATED)
async def create_chatbot_response(
    new_chatbot_request: schemas.ChatbotRequest,
    db: Session = Depends(get_db)
):
    input_text = new_chatbot_request.message.strip()
    model_type = new_chatbot_request.model_used
    
    if model_type == "LLaMA-2-7b-chat-hf":
        # Use LLaMA model
        chatbot_model = get_llama_model()
        
        # LLaMA-2 prompt formatting
        prompt = f"<s>[INST] {input_text} [/INST]"
        
        inputs = chatbot_model.tokenizer(prompt, return_tensors="pt")
        outputs = chatbot_model.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=chatbot_model.tokenizer.eos_token_id
        )
        
        response = chatbot_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up LLaMA response
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        if input_text in response:
            response = response.replace(input_text, "").strip()
    
    elif model_type == "facebook/blenderbot-400M-distill":
        # Use Facebook model
        chatbot_model = get_facebook_model()
        
        inputs = chatbot_model.tokenizer(input_text, return_tensors="pt")
        outputs = chatbot_model.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=chatbot_model.tokenizer.eos_token_id
        )
        
        response = chatbot_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up Facebook response
        if input_text in response:
            response = response.replace(input_text, "").strip()
    
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported model: {model_type}. Supported models: 'LLaMA-2-7b-chat-hf', 'facebook/blenderbot-400M-distill'"
        )
    
    # Save to database
    try:
        new_chatbot_response = models.Chatbot(
            message=input_text,
            response=response,
            model_used=model_type
        )
    except TypeError:
        # Fallback if model_used field doesn't exist in database
        new_chatbot_response = models.Chatbot(
            message=input_text,
            response=response
        )
    
    db.add(new_chatbot_response)
    db.commit()
    db.refresh(new_chatbot_response)
    
    # Create response with model_used field regardless of database schema
    return schemas.ChatbotResponse(
        id=new_chatbot_response.id,
        message=new_chatbot_response.message,
        response=new_chatbot_response.response,
        created_at=new_chatbot_response.created_at,
        model_used=model_type
    )



@router.get("/", response_model=list[schemas.ChatbotResponse] , status_code=status.HTTP_200_OK)
async def get_all_chatbot_responses(
    db: Session = Depends(get_db),
    limit: int = 10,
    skip: int = 0,
    search: Optional[str] = ""
):
    results = db.query(models.Chatbot).filter(
        models.Chatbot.message.contains(search)
    ).offset(skip).limit(limit).all()
    if not results:
        raise HTTPException(status_code=404, detail="No chatbot responses found")
    
    return results
