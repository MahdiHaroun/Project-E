from .. import models , schemas 
from ..database import get_db
from fastapi import APIRouter, Depends, status , HTTPException
from sqlalchemy.orm import Session



router = APIRouter(
    prefix="/image_cap",
    tags=["Image Captioning"]
)



@router.post("/" , response_model=schemas.ImageCaptioningResponse , status_code=status.HTTP_201_CREATED)
async def create_image_captioning(new_image_caption: schemas.ImageCaptioningRequest , db: Session = Depends(get_db)):
    from PIL import Image
    from transformers import AutoProcessor, BlipForConditionalGeneration
    
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    img_path = new_image_caption.image_path
    image = Image.open(img_path).convert('RGB')
    text = "i see a photo of"
    inputs = processor(images=image, text=text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    new_image_caption = models.image_Captioning(
        image_path=img_path,
        caption=caption
    )

    db.add(new_image_caption)
    db.commit()
    db.refresh(new_image_caption)

    return new_image_caption

    
@router.get("/{image_caption_id}", response_model=schemas.ImageCaptioningResponse)
async def get_image_captioning(image_caption_id: int, db: Session = Depends(get_db)):
    image_caption = db.query(models.image_Captioning).filter(models.image_Captioning.id == image_caption_id).first()
    if not image_caption:
        raise HTTPException(status_code=404, detail="Image caption not found")
    return image_caption

@router.get("/", response_model=list[schemas.ImageCaptioningResponse])
async def get_all_image_captions(db: Session = Depends(get_db)):
    image_captions = db.query(models.image_Captioning).all()
    return image_captions

@router.delete("/{image_caption_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_image_captioning(image_caption_id: int, db: Session = Depends(get_db)):
    image_caption = db.query(models.image_Captioning).filter(models.image_Captioning.id == image_caption_id).first()
    if not image_caption:
        raise HTTPException(status_code=404, detail="Image caption not found")
    
    db.delete(image_caption)
    db.commit()
    return {"detail": "Image caption deleted successfully"}



@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_all_image_captions(db: Session = Depends(get_db)):
    db.query(models.image_Captioning).delete()
    db.commit()
    return {"detail": "All image captions deleted successfully"}