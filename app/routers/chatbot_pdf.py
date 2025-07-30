from .. import models , schemas 
from ..database import get_db
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from functools import lru_cache
from ..config import settings


from fastapi import APIRouter, status, Depends
from fastapi.responses import JSONResponse
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from ..schemas import PDFPathRequest, QuestionRequest, QAResponse
from functools import lru_cache
from pathlib import Path
import os, torch, logging

router = APIRouter(
    prefix="/pdfqa",
    tags=["PDF QA"]
)

logger = logging.getLogger(__name__)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

conversation_retrieval_chain = None
chat_history = []

# === LangChain PDF QA Pipeline ===
class PDFQAPipeline:
    def __init__(self):
       

        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=256,
            openai_api_key=settings.openai_api_key
        )

        self.embeddings = HuggingFaceInstructEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": DEVICE}
        )

    def process_pdf(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(texts, embedding=self.embeddings)

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6}),
            return_source_documents=False,
            input_key="question"
        )

        return qa_chain


@lru_cache()
def get_pdfqa_pipeline():
    return PDFQAPipeline()



@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_pdf_from_path(
    req: PDFPathRequest,
    pipeline=Depends(get_pdfqa_pipeline)
):
    global conversation_retrieval_chain

    path = Path(req.pdf_path)
    if not path.exists() or not path.suffix == ".pdf":
        return JSONResponse(status_code=400, content={"error": f"Invalid PDF path: {req.pdf_path}"})

    conversation_retrieval_chain = pipeline.process_pdf(str(path.resolve()))
    return {"message": f"{path.name} processed successfully."}



@router.post("/ask", response_model=QAResponse)
async def ask_question(
    req: QuestionRequest,
    db: Session = Depends(get_db),  # Inject DB session
    pipeline=Depends(get_pdfqa_pipeline)
):
    global conversation_retrieval_chain, chat_history

    if conversation_retrieval_chain is None:
        return JSONResponse(status_code=400, content={"error": "No PDF has been processed yet."})

    result = conversation_retrieval_chain.invoke({
        "question": req.question,
        "chat_history": chat_history
    })

    answer = result["result"]
    chat_history.append((req.question, answer))

    # Save to DB
    entry = models.PDFChatbot(
        pdf_path="most_recent.pdf",  # You can store the real path if tracked
        question=req.question,
        answer=answer
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    return entry
