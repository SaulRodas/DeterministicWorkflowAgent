import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.llm import get_llm
from src.retrievers.csv_retriever import load_csv
from src.retrievers.txt_retriever import rag_txt
from src.workflow import deterministic_workflow

# App init
app = FastAPI(
    title="Question Agent API",
    version="1.0"
)

# Request Schema
class QuestionRequest(BaseModel):
    user_question: str


def build_pipeline_assets(data_dir: str = "data"):
    csv_path = os.path.join(data_dir, "datos_clima_mexico.csv")
    txt_path = os.path.join(data_dir, "GPT-41_PromptingGuide.txt")
    image_path = os.path.join(data_dir, "maiz_info.jpg")

    data = load_csv(csv_path)
    texts, index, metadatas = rag_txt(txt_path)

    return data, texts, index, metadatas, image_path


# Load assets ONCE (startup)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError(
        "No se encontro GOOGLE_API_KEY. Define la variable de entorno."
    )

chat_model = get_llm(api_key)
data, texts, index, metadatas, image_path = build_pipeline_assets()


# Endpoint
@app.post("/question_agent")
def question_agent(request: QuestionRequest):

    if not request.user_question.strip():
        raise HTTPException(
            status_code=400,
            detail="La pregunta no puede estar vacia."
        )

    result = deterministic_workflow(
        user_question=request.user_question,
        chat_model=chat_model,
        data=data,
        index=index,
        texts=texts,
        metadatas=metadatas,
        image_path=image_path,
    )

    return {
        "original_answer": result.original_answer,
        "dry_answer": result.dry_answer,
        "funny_answer": result.funny_answer,
    }
