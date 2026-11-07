
from io import BytesIO

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage
from agents.data_agent import data_agent


app = FastAPI(
    title="Data Agent API",
    description="API for the LangGraph-powered Data Agent",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    question: str
    answer: str


def get_agent_response(question: str, dataset=None) -> str:
    result = data_agent.invoke(
        {
            "messages": [HumanMessage(content=question)]
        },
        config={
            "recursion_limit": 20,
            "configurable": {"dataset": dataset},
        },
    )

    return next(
        (
            message.content
            for message in reversed(result["messages"])
            if isinstance(message, AIMessage)
            and isinstance(message.content, str)
            and message.content.strip()
        ),
        "No response was generated.",
    )


@app.get("/")
def home():
    return {
        "message": "Data Agent API is running",
        "status": "healthy",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid question.",
        )

    try:
        answer = get_agent_response(question)
        return ChatResponse(question=question, answer=answer)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Data Agent error: {type(error).__name__}",
        )


@app.post("/chat/upload", response_model=ChatResponse)
async def chat_with_uploaded_dataset(
    question: str = Form(...),
    file: UploadFile = File(...),
):
    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    filename = file.filename or ""
    contents = await file.read()

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(contents))

        elif filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(contents))

        else:
            raise HTTPException(
                status_code=400,
                detail="Upload a CSV or Excel file.",
            )

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded dataset is empty.",
            )

        answer = get_agent_response(question, dataset=df)

        return ChatResponse(
            question=question,
            answer=answer,
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset processing error: {type(error).__name__}",
        )
