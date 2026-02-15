from langchain_google_genai import ChatGoogleGenerativeAI
import os

def get_llm(api_key: str):

    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=0
    )
