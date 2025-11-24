import os
import uvicorn
from typing import List
from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

#Prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{text}")
    ]
)

#Modeli Oluştur
model = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=api_key)

#Parser oluştur
parser = StrOutputParser()

#Chain bağla
chain = prompt_template | model | parser

#FastAPI Uygulamasını tanımla
app = FastAPI(
    title="Translate AI",
    description="A simple API server using LangChain + Groq Llama 3.3",
    version="1.0.0",
)

#Rotayı ekle
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)