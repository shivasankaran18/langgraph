from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

llm = ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)