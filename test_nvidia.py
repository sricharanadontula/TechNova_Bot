from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

MODEL = "nvidia/nemotron-3-super-120b-a12b"

llm = ChatNVIDIA(
    model=MODEL,
    temperature=0
)

response = llm.invoke(
    "Hello! Explain what an AI chatbot is in one sentence."
)

print(response.content)