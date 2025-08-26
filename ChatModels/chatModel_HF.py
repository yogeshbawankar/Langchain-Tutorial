# chatModel_HF.py

from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load token from .env
load_dotenv()

# ✅ Replace this with the real URL you got from Hugging Face endpoint
ENDPOINT_URL = "https://your-real-endpoint-name.hf.space"

# Initialize the client
llm = HuggingFaceEndpoint(
    endpoint_url=ENDPOINT_URL,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.7,
    max_new_tokens=200,
)

# Prompt
prompt = "What is the capital of France?"

# Get result
result = llm.invoke(prompt)
print(result)