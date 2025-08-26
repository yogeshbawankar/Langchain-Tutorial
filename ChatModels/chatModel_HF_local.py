# It loccally loads a Hugging Face model using the langchain_huggingface library.

# Import necessary libraries
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

import os 

load_dotenv()

llm = HuggingFacePipeline(
    model_id= 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task="text-generation",
    pipeline=dict(
        temperature=0.7,
        max_length=512
    )
)
# Initialize the Hugging Face chat model
model = ChatHuggingFace(
    llm=llm
)   
# Invoke the model with a prompt
result = model.invoke("What is the capital of France?")

# Print the result
print(result)
