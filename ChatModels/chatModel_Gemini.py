# Import necessary libraries
from langchain_google_genai import ChatGoogleGenerativeAI           
from dotenv import load_dotenv
import os 

load_dotenv()

# GOOGLE_CHAT_MODEL = "gemini-1.5-flash"
MODEL = os.environ.get("GOOGLE_CHAT_MODEL")


# Initialize the Gemini chat model
chat_model = ChatGoogleGenerativeAI(
    model= MODEL,
    temperature=0.7
)
# Invoke the model with a prompt
result = chat_model.invoke("Tell about 'One Piece' anime.")

# Print the result
print(result.content)
