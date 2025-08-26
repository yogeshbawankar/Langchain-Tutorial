from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 

load_dotenv()

MODEL = os.environ.get("GOOGLE_CHAT_MODEL")

chat_model = ChatGoogleGenerativeAI(
    model = MODEL,
)

message = [
    SystemMessage(content = "You are a helpful assistant that can answer questions and help with tasks."),
    HumanMessage(content = "What is the capital of France?")
]

result = chat_model.invoke(message)
message.append(AIMessage(content = result.content))
print(message)











