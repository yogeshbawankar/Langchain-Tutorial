"""
    Chat Prompt Template
    - A chat prompt template is a template that is used to format the input to a chat model.
    - It is a list of messages that are used to format the input to a chat model.
    - It is a list of messages that are used to format the input to a chat model.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os 

load_dotenv()

MODEL = os.environ.get("GOOGLE_CHAT_MODEL")

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} assistant."),
    ("human", "Explain the concept of {topic} in a way that is easy to understand.")
])

prompt = chat_template.invoke({
    "domain": "AI",
    "topic": "AI"
})

print("PROMPT : ",prompt)

chat_model = ChatGoogleGenerativeAI(
    model = MODEL,
)










