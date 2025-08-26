from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import os 

import streamlit as st 

load_dotenv()

MODEL = os.environ.get("GOOGLE_CHAT_MODEL")

chat_model = ChatGoogleGenerativeAI(
    model = MODEL,
)
# LLM Memory in the form of a list of messages
chat_history = [SystemMessage(content = "You are a helpful assistant that can answer questions and help with tasks.")]

while True: 
    user_input = input('You: ')
    chat_history.append(HumanMessage(content = user_input))

    if user_input == 'exits':
        break
    result = chat_model.invoke(chat_history)
    
    chat_history.append(AIMessage(content = result.content))

    print("AI : ",result.content)

print(chat_history)

