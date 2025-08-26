from langchain_core import runnables
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

MODEL = os.environ.get('GOOGLE_CHAT_MODEL')

chat_model = ChatGoogleGenerativeAI(model = MODEL)

prompt = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables= ['topic']
)

parser = StrOutputParser()

# Calling RunnableSequence 
chain = RunnableSequence(prompt,chat_model,parser)

print(chain.invoke({'topic':'AI'}))