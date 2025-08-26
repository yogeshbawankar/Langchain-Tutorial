from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os 

load_dotenv()

MODEL = os.environ.get("GOOGLE_CHAT_MODEL")

chat_model = ChatGoogleGenerativeAI(model = MODEL)

# 1st prompt -> detailed prompt 

template1 = PromptTemplate(
    template = " Write a detailed report on {topic}.",
    input_variables= ['topic']
)

# 2st prompt -> summary

template2 = PromptTemplate(
    template = "Write a 5 line of summary on the following text. /n {text}",
    input_variables= ['text']
)

# prompt1 = template1.invoke({'topic':'black hole'})

# result1 = chat_model.invoke(prompt1 )

# prompt2 = template2.invoke({'text':result1.content})

# result2 = chat_model.invoke(prompt2)

# print(result1.content)

# print("-"*60)

# print(result2)

# ========== Write the same code with string output parser ===========

from langchain_core.output_parsers import StrOutputParser

# Parser 

parser = StrOutputParser()

# Parser gives us only the important text. not the unwanted metadata and other things. 

# Build chain this chain return summary

chain = template1 | chat_model | parser #| template2 | chat_model | parser

result = chain.invoke({'topic':'black hole'})

print(result)
