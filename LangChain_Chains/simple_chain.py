from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

MODEL = os.environ.get('GOOGLE_CHAT_MODEL')

# Model is loaded 
chat_model = ChatGoogleGenerativeAI(model = MODEL)

# Create a simple prompt 
prompt = PromptTemplate(
    template = 'Generate five intreasting facts about {topic}.', 
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | chat_model | parser 

result = chain.invoke({'topic' : 'cricket'})

print(result)

chain.get_graph().print_ascii()
