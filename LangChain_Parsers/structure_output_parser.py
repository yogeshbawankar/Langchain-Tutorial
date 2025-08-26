"""
In the structure output parser we can enforce schema so that we can get json format as we want. 
"""
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = os.environ.get("GOOGLE_CHAT_MODEL")

chat_model = ChatGoogleGenerativeAI(model=MODEL)

# schema 

schema = [
    ResponseSchema(name = 'fact_1',description = "Fact 1 about the topic "),
    ResponseSchema(name = 'fact_2',description = "Fact 2 about the topic "),
    ResponseSchema(name = 'fact_3',description = "Fact 3 about the topic ")
        
]

# Initialize our parser
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Give three fact about {topic} \n {format_instruction}",
    input_variables = ['topic'],
    partial_variables= {'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'black hole'})

chain = chat_model | parser 

result = chain.invoke(prompt)

print(result)



