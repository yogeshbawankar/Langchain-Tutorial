from langchain_core.output_parsers import PydanticOutputParser 
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = os.environ.get("GOOGLE_CHAT_MODEL")

# Define the model 

chat_model = ChatGoogleGenerativeAI(model = MODEL)

class Person(BaseModel):
    name : str = Field(description= "Name of the person")
    age : int = Field(description="Age of the person")
    city : str = Field(description="Name of city that person belongs to. ")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Generate the name, age and city of a fictional {place} person  \n {format_instructions}",
    input_variables=['place'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

# Without using chain 

prompt = template.invoke({'place':'indain'})

print(prompt)

result = chat_model.invoke(prompt)

print(type(result))

print(result.content)


# With using chain 

chain = template | chat_model | parser 

result = chain.invoke({'place':'Shrilanka'})

print(type(result))

print(result)


