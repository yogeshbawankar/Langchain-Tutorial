from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = os.environ.get("GOOGLE_CHAT_MODEL")

chat_model = ChatGoogleGenerativeAI(model=MODEL)

# Initialize the parser
parser = JsonOutputParser()

# Define the template
template = PromptTemplate(
    template="Give me the name, age and city of a fictional person.\n{format_instructions}",
    input_variables=[],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# Create the prompt from the template
prompt = template.format()

# Let's check how our prompt looks
print("------- PROMPT -------")
print(prompt)
print("----------------------\n")


# Create the chain
chain = chat_model | parser

# Invoke the chain
result = chain.invoke(prompt)

print("------- RESULT -------")
print(result)
print("----------------------")