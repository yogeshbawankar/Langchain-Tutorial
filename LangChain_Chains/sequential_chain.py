from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

MODEL = os.environ.get('GOOGLE_CHAT_MODEL')

# Loading model 
chat_model = ChatGoogleGenerativeAI(model = MODEL)

# Prompt1 -> Generating report on the topic 
prompt1 = PromptTemplate(
    template = 'Generate a detailed report on {topic}',
    input_variables=['topic']
)

# Prompt2 -> To generate summary for give report
prompt2 = PromptTemplate(
    template='Generate summary for the text : {text}..',
    input_variables=['text']
)

# Initializing parser 
parser = StrOutputParser()

# Chain 
chain = prompt1 | chat_model | parser | prompt2 | chat_model | parser 

# Invoking chain 
result = chain.invoke({'topic':'anime'})

print(result )

# Visualizing chain 
chain.get_graph().print_ascii()