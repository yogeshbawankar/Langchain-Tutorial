from langchain_core import runnables
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

MODEL = os.environ.get('GOOGLE_CHAT_MODEL')

chat_model = ChatGoogleGenerativeAI(model = MODEL)

prompt1 = PromptTemplate(
    template = " Generate a tweet about topic -> {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template = " Generate a linkedin post about topic ->{topic}",
    input_variables= ['topic']
)

parser = StrOutputParser()

# Initializing Runnable parallel
parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1,chat_model,parser),
    'linkedin':RunnableSequence(prompt2,chat_model,parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(result['tweet'])
print("="*50)
print(result['linkedin'])
