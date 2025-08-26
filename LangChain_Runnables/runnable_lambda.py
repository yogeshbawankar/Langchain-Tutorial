from langchain_core import runnables
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnableLambda,RunnablePassthrough

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

MODEL = os.environ.get('GOOGLE_CHAT_MODEL')

chat_model = ChatGoogleGenerativeAI(model = MODEL)

# A function for word count 
def wordCounter(text):
    return len(text.split())

# Using runnable lambda count word
runnable_word_counter = RunnableLambda(wordCounter)

prompt = PromptTemplate(
    template = " Generate a joke about topic -> {topic}",
    input_variables= ['topic']
)

parser = StrOutputParser()

# Joke gen chain give joke
joke_gen_chain = RunnableSequence(prompt,chat_model,parser)

# Paralled chain to runs both chain parallel.
paralled_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count': RunnableLambda(wordCounter)
})

# Final chain to combine output
final_chain = RunnableSequence(joke_gen_chain,paralled_chain)

result = final_chain.invoke({'topic':'AI'})

print(f" \n Word count -{result['word_count']} \n Joke - {result['joke']} ")
