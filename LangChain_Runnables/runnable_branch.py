from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field

import os

load_dotenv()

MODEL = os.environ.get('GOOGLE_CHAT_MODEL')

# YOU CAN USE DIFFERENT MODEL FOR THIS TASK. 

chat_model = ChatGoogleGenerativeAI(model = MODEL)

# validation for our output 
class Feedback(BaseModel):
    sentiment : Literal['Positive','Negative'] = Field(description="Give the sentiment of the feedback.")

# PydanticOutputParser for validation.
parser1 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt1 -> To get the sentiment of the feedback

prompt1 = PromptTemplate(
    template="Classify the sentiment of the given feedback into either positive or negative \n feedback : {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables= {'format_instructions':parser1.get_format_instructions()}
)


# StrOutputParser for textual output.
parser2 = StrOutputParser()

classifier_chain = prompt1 | chat_model | parser1

# Prompt2 -> It will write response to feedback based on sentiment. 

# Positive
prompt2 = PromptTemplate(
    template="You are a customer support agent. Write a brief, empathetic, and professional response to the following positive customer feedback. \n\nCustomer Feedback: {feedback}\n\nYour Response:",
    input_variables=['feedback']
)

# Negative
prompt3 = PromptTemplate(
    template="You are a customer support agent. Write a brief, empathetic, and professional response to the following negative customer feedback. Apologize for the poor experience and ask for more details. \n\nCustomer Feedback: {feedback}\n\nYour Response:",
    input_variables=['feedback']
)

# result = classifier_chain.invoke({'feedback':"This is worst phone."}).sentiment

# print(result)

# RunnablesBranch is similar to if else condition here is the format. 
# branch_chain = RunnableBranch(
#     (condition1,chain1),
#     (condition2,chain2),
#     default chain
# )

# Initializing runnable branch
branch_chain = RunnableBranch(
    (lambda x: x['sentiment'].sentiment == 'Positive', prompt2 | chat_model | parser2),
    (lambda x: x['sentiment'].sentiment == 'Negative', prompt3 | chat_model | parser2),
    RunnableLambda(lambda x: "Could not find sentiment.")
)

# RunnablePassthrough to pass our feedback not only the first classifier_chain as well as branch chains. 
chain = RunnablePassthrough.assign(
    sentiment = classifier_chain
) | branch_chain 


print(chain.invoke({'feedback':'This worst phone.'}))

chain.get_graph().print_ascii()