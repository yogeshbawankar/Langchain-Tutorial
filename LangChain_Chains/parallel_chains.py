from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

MODEL = os.environ.get('GOOGLE_CHAT_MODEL')

# YOU CAN USE DIFFERENT MODEL FOR THIS TASK. 

chat_model_1 = ChatGoogleGenerativeAI(model = MODEL)

# Assume this a other company chat model 

chat_model_2 = ChatGoogleGenerativeAI(model = MODEL)

# Prompt1 -> To Generate notes from given topic 

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following text : {text}",
    input_variables=['text']
)

# Prompt2 -> It will generate quize. 

prompt2 = PromptTemplate(
    template="Generate five short questions and answer from the following text : {text} ",
    input_variables=['text']
)

# Prompt3 -> It will merge both text. 
prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes','quiz']
)

# Initialize parser 
parser = StrOutputParser()

# Creating parallel chains using RunnablesPrallel
parallel_chain = RunnableParallel(
    {
        'notes':prompt1 | chat_model_1 | parser,
        'quiz' :prompt2 | chat_model_2 | parser
    }
)

# Merge chain that will merge our outputs 
merge_chain = prompt3 | chat_model_1 | parser

# Final chain to call all the chains
chain = parallel_chain | merge_chain 

# Input text 

text = """Linear regression is a fundamental statistical and machine learning technique used to model the relationship between a dependent variable and one or more independent variables. The core idea is to find the best-fitting straight line, known as the regression line, that represents the relationship between these variables on a scatter plot. . This method is primarily used for two main purposes: to understand the strength and nature of the relationship (e.g., how much do sales increase for every dollar spent on advertising?) and to make predictions about the dependent variable based on the values of the independent variables (e.g., predicting a house's price based on its size).

The model's output is an equation for a line, which is defined by a slope and an intercept. The slope represents the rate of change, indicating how much the dependent variable is expected to change for a one-unit increase in the independent variable. The intercept is the value of the dependent variable when the independent variable is zero. There are two main types of linear regression: simple linear regression, which involves only one independent variable to predict an outcome, and multiple linear regression, which uses two or more independent variables to create a more robust and often more accurate predictive model
"""
result = chain.invoke(text)

print(result)

chain.get_graph().print_ascii()
