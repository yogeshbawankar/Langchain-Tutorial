from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template with messages placeholder
chat_template = ChatPromptTemplate([
    ("system", "Your are a helpful customer support assistant."),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("human", "{query}")
])

chat_history = []
# load chat history
with open("chat_history.txt", "r") as file:
    chat_history.extend(file.readlines())

print("Chat History : ",chat_history)

prompt = chat_template.invoke({'chat_history':chat_history,'query':'Where is my refund'})

print("Prompt : ",prompt)



