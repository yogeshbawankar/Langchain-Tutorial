# Import necessary libraries
from langchain_anthropic import ChatAnthropic   
from dotenv import load_dotenv

load_dotenv()

# Initialize the Anthropic chat model
chat_model = ChatAnthropic(
    model="claude-2",
    temperature=0.7
)

# Invoke the model with a prompt
result = chat_model.invoke("What is the capital of France?")        

# Print the result
print(result)
