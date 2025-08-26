
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=32
)

# Invoke the embeddings model with a query
result = embeddings.embed_query("What is the capital of France?")   

# Print the result
print(str(result))