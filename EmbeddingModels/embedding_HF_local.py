# Downloading and using a local hugging face embedding model
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv

load_dotenv() 

# Load environment variables from .env file
embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
# Sample text to embed
text = "Paris is the capital of France."

# Invoke the embeddings model with a query
vector = embedding.embed_query(text)    

# Print the result
print(str(vector))