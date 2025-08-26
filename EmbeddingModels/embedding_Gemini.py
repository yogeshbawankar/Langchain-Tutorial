# We generate embeddings using GEMINI. 

# Import necessary libraries
from langchain_google_genai import ChatGoogleGenerativeAI           
from dotenv import load_dotenv
import google.generativeai as genai
import os 

load_dotenv()

# configure API_Key
key = os.environ.get("GOOGLE_API_KEY")

try: 
    genai.configure(api_key=key)
    print("API_Key Configured..")
except Exception as e: 
    print(e)

# step 1: Define the text you want to embed

my_text = "The launch of the James Webb Space Telescope has opened a new window into the early universe."

# step 2: Generate the embeddings

result = genai.embed_content(
    model="models/embedding-001",
    content = my_text,
    task_type= "RETRIEVAL_DOCUMENT",
    title = " A sentence about the space exploration"
)

# step 3: Inspect the result 
embed_vector = result['embedding']

print("Text :",my_text)

print("-" * 50)

# check the dimension of our vector 
print("Embedding Dimension:", len(embed_vector))

# let's check our vector 
print(embed_vector)