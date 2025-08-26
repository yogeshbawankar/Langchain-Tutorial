from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

load_dotenv()

# Document for embeddings
document = [
    "Virat kholi is an Indain cricketer know for his aggresive batting and leadership.",
    "MS Dhoni is a former Indain captain famous for his clam demeanor and finishing skills.",
    "Sachin Tendulkar, also know as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indain fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about virat kohli."

# first create embeddings model 
embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

# Generate document and query embeddings 
doc_embed = embeddings.embed_documents(document)

query_embed = embeddings.embed_query(query)

# cosine similarity always recive 2D List.
scores = cosine_similarity([query_embed],doc_embed)

# It sort embedding score and assign index
idx , score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(document[idx])
print("Similarity Score is : ",score)
