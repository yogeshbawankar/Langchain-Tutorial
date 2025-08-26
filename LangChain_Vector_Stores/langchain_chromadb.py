import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style.",
        metadata={"team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his elegant batting.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills are legendary.",
        metadata={"team": "Chennai Super Kings"}
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his deadly yorkers.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="KL Rahul is a stylish opening batsman known for his consistency and elegant stroke play. He has captained Lucknow Super Giants.",
        metadata={"team": "Lucknow Super Giants"}
    )
]


model_name = os.environ.get('GOOGLE_EMBEDDING_MODEL')
embed_model = GoogleGenerativeAIEmbeddings(model=model_name)

# Create and Populate the Vector Store 

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embed_model,
    persist_directory='chroma_db_new',  
    collection_name='ipl_players'
)

print("Vector Store created successfully!")

# Let's see our documents 
retrieved_docs = vector_store.get(include=['documents', 'metadatas'])

print("\n Stored Documents ")
print(retrieved_docs)

# Example of a similarity search 
query = "Who is a fast bowler?"
search_results = vector_store.similarity_search(query, k=1)
print(f"\n Similarity Search for '{query}' ")
print(search_results[0].page_content)

