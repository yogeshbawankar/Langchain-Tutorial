from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os 

load_dotenv()

MODEL = os.environ.get('GOOGLE_EMBEDDING_MODEL')

embed_model = GoogleGenerativeAIEmbeddings(model = MODEL)

text = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# If the distance between two sentences is greater than 1 standard deviation is consider as new semantic meaning text. (1 standard cause we set breakpoin threashold amount to 1.)

splitter = SemanticChunker(
    embed_model, 
    breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1
)

# 3. Call create_documents with a LIST of texts.
result = splitter.create_documents([text])

print(f"Number of chunks: {len(result)}")
print("-" * 20)
for i, doc in enumerate(result):
    print(f"Chunk {i+1}:\n{doc.page_content}\n")
    print("-" * 20)



