from langchain_community.document_loaders import WebBaseLoader

url = # Enter you website URL 

loader = WebBaseLoader()

docs = loader.load()

print(docs[0])