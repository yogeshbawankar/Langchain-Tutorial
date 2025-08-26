# Document loader demo

from langchain_community.document_loaders import TextLoader

loader = TextLoader(r'C:\Users\Lenovo\Desktop\LANGCHAIN_MODELS\LangChain_Loaders\dummy_data.txt')

# loading a loader 
docs = loader.load()

# docs type is python List
print(type(docs))

# printing the first page of the docs 
# print(docs[0])

# printing the length of the docs in pages 
print(len(docs))

# print the page content of the docs 
print(docs[0].page_content)

# print the metadata about the docs 
print(docs[0].metadata)