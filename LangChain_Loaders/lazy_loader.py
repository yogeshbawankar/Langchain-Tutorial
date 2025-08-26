from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "Path to the folder",
    glob = "Pattern to retrive file for ex '*.pdf' ",
    loader_cls = "Mention loader class for ex PyPDFLoader"
)
# Using lazy_load
docs = loader.lazy_load()

print(docs[0].page_content)
print(docs[0].metadata)