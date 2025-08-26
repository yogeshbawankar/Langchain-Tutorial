from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what's possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""
txt_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap = 0
)

result = txt_splitter.split_text(text)

# print(result)

pycode = """
    from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

txt_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap = 0
)

result = txt_splitter.split_text(text)

# print(result)

pycode = """

"""

py_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 10, 
    Language = Language.PYTHON
)

result = py_splitter.split_text(pycode)

"""

py_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size = 200,
    chunk_overlap = 10, 
    language = Language.PYTHON
)

result = py_splitter.split_text(pycode)

print(result[0])