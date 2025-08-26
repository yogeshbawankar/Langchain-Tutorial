"""
WARNING : This error occurs because the langchain-google-genai library's .with_structured_output() method requires a Pydantic BaseModel to correctly define and inspect the output schema. It cannot process a standard Python TypedDict, which it sees as a simple dictionary without a clear structure to analyze.

But it works with OpenAI Models.

Code is correct. 
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel,EmailStr,Field  # Pydantic is used for data validation.
from typing import Optional,Annotated, Literal,TypedDict
import json
import os 

load_dotenv()

MODEL = os.environ.get("GOOGLE_CHAT_MODEL")

chat_model = ChatGoogleGenerativeAI(model = MODEL)

data = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by Yogesh Bawankar
"""

# schema 
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]

    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]

    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]

    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]

    name: Annotated[Optional[str], "Write the name of the reviewer"]

# Passing our structure schema to the model
structure_model = chat_model.with_structured_output(Review)

# passing our data to custom structure schema
result = structure_model(data)
print(result)
