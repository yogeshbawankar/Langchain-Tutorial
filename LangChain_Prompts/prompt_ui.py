from langchain_google_genai import ChatGoogleGenerativeAI           
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

import os 

import streamlit as st

load_dotenv()

# GOOGLE_CHAT_MODEL = "gemini-1.5-flash"
MODEL = os.environ.get("GOOGLE_CHAT_MODEL")


# Initialize the Gemini chat model
chat_model = ChatGoogleGenerativeAI(
    model= MODEL,
)

# Heading for streamlit website
st.header("Research Tool")

paper_input = st.selectbox("Select Research Paper Name", ["Select...", "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# User prompt 
# user_input = st.text_input("Enter your prompt : ")

# Now let's use langchain prompt template 
template = PromptTemplate(
    template = 
    """
    Please summarize the research paper titled "{paper_input}" with the following specifications:
    Explanation Style: {style_input}
    Explanation length: {length_input}

    Mathematical Details:

    Include relevant mathematical equations if present in the paper.

    Explain the mathematical concepts using simple, intuitive code snippets where applicable.

    Analogies:

    Use relatable analogies to simplify complex ideas.

    If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
    Ensure the summary is clear, accurate, and aligned with the provided style and length.
    """,
    input_varibales = ['paper_input','style_input','length_input']
)

# prompt = template.invoke({
#         'paper_input':paper_input,
#         'style_input':style_input,
#         'length_input':length_input
#     })

if st.button('Summrize'):
    
    # result = chat_model.invoke(user_input)
    # st.write(result.content)
    # result = chat_model.invoke(prompt)

    # Try to use chain for same task
    chain = template | chat_model
    result = chain.invoke({       
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input   
    })
    st.write(result.content)


