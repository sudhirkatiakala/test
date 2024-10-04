from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
## langscmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"

## Promt Template
prompt= ChatPromptTemplate.from_messages(
    [
        ("system", "Please response to the user querries"),
        ("user", "Question: {question}")
    ]
)

## streamlit framework
st.title("LangChain OpenAI")
input_text = st.text_area("Enter your question here", "What is the capital of France?")

#openAI LLm
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    try: 
        response = chain.invoke({"question": input_text})
        st.write(response)
    except Exception as e:
        st.write(e)
        if "insufficient_quota" in str(e):
            st.write("Insufficient insufficient_quota, please check billing")

