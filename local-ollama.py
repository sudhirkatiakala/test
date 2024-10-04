from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Please respond to the user's queries")
    ]
)

## streamlit framework
st.title("LangChain OpenAI")
input_text = st.text_area("Enter your question here", "What is the capital of India?")

# OpenAI LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
output_parser = StrOutputParser()

# Create a SimpleChain
chain = LLMChain(prompt=prompt, llm=llm, output_parser=output_parser)

if input_text:
    response = chain.run(input_text)
    st.write(response)