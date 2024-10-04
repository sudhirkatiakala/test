import os
import numpy as np
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import requests
from requests.auth import HTTPBasicAuth
from langchain.chains import LLMChain
import faiss
import streamlit as st
from langchain.document_loaders import WebBaseLoader
import bs4
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Embeddings and FAISS
dimension = 1536  # The embedding dimension for OpenAI's embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
faiss_index = faiss.IndexFlatL2(dimension)
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Function to load data from a webpage using WebBaseLoader
def load_webpage_content():
    loader = WebBaseLoader(
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="mw-content-ltr mw-parser-output"))
    )
    text_documents = loader.load()
    return text_documents

# Function to query Jira
def query_jira(query):
    jira_url = "https://your-domain.atlassian.net/rest/api/3/search"
    auth = HTTPBasicAuth('your-email@example.com', 'your-api-token')
    jql = {'jql': query, 'maxResults': 10}
    response = requests.get(jira_url, params=jql, auth=auth)
    return response.json().get('issues', [])

# LangChain class for processing queries with advanced RAG
class AdvancedRAGChain(LLMChain):
    def __init__(self, prompt, llm):
        super().__init__(prompt=prompt, llm=llm)

    def call(self, inputs):
        query = inputs["query"]
        
        # Step 1: Load documents from the webpage using WebBaseLoader
        web_docs = load_webpage_content()
        web_doc_texts = [doc.page_content for doc in web_docs]

        # Step 2: Query Jira and retrieve documents
        jira_docs = query_jira(query)
        jira_doc_texts = [issue['fields'].get('description', '') for issue in jira_docs]

        # Combine and embed the documents
        doc_texts = web_doc_texts + jira_doc_texts
        embeddings = embedding_model.embed_documents(doc_texts)

        # Convert embeddings list to a NumPy array
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Add the embeddings to the FAISS index
        faiss_index.add(embeddings_np)
        
        # Step 3: Retrieve relevant documents from FAISS
        query_embedding = embedding_model.embed_query(query)
        query_embedding_np = np.array([query_embedding]).astype('float32')
        distances, indices = faiss_index.search(query_embedding_np, k=5)
        relevant_docs = [doc_texts[i] for i in indices[0]]
        context = " ".join(relevant_docs)

        # Step 4: Generate a response using OpenAI
        messages = [
            SystemMessage(content="Please respond to the user's queries."),
            HumanMessage(content=context + "\n\n" + query),
        ]
        response = self.llm.generate(messages, max_tokens=200)
        return response.generations[0].text  # Extract the content of the response

# Create the prompt (this was missing in the previous code)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Please respond to the user's queries")
])

# Create an instance of AdvancedRAGChain with the required prompt and llm
rag_chain = AdvancedRAGChain(prompt=prompt, llm=llm)

# Streamlit Client Interface
st.title("Advanced RAG Query System")
st.write("Enter your query below and get responses from a webpage and Jira.")

# User input query
user_query = st.text_input("Enter your query here:")

# If the user submits a query
if st.button("Submit"):
    if user_query:
        response = rag_chain.call({"query": user_query})
        st.write("### Response")
        st.write(response)
    else:
        st.write("Please enter a query to get a response.")
