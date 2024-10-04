import os
import numpy as np
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import requests
from requests.auth import HTTPBasicAuth
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import faiss
import streamlit as st  # Import Streamlit for the client interface

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Embeddings and FAISS
dimension = 1536  # The embedding dimension for OpenAI's embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
faiss_index = faiss.IndexFlatL2(dimension)
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Create a prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Please respond to the user's queries")
    ]
)

# Function to query Confluence
def query_confluence(query):
    confluence_url = "https://your-domain.atlassian.net/wiki/rest/api/content/search"
    auth = HTTPBasicAuth('your-email@example.com', 'your-api-token')
    params = {'cql': query, 'expand': 'body.storage'}
    response = requests.get(confluence_url, params=params, auth=auth)
    return response.json().get('results', [])

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
        
        # Step 1: Retrieve documents from Confluence and Jira
        confluence_docs = query_confluence(query)
        jira_docs = query_jira(query)

        # Combine and embed the documents
        doc_texts = [doc['body']['storage']['value'] for doc in confluence_docs] + \
                    [issue['fields'].get('description', '') for issue in jira_docs]
        embeddings = embedding_model.embed_documents(doc_texts)

        # Convert embeddings list to a NumPy array
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Add the embeddings to the FAISS index
        faiss_index.add(embeddings_np)
        
        # Step 2: Retrieve relevant documents from FAISS
        query_embedding = embedding_model.embed_query(query)
        query_embedding_np = np.array([query_embedding]).astype('float32')
        distances, indices = faiss_index.search(query_embedding_np, k=5)
        relevant_docs = [doc_texts[i] for i in indices[0]]
        context = " ".join(relevant_docs)

        # Step 3: Generate a response using OpenAI
        response = self.llm.generate([context + "\n\n" + query], max_tokens=200)
        return response[0]  # Adjusted to extract the correct content

# Create an instance of AdvancedRAGChain with the required prompt and llm
rag_chain = AdvancedRAGChain(prompt=prompt, llm=llm)

# Streamlit Client Interface
st.title("Advanced RAG Query System")
st.write("Enter your query below and get responses from Confluence and Jira.")

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
