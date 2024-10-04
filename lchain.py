from langchain_community.llms import OpenAI
from transformers import pipeline
import requests
from sentence_transformers import SentenceTransformer
import faiss
from requests.auth import HTTPBasicAuth

# Initialize Hugging Face and FAISS
dimension = 384  # Embedding dimension, depends on the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
faiss_index = faiss.IndexFlatL2(dimension)
generator = pipeline('text-generation', model='gpt-2')

# Function to query Confluence
def query_confluence(query):
    confluence_url = "https://your-domain.atlassian.net/wiki/rest/api/content/search"
    auth = HTTPBasicAuth('your-email@example.com', 'your-api-token')
    params = {'cql': query, 'expand': 'body.storage'}
    response = requests.get(confluence_url, params=params, auth=auth)
    return response.json()['results']

# Function to query Jira
def query_jira(query):
    jira_url = "https://your-domain.atlassian.net/rest/api/3/search"
    auth = HTTPBasicAuth('your-email@example.com', 'your-api-token')
    jql = {'jql': query, 'maxResults': 10}
    response = requests.get(jira_url, params=jql, auth=auth)
    return response.json()['issues']

# LangChain for processing queries with advanced RAG
class AdvancedRAGChain(Chain):
    def __init__(self):
        super().__init__()
        self.model = OpenAI(model_name="text-davinci-003")  # Replace with your preferred model

    def call(self, inputs):
        query = inputs["query"]
        
        # Step 1: Retrieve documents from Confluence and Jira
        confluence_docs = query_confluence(query)
        jira_docs = query_jira(query)

        # Combine and embed the documents
        combined_docs = confluence_docs + jira_docs
        doc_texts = [doc['body']['storage']['value'] for doc in confluence_docs] + \
                    [issue['fields']['description'] for issue in jira_docs]
        embeddings = model.encode(doc_texts)
        faiss_index.add(embeddings)
        
        # Step 2: Retrieve relevant documents from FAISS
        query_embedding = model.encode([query])
        distances, indices = faiss_index.search(query_embedding, top_k=5)
        relevant_docs = [doc_texts[i] for i in indices[0]]
        context = " ".join(relevant_docs)

        # Step 3: Generate a response using Hugging Face
        response = generator(context + "\n\n" + query, max_length=200, num_return_sequences=1)
        return response[0]['generated_text']

rag_chain = AdvancedRAGChain()

async def dispatch_to_langchain(context: TurnContext):
    user_query = context.activity.text
    response = rag_chain.call({"query": user_query})
    await context.send_activity(response)
