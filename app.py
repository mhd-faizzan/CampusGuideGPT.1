import os
import pandas as pd
import faiss
import requests
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch Groq API Key from the environment
api_key = os.getenv("GROQ_API_KEY")

# Ensure the API key is available
if not api_key:
    st.error("API key is missing. Please add it to the .env file.")
    st.stop()

# Fetch the model name from environment variables or use the provided model
model_name = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# Function to interact with Groq API
def get_groq_model_response(api_key, model_name, prompt):
    url = f"https://api.groq.com/openai/v1/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["text"].strip()
    else:
        return f"Error: {response.status_code}, {response.text}"

# Initialize FAISS and OpenAI Embeddings
embedding_model = OpenAIEmbeddings()

# Load Hochschule Harz data (replace with actual data path)
df = pd.read_csv("data/hochschule_harz_data.csv")
texts = df.apply(lambda row: f"{row['University']} - {row['Course']} - {row['Admission Deadline']} - {row['Language']} - {row['Fees']}", axis=1).tolist()

# Create FAISS index and save it locally
vector_store = FAISS.from_texts(texts, embedding_model)
vector_store.save_local("faiss_index")

# Streamlit app interface
st.title("🎓 CampusGuideGPT - Hochschule Harz Info")

# User input for query
user_query = st.text_input("Ask about Hochschule Harz admissions, courses, and deadlines:")

if user_query:
    # Query FAISS for relevant data
    results = vector_store.similarity_search(user_query, k=2)
    relevant_info = "\n".join([result.page_content for result in results])
    
    # Generate a response using Groq's LLaMA model
    groq_response = get_groq_model_response(api_key, model_name, f"Answer the question based on this information: {relevant_info}\n\nQuestion: {user_query}")
    
    # Display response in Streamlit
    st.write(groq_response)
