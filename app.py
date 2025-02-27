import os
import pandas as pd
import faiss
import requests
import streamlit as st
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Set the correct path for the data file
data_path = os.path.join(os.getcwd(), "data", "HS_Harz_data.csv")

# Debugging: Print current working directory and check files
st.write("🔹 Current Working Directory:", os.getcwd())
st.write("🔹 Files in Current Directory:", os.listdir())
st.write("🔹 Does 'data' folder exist?", os.path.exists("data"))
st.write("🔹 Files inside 'data' folder:", os.listdir("data") if os.path.exists("data") else "Folder not found")
st.write("🔹 Full file path:", os.path.abspath(data_path))
st.write("🔹 Does the file exist?", os.path.exists(data_path))

# Check if the data file exists
if not os.path.exists(data_path):
    st.error(f"Missing data file: {data_path}")
    st.stop()

# Load Hochschule Harz data
df = pd.read_csv(data_path)

# Ensure required columns exist
required_columns = {"University", "Course", "Admission Deadline", "Language", "Fees"}
if not required_columns.issubset(df.columns):
    st.error(f"CSV file must contain columns: {', '.join(required_columns)}")
    st.stop()

# Convert data into text format for embeddings
texts = df.apply(lambda row: f"{row['University']} - {row['Course']} - {row['Admission Deadline']} - {row['Language']} - {row['Fees']}", axis=1).tolist()

# Load sentence transformer model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize or load FAISS index
index_path = "faiss_index/index"
if os.path.exists(index_path + ".faiss"):
    vector_store = FAISS.load_local(index_path, embedding_model)
else:
    vector_store = FAISS.from_texts(texts, embedding_model)
    vector_store.save_local(index_path)

# Streamlit app interface
st.title("🎓 CampusGuideGPT - Hochschule Harz Info")

# User input
user_query = st.text_input("Ask about Hochschule Harz admissions, courses, and deadlines:")

if user_query:
    # Convert query to embeddings
    query_embedding = embedding_model.embed_query(user_query)

    # Retrieve relevant information
    results = vector_store.similarity_search_by_vector(query_embedding, k=2)
    relevant_info = "\n".join([result.page_content for result in results])

    # Generate response using Groq API
    def get_groq_model_response(api_key, model_name, prompt):
        url = "https://api.groq.com/openai/v1/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {"model": model_name, "prompt": prompt, "max_tokens": 200, "temperature": 0.7, "stop": ["\n"]}
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["choices"][0]["text"].strip()
        else:
            return f"Error: {response.status_code}, {response.text}"

    groq_response = get_groq_model_response(api_key, model_name, f"Answer the question based on this information: {relevant_info}\n\nQuestion: {user_query}")

    # Display response
    st.write(groq_response)
