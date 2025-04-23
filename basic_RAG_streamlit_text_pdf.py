'''
This version should now handle:
All file sizes (100KB to 50MB)
Both TXT and PDF formats
Proper newline formatting in prompts
'''
# Installation: pip install streamlit mistralai==0.4.2 numpy faiss-cpu langchain pymupdf tqdm
import streamlit as st
from mistralai.client import MistralClient
import numpy as np
import faiss
import fitz
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Initialize client
client = MistralClient(api_key="add the api_key")  # Replace with actual key

# Configuration
MAX_FILE_SIZE_MB = 100
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 256
EMBEDDING_BATCH_SIZE = 32

# File processing with PDF support
@st.cache_data(show_spinner=False)
def load_and_split_text(uploaded_file):
    file_size = sys.getsizeof(uploaded_file) / (1024 * 1024)
    if file_size > MAX_FILE_SIZE_MB:
        raise ValueError(f"File size exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB")

    file_bytes = uploaded_file.read()
    
    if uploaded_file.name.lower().endswith('.pdf'):
        text = []
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in tqdm(doc, desc="Processing PDF pages"):
                text.append(page.get_text())
        text = "\n".join(text)
    else:
        text = file_bytes.decode("utf-8", errors="replace")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return text_splitter.split_text(text)

# Embedding generation
@st.cache_data(show_spinner=False)
def get_embeddings(texts):
    embeddings = []
    progress_bar = st.progress(0)
    
    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Generating embeddings"):
        batch = texts[i:i+EMBEDDING_BATCH_SIZE]
        try:
            response = client.embeddings(model="mistral-embed", input=batch)
            embeddings.extend([e.embedding for e in response.data])
            progress_bar.progress(min((i+EMBEDDING_BATCH_SIZE)/len(texts), 1.0))
        except Exception as e:
            st.error(f"Error processing batch {i//EMBEDDING_BATCH_SIZE}: {str(e)}")
            raise
    
    progress_bar.empty()
    return np.array(embeddings, dtype=np.float32)

# FAISS index creation
def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    
    batch_size = 10000
    for i in range(0, embeddings.shape[0], batch_size):
        index.add(embeddings[i:i+batch_size])
    
    return index

# Fixed RAG query processing with correct string formatting
def rag_query(question, chunks, index, k=5):
    try:
        response = client.embeddings(model="mistral-embed", input=[question])
        q_embedding = np.array([response.data[0].embedding], dtype=np.float32)
        
        distances, indices = index.search(q_embedding, k)
        context = [chunks[i] for i in indices[0]]
        
        # Properly formatted prompt without backslash issues
        context_str = '\n'.join(context)
        prompt = (
            "Context:\n"
            f"{context_str}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        
        response = client.chat(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Streamlit UI
def main():
    st.title("📄 Document Chat with Mistral RAG")
    
    uploaded_file = st.file_uploader("Upload document (TXT/PDF)", type=["txt", "pdf"])
    
    if uploaded_file:
        try:
            with st.spinner("Processing document..."):
                chunks = load_and_split_text(uploaded_file)
                
                if not chunks:
                    st.error("No text extracted from document")
                    return
                
                with st.spinner(f"Generating embeddings ({len(chunks)} chunks)..."):
                    embeddings = get_embeddings(chunks)
                
                with st.spinner("Building search index..."):
                    index = create_faiss_index(embeddings)
                
                st.success(f"Ready! Processed {len(chunks)} chunks")
                st.session_state.chunks = chunks
                st.session_state.index = index

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return
        
        if "history" not in st.session_state:
            st.session_state.history = []
            
        question = st.chat_input("Ask about the document:")
        if question:
            try:
                answer = rag_query(question, st.session_state.chunks, st.session_state.index)
                st.session_state.history.append((question, answer))
            except Exception as e:
                st.error(f"Query failed: {str(e)}")
            
            for q, a in st.session_state.history:
                with st.chat_message("user"):
                    st.write(q)
                with st.chat_message("assistant"):
                    st.write(a)

if __name__ == "__main__":
    main()