import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
import PyPDF2
import os
import time
from PyPDF2 import PdfReader

st.title("RAG Chatbot for documnet querying")
uploaded_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)

def get_pdf_text(uploaded_files):
    text = ""
    all_text=[]
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        all_text.append(text)
        
    return all_text

    
def retrieve_vector_db(collection,query, n_results=3):
        start = time.time()
        results = collection.query(
        query_embeddings = st.embedding_model.encode(query).tolist(),
        n_results=n_results
        )
        end = time.time()
        elapsed = end - start
        return results['documents'],elapsed


def get_llama2_chat_response(question, context, max_new_tokens=500,end_token ="</s>"): 

        prompt = [
        {
            "role": "system",
            "content": "You are a friendly chatbot and answer the users question based on the given context. If the answer is not in the context provided, just answer `I don't know` please do not give any additional information."
        },
        {
            "role": "user",
           "content": f"Context: {context}\nQuestion: {question}"
        }
    ]   
              
        tokenized_chat = st.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")     
        start = time.time()
        outputs =st.model.generate(tokenized_chat, max_new_tokens=max_new_tokens, temperature=0.00001)
        end = time.time()
        response = st.tokenizer.decode(outputs[0])
        # Find the index of the answer part within the response
        start_idx = response.find("Question:") + len(f"Question: {question}")
        end_idx = response.find(end_token, start_idx)
    
        # Extract the actual response from the generated text
        if end_idx != -1:
            extracted_response = response[start_idx:end_idx].strip()
        else:
            extracted_response = response[start_idx:].strip()
    
        # Remove unwanted characters
        unwanted_characters = ['\n', '\n\n1', '\n2', '</s>','<|eot_id|>','<|end_header_id|>']  # Add other unwanted characters if needed
        for char in unwanted_characters:
            extracted_response = extracted_response.replace(char, '')
    
    # Clean up the response
        if "assistant" in extracted_response:
            extracted_response = extracted_response.split("assistant")[-1].strip()

    # Calculate the elapsed generation time
        elapsed_gen = end - start
        
        return extracted_response, elapsed_gen


def get_or_create_collection(chroma_client,collection_name):
    try:
        return chroma_client.create_collection(name=collection_name)
    except chromadb.db.base.UniqueConstraintError:
   
        return chroma_client.get_collection(collection_name)


def initialize():
    st.session_state.initialized = True
    st.write("Initialization complete.")
    # Load the pre-trained model and tokenizer
    model_name = "NousResearch/Meta-Llama-3-8B-Instruct"
    st.model = AutoModelForCausalLM.from_pretrained(model_name)
    st.tokenizer = AutoTokenizer.from_pretrained(model_name)

   
if 'initialized' not in st.session_state:
    initialize()

def initialized2():
    st.session_state.pdf = True
    # Create PdfReader object using the uploaded file
    cache_folder = "sentence_transformers_cache"

    # Ensure cache folder exists
    os.makedirs(cache_folder, exist_ok=True)

    # Initialize SentenceTransformer model for encoding queries with specified cache folder
    st.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder=cache_folder)
    
    all_text =get_pdf_text(uploaded_files)

    # Further processing of all_text can be done here
    document = '\n'.join(all_text)
   
    def get_overlapped_chunks(textin, chunksize, overlapsize):
        return [textin[a:a+chunksize] for a in range(0,len(textin), chunksize-overlapsize)]
    chunks = get_overlapped_chunks(document, 1000, 100)

 
    chunk_embeddings = st.embedding_model.encode(chunks)

    chroma_client = chromadb.Client()
    

    # Define collection name
    collection_name = "new_collection"

    # Call the function to create or recreate collection
    collection = get_or_create_collection(chroma_client, collection_name)

    collection.add(
    embeddings = chunk_embeddings,
    documents=chunks,
    ids= [str(i) for i in range(len(chunks))]
    )
    st.collection =collection
    print("collection saved :", collection)
    print(collection)

if 'pdf' not in st.session_state and st.button("Process"):
    initialized2()
    
       
query = st.text_input("Enter your question:")


if st.button("Ask"): 
    
    retrieved_results,retrieve_time = retrieve_vector_db(st.collection,query, n_results=5)
    context = '\n\n'.join(retrieved_results[0])
    response,generation_time = get_llama2_chat_response(query, context, max_new_tokens=500)
    st.write("Response:", response)
   
