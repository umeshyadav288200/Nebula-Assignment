import os
import streamlit as st
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from document_processors import load_multimodal_data, load_data_from_directory
from utils import set_environment_variables
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize Sentence Transformer model for embeddings
embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Set up the page configuration
st.set_page_config(layout="wide")

# Initialize a connection to Milvus
def connect_to_milvus():
    connections.connect(alias="default", host="127.0.0.1", port="19530")
    st.write("Connected to Milvus")

# Drop the existing collection (if exists) and create a new collection
def create_collection():
    collection_name = "my_collection"
    
    # Check if the collection exists, if yes, drop it
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.drop()
        st.write(f"Dropped existing collection: {collection_name}")
    
    # Define fields for your collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)  # 768 is the dimension from Sentence Transformer
    ]
    
    # Create a collection schema
    schema = CollectionSchema(fields, description="Embedding collection")
    
    # Create a collection in Milvus
    collection = Collection(collection_name, schema)
    st.write("Milvus collection created")
    return collection

# Insert embeddings into the collection and load collection into memory
def insert_embeddings(collection, embeddings):
    # Create unique IDs for each embedding
    ids = [i for i in range(len(embeddings))]
    
    # Insert embeddings into Milvus collection (ensure the embeddings are 2D lists of floats)
    collection.insert([ids, embeddings])
    st.write(f"Inserted {len(embeddings)} embeddings into Milvus")
    
    # Load the collection into memory to prepare for search
    collection.load()
    st.write("Milvus collection loaded into memory for search")

# Search embeddings in the collection
def search_embeddings(collection, query_embedding, top_k=5):
    # Ensure the query embedding is in the correct format (a 1D list of floats)
    query_embedding = np.array(query_embedding).flatten().tolist()  # Convert to 1D list if necessary
    
    # Conduct search on Milvus collection
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    
    # Search for the top K most similar embeddings
    result = collection.search([query_embedding], "vector", search_params, limit=top_k)
    return result

# Main function to run the Streamlit app
def main():
    # Load environment variables (set NVIDIA API keys or other settings)
    set_environment_variables()
    
    # Connect to Milvus
    connect_to_milvus()

    # Create or load Milvus collection
    collection = create_collection()

    # Initialize layout columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.title("Enterprise QUESTION AND ANSWER USING MULTI-MODAL DOCUMENTS")

        # Input method selection
        input_method = st.radio("Choose input method:", ("Upload Files", "Enter Directory Path"))

        if input_method == "Upload Files":
            uploaded_files = st.file_uploader("Drag and drop files here", accept_multiple_files=True)
            if uploaded_files and st.button("Process Files"):
                with st.spinner("Processing files..."):
                    # Process the uploaded files and extract documents
                    documents = load_multimodal_data(uploaded_files)

                    # Embed the document texts using Sentence Transformer
                    embeddings = [embedder.encode(doc.text).tolist() for doc in documents]
                    insert_embeddings(collection, embeddings)
                    st.session_state['documents'] = documents
                    st.session_state['history'] = []
                    st.success("Files processed and embeddings inserted into Milvus!")
        else:
            directory_path = st.text_input("Enter directory path:")
            if directory_path and st.button("Process Directory"):
                if os.path.isdir(directory_path):
                    with st.spinner("Processing directory..."):
                        # Process the files from the directory and extract documents
                        documents = load_data_from_directory(directory_path)

                        # Embed the document texts using Sentence Transformer
                        embeddings = [embedder.encode(doc.text).tolist() for doc in documents]
                        insert_embeddings(collection, embeddings)
                        st.session_state['documents'] = documents
                        st.session_state['history'] = []
                        st.success("Directory processed and embeddings inserted into Milvus!")
                else:
                    st.error("Invalid directory path. Please enter a valid path.")

    with col2:
        if 'documents' in st.session_state:
            st.title("Chat")
            if 'history' not in st.session_state:
                st.session_state['history'] = []

            # Chat interface
            user_input = st.chat_input("Enter your query:")

            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state['history'].append({"role": "user", "content": user_input})

                # Embed the user query and search in Milvus
                query_embedding = embedder.encode([user_input]).flatten().tolist()  # Convert query embedding to a 1D list
                results = search_embeddings(collection, query_embedding)

                with st.chat_message("assistant"):
                    response = "Top results:\n"
                    for result in results:
                        response += f"ID: {result.id}, Distance: {result.distance}\n"
                    st.markdown(response)
                st.session_state['history'].append({"role": "assistant", "content": response})

            # Display chat history
            chat_container = st.container()
            with chat_container:
                for message in st.session_state['history']:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # Clear Chat button
            if st.button("Clear Chat"):
                st.session_state['history'] = []
                st.rerun()

if __name__ == "__main__":
    main()
