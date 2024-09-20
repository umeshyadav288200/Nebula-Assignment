import os
import fitz  # PyMuPDF for PDFs
import pandas as pd
from pptx import Presentation
import streamlit as st
from openai import OpenAI

# Import functions to process documents
from document_processors import extract_text_from_pdf, extract_text_from_spreadsheet, extract_text_from_ppt
from utils import set_environment_variables

# Set up the OpenAI LLM client
def get_openai_client():
    api_key = os.getenv("NVIDIA_API_KEY")  # Fetch the API key from the environment variable
    if not api_key:
        st.error("API key not found! Please set the NVIDIA_API_KEY environment variable.")
        return None
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    return client


# Query LLM using OpenAI or NVIDIA's API
def query_llm(client, document_text, user_query):
    prompt = f"You have the following document text:\n\n{document_text}\n\nUser query: {user_query}\nPlease provide the best possible answer."
    
    # Send the request to the LLM model
    completion = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=False
    )
    
    # Access the content from the ChatCompletionMessage object correctly
    result = completion.choices[0].message.content  # Use the correct attribute for accessing the content
    return result


# Handle document upload and processing
def handle_uploaded_file(uploaded_file):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension == ".pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_extension in [".xlsx", ".xls", ".csv"]:
        return extract_text_from_spreadsheet(uploaded_file)
    elif file_extension in [".pptx", ".ppt"]:
        return extract_text_from_ppt(uploaded_file)
    else:
        st.error("Unsupported file format.")
        return None

# Main function to run the Streamlit app
def main():
    set_environment_variables()  # Set up any required environment variables

    st.title("Document Query System Using LLM")
    
    uploaded_file = st.file_uploader("Upload a PDF, Spreadsheet, or PPT file", type=['pdf', 'xlsx', 'xls', 'csv', 'pptx', 'ppt'])

    if uploaded_file:
        with st.spinner("Processing the file..."):
            document_text = handle_uploaded_file(uploaded_file)
        
        if document_text:
            st.write("File processed successfully!")
            st.text_area("Extracted Document Text", value=document_text, height=300)

            user_query = st.text_input("Enter your query:")
            if user_query:
                with st.spinner("Querying LLM for the answer..."):
                    client = get_openai_client()
                    answer = query_llm(client, document_text, user_query)
                    st.success(f"Answer: {answer}")

if __name__ == "__main__":
    main()
