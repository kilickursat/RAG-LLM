import os
import streamlit as st
from typing import List, Dict
import torch
import transformers
import logging
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from huggingface_hub import login

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeotechnicalRAG:
    def __init__(self, persist_directory: str = "db"): 
        self.persist_directory = persist_directory
        self.vector_store = None
        self.qa_chain = None

    def load_and_process_documents(self, docs_dir: str):
        logging.info("Loading and processing documents...")

        # Load documents from uploaded files
        documents = []
        for file_name in os.listdir(docs_dir):
            if file_name.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(docs_dir, file_name))
                documents.extend(loader.load())

        # Split documents with improved settings for better context management
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings()
        if os.path.exists(self.persist_directory):
            logging.info(f"Clearing existing vector store at {self.persist_directory} to avoid dimension mismatch.")
            os.system(f'rm -rf {self.persist_directory}')
        
        self.vector_store = Chroma.from_documents(splits, embeddings, persist_directory=self.persist_directory)
        self.vector_store.persist()
        logging.info("Document processing completed and vector store created.")

    def setup_qa_chain(self):
        logging.info("Setting up the QA chain...")

        # Create retriever with higher k for better retrieval results
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        # Set up Llama model from Hugging Face
        model_id = "meta-llama/Meta-Llama-3.1-8B"

        # Load the configuration and adjust rope_scaling
        config = transformers.AutoConfig.from_pretrained(model_id)
        config.rope_scaling = {"type": "linear", "factor": 8.0}

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            config=config,
            max_length=512,  # Control max_length to prevent overly long generations
            max_new_tokens=128,  # Adjust to generate reasonable output length
            temperature=0.7,  # Reduce randomness in the generation
            top_p=0.9  # Use nucleus sampling to focus on the most probable outputs
        )

        # Wrap the Hugging Face pipeline in a LangChain LLM
        llm = HuggingFacePipeline(pipeline=pipeline)

        # Enhanced prompt template with clearer instructions
        template = """
        You are an expert in geotechnical engineering. Use the following context to provide the most accurate and concise answer to the question. If you don't know the answer, explicitly say "I don't know". Ensure your response is relevant and informative.

        Context: {context}

        Question: {question}

        Expert Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Create QA chain with error handling and better context management
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        logging.info("QA chain successfully set up.")

    def query(self, question: str) -> str:
        if not self.qa_chain:
            raise ValueError("QA chain not set up. Call setup_qa_chain() first.")
        try:
            logging.info(f"Received query: {question}")
            response = self.qa_chain.run(question)
            logging.info(f"Generated response: {response}")
            return response
        except Exception as e:
            logging.error(f"Error during query processing: {e}")
            return "An error occurred while processing your query. Please try again."

# Streamlit Interface
st.title("Geotechnical RAG System")

# API Key Input
api_key = st.text_input("Enter your Hugging Face API Key:", type="password")

if api_key:
    login(api_key)  # Authenticate with the provided API key
    st.success("API Key authenticated successfully!")

    # Directory to save uploaded documents
    upload_dir = "uploaded_docs"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # File uploader
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Save uploaded files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded successfully!")

        # Initialize the RAG system and process documents
        rag_system = GeotechnicalRAG(persist_directory="db")
        rag_system.load_and_process_documents(upload_dir)
        rag_system.setup_qa_chain()

        # Query input
        user_query = st.text_input("Enter your geotechnical query:")

        if user_query:
            response = rag_system.query(user_query)
            st.write(f"Summary: {response}")
else:
    st.warning("Please enter your Hugging Face API Key to proceed.")
