# -*- coding: big5 -*-

import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

# Import necessary variables/objects from config
from config import VECTORSTORE_DIR, DATA_DIR, console, EMBEDDING # Assuming EMBEDDING is globally initialized

def initialize_rag_system(embedding: GoogleGenerativeAIEmbeddings):
    """Initializes the RAG system: loads or creates the vector store."""
    # Check if we already have a persisted vector store
    if os.path.exists(VECTORSTORE_DIR) and len(os.listdir(VECTORSTORE_DIR)) > 0:
        console.print(f"[yellow]Loading existing vector store from {VECTORSTORE_DIR}...[/yellow]")
        try:
            vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embedding)
            console.print("[green]Existing vector store loaded.[/green]")
            return vectorstore.as_retriever(search_kwargs={"k": 5})
        except Exception as e:
            console.print(f"[bold red]Error loading vector store: {e}. Rebuilding...[/bold red]")


    # If not, create a new one from documents
    console.print("[yellow]Creating new vector store from documents...[/yellow]")
    pdf_paths = glob(os.path.join(DATA_DIR, "*.pdf")) # Use os.path.join for compatibility

    if not pdf_paths:
        console.print(f"[bold red]No PDF documents found in {DATA_DIR} directory![/bold red]")
        console.print(f"[yellow]Please add PDF documents to the {DATA_DIR} directory[/yellow]")
        exit(1) 

    all_pages = []
    for path in pdf_paths:
        console.print(f"[green]Loading document: {path}[/green]")
        try:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            all_pages.extend(pages)
        except Exception as e:
            console.print(f"[bold red]Error loading PDF {path}: {e}[/bold red]")
            # Optionally skip this file and continue

    if not all_pages:
        console.print("[bold red]No content loaded from PDF files. Cannot build vector store.[/bold red]")
        exit(1) 

    text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_pages)

    console.print(f"[green]Created {len(chunks)} text chunks for retrieval[/green]")

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=VECTORSTORE_DIR
        )
        console.print(f"[green]New vector store created and persisted to {VECTORSTORE_DIR}.[/green]")
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        console.print(f"[bold red]Error creating vector store: {e}[/bold red]")
        exit(1) # Or return None

# Initialize retriever globally or call this function in main.py
RETRIEVER = initialize_rag_system(EMBEDDING)