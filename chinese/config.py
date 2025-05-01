# -*- coding: big5 -*-

import os
import nltk
from dotenv import load_dotenv
from rich.console import Console
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- Directory Setup ---
DATA_DIR = "chinese/data"
LOGS_DIR = "chinese/learning_logs"
PROFILES_DIR = "chinese/student_profiles"
VECTORSTORE_DIR = "chinese/vectorstore"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# --- Console Setup ---
console = Console()

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    console.print("[yellow]NLTK 'punkt' tokenizer not found. Downloading...[/yellow]")
    nltk.download('punkt')
    console.print("[green]NLTK 'punkt' downloaded.[/green]")

# --- API Key Setup ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    console.print("[bold red]ERROR: Google API Key not found in .env file[/bold red]")
    console.print("[yellow]Please create a .env file with GOOGLE_API_KEY=your_api_key[/yellow]")
    exit(1)
else:
    console.print("[green]Google API Key loaded successfully.[/green]")

# --- Model and Embedding Initialization ---
def initialize_models(api_key: str):
    """Initializes the Chat Model and Embeddings."""
    try:
        chat_model = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model="gemini-1.5-flash-latest",
            temperature=0.2
        )

        embedding = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model="models/embedding-001"
        )
        console.print("[green]Chat Model and Embeddings initialized.[/green]")
        return chat_model, embedding
    except Exception as e:
        console.print(f"[bold red]Error initializing models: {e}[/bold red]")
        exit(1)

# Initialize models globally or pass API_KEY around
CHAT_MODEL, EMBEDDING = initialize_models(API_KEY)