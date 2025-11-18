import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub  # <--- CORRETTO PER EVITARE ERRORI
from langchain.chains import RetrievalQA

# --- Configurazione della Pagina ---
st.set_page_config(page_title="Unipol Bot", page_icon="🤗")
st.title("🤖 Assistente Unipol (Hugging Face)")

# --- Recupero Token Segreto ---
try:
    hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
except:
    st.warning("Manca il token Hugging Face nei Secrets!")
    st.stop()

# --- Configurazione Percorsi e URL ---
FAISS_PATH = "faiss_db_cloud"
URLS = [
    "https://www.safeassicurazioni.com/",
    "https://www.unipolmove.it/",
    "https://www.unipol.it/homepage",
    "https://www.unipolglass.it/",
    "https://www.unipolservice.it/"
]

# --- 1. Funzione per creare/caricare la Conoscenza ---
@st.cache_resource
def setup_knowledge_base():
    # Usiamo un modello di embedding leggero e gratuito
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Se il database non esiste ancora sul server, lo creiamo ora
    if not os.path.exists(FAISS_PATH):
        with st.status("🚀 Primo avvio: Sto scaricando i dati Unipol...", expanded=True) as status:
            st.write("Scaricamento pagine web in corso...")
            loader = WebBaseLoader(URLS)
            documents = loader.load()
            
            st.write("Suddivisione del testo...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)
            
            st.write(f"Creazione del cervello (Embeddings) su {len(texts)} frammenti...")
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(FAISS_PATH)
            
            status.update(label="✅ Database creato con successo!", state="complete", expanded=False)
    else:
        # Se esiste già, lo carichiamo
        db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
    return db

# --- 2. Configurazione del Modello AI (LLM) ---
def get_llm():
    # Usiamo HuggingFaceHub che è più stabile per questa configurazione
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=hf_token
    )
    return llm

# --- Esecuzione Principale ---
try:
    # Carica DB e Modello
    db = setup_knowledge_base()
    llm = get_llm()
    
    # Crea la
