import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# --- Configurazione della Pagina ---
st.set_page_config(page_title="Unipol Bot", page_icon="🤗")
st.title("🤖 Assistente Unipol (Hugging Face)")

# --- Recupero Token Segreto ---
# Questo prende la chiave che hai messo nei "Secrets" di Streamlit
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
    # Usiamo un modello di embedding leggero e gratuito che gira sulla CPU
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
    # Usiamo Mistral-7B Instruct via API gratuita
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, 
        max_length=512,
        temperature=0.5, 
        token=hf_token
    )
    return llm

# --- Esecuzione Principale ---
try:
    # Carica DB e Modello
    db = setup_knowledge_base()
    llm = get_llm()
    
    # Crea la catena di risposta
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # --- Interfaccia Chat ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostra cronologia
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input utente
    if prompt := st.chat_input("Chiedi informazioni sui servizi Unipol..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulto i documenti Unipol..."):
                res = qa_chain.invoke({"query": prompt})
                response = res["result"]
                
                # Pulizia della risposta (a volte i modelli open source ripetono il prompt)
                if "Helpful Answer:" in response:
                    response = response.split("Helpful Answer:")[-1]
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"Si è verificato un errore: {e}")