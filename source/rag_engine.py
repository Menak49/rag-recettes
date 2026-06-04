import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import de la fonction du fichier prompts.py
from prompts import get_rag_prompt

# Chargement du fichier .env situé à la racine du projet
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

def format_docs(docs):
    """Combine le contenu des documents trouvés en une seule chaîne de caractères."""
    return "\n\n".join(doc.page_content for doc in docs)

def init_rag_chain():
    """Initialise les composants et retourne la chaîne RAG et le retriever."""
    
    # 1. Configuration des embeddings
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 2. Connexion à la base Chroma locale
    # Le chemin remonte d'un cran par rapport à 'src/' pour trouver 'chroma_db/'
    chroma_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    vectorstore_disk = Chroma(
        persist_directory=chroma_path,
        embedding_function=gemini_embeddings
    )
    
    # Configuration du retriever
    retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 6})
    
    # 3. Initialisation du LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # 4. Récupération du prompt
    llm_prompt = get_rag_prompt()
    
    # 5. Construction de la chaîne LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever