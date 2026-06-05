import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from prompts import get_rag_prompt, get_reformulation_prompt

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def init_rag_chain():
    """Initialise et retourne le llm et le retriever."""

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    chroma_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    vectorstore_disk = Chroma(
        persist_directory=chroma_path,
        embedding_function=gemini_embeddings
    )

    retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 6})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    return llm, retriever


def chat(question: str, llm, retriever, chat_history: list) -> str:
    """Génère une réponse en tenant compte de l'historique de conversation."""

    reformulation_prompt = get_reformulation_prompt()
    answer_prompt = get_rag_prompt()

    # Conversion de l'historique en texte
    history_str = "\n".join([
        f"Humain: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}"
        for m in chat_history
    ])

    # Reformulation de la question si historique existant
    if chat_history:
        standalone_question = (llm | StrOutputParser()).invoke(
            reformulation_prompt.invoke({"question": question, "chat_history": history_str})
        )
        standalone_question = standalone_question.strip()[:200]
        print(f"Question reformulée : {standalone_question}")
    else:
        standalone_question = question

    # Retrieval
    retrieved_docs = retriever.invoke(standalone_question)
    context = format_docs(retrieved_docs)

    # Génération de la réponse
    answer = (llm | StrOutputParser()).invoke(
        answer_prompt.invoke({
            "question": question,
            "context": context,
            "chat_history": history_str
        })
    )

    return answer