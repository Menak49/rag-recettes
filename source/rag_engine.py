import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from prompts import get_rag_prompt, get_reformulation_prompt

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

K_RETRIEVAL = 6
SEUIL_RELATIF = 0.01
AFFICHER_SCORES_RETRIEVAL = True # Permet d'afficher les scores et les documents gardés dans le terminal.

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class RelativeThresholdRetriever:
    """
    Retriever personnalisé basé sur un seuil relatif.

    Au lieu de garder toujours exactement k documents, on récupère d'abord
    K_RETRIEVAL documents avec leurs scores, puis on garde uniquement ceux
    qui sont proches du meilleur score.

    Cela évite de donner au LLM des documents peu pertinents juste parce qu'il
    faut remplir un top-k fixe.
    """

    def __init__(
        self,
        vectorstore,
        k: int = K_RETRIEVAL,
        seuil_relatif: float = SEUIL_RELATIF,
        afficher_scores: bool = AFFICHER_SCORES_RETRIEVAL,
    ):
        self.vectorstore = vectorstore
        self.k = k
        self.seuil_relatif = seuil_relatif
        self.afficher_scores = afficher_scores

    def invoke(self, query: str):
        """
        Rend le retriever compatible avec le reste du projet.

        evaluation.py et chat() appellent déjà retriever.invoke(question),
        donc on garde cette interface.
        """
        docs_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=self.k,
        )

        if not docs_scores:
            return []

        meilleur_score = docs_scores[0][1]
        score_minimum = meilleur_score - self.seuil_relatif

        docs_filtres = [
            doc
            for doc, score in docs_scores
            if score >= score_minimum
        ]

        if self.afficher_scores:
            print("\n=== Scores du retrieval ===")
            print(f"Question : {query}")
            print(f"Meilleur score : {meilleur_score:.4f}")
            print(f"Seuil minimum gardé : {score_minimum:.4f}")
            print(f"Documents gardés : {len(docs_filtres)}/{len(docs_scores)}")

            for i, (doc, score) in enumerate(docs_scores, start=1):
                statut = "GARDÉ" if score >= score_minimum else "REJETÉ"
                extrait = doc.page_content[:200].replace("\n", " ")
                print(f"Doc {i} | score={score:.4f} | {statut}")
                print(extrait)
                print("-----")

        return docs_filtres



def generate_hypothetical_document(question: str, llm) -> str:
    """
    HyDE : génère une recette hypothétique qui répond à la question.
    C'est ce document fictif qu'on va embedder, pas la question brute.
    """
    hyde_prompt = PromptTemplate.from_template("""Tu es un chef cuisinier. 
Génère une courte recette fictive (titre + ingrédients + 3-4 étapes) qui correspond exactement à cette demande.
Respecte ABSOLUMENT toutes les contraintes mentionnées (sans four, temps limité, ingrédients disponibles...).
Réponds uniquement avec la recette, sans intro ni explication.

Demande : {question}
Recette :""")
    
    from langchain_core.output_parsers import StrOutputParser
    return (llm | StrOutputParser()).invoke(
        hyde_prompt.invoke({"question": question})
    ).strip()


def init_rag_chain():
    """Initialise et retourne le llm et le retriever."""

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    chroma_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    vectorstore_disk = Chroma(
        persist_directory=chroma_path,
        embedding_function=gemini_embeddings
    )

    # Ancienne version :
    # retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 6})
    #
    # Nouvelle version :
    # on utilise un retriever personnalisé qui applique le seuil relatif.
    retriever = RelativeThresholdRetriever(
        vectorstore=vectorstore_disk,
        k=K_RETRIEVAL,
        seuil_relatif=SEUIL_RELATIF,
        afficher_scores=AFFICHER_SCORES_RETRIEVAL,
    )
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
    hypothetical_doc = generate_hypothetical_document(standalone_question, llm)
    print(f"Document hypothétique : {hypothetical_doc[:200]}")  # debug
    retrieved_docs = retriever.invoke(hypothetical_doc)
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