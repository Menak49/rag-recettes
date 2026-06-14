import os
import re
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from prompts import get_rag_prompt, get_reformulation_prompt

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

K_RETRIEVAL = 8
#SEUIL_RELATIF = 0.01
MAX_DOCS_GARDES = 3
AFFICHER_SCORES_RETRIEVAL = True # Permet d'afficher les scores et les documents gardés dans le terminal.

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

"""
class RelativeThresholdRetriever:
    
    #Retriever personnalisé basé sur un seuil relatif.

    #Au lieu de garder toujours exactement k documents, on récupère d'abord
    #K_RETRIEVAL documents avec leurs scores, puis on garde uniquement ceux
    #qui sont proches du meilleur score.

    #Cela évite de donner au LLM des documents peu pertinents juste parce qu'il
    #faut remplir un top-k fixe.
    

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
        
        #Rend le retriever compatible avec le reste du projet.

        #evaluation.py et chat() appellent déjà retriever.invoke(question),
        #donc on garde cette interface.
        
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
"""
class CRAGRetriever:
    """
    Retriever inspiré de CRAG.

    Étape 1 : Chroma récupère plusieurs documents candidats avec une recherche vectorielle.
    Étape 2 : un LLM joue le rôle de juge et garde uniquement les documents vraiment
    pertinents par rapport à la question utilisateur.

    Cette approche est plus coûteuse qu'un simple top-k ou qu'un seuil relatif,
    mais elle réduit les faux positifs lorsque des recettes sont proches
    sémantiquement sans répondre précisément à la demande.
    """

    def __init__(
        self,
        vectorstore,
        llm,
        k: int = K_RETRIEVAL,
        max_docs_gardes: int = MAX_DOCS_GARDES,
        afficher_scores: bool = AFFICHER_SCORES_RETRIEVAL,
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.k = k
        self.max_docs_gardes = max_docs_gardes
        self.afficher_scores = afficher_scores

    def _build_filter_prompt(self, query: str, docs):
        docs_text = ""

        for i, doc in enumerate(docs, start=1):
            extrait = doc.page_content[:1200].replace("\n", " ")
            docs_text += f"\nDOCUMENT {i}:\n{extrait}\n"

        return f"""
Tu es un évaluateur de pertinence pour un système RAG de recettes de cuisine.

Question utilisateur :
{query}

Documents candidats récupérés par recherche vectorielle :
{docs_text}

Ta tâche :
- garde uniquement les documents qui répondent vraiment à la demande utilisateur ;
- rejette les documents seulement proches lexicalement ou sémantiquement mais non pertinents ;
- si la question impose une contrainte, par exemple sans four, végétarien, rapide, poêle, etc., rejette les documents qui ne respectent pas cette contrainte ;
- garde au maximum {self.max_docs_gardes} documents ;
- réponds uniquement avec les numéros des documents à garder, séparés par des virgules ;
- si aucun document n'est pertinent, réponds uniquement : AUCUN.

Exemples de réponses valides :
1,2
3
AUCUN
"""

    def _parse_llm_selection(self, response: str, nb_docs: int):
        response = response.strip()

        if "AUCUN" in response.upper():
            return []

        indices = []
        for number in re.findall(r"\d+", response):
            index = int(number)
            if 1 <= index <= nb_docs and index not in indices:
                indices.append(index)

        return indices[:self.max_docs_gardes]

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

        docs = [doc for doc, score in docs_scores]

        filter_prompt = self._build_filter_prompt(query, docs)

        llm_result = self.llm.invoke(filter_prompt)
        llm_content = llm_result.content

        if isinstance(llm_content, list):
            llm_response = " ".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in llm_content
            ).strip()
        else:
            llm_response = str(llm_content).strip()

        indices_gardes = self._parse_llm_selection(llm_response, len(docs))

        docs_filtres = [
            docs[index - 1]
            for index in indices_gardes
        ]

        if self.afficher_scores:
            print("\n=== Retrieval vectoriel + filtrage LLM ===")
            print(f"Question : {query}")
            print(f"Réponse du LLM filtre : {llm_response}")
            print(f"Documents gardés : {len(docs_filtres)}/{len(docs_scores)}")

            for i, (doc, score) in enumerate(docs_scores, start=1):
                statut = "GARDÉ" if i in indices_gardes else "REJETÉ"
                extrait = doc.page_content[:200].replace("\n", " ")
                print(f"Doc {i} | score={score:.4f} | {statut}")
                print(extrait)
                print("-----")

        return docs_filtres

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
    """
    retriever = RelativeThresholdRetriever(
        vectorstore=vectorstore_disk,
        k=K_RETRIEVAL,
        seuil_relatif=SEUIL_RELATIF,
        afficher_scores=AFFICHER_SCORES_RETRIEVAL,
    )
    """
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
    # Version finale : approche inspirée de CRAG.
    # On récupère plusieurs documents candidats, puis le LLM filtre les faux positifs.
    retriever = CRAGRetriever(
        vectorstore=vectorstore_disk,
        llm=llm,
        k=K_RETRIEVAL,
        max_docs_gardes=MAX_DOCS_GARDES,
        afficher_scores=AFFICHER_SCORES_RETRIEVAL,
    )

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

    return {
        "answer": answer,
        "standalone_question": standalone_question,
        "retrieved_docs": retrieved_docs,
    }