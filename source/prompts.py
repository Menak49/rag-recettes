from langchain_core.prompts import PromptTemplate

REFORMULATION_TEMPLATE = """Reformule la question suivante en une question autonome et complète,
en utilisant l'historique de conversation si nécessaire.
Si la question est déjà claire sans contexte, retourne-la telle quelle.
Retourne UNIQUEMENT la question reformulée, sans explication.

Historique:
{chat_history}

Question: {question}

Question reformulée:"""

RAG_PROMPT_TEMPLATE = """Tu es un assistant culinaire expert.
Utilise le contexte suivant pour répondre à la question.
Si tu ne connais pas la réponse, dis-le simplement.
Sois précis et concis (5 phrases maximum).

Historique de conversation:
{chat_history}

Contexte (extraits de recettes):
{context}

Question: {question}

Réponse:"""


def get_rag_prompt():
    return PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def get_reformulation_prompt():
    return PromptTemplate.from_template(REFORMULATION_TEMPLATE)