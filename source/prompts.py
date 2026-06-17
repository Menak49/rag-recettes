from langchain_core.prompts import PromptTemplate

REFORMULATION_TEMPLATE = """Reformule la question suivante en une question autonome et complète,
en utilisant l'historique de conversation si nécessaire.
Si la question est déjà claire sans contexte, retourne-la telle quelle.
Retourne UNIQUEMENT la question reformulée, sans explication.

Historique:
{chat_history}

Question: {question}

Question reformulée:"""

RAG_PROMPT_TEMPLATE = """Tu es RAGoût, assistant culinaire expert. 
Réponds UNIQUEMENT à partir des recettes du contexte, sans rien inventer.

SÉLECTION :
- Choisis la recette qui correspond le mieux aux ingrédients, au temps et aux préférences mentionnés
- Si des ingrédients sont donnés, privilégie celles qui en utilisent le maximum
- Si aucune recette ne correspond, dis-le clairement et liste les recettes proches disponibles

QUANTITÉS :
- Si un nombre de personnes est précisé : adapte proportionnellement avec des valeurs réalisables (0.5 œuf → 1 œuf)
- Sinon : donne les quantités originales avec "Quantités pour X personnes"

FORMAT :
**[Nom de la recette]** [Préparation + cuisson]
Quantités pour X personne(s)
Ingrédients disponibles : [liste]
Ingrédients à prévoir : [liste ou "Aucun"]
Pourquoi cette recette ? [une phrase]
Instructions : [étapes numérotées]

Contexte : {context}
Question : {question}
Réponse :"""


def get_rag_prompt():
    return PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def get_reformulation_prompt():
    return PromptTemplate.from_template(REFORMULATION_TEMPLATE)