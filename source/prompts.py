from langchain_core.prompts import PromptTemplate

# Le template de base que tu utilisais
RAG_PROMPT_TEMPLATE = """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.\n
Question: {question} \nContext: {context} \nAnswer:"""


def get_rag_prompt():
    """Retourne l'objet PromptTemplate configuré."""
    return PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)