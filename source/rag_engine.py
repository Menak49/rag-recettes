from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

reformulation_prompt = PromptTemplate.from_template("""Reformule la question suivante en une question autonome et complète,
en utilisant l'historique de conversation si nécessaire.
Si la question est déjà claire sans contexte, retourne-la telle quelle.
Retourne UNIQUEMENT la question reformulée, en 10 mots maximum, sans explication.

Historique:
{chat_history}

Question: {question}

Question reformulée:""")

answer_prompt = PromptTemplate.from_template("""Tu es un assistant culinaire expert. Réponds à la question en te basant sur le contexte fourni.
Si le contexte ne contient pas la réponse, utilise tes connaissances générales en cuisine.
Sois précis, chaleureux et pratique dans tes réponses.

Historique de conversation:
{chat_history}

Contexte (extraits de recettes):
{context}

Question: {question}

Réponse:""")


def chat(question: str, llm, retriever, chat_history: list) -> str:
    history_str = "\n".join([
        f"Humain: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}"
        for m in chat_history
    ])

    # Reformulation si historique existant
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
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Génération de la réponse
    answer = (llm | StrOutputParser()).invoke(
        answer_prompt.invoke({
            "question": question,
            "context": context,
            "chat_history": history_str
        })
    )

    return answer


def init_rag_chain():
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return llm, retriever