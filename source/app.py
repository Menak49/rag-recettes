import streamlit as st
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

from rag_engine import init_rag_chain, chat

DOSSIER_COURANT = Path(__file__).parent
chemin_image = DOSSIER_COURANT / "img" / "chefkiss.png"

st.set_page_config(page_title="RAGoût Recettes", page_icon="👩‍🍳")
col1, col2 = st.columns([1, 5])
with col1:
    st.image(str(chemin_image), use_container_width=True)
with col2:
    st.title("RAGoût")

@st.cache_resource
def load_application():
    return init_rag_chain()

llm, retriever = load_application()

# Initialisation de l'historique
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("Posez une question à votre assistant RAGoût basé sur une base de données de recettes.")

# Affichage de l'historique
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

# Saisie
question = st.chat_input("Ex: J'ai des tomates, j'ai 20 minutes, et j'adore la cuisine italienne")

if question:
    with st.chat_message("user"):
        st.write(question)

    erreur_rencontree = None
    reponse = None

    with st.spinner("Recherche et mijotage en cours..."):
        try:
            reponse = chat(
                question=question,
                llm=llm,
                retriever=retriever,
                chat_history=st.session_state.chat_history
            )
        except Exception as e:
            # On stocke l'erreur pour la traiter plus tard et laisser le spinner se fermer
            erreur_rencontree = e

    # --- Gestion des erreurs en dehors du spinner ---
    if erreur_rencontree:
        erreur_str = str(erreur_rencontree).lower()
        if "token" in erreur_str or "context length" in erreur_str or "limit" in erreur_str or "rate" in erreur_str:
            st.error("🥘 **La casserole déborde !** \n\nNotre assistant a atteint sa limite de tokens.")
        else:
            # Si c'est une autre erreur (ex: problème de connexion)
            st.error(f"Une erreur inattendue a ruiné la recette : {erreur_rencontree}")
        
        # On arrête le script ici de façon propre, sans bloquer l'interface
        st.stop()

    # --- Suite normale si aucune erreur n'a été levée ---

    # Affichage de la réponse si tout s'est bien passé
    with st.chat_message("assistant"):
        st.write(reponse)

    # Mise à jour de l'historique (uniquement si la génération a réussi)
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=reponse))

    with st.expander("Voir les extraits de recettes utilisés (Contexte)"):
        # Attention : le retriever peut aussi crasher si la question seule dépasse la limite, 
        # mais c'est généralement le LLM avec tout l'historique qui bloque en premier.
        try:
            sources = retriever.invoke(question)
            for i, doc in enumerate(sources):
                st.markdown(f"**Extrait {i+1} :**")
                st.info(doc.page_content)
        except Exception:
            st.warning("Impossible de récupérer les sources pour le moment.")