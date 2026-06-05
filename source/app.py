import streamlit as st
from pathlib import Path

# Import de la fonction d'initialisation depuis le moteur
from rag_engine import init_rag_chain

DOSSIER_COURANT = Path(__file__).parent
chemin_image = DOSSIER_COURANT / "img" / "chefkiss.png"

# Configuration de la page
st.set_page_config(page_title="RAGoût Recettes", page_icon="👩‍🍳")
col1, col2 = st.columns([1, 5])

with col1:
    # Ajustez le chemin de votre image et la largeur
    st.image(str(chemin_image), use_container_width=True)
with col2:
    st.title("RAGoût")

# Utilisation du cache Streamlit pour ne pas recharger les modèles à chaque clic
@st.cache_resource
def load_application():
    return init_rag_chain()

# Chargement des composants du RAG
rag_chain, retriever = load_application()

# --- Interface Graphique ---
st.markdown("Posez une question à votre assistant RAGoût basé sur une base de données de recettes.")

question = st.text_input(
    "Votre question :", 
    placeholder="Ex: J'ai des tomates, j'ai 20 minutes, et j'adore la cuisine italienne"
)

if st.button("Générer une réponse", type="primary"):
    if question:
        with st.spinner("Recherche et génération en cours..."):
            # Appel de la chaîne de traitement
            reponse = rag_chain.invoke(question)
            
            # Affichage du résultat
            st.subheader("Réponse de l'assistant")
            st.write(reponse)
            
            # Section extensible pour le débogage ou voir les sources
            with st.expander("Voir les extraits de recettes utilisés (Contexte)"):
                sources = retriever.invoke(question)
                for i, doc in enumerate(sources):
                    st.markdown(f"**Extrait {i+1} :**")
                    st.info(doc.page_content)
    else:
        st.warning("Veuillez entrer une question avant de générer une réponse.")