# LO17 - projet RAG

### Description :
Ce projet a pour objectif la conception et l’évaluation d’un système de Génération Augmentée
par la Recherche, plus couramment appelé RAG pour Retrieval-Augmented Generation. Un système
RAG combine deux approches complémentaires : d’une part, un module de recherche d’information,
chargé d’identifier dans une base documentaire les contenus les plus pertinents par rapport à une
requête utilisateur ; d’autre part, un modèle de génération de texte, qui exploite ces informations
pour produire une réponse contextualisée. Contrairement à un modèle de langage utilisé seul, le
RAG permet donc d’ancrer la génération dans des documents externes, ce qui améliore la précision
des réponses et limite en partie le risque d’hallucinations.
Nous avons choisi d’appliquer cette approche au domaine culinaire, en nous appuyant sur un
corpus de recettes extraites du site web 750g.

### Rapport et Démo
Le rapport se trouve dans le dossier `Docs/Rapport_de_Projet.pdf`
La démo se trouve dans le dossier `Docs/Demo.mp4`

## Prérequis : 
- Installer les requirements avec :
```bash
    pip install -r requirements.txt
```

- Ajouter un .env avec une clé API Gemini sous le format:
```bash
    GOOGLE_API_KEY="AIzaxxxxxxxxxxxxxxxxx"
```

### Lancement du moteur en local: 
Pour lancer le RAG en local dans le dossier source:
```bash
    python streamlit run app.py
```

### Version déployée: 

https://duflan.streamlit.app/