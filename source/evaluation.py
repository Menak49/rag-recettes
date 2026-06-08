import os
from dotenv import load_dotenv
import time
import random
import logging
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# google.api_core n'est pas toujours installé en standalone ; on essaie de
# l'importer et on replie sur une détection par message d'erreur si absent.
try:
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError as GoogleInternalServerError
    RETRYABLE_EXCEPTIONS = (ResourceExhausted, ServiceUnavailable, GoogleInternalServerError)
except ModuleNotFoundError:
    # Fallback : on attrape toutes les exceptions dont le message contient les
    # codes HTTP typiques des rate-limits Gemini (429, 500, 503).
    RETRYABLE_EXCEPTIONS = ()   # redéfini juste en dessous via la fonction

# Import de tes fonctions
from rag_engine import init_rag_chain, format_docs

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exponential backoff
# ---------------------------------------------------------------------------

# Import du wrapper LangChain pour Gemini — toujours disponible si
# langchain-google-genai est installé.
try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
    _LANGCHAIN_GENAI_ERROR = ChatGoogleGenerativeAIError
except ImportError:
    _LANGCHAIN_GENAI_ERROR = None

# Mots-clés présents dans les messages d'erreur Gemini (rate-limit / serveur)
_RETRYABLE_KEYWORDS = (
    "429", "quota", "resource_exhausted", "resource exhausted",
    "503", "500", "unavailable", "internal server error", "rate limit",
)

# Regex pour extraire le retryDelay suggéré par l'API ("retryDelay": "51s")
import re as _re
_RETRY_DELAY_RE = _re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)\s*s", _re.IGNORECASE)


def _is_retryable(exc: Exception) -> bool:
    """Retourne True si l'exception est une erreur transitoire Gemini."""
    # 1. ChatGoogleGenerativeAIError (wrapper LangChain) — couvre le cas observé
    if _LANGCHAIN_GENAI_ERROR and isinstance(exc, _LANGCHAIN_GENAI_ERROR):
        return True
    # 2. Types google.api_core si disponibles
    if RETRYABLE_EXCEPTIONS and isinstance(exc, RETRYABLE_EXCEPTIONS):
        return True
    # 3. Fallback textuel universel
    msg = str(exc).lower()
    return any(kw in msg for kw in _RETRYABLE_KEYWORDS)


def _extract_retry_delay(exc: Exception) -> float | None:
    """
    Tente d'extraire le délai conseillé par l'API Gemini dans le message
    d'erreur (ex. 'retryDelay': '51s'). Retourne None si non trouvé.
    """
    m = _RETRY_DELAY_RE.search(str(exc))
    if m:
        return float(m.group(1)) + 2.0   # +2 s de marge de sécurité
    return None


def call_with_backoff(fn, *args, max_retries: int = 8, base_delay: float = 30.0,
                      max_delay: float = 360.0, jitter: bool = True, **kwargs):
    """
    Appelle fn(*args, **kwargs) avec un exponential backoff en cas d'erreur
    de rate-limit ou d'erreur serveur Gemini.

    Priorité du délai :
      1. retryDelay indiqué par l'API dans le message d'erreur (le plus fiable)
      2. Calcul exponentiel : min(base_delay * 2^attempt ± 20 % jitter, max_delay)

    Args:
        fn            : callable à exécuter.
        max_retries   : nombre maximum de tentatives (défaut : 8).
        base_delay    : délai initial en secondes (défaut : 30 s).
        max_delay     : plafond du délai en secondes (défaut : 360 s = 6 min).
        jitter        : ajoute ±20 % de bruit sur le délai exponentiel.
    """
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)

        except Exception as e:
            if not _is_retryable(e):
                logger.error("Erreur non-retriable : %s", e)
                raise

            attempt += 1
            if attempt > max_retries:
                logger.error("Nombre maximum de tentatives atteint (%d). Abandon.", max_retries)
                raise

            # Utilise le délai suggéré par l'API si disponible, sinon backoff exponentiel
            api_delay = _extract_retry_delay(e)
            if api_delay is not None:
                delay = min(api_delay, max_delay)
                logger.warning(
                    "Rate-limit Gemini. Tentative %d/%d — délai API : %.1f s...",
                    attempt, max_retries, delay,
                )
            else:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                if jitter:
                    delay *= 1 + random.uniform(-0.2, 0.2)
                logger.warning(
                    "Erreur API Gemini (%s). Tentative %d/%d — backoff : %.1f s...",
                    type(e).__name__, attempt, max_retries, delay,
                )

            time.sleep(delay)


# ---------------------------------------------------------------------------
# Wrapper Ragas : on surcharge les appels LLM internes avec le backoff
# ---------------------------------------------------------------------------

class BackoffChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """
    Sous-classe de ChatGoogleGenerativeAI qui applique automatiquement
    un exponential backoff sur chaque appel .invoke() et ._generate().
    """

    def invoke(self, *args, **kwargs):
        return call_with_backoff(super().invoke, *args, **kwargs)

    def _generate(self, *args, **kwargs):
        return call_with_backoff(super()._generate, *args, **kwargs)


# ---------------------------------------------------------------------------
# Dataset de test
# ---------------------------------------------------------------------------

EVAL_SET = [
    # Questions de Charlotte
    # {
    #     "question": "J'ai une poêle mais pas de four, qu'est-ce que je peux faire ?",
    #     "ground_truth": (
    #         "Sans four, plusieurs recettes se cuisinent entièrement à la poêle. "
    #         "Vous pouvez préparer des pommes de terre sautées (20 min de préparation, 20 min "
    #         "de cuisson avec du beurre, de l'huile d'olive et du persil haché), des crêpes sans "
    #         "œufs à base de farine, lait et huile, des pancakes moelleux (farine, sucre, beurre, "
    #         "lait, œufs), ou encore une harcha — galette de semoule nord-africaine prête en 20 min. "
    #         "La sauce bolognaise et les samoussa de galettes de sarrasin sont aussi réalisables "
    #         "uniquement à la poêle ou à la casserole."
    #     ),
    # },
    {
        "question": "J'ai 20 minutes, des tomates et je veux faire une recette italienne.",
        "ground_truth": (
            "En 20 minutes avec des tomates et une inspiration italienne, vous pouvez réaliser "
            "une salade de pâtes comme en Italie : faites cuire des pâtes al dente, puis mélangez-les "
            "avec des tomates fraîches, de la mozzarella, des olives vertes, de l'huile d'olive et "
            "de l'origan. Vous pouvez aussi préparer une salade de pâtes à l'avocat avec tomates "
            "cerise, feta et olives, ou un gaspacho tomates-poivrons si vous préférez une entrée froide."
        ),
    },
    # {
    #     "question": "Quels ingrédients faut-il pour faire une ratatouille ?",
    #     "ground_truth": (
    #         "La recette présente dans la base est une ratatouille niçoise aux herbes de Provence. "
    #         "Il faut : 4 aubergines, 4 courgettes, 2 poivrons, 500 g de tomates, 3 gros oignons, "
    #         "2 gousses d'ail, 1 verre d'huile d'olive, 1 bouquet garni, du sel et du poivre. "
    #         "Tous les légumes sont coupés en morceaux et mijotés en cocotte pendant 1 h à 1 h 30."
    #     ),
    # },
    # {
    #     "question": "J'ai que des pommes de terre et des oignons, que puis-je cuisiner ?",
    #     "ground_truth": (
    #         "Avec des pommes de terre et des oignons, la base de données contient notamment une "
    #         "salade de pommes de terre façon grand-mère : pommes de terre à chair ferme cuites et "
    #         "refroidies, émincées avec de l'oignon, assaisonnées d'une vinaigrette moutardée "
    #         "(moutarde de Dijon, vinaigre d'alcool, huile) et de ciboulette. C'est une recette "
    #         "très facile, prête en 35 min. Si vous disposez aussi de crème et de lardons, une "
    #         "tartiflette au Cookeo (25 min) ou un gratin lyonnais sont envisageables."
    #     ),
    # },
    # {
    #     "question": "Comment on fait un karkkouf ?",
    #     "ground_truth": (
    #         "Le terme « karkkouf » n'existe pas dans la base de données de recettes, et il ne "
    #         "correspond à aucune préparation culinaire connue dans les traditions répertoriées. "
    #         "Cette recette n'est pas disponible."
    #     ),
    # },
    # Contrainte de matériel
    # {
    #     "question": "Je veux faire des pancakes, quels ingrédients me faut-il ?",
    #     "ground_truth": (
    #         "La recette de pancakes facile et moelleuse de la base nécessite (pour environ 4 pers.) : "
    #         "300 g de farine, 90 g de sucre, 1 sachet de sucre vanillé, 1 pincée de sel, "
    #         "½ sachet de levure chimique, 3 œufs (blancs et jaunes séparés), 30 cl de lait, "
    #         "30 g de beurre fondu. En option : 2 c. à c. de rhum et de kirsch. "
    #         "Les blancs sont montés en neige et incorporés délicatement avant cuisson à la poêle."
    #     ),
    # },
    # # Contrainte ingrédients + temps
    # {
    #     "question": "J'ai du saumon fumé et 20 minutes, qu'est-ce que je peux faire ?",
    #     "ground_truth": (
    #         "Avec du saumon fumé et 20 minutes, vous avez le choix entre plusieurs recettes de la base. "
    #         "Les wraps au fromage frais et saumon fumé (10 min) : tartiner des tortillas d'un mélange "
    #         "fromage à tartiner/crème/ciboulette/échalote, puis garnir de saumon fumé, laitue et "
    #         "concombre. La verrine avocat-saumon-boursin (15 min) : écrasé d'avocat citronné, "
    #         "fromage Boursin crémé, saumon fumé et oeufs de lompe. "
    #         "Enfin, le tartare de saumon frais rehaussé de saumon fumé (20 min) convient si vous "
    #         "disposez aussi de saumon frais, échalotes, câpres et huile d'olive."
    #     ),
    # },
    # # Technique de base
    # {
    #     "question": "Comment faire une sauce béchamel ?",
    #     "ground_truth": (
    #         "La base de données contient une sauce béchamel préparée au Companion. "
    #         "Les ingrédients sont : du lait, de la farine, du beurre, une noix de muscade et du sel. "
    #         "La technique consiste à mélanger farine et lait, ajouter une pincée de sel et de la muscade "
    #         "râpée, puis incorporer le beurre en morceaux et cuire à 90 °C jusqu'à épaississement. "
    #         "Cette béchamel s'utilise ensuite pour un gratin, un croque-monsieur ou des lasagnes."
    #     ),
    # },
    # # Question sur la difficulté / accessibilité
    # {
    #     "question": "Quelle recette sucrée très facile peut-on faire en moins de 20 minutes ?",
    #     "ground_truth": (
    #         "Parmi les recettes sucrées très faciles et réalisables en moins de 20 minutes, "
    #         "on trouve notamment : le bavarois aux fraises sur biscuit spéculoos (20 min, sans cuisson), "
    #         "les pancakes à la banane et aux flocons d'avoine (quelques minutes à la poêle, "
    #         "seulement 3 ingrédients : banane, œuf, flocons d'avoine), "
    #         "ou la granola bio à l'huile de coco (15 min, noté facile). "
    #         "Ces recettes ne nécessitent pas de compétences particulières."
    #     ),
    # },
    # # Requête par légume (courgette)
    # {
    #     "question": "Que faire avec des courgettes ?",
    #     "ground_truth": (
    #         "La base contient 70 recettes à base de courgettes. Quelques idées accessibles : "
    #         "les courgettes à la crème fraîche et au jambon (55 min, très facile) : courgettes revenues "
    #         "à l'huile d'olive, crème fraîche, jambon et gruyère râpé ; "
    #         "les courgettes au chorizo (40 min, très facile) : courgettes sautées à l'huile d'olive "
    #         "avec du chorizo fort et du thym ; "
    #         "la quiche à la ricotta et à la courgette (60 min, très facile) avec pâte feuilletée, "
    #         "ricotta, dés de jambon et œufs ; "
    #         "ou encore le gaspacho courgettes-menthe (40 min) pour une option froide et rafraîchissante."
    #     ),
    # },
    # # Requête soupe / velouté
    # {
    #     "question": "Je voudrais faire une soupe de carottes, vous avez une idée ?",
    #     "ground_truth": (
    #         "La base propose un velouté de carottes et orange aux épices (30 min, facile, 4 pers.). "
    #         "Ingrédients : 800 g de carottes, 1 oignon, 1,2 l de bouillon de légumes, 150 g de crème "
    #         "liquide, 1 orange (jus + zeste), 1 c. à c. de cumin, 1 c. à c. de curcuma, huile d'olive, "
    #         "sel, poivre. Méthode : faire revenir l'oignon, ajouter les carottes en rondelles, "
    #         "les épices, le jus et les zestes d'orange, la crème et le bouillon. "
    #         "Cuire 10-15 min puis mixer."
    #     ),
    # },
    # # Question hors base (ingrédient absent)
    # {
    #     "question": "Comment préparer un pad thaï ?",
    #     "ground_truth": (
    #         "Il n'existe pas de recette de pad thaï dans la base de données de recettes disponible. "
    #         "Cette recette thaïlandaise à base de nouilles de riz, crevettes ou poulet, sauce tamarin "
    #         "et cacahuètes ne figure pas parmi les 1 674 recettes référencées."
    #     ),
    # },
    # # Requête par occasion / grand groupe
    # {
    #     "question": "Je reçois 8 personnes pour un dîner, quelle entrée facile puis-je préparer ?",
    #     "ground_truth": (
    #         "La base contient des recettes prévues pour 8 personnes ou plus. Pour une entrée facile : "
    #         "les verrines de saumon et mascarpone (23 min, très facile) — mélange de mascarpone, "
    #         "fromage blanc, crème, saumon fumé, œufs de saumon et ciboulette, servi sur blini ou pain "
    #         "toast — sont particulièrement adaptées. Une terrine de poulet facile et rapide (pour 8) "
    #         "peut aussi convenir si vous souhaitez une entrée à préparer à l'avance."
    #     ),
    # },
    # # Requête végétale express
    # {
    #     "question": "Je cherche une recette froide à base de tomates sans cuisson.",
    #     "ground_truth": (
    #         "La base propose plusieurs recettes froides à base de tomates ne nécessitant pas de cuisson. "
    #         "Le gaspacho au Monsieur Cuisine (10 min) : tomates mûres, concombre, poivrons rouge et vert, "
    #         "oignon, mixés ensemble. Le gaspacho de concombre à la feta (10 min) : concombres, feta, "
    #         "ail, oignon nouveau et yaourt nature. La salade grecque classique (15 min) : tomates, "
    #         "concombre, poivron vert, oignon rouge, olives et feta. Ces trois recettes sont très rapides "
    #         "et notées faciles."
    #     ),
    # },
]


# ---------------------------------------------------------------------------
# Génération RAG avec backoff
# ---------------------------------------------------------------------------

def generate_answer(llm, retriever, question: str) -> tuple[str, list[str]]:
    """Récupère les docs et génère une réponse. Chaque appel LLM passe par le backoff."""
    from prompts import get_rag_prompt
    from langchain_core.output_parsers import StrOutputParser

    docs = call_with_backoff(retriever.invoke, question)
    context_str = format_docs(docs)
    answer_prompt = get_rag_prompt()

    answer = call_with_backoff(
        (llm | StrOutputParser()).invoke,
        answer_prompt.invoke({
            "question": question,
            "context": context_str,
            "chat_history": "",
        }),
    )
    return answer, [doc.page_content for doc in docs]


# ---------------------------------------------------------------------------
# Évaluation principale
# ---------------------------------------------------------------------------

def run_evaluation():
    logger.info("Initialisation de la chaîne RAG...")
    llm, retriever = init_rag_chain()

    questions     = [item["question"]    for item in EVAL_SET]
    ground_truths = [item["ground_truth"] for item in EVAL_SET]

    generated_answers  = []
    retrieved_contexts = []

    logger.info("Génération des réponses sur le dataset de test (%d questions)...", len(questions))
    for i, q in enumerate(questions):
        logger.info("Question %d/%d : %s", i + 1, len(questions), q[:60])
        answer, contexts = generate_answer(llm, retriever, q)
        generated_answers.append(answer)
        retrieved_contexts.append(contexts)
        # Pause de courtoisie entre chaque question pour ménager le quota free tier
        # (uniquement si ce n'est pas la dernière question)
        if i < len(questions) - 1:
            pause = 15  # secondes — ajuste si nécessaire
            logger.info("Pause de %d s avant la prochaine question...", pause)
            time.sleep(pause)

    # -----------------------------------------------------------------------
    # Préparation du dataset Ragas
    # -----------------------------------------------------------------------
    dataset = Dataset.from_dict({
        "user_input":         questions,
        "response":           generated_answers,
        "retrieved_contexts": retrieved_contexts,
        "reference":          ground_truths,
    })

    # LLM et embeddings avec backoff intégré pour Ragas
    eval_llm = BackoffChatGoogleGenerativeAI(
        model=llm.model,           # réutilise le même modèle que ton RAG
        temperature=0,
    )
    eval_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    logger.info("Lancement de l'évaluation avec Ragas...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,   
    )

    # -----------------------------------------------------------------------
    # Affichage des résultats
    # -----------------------------------------------------------------------

    df = result.to_pandas()
 
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_cols = [c for c in metric_cols if c in df.columns]
 
    print("\n=== Résultats globaux de l'évaluation ===")
    for col in metric_cols:
        print(f"  {col:<25} : {df[col].mean():.4f}")
 
    print("\n=== Détail par question ===")
    detail_cols = ["user_input"] + metric_cols
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 60)
    print(df[detail_cols].to_string(index=False))


if __name__ == "__main__":
    run_evaluation()