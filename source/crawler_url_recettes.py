"""
Crawler 750g — Extraction des liens de recettes par catégorie
Usage:
    python crawler_750g.py
    python crawler_750g.py --output recettes.json --delay 1.5
    python crawler_750g.py --max-per-category 50 --max-total 200
"""

import re
import json
import time
import logging
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CATEGORY_URLS = [
    "https://www.750g.com/categorie_accompagnements.htm",
    "https://www.750g.com/recettes-aperitifs/",
    "https://www.750g.com/recettes-boissons/",
    "https://www.750g.com/recettes-bases/",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9",
}

# ---------------------------------------------------------------------------
# Modèles de données
# ---------------------------------------------------------------------------

@dataclass
class RecipeLink:
    url: str
    titre: str
    categorie: str


@dataclass
class CrawlResult:
    categorie_url: str
    categorie_nom: str
    nb_pages: int
    recettes: list[RecipeLink] = field(default_factory=list)

    def to_dict(self):
        return {
            "categorie_url": self.categorie_url,
            "categorie_nom": self.categorie_nom,
            "nb_pages": self.nb_pages,
            "nb_recettes": len(self.recettes),
            "recettes": [asdict(r) for r in self.recettes],
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_recipe_links(soup: BeautifulSoup, categorie_nom: str) -> list[RecipeLink]:
    """Extrait tous les liens de recettes d'une page de catégorie."""
    recettes = []

    # Les recettes sont dans des balises <h2> ou <h3> avec un lien
    # Pattern URL : /nom-recette-rXXXXX.htm
    recipe_pattern = re.compile(r"^https://www\.750g\.com/[a-z0-9\-]+-r\d+\.htm$")

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]

        # Normalisation de l'URL (relative → absolue)
        if href.startswith("/"):
            href = "https://www.750g.com" + href

        if recipe_pattern.match(href):
            titre = a_tag.get_text(strip=True)
            if titre and len(titre) > 3:  # filtre les liens sans texte utile
                recettes.append(RecipeLink(url=href, titre=titre, categorie=categorie_nom))

    # Dédoublonnage (même URL peut apparaître plusieurs fois dans la page)
    seen = set()
    unique = []
    for r in recettes:
        if r.url not in seen:
            seen.add(r.url)
            unique.append(r)

    return unique


def get_last_page(soup: BeautifulSoup) -> int:
    """Détermine le numéro de la dernière page de pagination."""
    # 750g affiche les numéros de page dans la pagination
    last_page = 1

    # Cherche tous les liens de pagination contenant ?page=N
    pagination_links = soup.find_all("a", href=re.compile(r"\?page=\d+"))
    for link in pagination_links:
        match = re.search(r"\?page=(\d+)", link["href"])
        if match:
            page_num = int(match.group(1))
            if page_num > last_page:
                last_page = page_num

    # Fallback : cherche un texte "page X sur Y" ou similaire
    if last_page == 1:
        text = soup.get_text()
        match = re.search(r"(\d+)\s*/\s*(\d+)", text)
        if match:
            last_page = int(match.group(2))

    return last_page


# ---------------------------------------------------------------------------
# Fetching avec retry
# ---------------------------------------------------------------------------

def fetch_page(url: str, session: requests.Session, retries: int = 3) -> Optional[BeautifulSoup]:
    """Télécharge une page et retourne le BeautifulSoup, ou None en cas d'échec."""
    for attempt in range(retries):
        try:
            response = session.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            logging.warning(f"Tentative {attempt + 1}/{retries} échouée pour {url} : {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # backoff exponentiel
    logging.error(f"Impossible de fetcher : {url}")
    return None


# ---------------------------------------------------------------------------
# Crawler principal
# ---------------------------------------------------------------------------

def crawl_category(
    category_url: str,
    session: requests.Session,
    delay: float = 1.0,
    max_recipes: Optional[int] = None,
) -> CrawlResult:
    """Crawle toutes les pages d'une catégorie et retourne les liens de recettes.

    Args:
        max_recipes: Nombre maximum de recettes à collecter pour cette catégorie.
                     None = pas de limite.
    """
    logging.info(f"▶ Catégorie : {category_url}")
    if max_recipes is not None:
        logging.info(f"  (limité à {max_recipes} recettes)")

    # Nom de la catégorie depuis l'URL (ex: "accompagnements")
    match = re.search(r"categorie_([^.]+)\.htm", category_url)
    categorie_nom = match.group(1).replace("_", " ") if match else "inconnue"

    # --- Page 1 : on détecte aussi le nombre de pages ---
    soup = fetch_page(category_url, session)
    if soup is None:
        return CrawlResult(category_url, categorie_nom, 0)

    last_page = get_last_page(soup)
    logging.info(f"  → {last_page} page(s) détectée(s)")

    result = CrawlResult(
        categorie_url=category_url,
        categorie_nom=categorie_nom,
        nb_pages=last_page,
    )

    # Recettes page 1
    result.recettes.extend(parse_recipe_links(soup, categorie_nom))
    logging.info(f"  Page 1 : {len(result.recettes)} recettes")

    # Limite atteinte dès la page 1 ?
    if max_recipes is not None and len(result.recettes) >= max_recipes:
        result.recettes = result.recettes[:max_recipes]
        logging.info(f"  Limite par catégorie atteinte ({max_recipes}) — arrêt")
        return result

    # --- Pages suivantes ---
    for page_num in range(2, last_page + 1):
        time.sleep(delay)  # politesse envers le serveur
        page_url = f"{category_url}?page={page_num}"
        soup = fetch_page(page_url, session)

        if soup is None:
            logging.warning(f"  Page {page_num} ignorée (erreur)")
            continue

        new_recipes = parse_recipe_links(soup, categorie_nom)
        result.recettes.extend(new_recipes)
        logging.info(f"  Page {page_num} : +{len(new_recipes)} recettes (total : {len(result.recettes)})")

        # Sécurité : si une page ne retourne rien, on arrête
        if not new_recipes:
            logging.info(f"  Page {page_num} vide — arrêt anticipé")
            break

        # Limite par catégorie atteinte ?
        if max_recipes is not None and len(result.recettes) >= max_recipes:
            result.recettes = result.recettes[:max_recipes]
            logging.info(f"  Limite par catégorie atteinte ({max_recipes}) — arrêt")
            break

    return result


def crawl_all(
    category_urls: list[str],
    output_path: str = "recettes_links.json",
    delay: float = 1.0,
    max_recipes_per_category: Optional[int] = None,
    max_recipes_total: Optional[int] = None,
) -> dict:
    """Crawle toutes les catégories et sauvegarde les résultats en JSON.

    Args:
        max_recipes_per_category: Limite le nombre de recettes collectées par catégorie.
                                   None = pas de limite.
        max_recipes_total:        Stoppe le crawl global dès que ce seuil est atteint
                                   (toutes catégories confondues, après dédoublonnage).
                                   None = pas de limite.
    """
    if max_recipes_total is not None:
        logging.info(f"⚙ Limite globale : {max_recipes_total} recettes au total")
    if max_recipes_per_category is not None:
        logging.info(f"⚙ Limite par catégorie : {max_recipes_per_category} recettes")

    session = requests.Session()
    all_results = []
    all_urls: set[str] = set()  # dédoublonnage global inter-catégories
    global_limit_reached = False

    for cat_url in category_urls:
        # Calcul de la limite effective pour cette catégorie :
        # on prend le min entre la limite par catégorie et le reste disponible globalement
        remaining_global = (
            max_recipes_total - len(all_urls) if max_recipes_total is not None else None
        )

        if remaining_global is not None and remaining_global <= 0:
            logging.info("  Limite globale atteinte — catégories restantes ignorées")
            global_limit_reached = True
            break

        # Limite effective = la plus contraignante des deux
        effective_limit: Optional[int] = None
        if max_recipes_per_category is not None and remaining_global is not None:
            effective_limit = min(max_recipes_per_category, remaining_global)
        elif max_recipes_per_category is not None:
            effective_limit = max_recipes_per_category
        elif remaining_global is not None:
            effective_limit = remaining_global

        result = crawl_category(cat_url, session, delay=delay, max_recipes=effective_limit)

        # Dédoublonnage inter-catégories
        before = len(result.recettes)
        result.recettes = [r for r in result.recettes if r.url not in all_urls]
        all_urls.update(r.url for r in result.recettes)

        duplicates = before - len(result.recettes)
        if duplicates:
            logging.info(f"  {duplicates} doublon(s) inter-catégories supprimé(s)")

        all_results.append(result.to_dict())
        logging.info(f"  ✓ {len(result.recettes)} recettes uniques pour '{result.categorie_nom}' (total global : {len(all_urls)})\n")

        # Vérification limite globale après dédoublonnage
        if max_recipes_total is not None and len(all_urls) >= max_recipes_total:
            logging.info(f"  Limite globale atteinte ({max_recipes_total}) — arrêt")
            global_limit_reached = True
            break

        time.sleep(delay)  # pause entre catégories

    # Résumé global
    summary = {
        "total_recettes": len(all_urls),
        "limite_par_categorie": max_recipes_per_category,
        "limite_totale": max_recipes_total,
        "limite_globale_atteinte": global_limit_reached,
        "categories": all_results,
        # Liste plate de toutes les URLs pour un usage direct dans ton RAG
        "all_recipe_urls": list(all_urls),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info(f"\n✅ Terminé — {len(all_urls)} recettes sauvegardées dans '{output_path}'")
    return summary


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Crawler de liens de recettes 750g")
    parser.add_argument(
        "--output", default="recettes_links.json", help="Fichier JSON de sortie"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Délai entre les requêtes (secondes)"
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        metavar="N",
        help="Nombre maximum de recettes à collecter par catégorie (défaut : illimité)",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        metavar="N",
        help="Nombre maximum de recettes au total, toutes catégories confondues (défaut : illimité)",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="URLs de catégories (optionnel, remplace la liste par défaut)",
    )

    delay = 1.0
    max_per_category = 50
    max_total = 2000
    output = "recettes_links.json"

    
    categories = CATEGORY_URLS
    crawl_all(
        categories,
        output_path=output,
        delay=delay,
        max_recipes_per_category=max_per_category,
        max_recipes_total=max_total,
    )