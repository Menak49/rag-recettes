"""
Scraper 750g — Extraction complète des détails des recettes

Ce script :
- Lit un fichier JSON contenant les URLs des recettes
- Scrape les informations utiles :
    - titre
    - ingrédients
    - quantités
    - difficulté
    - temps préparation
    - temps cuisson
    - matériel
    - url
- Sauvegarde le tout dans un nouveau fichier JSON

Usage :
    python scraper_750g_details.py

Options :
    python scraper_750g_details.py --input recettes_links.json --output recettes_details.json
"""

import json
import time
import logging
import argparse
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9",
}


# -------------------------------------------------------------------
# FETCH
# -------------------------------------------------------------------

def fetch_page(url: str, session: requests.Session, retries: int = 3) -> Optional[BeautifulSoup]:
    """
    Télécharge une page et retourne BeautifulSoup.
    """

    for attempt in range(retries):

        try:
            response = session.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()

            return BeautifulSoup(response.text, "html.parser")

        except requests.RequestException as e:

            logging.warning(
                f"[{attempt + 1}/{retries}] Erreur pour {url} : {e}"
            )

            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    logging.error(f"Impossible de récupérer : {url}")

    return None


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Nettoie un texte.
    """

    return re.sub(r"\s+", " ", text).strip()


# -------------------------------------------------------------------
# FILTRES / PARASITES
# -------------------------------------------------------------------

PARASITE_MATERIALS = {
    "Acheter",
    'En cliquant sur "Acheter", vous serez redirigé vers un site externe.',
}

PARASITE_INGREDIENTS = [
    "less icon",
    "plus icon",
]


def is_valid_material(text: str) -> bool:
    """
    Vérifie si un matériel est valide.
    """

    if not text:
        return False

    text = clean_text(text)

    if text in PARASITE_MATERIALS:
        return False

    if "redirigé vers un site externe" in text.lower():
        return False

    if len(text) < 2:
        return False

    return True


def is_valid_ingredient(text: str) -> bool:
    """
    Vérifie si une ligne est un vrai ingrédient.
    """

    if not text:
        return False

    text_lower = text.lower()

    for parasite in PARASITE_INGREDIENTS:
        if parasite in text_lower:
            return False

    if "personnes" in text_lower and "icon" in text_lower:
        return False

    if "acheter" in text_lower:
        return False

    if len(text) < 2:
        return False

    return True


# -------------------------------------------------------------------
# EXTRACTION TEMPS
# -------------------------------------------------------------------

def extract_time(soup: BeautifulSoup, keyword: str) -> Optional[str]:
    """
    Extrait un temps (préparation/cuisson).
    """

    text = soup.get_text(" ", strip=True)

    patterns = [
        rf"{keyword}\s*:?\s*(\d+\s*h\s*\d*\s*min|\d+\s*min|\d+\s*h)",
        rf"{keyword}\s*(\d+\s*h\s*\d*\s*min|\d+\s*min|\d+\s*h)",
    ]

    for pattern in patterns:

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return clean_text(match.group(1)).replace(" ", "")

    return None


# -------------------------------------------------------------------
# DIFFICULTE
# -------------------------------------------------------------------

def extract_difficulty(soup: BeautifulSoup) -> Optional[str]:
    """
    Extrait la difficulté.
    """

    text = soup.get_text(" ", strip=True)

    difficulties = [
        "Très facile",
        "Facile",
        "Moyen",
        "Difficile",
    ]

    for diff in difficulties:

        if diff.lower() in text.lower():
            return diff

    return None


# -------------------------------------------------------------------
# INGREDIENTS
# -------------------------------------------------------------------

def extract_ingredients(soup: BeautifulSoup) -> list[dict]:
    """
    Extrait les ingrédients et quantités.
    """

    ingredients = []

    ingredient_blocks = soup.find_all(
        lambda tag:
        tag.name in ["li", "div"]
        and (
            "ingredient" in " ".join(tag.get("class", [])).lower()
            or "recipe-ingredient" in " ".join(tag.get("class", [])).lower()
        )
    )

    for block in ingredient_blocks:

        text = clean_text(block.get_text(" ", strip=True))

        if not is_valid_ingredient(text):
            continue

        match = re.match(
            r"^([\d\/\.,]+\s*(?:g|kg|cl|ml|l|c\.?\s*à\s*s\.?|pincées?|tour|tranches?)?\s*)?(.*)$",
            text,
            re.IGNORECASE
        )

        if match:

            quantity = clean_text(match.group(1) or "")
            ingredient = clean_text(match.group(2))

            # supprime "de", "d'", etc.
            ingredient = re.sub(
                r"^(de|d'|du|des)\s+",
                "",
                ingredient,
                flags=re.IGNORECASE
            )

            # suppression pubs parasites
            ingredient = re.sub(
                r"Le Sucre Cristal Daddy.*$",
                "",
                ingredient,
                flags=re.IGNORECASE
            ).strip()

            ingredient = ingredient.replace(". à s.", "").strip()

            if ingredient:

                ingredients.append({
                    "ingredient": ingredient,
                    "quantite": quantity,
                })

    # dédoublonnage
    unique = []
    seen = set()

    for ing in ingredients:

        key = (ing["ingredient"], ing["quantite"])

        if key not in seen:
            seen.add(key)
            unique.append(ing)

    return unique


# -------------------------------------------------------------------
# MATERIEL
# -------------------------------------------------------------------

def extract_materials(soup: BeautifulSoup) -> list[str]:
    materials = []

    # 1) chercher le titre "Matériel / Ustensiles"
    headings = soup.find_all(["h2", "h3", "h4", "p", "div", "span"])

    material_section = None

    for h in headings:
        text = clean_text(h.get_text(" ", strip=True)).lower()

        if "matériel" in text or "ustensiles" in text or "équipement" in text:
            material_section = h
            break

    if not material_section:
        return []

    # 2) remonter au parent contenant la liste
    container = material_section.find_parent()

    if not container:
        return []

    # 3) extraire uniquement les éléments proches
    candidates = container.find_all(["li", "span", "div"])

    for c in candidates:
        text = clean_text(c.get_text(" ", strip=True))

        if not text:
            continue

        lower = text.lower()

        # filtres parasites
        if any(x in lower for x in [
            "connexion",
            "inscription",
            "acheter",
            "redirigé",
            "site externe",
            "personnes",
            "plus icon",
            "less icon"
        ]):
            continue

        # éviter les textes trop longs (pub / footer)
        if len(text) > 40:
            continue

        # garder uniquement objets réalistes cuisine
        if any(tool in lower for tool in [
            "saladier", "casserole", "four", "poêle", "mixeur",
            "fouet", "maryse", "spatule", "airfryer", "cookeo",
            "bol", "torchon", "plaque", "moule", "presse"
        ]):
            materials.append(text)

    return list(dict.fromkeys(materials))


# -------------------------------------------------------------------
# PARSE RECIPE
# -------------------------------------------------------------------

def parse_recipe(url: str, soup: BeautifulSoup) -> dict:
    """
    Parse une recette complète.
    """

    title = None

    h1 = soup.find("h1")

    if h1:
        title = clean_text(h1.get_text())

    ingredients = extract_ingredients(soup)

    prep_time = extract_time(soup, "préparation")
    cook_time = extract_time(soup, "cuisson")

    difficulty = extract_difficulty(soup)

    materials = extract_materials(soup)

    return {
        "url": url,
        "titre": title,
        "difficulte": difficulty,
        "temps_preparation": prep_time,
        "temps_cuisson": cook_time,
        "materiel": materials,
        "ingredients": ingredients,
    }


# -------------------------------------------------------------------
# LOAD URLS
# -------------------------------------------------------------------

def load_recipe_urls(input_file: str) -> list[str]:
    """
    Charge les URLs depuis le JSON.
    """

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("all_recipe_urls", [])


# -------------------------------------------------------------------
# MAIN SCRAPER
# -------------------------------------------------------------------

def scrape_recipes(
    input_file: str,
    output_file: str,
    delay: float = 1.0,
):
    """
    Scrape toutes les recettes.
    """

    urls = load_recipe_urls(input_file)

    logging.info(f"{len(urls)} recettes à scraper")

    session = requests.Session()

    results = []

    for i, url in enumerate(urls, start=1):

        logging.info(f"[{i}/{len(urls)}] {url}")

        soup = fetch_page(url, session)

        if soup is None:
            continue

        try:

            recipe_data = parse_recipe(url, soup)

            results.append(recipe_data)

        except Exception as e:

            logging.error(f"Erreur parsing {url} : {e}")

        time.sleep(delay)

    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info(
        f"\n✅ {len(results)} recettes sauvegardées dans '{output_file}'"
    )


# -------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Scraper de détails des recettes 750g"
    )

    parser.add_argument(
        "--input",
        default="recettes_links.json",
        help="Fichier JSON contenant les URLs"
    )

    parser.add_argument(
        "--output",
        default="recettes_details.json",
        help="Fichier JSON de sortie"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Délai entre les requêtes"
    )

    args = parser.parse_args()

    scrape_recipes(
        input_file=args.input,
        output_file=args.output,
        delay=args.delay,
    )