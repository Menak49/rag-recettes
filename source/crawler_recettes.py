"""
Scraper 750g — Extraction complète des détails des recettes

Ce script :
- Lit un fichier JSON contenant les URLs des recettes
- Scrape les informations utiles :
    - titre
    - ingrédients
    - instructions (étapes de préparation)
    - difficulté
    - temps préparation
    - temps cuisson
    - matériel
    - nb_personnes (nombre de personnes/pièces pour les quantités affichées)
    - url
- Sauvegarde le tout dans un nouveau fichier JSON

Usage :
    python scraper_750g_details.py
"""

import json
import time
import logging
import re
import html
from typing import Optional

import requests
from bs4 import BeautifulSoup



HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9",
}



def fetch_page(url: str, session: requests.Session, retries: int = 3) -> Optional[BeautifulSoup]:
    """Télécharge une page et retourne BeautifulSoup."""
    for attempt in range(retries):
        try:
            response = session.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            logging.warning(f"[{attempt + 1}/{retries}] Erreur pour {url} : {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    logging.error(f"Impossible de récupérer : {url}")
    return None

def clean_text(text: str) -> str:
    """Nettoie un texte et décode les entités HTML (double passe pour les entités doublées)."""
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()



PARASITE_MATERIALS = {
    "Acheter",
    'En cliquant sur "Acheter", vous serez redirigé vers un site externe.',
}

PARASITE_INGREDIENTS = [
    "less icon",
    "plus icon",
]


def is_valid_material(text: str) -> bool:
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



SERVING_TYPE_PERSONS = "personnes"
SERVING_TYPE_PIECES  = "pieces"
SERVING_TYPE_UNKNOWN = None

PIECES_KEYWORDS = [
    "pièces?", "biscuits?", "cookies?", "muffins?", "cupcakes?",
    "macarons?", "tartelettes?", "gâteaux?", "cakes?", "éclairs?",
    "choux?", "cannelés?", "financiers?", "madeleines?", "brioches?",
    "pains?", "petits pains?", "galettes?", "crêpes?", "pancakes?",
    "verrines?", "bocaux?", "bocal",
]
PERSON_KEYWORDS = [
    "personnes?", "portions?", "convives?",
]


def detect_serving_type(label: str) -> Optional[str]:
    label_lower = label.lower()
    for kw in PERSON_KEYWORDS:
        if re.search(kw, label_lower):
            return SERVING_TYPE_PERSONS
    for kw in PIECES_KEYWORDS:
        if re.search(kw, label_lower):
            return SERVING_TYPE_PIECES
    return SERVING_TYPE_UNKNOWN


def extract_servings(soup: BeautifulSoup) -> tuple[Optional[int], Optional[str]]:
    """
    Extrait le nombre de portions et leur type.

    Retourne (valeur, type) :
        (6, "personnes")  → recette pour 6 personnes
        (12, "pieces")    → recette pour 12 pièces
        (None, None)      → rien trouvé
    """

    # jSON-LD schema.org (le plus fiable)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                data = next((d for d in data if d.get("@type") == "Recipe"), {})
            recipe_yield = str(data.get("recipeYield", ""))
            if recipe_yield:
                m = re.search(r"(\d+)", recipe_yield)
                if m:
                    return int(m.group(1)), detect_serving_type(recipe_yield)
        except Exception:
            pass

    # balises avec classe contenant "serving" / "person" / "portion"
    candidates = soup.find_all(
        lambda tag: any(
            kw in " ".join(tag.get("class", [])).lower()
            for kw in ["serving", "person", "portion", "convive", "nombre"]
        )
    )
    for c in candidates:
        text = c.get_text(" ", strip=True)
        m = re.search(r"(\d+)", text)
        if m:
            return int(m.group(1)), detect_serving_type(text)

    # recherche textuelle dans la page entière
    text = soup.get_text(" ", strip=True)

    typed_patterns = [
        (r"(\d+)\s*(personnes?|convives?|portions?)", SERVING_TYPE_PERSONS),
        (r"pour\s+(\d+)\s*(personnes?|convives?)", SERVING_TYPE_PERSONS),
    ] + [
        (rf"(\d+)\s*({kw})", SERVING_TYPE_PIECES)
        for kw in PIECES_KEYWORDS
    ] + [
        (r"pour\s+(\d+)", SERVING_TYPE_UNKNOWN),
    ]

    for pattern, stype in typed_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 100:
                return val, stype

    logging.warning("Nombre de personnes/pièces introuvable.")
    return None, SERVING_TYPE_UNKNOWN


def extract_time(soup: BeautifulSoup, keyword: str) -> Optional[str]:
    """Extrait un temps (préparation/cuisson)."""
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


def extract_difficulty(soup: BeautifulSoup) -> Optional[str]:
    """Extrait la difficulté."""
    text = soup.get_text(" ", strip=True)
    for diff in ["Très facile", "Facile", "Moyen", "Difficile"]:
        if diff.lower() in text.lower():
            return diff
    return None



def extract_instructions(soup: BeautifulSoup) -> list[str]:
    """
    Extrait les étapes de préparation.

    Stratégie (par ordre de priorité) :
    1. JSON-LD schema.org → recipeInstructions
    2. HTML : ol > li sous la section "Préparation"
    3. HTML : premier ol avec plusieurs li dans la page
    """

    # JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                data = next((d for d in data if d.get("@type") == "Recipe"), {})
            instructions_raw = data.get("recipeInstructions", [])
            if instructions_raw:
                steps = []
                for item in instructions_raw:
                    if isinstance(item, str):
                        step = clean_text(item)
                    elif isinstance(item, dict):
                        step = clean_text(item.get("text", "") or item.get("name", ""))
                    else:
                        continue
                    if step:
                        steps.append(step)
                if steps:
                    return steps
        except Exception:
            pass

    # HTML : section Préparation puis ol suivant
    prep_heading = None
    for tag in soup.find_all(["h2", "h3", "h4"]):
        if "préparation" in clean_text(tag.get_text()).lower():
            prep_heading = tag
            break

    if prep_heading:
        for sibling in prep_heading.find_next_siblings():
            if sibling.name == "ol":
                steps = _extract_steps_from_ol(sibling)
                if steps:
                    return steps
            if sibling.name in ["h2", "h3", "h4"]:
                break

    #  Fallback : premier ol avec au moins 2 li
    for ol in soup.find_all("ol"):
        steps = _extract_steps_from_ol(ol)
        if len(steps) >= 2:
            return steps

    return []


def _extract_steps_from_ol(ol_tag) -> list[str]:
    """Extrait et nettoie les étapes d'un tag <ol>."""
    steps = []
    for li in ol_tag.find_all("li", recursive=False):
        text = clean_text(li.get_text(" ", strip=True))
        text = re.sub(r"^\d+\s+", "", text)
        if not text or len(text) < 10:
            continue
        if any(kw in text.lower() for kw in ["connexion", "inscription", "acheter"]):
            continue
        steps.append(text)
    return steps


def extract_ingredients(soup: BeautifulSoup) -> list[dict]:
    """Extrait les ingrédients et quantités tels qu'affichés sur la page."""
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
            r"^([\d\/\.,½⅓¼⅔¾⅛⅜⅝⅞]+\s*(?:g|kg|cl|ml|l|c\.?\s*à\s*s\.?|pincées?|tour|tranches?)?\s*)?(.*)$",
            text,
            re.IGNORECASE
        )

        if match:
            quantity   = clean_text(match.group(1) or "")
            ingredient = clean_text(match.group(2))

            ingredient = re.sub(r"^(de|d'|du|des)\s+", "", ingredient, flags=re.IGNORECASE)
            ingredient = re.sub(r"Le Sucre Cristal Daddy.*$", "", ingredient, flags=re.IGNORECASE).strip()
            ingredient = ingredient.replace(". à s.", "").strip()

            if ingredient:
                ingredients.append({
                    "ingredient": ingredient,
                    "quantite": quantity,
                })

    # dédoublonnage
    unique = []
    seen   = set()
    for ing in ingredients:
        key = (ing["ingredient"], ing["quantite"])
        if key not in seen:
            seen.add(key)
            unique.append(ing)

    return unique


def extract_materials(soup: BeautifulSoup) -> list[str]:
    # Le nom de chaque ustensile est toujours dans un <span class="recipe-equipments-item-label">
    labels = soup.find_all("span", class_="recipe-equipments-item-label")
    
    materials = []
    for span in labels:
        name = clean_text(span.get_text(" ", strip=True))
        if name and len(name) >= 2:
            materials.append(name)
    
    return list(dict.fromkeys(materials))

def parse_recipe(url: str, soup: BeautifulSoup) -> dict:
    """Parse une recette complète."""

    title = None
    h1    = soup.find("h1")
    if h1:
        title = clean_text(h1.get_text())

    servings_count, servings_type = extract_servings(soup)

    return {
        "url":               url,
        "titre":             title,
        "difficulte":        extract_difficulty(soup),
        "temps_preparation": extract_time(soup, "préparation"),
        "temps_cuisson":     extract_time(soup, "cuisson"),
        "nb_personnes":      servings_count,   # null si introuvable
        "type_quantite":     servings_type,    # "personnes", "pieces" ou null
        "materiel":          extract_materials(soup),
        "ingredients":       extract_ingredients(soup),
        "instructions":      extract_instructions(soup),
    }



def load_recipe_urls(input_file: str) -> list[str]:
    """Charge les URLs depuis le JSON."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("all_recipe_urls", [])



def scrape_recipes(input_file: str, output_file: str, delay: float = 1.0):
    """Scrape toutes les recettes."""
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
            results.append(parse_recipe(url, soup))
        except Exception as e:
            logging.error(f"Erreur parsing {url} : {e}")
        time.sleep(delay)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info(f"\n✅ {len(results)} recettes sauvegardées dans '{output_file}'")




INPUT_FILE = "data/recettes_links.json"
OUTPUT_FILE = "data/recettes_details.json"
DELAY = 1

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    scrape_recipes(input_file=INPUT_FILE, output_file=OUTPUT_FILE, delay=DELAY)