"""
Scraper 750g — Extraction complète des détails des recettes

Ce script :
- Lit un fichier JSON contenant les URLs des recettes
- Scrape les informations utiles :
    - titre
    - ingrédients (quantités normalisées pour 1 personne)
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
import re
from fractions import Fraction
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


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Nettoie un texte."""
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


# -------------------------------------------------------------------
# EXTRACTION DU NOMBRE DE PERSONNES
# -------------------------------------------------------------------

# Types de quantité de recette
SERVING_TYPE_PERSONS = "personnes"
SERVING_TYPE_PIECES  = "pieces"
SERVING_TYPE_UNKNOWN = None


def extract_servings(soup: BeautifulSoup) -> tuple[Optional[int], Optional[str]]:
    """
    Extrait le nombre de portions et leur type.

    Retourne un tuple (valeur, type) :
        (6, "personnes")  → recette pour 6 personnes  → on divise
        (12, "pieces")    → recette pour 12 pièces     → on NE divise PAS (nb_personnes=null)
        (None, None)      → rien trouvé                → on NE divise PAS (nb_personnes=null)

    Logique de détection du type :
    - Si le texte contient "personnes", "convives", "portions" → type personnes
    - Si le texte contient "pièces", "biscuits", "cookies", "muffins",
      "cupcakes", "macarons", "tartelettes", "gâteaux", "cakes", etc. → type pièces
    - Sinon → inconnu
    """

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

    def detect_type(label: str) -> Optional[str]:
        """Détecte le type depuis un label textuel."""
        label_lower = label.lower()
        for kw in PERSON_KEYWORDS:
            if re.search(kw, label_lower):
                return SERVING_TYPE_PERSONS
        for kw in PIECES_KEYWORDS:
            if re.search(kw, label_lower):
                return SERVING_TYPE_PIECES
        return SERVING_TYPE_UNKNOWN

    # 1) JSON-LD schema.org (le plus fiable)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                data = next((d for d in data if d.get("@type") == "Recipe"), {})
            recipe_yield = str(data.get("recipeYield", ""))
            if recipe_yield:
                m = re.search(r"(\d+)", recipe_yield)
                if m:
                    val = int(m.group(1))
                    stype = detect_type(recipe_yield)
                    return val, stype
        except Exception:
            pass

    # 2) Balises avec classe contenant "serving" / "person" / "portion"
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
            val = int(m.group(1))
            stype = detect_type(text)
            return val, stype

    # 3) Recherche textuelle dans la page entière
    text = soup.get_text(" ", strip=True)

    # Patterns ordonnés : les plus précis d'abord
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

    # 4) Introuvable
    logging.warning("Nombre de personnes/pièces introuvable.")
    return None, SERVING_TYPE_UNKNOWN


# -------------------------------------------------------------------
# NORMALISATION D'UNE QUANTITE VERS 1 PERSONNE
# -------------------------------------------------------------------

# Unités reconnues (pour ne pas les confondre avec des nombres)
UNITS = r"(?:g|kg|cl|ml|l|c\.?\s*à\s*s\.?|c\.?\s*à\s*c\.?|pincées?|tours?|tranches?|sachets?|boîtes?|boites?|gousses?|branches?)"

def parse_quantity_value(qty_str: str) -> Optional[float]:
    """
    Convertit une chaîne de quantité en float.
    Gère : "200", "1/2", "½", "1 1/2", "0,5", "2.5"
    Retourne None si non parsable.
    """
    if not qty_str:
        return None

    s = qty_str.strip()

    # Fractions unicode : ½ ⅓ ¼ ⅔ ¾ etc.
    unicode_fractions = {
        "½": 0.5, "⅓": 1/3, "¼": 0.25, "⅔": 2/3,
        "¾": 0.75, "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
    }
    for uf, val in unicode_fractions.items():
        s = s.replace(uf, f" {val} ")

    s = s.replace(",", ".")

    # "1 1/2" → somme de parties
    # On cherche d'abord un entier + fraction : "1 1/2"
    m = re.match(r"^(\d+)\s+(\d+)/(\d+)$", s.strip())
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))

    # Fraction simple "3/4"
    m = re.match(r"^(\d+)/(\d+)$", s.strip())
    if m:
        return int(m.group(1)) / int(m.group(2))

    # Nombre simple (peut contenir un espace dû au remplacement unicode)
    s_clean = s.strip()
    # Si plusieurs nombres, prendre le premier
    m = re.search(r"[\d.]+", s_clean)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass

    return None


def format_quantity(value: float) -> str:
    """
    Formate un float de façon lisible :
    - Entier si valeur entière
    - 2 décimales max sinon, sans zéros trailing
    """
    if value == int(value):
        return str(int(value))
    # arrondi à 2 décimales
    rounded = round(value, 2)
    # supprimer les zéros inutiles
    return f"{rounded:g}"


def normalize_quantity(qty_str: str, servings: Optional[int]) -> str:
    """
    Divise la partie numérique d'une quantité par `servings`.
    Conserve l'unité intacte.
    Exemples :
        "200 g", 4  →  "50 g"
        "3",     6  →  "0.5"
        "1/2",   2  →  "0.25"
        "",      4  →  ""      (inchangé si vide)
        "sel",   4  →  "sel"   (inchangé si pas de nombre)
        "200 g", None → "200 g" (inchangé si pas de division)
    """
    if not qty_str or not servings or servings <= 1:
        return qty_str

    # Séparer la partie numérique de l'unité
    # Regex : optionnel entier+espace, puis fraction ou décimal
    num_pattern = r"^((?:\d+\s+)?\d+[/.,]\d+|\d+[/.,]\d+|\d+)"
    m = re.match(num_pattern, qty_str.strip())

    if not m:
        # Pas de partie numérique détectée (ex: "sel", "QS")
        return qty_str

    num_str = m.group(1)
    rest = qty_str[m.end():]  # unité + reste

    value = parse_quantity_value(num_str)
    if value is None:
        return qty_str

    new_value = value / servings
    return format_quantity(new_value) + rest


# -------------------------------------------------------------------
# EXTRACTION TEMPS
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# DIFFICULTE
# -------------------------------------------------------------------

def extract_difficulty(soup: BeautifulSoup) -> Optional[str]:
    """Extrait la difficulté."""
    text = soup.get_text(" ", strip=True)
    for diff in ["Très facile", "Facile", "Moyen", "Difficile"]:
        if diff.lower() in text.lower():
            return diff
    return None


# -------------------------------------------------------------------
# INGREDIENTS
# -------------------------------------------------------------------

def extract_ingredients(soup: BeautifulSoup, servings: Optional[int]) -> list[dict]:
    """
    Extrait les ingrédients et quantités.
    Si servings est None ou <= 1 : quantités conservées telles quelles.
    Sinon : quantités normalisées pour 1 personne.
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
            r"^([\d\/\.,½⅓¼⅔¾⅛⅜⅝⅞]+\s*(?:g|kg|cl|ml|l|c\.?\s*à\s*s\.?|pincées?|tour|tranches?)?\s*)?(.*)$",
            text,
            re.IGNORECASE
        )

        if match:
            quantity = clean_text(match.group(1) or "")
            ingredient = clean_text(match.group(2))

            # supprime "de", "d'", etc.
            ingredient = re.sub(r"^(de|d'|du|des)\s+", "", ingredient, flags=re.IGNORECASE)

            # suppression pubs parasites
            ingredient = re.sub(
                r"Le Sucre Cristal Daddy.*$", "", ingredient, flags=re.IGNORECASE
            ).strip()
            ingredient = ingredient.replace(". à s.", "").strip()

            if ingredient:
                # Normalisation de la quantité pour 1 personne
                normalized_qty = normalize_quantity(quantity, servings)

                ingredients.append({
                    "ingredient": ingredient,
                    "quantite": normalized_qty,
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

    headings = soup.find_all(["h2", "h3", "h4", "p", "div", "span"])
    material_section = None

    for h in headings:
        text = clean_text(h.get_text(" ", strip=True)).lower()
        if "matériel" in text or "ustensiles" in text or "équipement" in text:
            material_section = h
            break

    if not material_section:
        return []

    container = material_section.find_parent()
    if not container:
        return []

    candidates = container.find_all(["li", "span", "div"])

    for c in candidates:
        text = clean_text(c.get_text(" ", strip=True))
        if not text:
            continue
        lower = text.lower()
        if any(x in lower for x in [
            "connexion", "inscription", "acheter", "redirigé",
            "site externe", "personnes", "plus icon", "less icon"
        ]):
            continue
        if len(text) > 40:
            continue
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
    """Parse une recette complète, quantités normalisées pour 1 personne."""

    title = None
    h1 = soup.find("h1")
    if h1:
        title = clean_text(h1.get_text())

    # Extraire le nombre de portions et son type AVANT les ingrédients
    servings_count, servings_type = extract_servings(soup)

    # On ne divise les quantités QUE si c'est un nombre de personnes/portions
    divisor = servings_count if servings_type == SERVING_TYPE_PERSONS else None

    ingredients = extract_ingredients(soup, divisor)

    prep_time = extract_time(soup, "préparation")
    cook_time = extract_time(soup, "cuisson")
    difficulty = extract_difficulty(soup)
    materials = extract_materials(soup)

    # nb_personnes : nombre si recette pour X personnes, None sinon
    nb_personnes = servings_count if servings_type == SERVING_TYPE_PERSONS else None

    return {
        "url": url,
        "titre": title,
        "difficulte": difficulty,
        "temps_preparation": prep_time,
        "temps_cuisson": cook_time,
        "nb_personnes": nb_personnes,      # null si pièces ou inconnu
        "materiel": materials,
        "ingredients": ingredients,        # quantités pour 1 personne (si nb_personnes défini)
    }


# -------------------------------------------------------------------
# LOAD URLS
# -------------------------------------------------------------------

def load_recipe_urls(input_file: str) -> list[str]:
    """Charge les URLs depuis le JSON."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("all_recipe_urls", [])


# -------------------------------------------------------------------
# MAIN SCRAPER
# -------------------------------------------------------------------

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
            recipe_data = parse_recipe(url, soup)
            results.append(recipe_data)
        except Exception as e:
            logging.error(f"Erreur parsing {url} : {e}")
        time.sleep(delay)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info(f"\n✅ {len(results)} recettes sauvegardées dans '{output_file}'")


# -------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------

# Fichier JSON contenant les URLs des recettes
INPUT_FILE = "data/recettes_links.json"

# Fichier JSON de sortie
OUTPUT_FILE = "data/recettes_details.json"

# Délai en secondes entre chaque requête
DELAY = 0.1


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    scrape_recipes(input_file=INPUT_FILE, output_file=OUTPUT_FILE, delay=DELAY)