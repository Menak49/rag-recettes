import re
import json
import time
import logging
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

START_URLS = [
    "https://www.750g.com/categorie_accompagnements.htm",
    "https://www.750g.com/recettes-aperitifs/",
    "https://www.750g.com/recettes-bases/",
    "https://www.750g.com/recettes-boissons/",
    "https://www.750g.com/recettes-confiseries/",
    "https://www.750g.com/categorie_confitures.htm",
    "https://www.750g.com/recettes-desserts/biscuits/",
    "https://www.750g.com/recettes-pdg/",
    "https://www.750g.com/recettes-desserts/patisseries/",
    "https://www.750g.com/recettes-salades/",
    "https://www.750g.com/recettes-sauces/",
    "https://www.750g.com/recettes-potages/",
    "https://www.750g.com/recettes-desserts/tartes/",
    "https://www.750g.com/recettes-plats/",
    "https://www.750g.com/recettes-plats/traditionnels/",
    "https://www.750g.com/recettes-desserts/",
    "https://www.750g.com/recettes-entrees/",
]

# Fichier de sortie
OUTPUT_FILE = "data/recettes_links.json"

# Délai entre les requêtes
DELAY = 0.5

RECIPE_PATTERN = re.compile(r"^https://www\.750g\.com/[a-z0-9\-]+-r\d+\.htm$")


def fetch_page(url, session):
    try:
        r = session.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 404:
            logging.warning(f"⚠️ Page introuvable (404) : {url}")
            return None
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except requests.RequestException as e:
        logging.warning(f"❌ Erreur sur {url} : {e}")
        return None


def extract_recipes_from_soup(soup):
    recipes = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("/"):
            href = "https://www.750g.com" + href
        if RECIPE_PATTERN.match(href):
            recipes.add(href)
    return recipes


def get_last_page(soup):
    last_page = 1
    for link in soup.find_all("a", href=re.compile(r"\?page=\d+")):
        match = re.search(r"\?page=(\d+)", link["href"])
        if match:
            last_page = max(last_page, int(match.group(1)))
    return last_page


def clean_pagination_url(base_url, page_num):
    if ".htm" in base_url:
        return f"{base_url}?page={page_num}"
    return f"{base_url}?page={page_num}" if base_url.endswith("/") else f"{base_url}/?page={page_num}"


def scrape_single_url(url, session):
    recipes_found = set()

    logging.info(f"🌐 Lecture de la page source : {url}")
    soup = fetch_page(url, session)
    if not soup:
        return set()

    initial_recipes = extract_recipes_from_soup(soup)
    recipes_found.update(initial_recipes)
    logging.info(f"   📈 +{len(initial_recipes)} recettes trouvées sur la page principale.")

    if len(recipes_found) >= 200:
        return recipes_found

    last_page = get_last_page(soup)

    if last_page > 1:
        logging.info(f"   📄 Pagination détectée ! {last_page} pages à faire.")
        for page_num in range(2, last_page + 1):
            if len(recipes_found) >= 200:
                logging.info("   🛑 Quota de 200 recettes atteint. Arrêt.")
                break

            page_url = clean_pagination_url(url, page_num)
            time.sleep(DELAY)

            logging.info(f"   📄 Page pagination : {page_url}")
            soup_p = fetch_page(page_url, session)
            if soup_p:
                p_recipes = extract_recipes_from_soup(soup_p)
                p_new = p_recipes - recipes_found
                recipes_found.update(p_new)
                logging.info(f"   📈 +{len(p_new)} nouvelles recettes. Total: {len(recipes_found)}")
    else:
        logging.info("   🛑 Pas de pagination. Fin pour cette URL.")

    return recipes_found


def crawl_all(start_urls, output_path=OUTPUT_FILE):
    session = requests.Session()

    # Set global pour dédoublonner toutes catégories confondues
    all_urls: set[str] = set()

    for idx, url in enumerate(start_urls, 1):
        logging.info(f"\n🔥 === DEBUT URL {idx}/{len(start_urls)} ===")
        category_recipes = scrape_single_url(url, session)
        new = category_recipes - all_urls
        all_urls.update(new)
        logging.info(f"🎯 +{len(new)} nouvelles. Total global : {len(all_urls)}")
        time.sleep(DELAY)

    # Format attendu par scraper_750g_details.py
    output_data = {
        "total": len(all_urls),
        "all_recipe_urls": list(all_urls),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logging.info(f"\n✅ Terminé ! {len(all_urls)} URLs sauvegardées dans '{output_path}'")
    return output_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    crawl_all(START_URLS)