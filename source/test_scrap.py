from recipe_scrapers import scrape_me
scraper = scrape_me("https://www.750g.com/gratin-dauphinois-r99820.htm")
scraper = scrape_me("https://www.750g.com/gratin-dauphinois-r99820.htm")
print(scraper.ingredients())  # liste prête
print(scraper.instructions())
