import csv
import os
import argparse
from playwright.sync_api import sync_playwright, TimeoutError

# Função para fazer o scraping
def scrape_ge_globo(team_name):

    team_url = f'https://ge.globo.com/futebol/times/{team_name}/'

    if not os.path.exists('CSVs'):
        os.makedirs('CSVs')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        try:
            page.goto(team_url, timeout=100000)
        except TimeoutError:
            print(f"Não foi possível carregar a página para o time '{team_name}'.")
            browser.close()
            return

        if "Página não encontrada" in page.content():
            print(f"A página para o time '{team_name}' não foi encontrada.")
            browser.close()
            return

        # Scroll até o final da página
        previous_height = 0
        while True:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(3000)
            new_height = page.evaluate("document.body.scrollHeight")
            if new_height == previous_height:
                break
            previous_height = new_height

        # Scroll de volta para o topo da página
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(2000)

        # Coleta de links de notícias
        news_links = set()
        scroll_increment = 500
        page_height = page.evaluate("document.body.scrollHeight")

        for position in range(0, page_height, scroll_increment):
            page.evaluate(f"window.scrollTo(0, {position})")
            page.wait_for_timeout(2000)
            current_links = page.locator(f'a[href*="/futebol/times/{team_name}/noticia"]').all()
            news_links.update([link.get_attribute('href') for link in current_links if link.get_attribute('href')])

        news_urls = list(news_links)
        news_data = []
        for url in news_urls:
            page.goto(url)

            # Captura o título
            try:
                title = page.locator('body > div.glb-grid > main > div.row.content-head.non-featured > div.title > h1').inner_text()
            except:
                title = "Título não encontrado"

            # Captura o subtítulo
            try:
                subtitle = page.locator('body > div.glb-grid > main > div.row.content-head.non-featured > div.medium-centered.subtitle > h2').inner_text()
            except:
                subtitle = "Sem subtítulo"

            # Captura o autor e remove o local
            try:
                author_full = page.locator('body > div.glb-grid > main > div.content__signa-share > div.content__signature > div > div > p.content-publication-data__from').inner_text()
                # Remove "Por " e qualquer coisa depois de "—"
                if author_full.startswith("Por "):
                    author = author_full[3:].split("—")[0].strip()
                else:
                    author = author_full.split("—")[0].strip()
            except:
                author = "Autor não encontrado"

            # Captura o conteúdo completo da notícia
            try:
                article_content = page.locator('div[id^="chunk-"] > div > p').all_inner_texts()
                article_content = "\n".join(article_content)
            except:
                article_content = "Conteúdo não encontrado"

            news_data.append({
                "title": title,
                "subtitle": subtitle,
                "author": author,
                "article_content": article_content
            })
        browser.close()

        # Verifica se há dados para salvar antes de criar o arquivo CSV
        if news_data:
            csv_filename = f'CSVs/news_data_{team_name}.csv'
            with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=["title", "subtitle", "author", "article_content"])
                writer.writeheader()
                for data in news_data:
                    writer.writerow(data)
        else:
            print(f"Não foram encontradas notícias para o time '{team_name}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape news articles for a given football team from GE Globo.')
    parser.add_argument('team_name', type=str, help='Name of the football team to scrape news for')
    args = parser.parse_args()

    scrape_ge_globo(args.team_name)