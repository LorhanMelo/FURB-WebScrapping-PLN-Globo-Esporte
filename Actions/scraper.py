import csv
import os
import argparse
from playwright.sync_api import sync_playwright, TimeoutError
import time

# Função para tentar acessar uma página com retry
def load_page_with_retry(page, url, retries=3, timeout=100000):
    for attempt in range(retries):
        try:
            print(f"Tentando carregar {url} - Tentativa {attempt + 1}")
            page.goto(url, timeout=timeout)
            return True
        except TimeoutError:
            print(f"Erro ao carregar {url}: Timeout na tentativa {attempt + 1}")
            if attempt < retries - 1:
                print(f"Tentando novamente...")
                time.sleep(2)
            else:
                print(f"Falha ao carregar {url} após {retries} tentativas. Pulando...")
                return False

# Função para fazer o scraping
def scrape_ge_globo(team_name):

    # Ajustando a URL para diferentes times
    if team_name == "cuiaba":
        team_url = f'https://ge.globo.com/mt/futebol/times/{team_name}/'
    elif team_name == "athletico-pr":
        team_url = f'https://ge.globo.com/pr/futebol/times/{team_name}/'
    else:
        team_url = f'https://ge.globo.com/futebol/times/{team_name}/'

    if not os.path.exists('../CSVs/singleCSVs'):
        os.makedirs('../CSVs/singleCSVs')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        if not load_page_with_retry(page, team_url):
            browser.close()
            return

        if "Página não encontrada" in page.content():
            print(f"A página para o time '{team_name}' não foi encontrada.")
            browser.close()
            return

        # Limite de cliques no botão "Veja Mais" (Altere aqui se quiser mais noticias, agora retorna +- 1400-1500)
        max_clicks = 250
        click_count = 0

        while click_count < max_clicks:
            see_more_button = page.locator("#feed-placeholder > div > div > div.load-more.gui-color-primary-bg")

            if see_more_button.is_visible():
                try:
                    see_more_button.click()
                    print(f"Clicou no botão 'Veja Mais' ({click_count + 1}/{max_clicks})")
                except Exception as e:
                    print(f"Erro ao clicar no botão 'Veja Mais': {e}")
                    break
                click_count += 1
                page.wait_for_timeout(3000)
            else:
                print("Botão 'Veja Mais' não encontrado. Continuando...")
                break

        news_links = set()
        current_links = page.locator(f'a[href*="/futebol/times/{team_name}/noticia"]').all()
        news_links.update([link.get_attribute('href') for link in current_links if link.get_attribute('href')])

        if not news_links:
            print("Não foram encontrados links suficientes, tentando realizar scroll para carregar mais conteúdo.")
            previous_height = page.evaluate("document.body.scrollHeight")
            while True:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(3000)
                new_height = page.evaluate("document.body.scrollHeight")
                if new_height == previous_height:
                    break
                previous_height = new_height

            current_links = page.locator(f'a[href*="/futebol/times/{team_name}/noticia"]').all()
            news_links.update([link.get_attribute('href') for link in current_links if link.get_attribute('href')])

        if not news_links:
            print(f"Não foram encontradas notícias para o time '{team_name}'.")
            browser.close()
            return

        news_urls = list(news_links)
        news_data = []
        for url in news_urls:
            if not load_page_with_retry(page, url):
                continue

            try:
                title = page.locator('body > div.glb-grid > main > div.row.content-head.non-featured > div.title > h1').inner_text()
            except:
                title = "Título não encontrado"

            try:
                subtitle = page.locator('body > div.glb-grid > main > div.row.content-head.non-featured > div.medium-centered.subtitle > h2').inner_text()
            except:
                subtitle = "Sem subtítulo"

            try:
                author_full = page.locator('body > div.glb-grid > main > div.content__signa-share > div.content__signature > div > div > p.content-publication-data__from').inner_text()
                if author_full.startswith("Por "):
                    author = author_full[3:].split("—")[0].strip()
                else:
                    author = author_full.split("—")[0].strip()
            except:
                author = "Autor não encontrado"

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

        if news_data:
            csv_filename = f'../CSVs/singleCSVs/news_data_{team_name}.csv'
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
