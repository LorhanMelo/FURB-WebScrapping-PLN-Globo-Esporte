import spacy
import pandas as pd
import os

# Carregar o modelo de português do spaCy
nlp = spacy.load('pt_core_news_sm')

class Cleaner:
    def __init__(self, directory):
        self.directory = directory

    def clean_text(self, text):
        # Transformar texto em String
        if not isinstance(text, str):
            text = str(text)

        text = text.lower()

        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)

    def process_csv(self, file_path):
        df = pd.read_csv(file_path)

        if 'article_content' not in df.columns:
            print(f"Coluna 'article_content' não encontrada em {file_path}. Pulando arquivo.")
            return

        # Double check para o conteudo ser uma String
        df['article_content'] = df['article_content'].astype(str)

        # Cria coluna com o conteúdo limpo
        df['cleaned_article_content'] = df['article_content'].apply(self.clean_text)

        # Sobrescreve o arquivo CSV com os resultados
        df.to_csv(file_path, index=False)
        print(f"Arquivo processado e salvo: {file_path}")

    def process_all_csvs(self):
        # Passar por todos os CSVs
        csv_files = [f for f in os.listdir(self.directory) if f.endswith('.csv')]

        # Verifica se existem arquivos CSV no diretório
        if not csv_files:
            print(f"Nenhum arquivo CSV encontrado no diretório {self.directory}.")
            return

        # Processa todos os arquivos CSV encontrados
        for file_name in csv_files:
            file_path = os.path.join(self.directory, file_name)
            self.process_csv(file_path)

if __name__ == "__main__":
    directory = "C:/Users/lomelo/AquaProjects/WebScrappingPLNGloboEsporte/CSVs"  # Substitua pelo seu caminho
    cleaner = Cleaner(directory)
    cleaner.process_all_csvs()