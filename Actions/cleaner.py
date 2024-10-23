import spacy
import pandas as pd
import os

nlp = spacy.load('pt_core_news_sm')

class Cleaner:
    def __init__(self, directory):
        self.directory = directory

    #Limpeza dos dados com Lematização e remoção de stopwords e pontuação
    def clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)

        doc = nlp(text)

        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)

    def process_csv(self, file_path):
        df = pd.read_csv(file_path)

        if 'article_content' not in df.columns:
            print(f"Coluna 'article_content' não encontrada em {file_path}. Pulando arquivo.")
            return

        df['article_content'] = df['article_content'].astype(str)

        print(f"Linhas antes da limpeza: {df.shape[0]}")

        df['cleaned_article_content'] = df['article_content'].apply(self.clean_text)

        print(f"Linhas após a limpeza: {df['cleaned_article_content'].notnull().sum()}")

        df.to_csv(file_path, index=False)
        print(f"Arquivo processado e salvo: {file_path}")

    def process_all_csvs(self):
        csv_files = [f for f in os.listdir(self.directory) if f.endswith('.csv')]

        if not csv_files:
            print(f"Nenhum arquivo CSV encontrado no diretório {self.directory}.")
            return

        for file_name in csv_files:
            file_path = os.path.join(self.directory, file_name)
            self.process_csv(file_path)

if __name__ == "__main__":
    # Substitua pelo caminho do seu diretório com os CSVs
    directory = "C:/Users/lomelo/AquaProjects/WebScrappingPLNGloboEsporte/CSVs/combinedCSV"
    cleaner = Cleaner(directory)
    cleaner.process_all_csvs()
