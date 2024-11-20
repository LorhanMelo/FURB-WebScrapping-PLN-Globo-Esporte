import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

nlp = spacy.load('pt_core_news_lg')
csv_path = r'C:\Users\lomelo\AquaProjects\WebScrappingPLNGloboEsporte\CSVs\combinedCSV\combined_news_data.csv'
data = pd.read_csv(csv_path)
text_column = 'cleaned_article_content'

if text_column not in data.columns:
    raise ValueError(f"Coluna '{text_column}' não encontrada no CSV.")

texts = data[text_column].fillna('').tolist()

# Processar textos com spaCy e extrair entidades
print("Rodando NER na base completa...")
entities = {
    "PERSON": [],
    "ORG": [],
    "LOC": [],
    "EVENT": [],
    "DATE": [],
    "TIME": [],
    "MONEY": [],
    "PERCENT": [],
    "QUANTITY": [],
    "ORDINAL": [],
    "CARDINAL": [],
    "GPE": [],
    "WORK_OF_ART": [],
    "LAW": [],
    "LANGUAGE": [],
    "PRODUCT": [],
    "NORP": [],
    "FAC": []
}

for doc in nlp.pipe(texts, batch_size=10):
    for ent in doc.ents:
        if ent.label_ in entities:
            if len(ent.text.split()) > 1 or ent.label_ != "LOC" or not ent.text.isnumeric():
                entities[ent.label_].append(ent.text)

# Recalcular as frequências, garantindo que não sejam vazias
entity_counts = {label: Counter(entities[label]).most_common(10) if entities[label] else [] for label in entities}

# Exibir resultados
print("\nFrequência de entidades nomeadas (filtradas):")
for label, counts in entity_counts.items():
    print(f"\n{label}:")
    if not counts:
        print("  Nenhuma entidade encontrada.")
    for entity, count in counts:
        print(f"  {entity}: {count}")

# Visualizar resultados com gráficos
for label, counts in entity_counts.items():
    if counts:
        labels, values = zip(*counts)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(values), y=list(labels), palette="viridis")
        plt.title(f"Top 10 {label} mencionadas")
        plt.xlabel("Frequência")
        plt.ylabel("Entidade")
        plt.show()
    else:
        print(f"\nNenhuma entidade encontrada para {label}.")
