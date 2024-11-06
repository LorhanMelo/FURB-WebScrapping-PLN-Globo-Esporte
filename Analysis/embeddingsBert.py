import os
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import time

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando o dispositivo: {device}")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

# Função para processar os textos em batches e obter embeddings BERT
def get_bert_embeddings(texts, batch_size=4):
    embeddings = []
    print("Iniciando extração de embeddings BERT...")
    start_time = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)

        if i % (batch_size * 10) == 0:
            print(f"Processados {i + batch_size} textos em {time.time() - start_time:.2f}s")

    print(f"Extração de embeddings concluída em {time.time() - start_time:.2f}s")
    return np.vstack(embeddings)

# Função para carregar o CSV e obter embeddings BERT
def load_data_and_get_embeddings(csv_path, text_column):
    data = pd.read_csv(csv_path)

    if text_column not in data.columns:
        raise ValueError(f"Coluna '{text_column}' não encontrada no CSV.")

    texts = data[text_column].fillna('').tolist()
    print("Carregando dados e obtendo embeddings BERT...")
    document_vectors = get_bert_embeddings(texts)
    print("Dados carregados e embeddings obtidos.")

    return data, document_vectors

# Função para calcular a similaridade entre os documentos
def calculate_similarity(doc_vectors):
    print("Calculando matriz de similaridade...")
    similarities = cosine_similarity(doc_vectors)
    print("Matriz de similaridade calculada.")
    return similarities

# Função para clusterizar os documentos
def cluster_documents(data, doc_vectors, team_column, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, max_iter=300, random_state=42)
    data['Cluster'] = kmeans.fit_predict(doc_vectors)

    cluster_labels = {}

    for cluster in range(num_clusters):
        cluster_indices = np.where(data['Cluster'] == cluster)[0]

        if len(cluster_indices) > 0:
            most_common_team = data.iloc[cluster_indices][team_column].mode()[0]
            cluster_labels[cluster] = most_common_team
        else:
            cluster_labels[cluster] = "Cluster Vazio"

    data['Cluster_Label'] = data['Cluster'].map(cluster_labels)

    return data, cluster_labels


# Função para plotar a visualização dos clusters com PCA
def plot_pca_clusters(doc_vectors, labels):
    print("Reduzindo dimensionalidade para visualização com PCA...")
    start_time = time.time()

    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(doc_vectors)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=labels, palette="viridis")
    plt.title('Visualização de Clusters com PCA')
    plt.xlabel('Dimensão 1')
    plt.ylabel('Dimensão 2')
    plt.show()

    print(f"Visualização com PCA concluída em {time.time() - start_time:.2f}s")

# Função para plotar a matriz de similaridade
def plot_similarity_matrix(similarity_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='coolwarm', square=True)
    plt.title('Matriz de Similaridade entre os Documentos')
    plt.xlabel('Documentos')
    plt.ylabel('Documentos')
    plt.show()

# Função para verificar a acurácia do cluster e plotar a matriz de confusão
def plot_confusion_matrix(data, team_column):
    confusion = pd.crosstab(data['Cluster_Label'], data[team_column], rownames=['Cluster'], colnames=['Time'])

    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão entre Clusters e Times')
    plt.ylabel('Cluster')
    plt.xlabel('Time')
    plt.show()

# Caminho para o CSV
csv_path = r'C:\Users\lomelo\AquaProjects\WebScrappingPLNGloboEsporte\CSVs\combinedCSV\combined_news_data.csv'
text_column = 'cleaned_article_content'
team_column = 'team'

# Definir o número de clusters
num_clusters = 400

data, document_vectors = load_data_and_get_embeddings(csv_path, text_column)
similarity_matrix = calculate_similarity(document_vectors)
plot_similarity_matrix(similarity_matrix)
data, cluster_labels = cluster_documents(data, document_vectors, team_column, num_clusters)
plot_pca_clusters(document_vectors, data['Cluster_Label'])
plot_confusion_matrix(data, team_column)
accuracy = accuracy_score(data[team_column], data['Cluster_Label'])
print(f"\nAcurácia com {num_clusters} clusters: {accuracy:.2f}")
