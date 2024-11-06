import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Função para carregar o CSV e aplicar Word2Vec
def apply_word2vec_and_get_data(csv_path, text_column):
    data = pd.read_csv(csv_path)

    if text_column not in data.columns:
        raise ValueError(f"Coluna '{text_column}' não encontrada no CSV.")

    texts = data[text_column].fillna('').str.split()

    word2vec_model = Word2Vec(sentences=texts, vector_size=400, window=40, min_count=4, workers=40)

    document_vectors = []
    for tokens in texts:
        valid_tokens = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]

        if valid_tokens:
            vector = np.mean(valid_tokens, axis=0)
            document_vectors.append(vector)
        else:
            document_vectors.append(np.zeros(word2vec_model.vector_size))

    document_vectors = np.array(document_vectors)
    return data, document_vectors, word2vec_model

# Função para calcular a similaridade entre os documentos
def calculate_similarity(doc_vectors):
    similarities = cosine_similarity(doc_vectors)
    return similarities

# Função para clusterizar os documentos
def cluster_documents(data, doc_vectors, team_column, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, max_iter=300, random_state=42)
    data['Cluster'] = kmeans.fit_predict(doc_vectors)

    cluster_labels = {}
    for cluster in range(num_clusters):
        cluster_indices = np.where(data['Cluster'] == cluster)[0]  # Pegue os índices dos documentos no cluster
        most_common_team = data.iloc[cluster_indices][team_column].mode()[0]  # Encontre o time mais comum
        cluster_labels[cluster] = most_common_team

    data['Cluster_Label'] = data['Cluster'].map(cluster_labels)

    return data, cluster_labels

# Função para plotar a visualização dos clusters com TSNE (diferenciar do PCA)
def plot_tsne_clusters(doc_vectors, labels):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_vectors = tsne.fit_transform(doc_vectors)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=labels, palette="viridis")
    plt.title('Visualização de Clusters com TSNE')
    plt.xlabel('Dimensão 1')
    plt.ylabel('Dimensão 2')
    plt.show()

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

data, document_vectors, word2vec_model = apply_word2vec_and_get_data(csv_path, text_column)
similarity_matrix = calculate_similarity(document_vectors)
plot_similarity_matrix(similarity_matrix)
pca = PCA(n_components=50)  # Reduzindo para 50 dimensões
reduced_vectors = pca.fit_transform(document_vectors)
data, cluster_labels = cluster_documents(data, reduced_vectors, team_column, num_clusters)
plot_tsne_clusters(reduced_vectors, data['Cluster_Label'])
plot_confusion_matrix(data, team_column)
accuracy = accuracy_score(data[team_column], data['Cluster_Label'])
print(f"\nAcurácia com {num_clusters} clusters: {accuracy:.2f}")

