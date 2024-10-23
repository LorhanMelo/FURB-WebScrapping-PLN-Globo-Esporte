import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np

# Função para aplicar TF-IDF e obter dados
def apply_tfidf_and_get_data(csv_path, text_column):
    data = pd.read_csv(csv_path)

    if text_column not in data.columns:
        raise ValueError(f"Coluna '{text_column}' não encontrada no CSV.")

    texts = data[text_column].fillna('')

    # Adicionando n-gramas (unigrams, bigrams e trigrams)
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
    X_tfidf = vectorizer.fit_transform(texts)

    feature_names = vectorizer.get_feature_names_out()

    return data, X_tfidf, vectorizer, feature_names

# Função para plotar os N-Gramas mais frequentes (excluindo unigrams)
def plot_most_frequent_ngrams(tfidf_matrix, feature_names, top_n=20):
    tfidf_sums = tfidf_matrix.sum(axis=0).A1

    ngram_frequencies = pd.DataFrame({'ngram': feature_names, 'tfidf': tfidf_sums})

    ngram_frequencies = ngram_frequencies[ngram_frequencies['ngram'].str.contains(' ')]

    top_ngrams = ngram_frequencies.nlargest(top_n, 'tfidf')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='tfidf', y='ngram', data=top_ngrams)
    plt.title(f'Top {top_n} Bigrams e Trigrams mais Frequentes')
    plt.xlabel('Frequência TF-IDF')
    plt.ylabel('N-Gramas (Bigrams e Trigrams)')
    plt.show()

# Função para calcular a similaridade entre os documentos
def calculate_similarity(tfidf_matrix):
    similarities = cosine_similarity(tfidf_matrix)
    return similarities

# Função para plotar a Matriz de Similaridade entre os Documentos
def plot_similarity_matrix(similarity_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='coolwarm', square=True)
    plt.title('Matriz de Similaridade entre os Documentos')
    plt.xlabel('Documentos')
    plt.ylabel('Documentos')
    plt.show()

# Função para clusterizar as notícias e rotular clusters com a coluna 'team'
def cluster_news_and_label(data, tfidf_matrix, team_column, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)

    data['Cluster'] = kmeans.labels_

    cluster_labels = {}
    for cluster in range(num_clusters):
        most_common_team = data[data['Cluster'] == cluster][team_column].mode()[0]
        cluster_labels[cluster] = most_common_team

    data['Cluster_Label'] = data['Cluster'].map(cluster_labels)

    return data, cluster_labels

# Função para agrupar clusters em intervalos
def group_clusters(data, num_clusters, interval=20):  # Alterado para 20
    max_cluster = num_clusters // interval + 1
    data['Grouped_Cluster'] = (data['Cluster'] // interval).clip(0, max_cluster - 1)  # Agrupando por intervalo
    return data

# Função para aplicar PCA e visualizar os clusters
def plot_pca_grouped_clusters(tfidf_matrix, labels):
    pca = PCA(n_components=20)  # Reduzir para 20 componentes
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette="viridis", legend='full')
    plt.title('Visualização dos Clusters Agrupados de 20 em 20 com PCA')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()

# Função para verificar a acurácia dos clusters
def check_cluster_accuracy(data, team_column):
    confusion = pd.crosstab(data['Cluster_Label'], data[team_column], rownames=['Cluster'], colnames=['Team'])
    return confusion

# Função para plotar a matriz de confusão
def plot_confusion_matrix(confusion):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão')
    plt.xlabel('Time Real')
    plt.ylabel('Cluster Predito')
    plt.show()

# Função para plotar a distribuição de notícias por time
def plot_news_distribution(data, team_column):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=team_column, order=data[team_column].value_counts().index)
    plt.title('Distribuição de Notícias por Time')
    plt.xticks(rotation=90)
    plt.xlabel('Time')
    plt.ylabel('Número de Notícias')
    plt.show()

# Função para plotar a distribuição de pesos TF-IDF por documento
def plot_tfidf_weight_distribution(tfidf_matrix):
    tfidf_weights = tfidf_matrix.mean(axis=1).A1
    plt.figure(figsize=(10, 6))
    sns.histplot(tfidf_weights, bins=30, kde=False)
    plt.title('Distribuição dos Pesos TF-IDF por Documento')
    plt.xlabel('Peso Médio TF-IDF')
    plt.ylabel('Frequência')
    plt.show()

# Caminho para o CSV
csv_path = r'C:\Users\lomelo\AquaProjects\WebScrappingPLNGloboEsporte\CSVs\combinedCSV\combined_news_data.csv'
text_column = 'cleaned_article_content'
team_column = 'team'

# Definir o número de clusters desejado
num_clusters = 400

data, tfidf_matrix, vectorizer, feature_names = apply_tfidf_and_get_data(csv_path, text_column)
plot_most_frequent_ngrams(tfidf_matrix, feature_names, top_n=20)
similarity_matrix = calculate_similarity(tfidf_matrix)
plot_similarity_matrix(similarity_matrix)
data, cluster_labels = cluster_news_and_label(data, tfidf_matrix, team_column, num_clusters)
data = group_clusters(data, num_clusters, interval=20)
confusion = check_cluster_accuracy(data, team_column)
accuracy = accuracy_score(data[team_column], data['Cluster_Label'])
print(f"\nAcurácia com {num_clusters} clusters: {accuracy:.2f}")
plot_confusion_matrix(confusion)
plot_news_distribution(data, team_column)
plot_tfidf_weight_distribution(tfidf_matrix)
plot_pca_grouped_clusters(tfidf_matrix, data['Grouped_Cluster'])
