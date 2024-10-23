import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score

# Função para aplicar Bag of Words e obter dados
def apply_bow_and_get_data(csv_path, text_column):
    data = pd.read_csv(csv_path)

    if text_column not in data.columns:
        raise ValueError(f"Coluna '{text_column}' não encontrada no CSV.")

    texts = data[text_column].fillna('')

    vectorizer = CountVectorizer()
    X_bow = vectorizer.fit_transform(texts)

    return data, X_bow, vectorizer

# Função para exibir termos mais frequentes
def plot_most_frequent_terms(vectorizer, bow_matrix, top_n=20):
    term_frequencies = bow_matrix.toarray().sum(axis=0)
    terms = vectorizer.get_feature_names_out()
    term_freq_df = pd.DataFrame({'term': terms, 'frequency': term_frequencies})
    term_freq_df = term_freq_df.sort_values(by='frequency', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='frequency', y='term', data=term_freq_df)
    plt.title(f'Termos Mais Frequentes (Top {top_n})')
    plt.xlabel('Frequência')
    plt.ylabel('Termo')
    plt.show()

# Função para clusterizar as notícias e rotular clusters com a coluna 'team'
def cluster_news_and_label(data, bow_matrix, team_column, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(bow_matrix)

    data['Cluster'] = kmeans.labels_

    cluster_labels = {}
    for cluster in range(num_clusters):
        most_common_team = data[data['Cluster'] == cluster][team_column].mode()[0]
        cluster_labels[cluster] = most_common_team

    data['Cluster_Label'] = data['Cluster'].map(cluster_labels)

    return data, cluster_labels

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

# Função para plotar o número de palavras únicas
def plot_unique_word_distribution(vectorizer, bow_matrix):
    unique_word_counts = bow_matrix.toarray().sum(axis=1)

    plt.figure(figsize=(10, 6))
    sns.histplot(unique_word_counts, bins=30, kde=False)
    plt.title('Distribuição do Número de Palavras Únicas nas Notícias')
    plt.xlabel('Número de Palavras Únicas')
    plt.ylabel('Frequência')
    plt.show()

# Caminho para o CSV
csv_path = r'C:\Users\lomelo\AquaProjects\WebScrappingPLNGloboEsporte\CSVs\combinedCSV\combined_news_data.csv'
text_column = 'cleaned_article_content'
team_column = 'team'

# Definir o número de clusters desejado
num_clusters = 400

data, bow_matrix, vectorizer = apply_bow_and_get_data(csv_path, text_column)
plot_most_frequent_terms(vectorizer, bow_matrix)
data, cluster_labels = cluster_news_and_label(data, bow_matrix, team_column, num_clusters)
confusion = check_cluster_accuracy(data, team_column)
accuracy = accuracy_score(data[team_column], data['Cluster_Label'])
print(f"\nAcurácia com {num_clusters} clusters: {accuracy:.2f}")
plot_confusion_matrix(confusion)
plot_news_distribution(data, team_column)
plot_unique_word_distribution(vectorizer, bow_matrix)
