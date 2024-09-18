# Scraping de Notícias do GE Globo

Este projeto é uma ferramenta de web scraping que coleta notícias sobre times de futebol Brasileiros do site GE Globo. Utiliza a biblioteca Playwright para acessar e extrair dados de notícias sobre times específicos.

## Requisitos

Para usar este projeto, você precisa ter o seguinte instalado:

- [Python 3.7+](https://www.python.org/downloads/)
- [Playwright](https://playwright.dev/python/docs/intro)

## Instalação

1. **Clone o repositório:**

   ```bash
   git clone <URL_DO_SEU_REPOSITORIO>
   cd <NOME_DA_PASTA_CLONADA>
   ```
## Instale as dependências:

   2. **Depois, você precisa instalar o Playwright e suas dependências:**

   ```bash
    pip install playwright
    playwright install
   ```

## Uso
**Para rodar o script e coletar notícias para um time específico, siga estas etapas:**

Execute o script com o nome do time como argumento em letra minuscula:

```bash
python scraper.py <nome_do_time>
```
1. Substitua <nome_do_time> pelo nome do time de futebol que você deseja coletar notícias. Por exemplo, para coletar notícias sobre o time "vasco", você usaria:

```bash
python sraper.py vasco
```
**Para times com espaço: Atlético Mineiro**

**Use abreviações: atletico-mg** 


## Verifique a pasta CSVs:

2. Após a execução do script, um arquivo CSV com as notícias será gerado na pasta CSVs. O arquivo será nomeado news_data_<nome_do_time>.csv, onde <nome_do_time> é o nome do time fornecido.

# Limpeza do Scrapping coletado

Este projeto inclui um script para processar arquivos CSV contendo dados textuais que foram obtidos via web scrapping, removendo stopwords, pontuações, e realizando lematização usando o spaCy. O objetivo é limpar e preparar o texto para análise posterior.

## Requisitos

Certifique-se de ter os seguintes pacotes instalados:

- `spacy`
- `pandas`
- `os`

Você pode instalar os pacotes necessários usando `pip`:

```bash
pip install spacy pandas
```
Além disso, é necessário baixar o modelo de linguagem em português do spaCy:
   ```bash
python -m spacy download pt_core_news_sm
   ```
## Estrutura Passo-a-Passo
O projeto contém um script Python chamado cleaner.py que realiza as seguintes operações:

- `Carrega o modelo de linguagem em português do spaCy.`
- `Processa arquivos CSV em um diretório especificado.`
- `Limpa o texto em cada arquivo CSV, removendo stopwords e pontuações e realizando lematização.`
- `Salva os resultados limpos em novas colunas dentro dos arquivos CSV originais.`

## Como usar
### 1. Prepare o Diretório:

Gere seus arquivos CSV com o scrapper.

### 2. Configure o Diretório no Script:

No script cleaner.py, defina o caminho para o diretório onde seus arquivos CSV estão localizados, substituindo o valor de directory no código:

```bash
directory = "C:/Users/lomelo/AquaProjects/WebScrappingPLNGloboEsporte/CSVs"  # Substitua pelo seu caminho
```

### 3. Execute o Script.
```bash
python cleaner.py
```

