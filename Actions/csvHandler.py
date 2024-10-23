import os
import pandas as pd
import shutil

csv_folder = r'C:\Users\lomelo\AquaProjects\WebScrappingPLNGloboEsporte\CSVs\singleCSVs' # Pasta geral
single_csv_folder = os.path.join(csv_folder, r'C:\Users\lomelo\AquaProjects\WebScrappingPLNGloboEsporte\CSVs\singleCSVs')  # Pasta para CSVs individuais
combined_csv_folder = os.path.join(csv_folder, r'C:\Users\lomelo\AquaProjects\WebScrappingPLNGloboEsporte\CSVs\combinedCSV')  # Pasta para o CSV combinado
os.makedirs(single_csv_folder, exist_ok=True)
os.makedirs(combined_csv_folder, exist_ok=True)

df_list = []


for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        if filename != 'combined_news_data.csv':
            file_path = os.path.join(csv_folder, filename)
            df = pd.read_csv(file_path)

            team_name = filename.split('_')[2].replace('.csv', '')

            df['team'] = team_name

            df_list.append(df)

            shutil.move(file_path, os.path.join(single_csv_folder, filename))

combined_df = pd.concat(df_list, ignore_index=True)


combined_csv_path = os.path.join(combined_csv_folder, 'combined_news_data.csv')
combined_df.to_csv(combined_csv_path, index=False)

print(f"Todos os CSVs foram combinados com sucesso e salvos em {combined_csv_path}!")
