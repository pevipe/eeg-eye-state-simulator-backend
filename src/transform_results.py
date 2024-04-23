import os

import pandas as pd

folder_path = '../out/results/individual/8s/pure_windows'

dataframe = pd.DataFrame()

# for i, f in enumerate(os.listdir(folder_path)):
#     if f.endswith('.csv'):
#         df = pd.read_csv(folder_path + '/' + f)[['classifier', 'mean_accuracy', 'std_accuracy']]
#         df.insert(0, 'subject', i + 1)
#         dataframe = pd.concat([dataframe, df], ignore_index=True)

for i, f in enumerate(os.listdir(folder_path)):
    if f.endswith('.csv'):
        df = pd.read_csv(folder_path + '/' + f)[
            ['classifier', 'mean_precision_0', 'mean_f1_0', 'mean_precision_1', 'mean_f1_1', 'accuracy']]
        df.insert(0, 'subject', i + 1)
        dataframe = pd.concat([dataframe, df], ignore_index=True)

# Pivotar
df_pivot = dataframe.pivot(index='subject', columns='classifier')

# Aplanar las columnas multi-index
df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

# Restablecer el índice
df_pivot.reset_index(inplace=True)

# Exportar a CSV
df_pivot.to_csv('../out/results/pure_windows/full_results_hyperopt_size_8/full_results_hyperopt_size_8.csv',
                index=False)
