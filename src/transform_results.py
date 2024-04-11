import os

import pandas as pd

folder_path = '../out/results/full_results_hyperopt_10'

dataframe = pd.DataFrame()

for i, f in enumerate(os.listdir(folder_path)):
    if f.endswith('.csv'):
        df = pd.read_csv(folder_path+'/'+f)[['classifier', 'mean_accuracy', 'std_accuracy']]
        df.insert(0, 'subject', i+1)
        dataframe = pd.concat([dataframe, df], ignore_index=True)

# Pivotar
df_pivot = dataframe.pivot(index='subject', columns='classifier')

# Aplanar las columnas multi-index
df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

# Restablecer el índice
df_pivot.reset_index(inplace=True)

# Exportar a CSV
df_pivot.to_csv('../out/results/full_results_hyperopt_10.csv', index=False)
