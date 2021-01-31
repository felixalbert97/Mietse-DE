import pandas as pd
import seaborn as sns
import numpy as np
import tools
import math as m

# Zeige immer alle Spalten eines DataFrames an
pd.set_option('display.max_columns', None)

df = pd.read_csv('data_imputed_1.csv', index_col=0)

df['firingTypes'].value_counts()

sns.violinplot(x='firingTypes', y='baseRent', data=df.loc[df.firingTypes.isin(['gas', 'district_heating','oil'])])