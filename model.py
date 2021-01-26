import pandas as pd
import utils

df = pd.read_csv('data_imputed_1.csv', index_col=0)

# Feature Auswahl 
features = ['livingSpace', 'noRooms', 'yearConstructedRange_new']
# Davon kategorisch:
features_categorical = ['yearConstructedRange_new']

df = df[features]

# Erstelle Dummievariablen 
df = utils.generate_dummies(df,features_categorical,trap=True)


# Split training- und test-set
X = df.loc[:, ~df.columns.isin(['baseRent'])]

