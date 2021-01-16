import pandas as pd

# Zeige immer alle Spalten eines DataFrames an
#pd.set_option('display.max_columns', None)

df = pd.read_csv('immo_data.csv')

# Droppe Spalten mit unbrauchbarer Info / zu viele NaN-values:

# description, facilities (Fließtext)
# street (komisches Format, dafür streetPlain)
# serviceCharge, electricityBasePrice, electricityKwhPrice, heatingCosts (nur für Warmmiete relevant)
# energyEfficiencyClass, lastRefurbish, telekomHybridUploadSpeed (zu viele fehlende Werte)

columns_drop = ['description', 'facilities', 'street', 'serviceCharge', 'electricityBasePrice', 'electricityKwhPrice','heatingCosts', 'energyEfficiencyClass', 'lastRefurbish', 'telekomHybridUploadSpeed']

df_clean = df.drop(columns_drop, axis=1)

df_clean.to_csv('immo_data_clean.csv')

