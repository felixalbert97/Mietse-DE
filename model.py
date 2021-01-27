import pandas as pd
import tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

df = pd.read_csv('data_imputed_1.csv', index_col=0)


###################
# Feature Auswahl #
###################

# alle relevanten Spalten
cols = ['baseRent', 'livingSpace', 'yearConstructedRange_new', 'interiorQual_new']
# Davon kategorische features:
features_categorical = ['yearConstructedRange_new', 'interiorQual_new']
# Davon numerische features:
features_numerical = ['livingSpace']

# Reduziere Datensatz auf relevante Spalten
df = df[cols]

# Erstelle Dummievariablen für kategorische Features
df = tools.generate_dummies(df,features_categorical,trap=True)


#################
# Preprocessing #
#################

# Split training- und test-set
X = df.loc[:, ~df.columns.isin(['baseRent'])]
y = df.loc[:,['baseRent']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)

# Skaliere Zielvariable (füge y_train und y_test eine skalierte Spalte hinzu)
sc_baseRent, y_train_sc, y_test_sc = tools.scale_targets(y_train, y_test)

# Skaliere Features
sc_features, X_train[features_numerical], X_test[features_numerical] = tools.scale_features(X_train[features_numerical], X_test[features_numerical])


########################
# Modelle im Vergleich #
########################

# Lineare Regression
X_train_lg, X_test_lg, y_train_lg, y_test_lg = X_train.copy(), X_test.copy(), y_train_sc.copy(), y_test_sc.copy()
lg = LinearRegression()
lg.fit(X_train_lg,y_train_lg[['target_sc']])
y_test_lg['predict_sc'] = lg.predict(X_test_lg)
y_test_lg['prediction'] = sc_baseRent.inverse_transform(y_test_lg['predict_sc'])

# Modellgüte berechnen durch RMSE
score_lg = mse(y_test_lg['baseRent'], y_test_lg['prediction'], squared=False)
# Modellgüte berechnen durch R^2 Wert
R2_lg = lg.score(X_test_lg, y_test_lg['target_sc'])

# DecisionTree 
X_train_rfr, X_test_rfr, y_train_rfr, y_test_rfr = X_train.copy(), X_test.copy(), y_train_sc.copy(), y_test_sc.copy()
rfr = RandomForestRegressor(max_depth=15, random_state=1)
rfr.fit(X_train_rfr,y_train_rfr[['target_sc']])
y_test_rfr['predict_sc'] = rfr.predict(X_test_rfr)
y_test_rfr['prediction'] = sc_baseRent.inverse_transform(y_test_rfr['predict_sc'])

# Modellgüte berechnen durch RMSE
score_rfr = mse(y_test_rfr['baseRent'], y_test_rfr['prediction'], squared=False)


# AdaBoost (Beispiel ohne Skalierung)
X_train_ada, X_test_ada, y_train_ada, y_test_ada = X_train.copy(), X_test.copy(), y_train_sc.copy(), y_test_sc.copy()
ada = AdaBoostRegressor(learning_rate=0.5, random_state=1)
ada.fit(X_train_ada,y_train_ada['baseRent'])
y_test_ada['prediction'] = ada.predict(X_test_ada)

# Modellgüte berechnen durch RMSE
score_ada = mse(y_test_ada['baseRent'], y_test_ada['prediction'], squared=False)
