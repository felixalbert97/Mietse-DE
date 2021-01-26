import pandas as pd
import tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

df = pd.read_csv('data_imputed_1.csv', index_col=0)

# Feature Auswahl + target
features = ['baseRent', 'livingSpace', 'noRooms', 'yearConstructedRange_new']
# Davon kategorisch:
features_categorical = ['yearConstructedRange_new']

df = df[features]

# Erstelle Dummievariablen für kategorische Features
df = tools.generate_dummies(df,features_categorical,trap=True)


# Split training- und test-set
X = df.loc[:, ~df.columns.isin(['baseRent'])]
y = df.loc[:,['baseRent']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 30, random_state=1)

########################
# Modelle im Vergleich #
########################

# Lineare Regression
lg = LinearRegression()
lg.fit(X_train,y_train)
y_predict_lg = lg.predict(X_test)

# Modellgüte berechnen durch RMSE
score_lg = mse(y_test, y_predict_lg, squared=False)

# DecisionTree 
rfr = RandomForestRegressor(max_depth=15, random_state=1)
rfr.fit(X_train,y_train)
y_predict_rfr = rfr.predict(X_test)

# Modellgüte berechnen durch RMSE
score_rfr = mse(y_test, y_predict_rfr, squared=False)

# AdaBoost 
ada = AdaBoostRegressor(learning_rate=1, random_state=1)
ada.fit(X_train,y_train)
y_predict_ada = ada.predict(X_test)

# Modellgüte berechnen durch RMSE
score_ada = mse(y_test, y_predict_ada, squared=False)
