import pandas as pd
import tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

def preprocessing_pipepline(df=None, target=None, features_categorical=None, features_numerical=None, test_set_size=0.3, random_state=1, dummy_trap=True):

    # Selektiere alle relevanten Spalten und reduziere Datensatz 
    cols = [target] + features_categorical + features_numerical
    df = df[cols]

    # Erstelle Dummievariablen für kategorische Features
    df = tools.generate_dummies(df,features_categorical,trap=dummy_trap)

    # Erstelle Training- und Test-set
    X = df.loc[:, ~df.columns.isin([target])]
    y = df.loc[:,[target]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_set_size, random_state=random_state)

    # Skaliere Features
    feature_scaler, X_train[features_numerical], X_test[features_numerical] = tools.scale_features(X_train[features_numerical], X_test[features_numerical])

    # Skaliere Zielvariable (füge y_train und y_test eine skalierte Spalte hinzu)
    target_scaler, y_train, y_test = tools.scale_targets(y_train, y_test)

    return X_train, X_test, y_train, y_test, target_scaler


def model_scores(model = None, X_train=None, X_test=None, y_train=None, y_test=None, target_scaler=None, scale_target=True):

    # Erstelle Kopien der Daten, damit sie nicht verändert werden
    X_train_copy, X_test_copy, y_train_copy, y_test_copy = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()

    # Berechne Vorhersagen für skalierte oder unskalierte Zielvariable
    if scale_target:
        model.fit(X_train_copy,y_train_copy[['target_sc']])
        y_test_copy['predict_sc'] = model.predict(X_test_copy)
        y_test_copy['prediction'] = target_scaler.inverse_transform(y_test_copy['predict_sc'])
    else:
        model.fit(X_train_copy,y_train_copy[target])
        y_test_copy['prediction'] = model.predict(X_test_copy)

    # Modellgüte berechnen 
    mae_score = mae(y_test_copy[target], y_test_copy['prediction'])
    mse_score = mse(y_test_copy[target], y_test_copy['prediction'], squared=False)
    R2_score = model.score(X_test_copy, y_test_copy['target_sc'])

    results = [mae_score, mse_score, R2_score]
    return results


if __name__ == "__main__":

    ###############
    # Import Data #
    ###############
    df = pd.read_csv('data_imputed_1.csv', index_col=0)

    ###################
    # Feature Auswahl #
    ###################

    # Zielvariable
    target = 'baseRent'
    # Kategorische features:
    features_categorical = ['yearConstructedRange_new','noRooms', 'interiorQual_new', 'condition_new']
    # Numerische features:
    features_numerical = ['livingSpace']

    #################
    # Preprocessing #
    #################

    X_train, X_test, y_train, y_test, target_scaler = preprocessing_pipepline(df=df, target=target, features_categorical=features_categorical, features_numerical=features_numerical, test_set_size=0.3, random_state=1)

    ################
    # Modellscores #
    ################

    #Lineare Regression
    lg = LinearRegression()
    results_lg = model_scores(model=lg, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, target_scaler=target_scaler)

    #RandomForest
    rf = RandomForestRegressor(max_depth=30, random_state=1)
    results_rf = model_scores(model=rf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, target_scaler=target_scaler)

    #AdaBoost
    ada = AdaBoostRegressor(learning_rate=0.3, n_estimators=200, random_state=1)
    results_ada = model_scores(model=ada, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, target_scaler=target_scaler)

