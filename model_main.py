# def feature_selection_pipeline(df=None, target=None, features_categorical=None, features_numerical=None):
#     cols = [target] + features_numerical
#     df = df[cols]
#     X = df.loc[:, ~df.columns.isin([target])]
#     y = df.loc[:,[target]]
#     # Feature Selection mit mutual_info_regression für  
#     mutual_info_regression(X,y)


def preprocessing_pipepline(df=None, df_new=None, target=None, features_categorical=None, features_numerical=None, exponents=[], test_set_size=0.3, random_state=1, dummy_trap=True):

    # Selektiere alle relevanten Spalten und reduziere Datensatz 
    cols = [target] + features_categorical + features_numerical
    df = df[cols]

    # Erweitere Datensatz um neue Daten
    df = df.append(df_new, ignore_index=True)

    # Erstelle Dummievariablen für kategorische Features
    df = tools.generate_dummies(df,features_categorical,trap=dummy_trap)

    # Erstelle polynomielle Spalten für numerische features (Potenzen aus exponents)
    df = tools.generate_polynomials(df,features_numerical,exponents=exponents)

    # Splitte target und features
    X = df.iloc[:-df_new.shape[0]].loc[:, ~df.columns.isin([target])].copy()
    y = df.iloc[:-df_new.shape[0]].loc[:,[target]].copy()
    
    X_new = df.iloc[-df_new.shape[0]:].loc[:, ~df.columns.isin([target])].copy()
    y_new = df.iloc[-df_new.shape[0]:].loc[:,[target]].copy()

    # Erstelle Training- und Test-set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_set_size, random_state=random_state)

    # Skaliere Features
    feature_scaler, X_train[features_numerical], X_test[features_numerical] = tools.scale_features(X_train[features_numerical], X_test[features_numerical])

    # Skaliere Zielvariable (füge y_train und y_test eine skalierte Spalte hinzu)
    target_scaler, y_train, y_test = tools.scale_targets(y_train, y_test)

    # Skaliere eigene Daten 
    X_new_sc_values = feature_scaler.transform(X_new[features_numerical].values)
    X_new[features_numerical] = pd.DataFrame(X_new_sc_values, index=X_new.index, columns=X_new[features_numerical].columns)

    y_new['target_sc'] = target_scaler.transform(y_new.values)

    return df, X_train, X_test, X_new, y_train, y_test, y_new, target_scaler

def model_scores(model=None, X_train=None, X_test=None, y_train=None, y_test=None, target_scaler=None, scale_target=True):

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
    mape_score = np.mean(np.abs((y_test_copy[target] - y_test_copy['prediction']) / y_test_copy[target])) *100
    mae_score = mae(y_test_copy[target], y_test_copy['prediction'])
    mse_score = mse(y_test_copy[target], y_test_copy['prediction'], squared=False)
    R2_score = model.score(X_test_copy, y_test_copy['target_sc'])

    return model, mape_score, mae_score, mse_score, R2_score

def predict_new(model=None, X_new=None, y_new=None, target_scaler=None, scaled_target=True):

    # Vorhersage für eigene Daten
    y_new_copy = y_new.copy()
    if scaled_target:
        y_new_copy['predict_sc'] = model.predict(X_new)
        y_new_copy['prediction'] = target_scaler.inverse_transform(y_new_copy['predict_sc'])
    else:
        y_new_copy['prediction'] = model.predict(X_new)

    return y_new_copy

if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    import tools
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.feature_selection import mutual_info_regression

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
    features_categorical = [
                            'regio2',
                            #'geo_plz',
                            'yearConstructedRange_new',
                            'interiorQual_new',
                            #'condition_new',
                            'lift',
                            'cellar',
                            'hasKitchen',
                            'balcony',
                            'newlyConst',
                            #'firingTypes_new',
                            #'floor_new'
                            ]
    # Numerische features:
    features_numerical = [
                          'livingSpace',
                          #'picturecount'
                          ]

    ################
    # Eigene Daten #
    ################
    data_new = [[355, 'Münster', 1, 0,  False, True, False, True, False, 25],
                [810, 'Münster', 1, 0,  False, True, False, True, False,  63],
                [580, 'Münster', 1, 0,  False, True, False, True, False,  80],
                [550, 'Münster', 2, 0,  True, True, True, True, False, 52],
                [1015, 'Hamburg', 1, 1,  False, False, True, False, False,  78],
                [940, 'Münster', 1, 0,  False, True, False, True, False,  70]]
                
    df_new = pd.DataFrame(data_new, columns=[target] + features_categorical + features_numerical)

    #################
    # Preprocessing #
    #################

    df, X_train, X_test, X_new, y_train, y_test, y_new, target_scaler = preprocessing_pipepline(df=df, df_new=df_new, target=target, features_categorical=features_categorical, features_numerical=features_numerical, exponents=[0.5,2], test_set_size=0.3, random_state=42)

    ################
    # Modellscores #
    ################

    #Lineare Regression
    lg = LinearRegression()
    model_lg, mape_score_lg, mae_score_lg, mse_score_lg, R2_score_lg = model_scores(model=lg, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, target_scaler=target_scaler)

    # Koeffizienten (Achtung, skaliert!)
    lg_coef = pd.DataFrame(model_lg.coef_, columns = X_train.columns)

    #RandomForest
    # rf = RandomForestRegressor(n_estimators= 50, max_depth=20, random_state=1)
    # results_rf = model_scores(model=rf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, target_scaler=target_scaler)

    #AdaBoost
    #ada = AdaBoostRegressor(learning_rate=0.3, n_estimators=50, random_state=1)
    #results_ada = model_scores(model=ada, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, target_scaler=target_scaler)

    ###########################
    # Vorhersage eigene Daten #
    ###########################

    y_new_predict_lg = predict_new(model=model_lg, X_new=X_new, y_new=y_new, target_scaler=target_scaler, scaled_target=True)
    ###########
    # Ausgabe #
    ###########
   
    print("""Mean Absolute Percentage Error: {3:.3f} % \n
Mean Absolute Error: {0:.3f} €\n
Root Mean Squared Error: {1:.3f} €\n
R^2 - Score: {2:.3f}%""".format(mae_score_lg, mse_score_lg, 100*R2_score_lg, mape_score_lg))
    print(y_new_predict_lg[['baseRent','prediction']])

#schlechte_plz = df.geo_plz.value_counts().loc[df.geo_plz.value_counts() < 70]
#df['geo_plz'].loc[df['geo_plz'].isin(schlechte_plz.index)] = 00000

# plz crazy: 17033, 17034, 17036
