import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def categorical_dist_imputer(df, column, inplace = False):
    """Imputes NaN-values from the categorical column named column in the DataFrame df
    by picking random values distributed according to the empirical distribution of the
    non-NaN-values """
    null_values = df.loc[df[column].isnull()].shape[0]
    non_null_values = df.size - null_values

    category_sizes = df.groupby(column).size()
    non_null_values = category_sizes.sum()
    category_percentage = category_sizes/non_null_values
    category_quantiles = category_percentage.cumsum()

    random_numbers = np.random.uniform(0,1,null_values)

    random_floors = [None]*null_values
    for i in range(len(random_numbers)):
        for (category, quantile) in category_quantiles.iteritems():
            if random_numbers[i] <= quantile:
                random_floors[i] = category
                break
    
    if inplace == False:
        df_copy = df.copy()
        df_copy.loc[df_copy[column].isnull(), column] = random_floors
        return df_copy
    else:
        df.loc[df[column].isnull(), column] = random_floors
        return df

def generate_dummies(ts, cols, trap=False):
    # iterate over each column which you want to dummify
    for col in cols:
        # create dummy variables out of the column
        dummies = pd.get_dummies(ts[col], prefix = col)
        # if trap is true we will drop one of the 
        # created dummies to avoid the dummy trap problem
        if trap:
            ts = ts.join(dummies.iloc[:, :-1])
        else:
            ts = ts.join(dummies)
    # drop the normal columns since you now have dummy variables
    ts.drop(cols, axis = 1, inplace = True)
    return ts

def scale_targets(y_train, y_test):
    """ Input: target columns ((:,1)-shaped DataFrame)
        Output: target DataFrames appended by a scaled column"""
    sc = StandardScaler()
    # fit and transfrom the target values of your train data
    y_train['target_sc'] = sc.fit_transform(y_train.values)
    
    # transfrom the target values of your test data
    # and save it as target_sc
    y_test['target_sc'] = sc.transform(y_test.values)
    
    # output the scaler and both training and testing datasets
    # you will need the scaler later to retransfrom your predictions 
    # back to real target values
    return sc, y_train, y_test

def scale_features(X_train,X_test):
    """ Input: DataFrames to scale 
        Output: used Scaler, scaled DataFrames """
    sc = StandardScaler()
    # fit and transform the train data
    X_train_sc = sc.fit_transform(X_train.values)
    # reassign scaled data to a Dataframe
    X_train_sc_df = pd.DataFrame(X_train_sc, index=X_train.index, columns=X_train.columns)

    # fit and transform the test data, with the same(!) Scaler
    X_test_sc = sc.transform(X_test.values)
    # reassign scaled data to a Dataframe
    X_test_sc_df = pd.DataFrame(X_test_sc, index=X_test.index, columns=X_test.columns)

    return sc, X_train_sc_df, X_test_sc_df