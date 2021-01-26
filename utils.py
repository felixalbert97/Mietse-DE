import numpy as np
import pandas as pd

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