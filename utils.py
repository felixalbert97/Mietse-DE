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
