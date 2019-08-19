import pandas as pd
import numpy as np
from c2_def import load_housing_data, create_train_set, data_cleaning

housing = load_housing_data()

housing["income_cat"] =np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5,5.0,inplace=True)

strain_train_set, strain_test_set = create_train_set(housing)

housing=strain_train_set.copy()

print(pd.DataFrame(housing).head())

from sklearn.base import BaseEstimator, TransformerMixin

# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

print(rooms_ix)

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

print(pd.DataFrame(housing_extra_attribs).head())



# housing["rooms_per_household"]= housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"]= housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"]= housing["population"]/housing["households"]

# stats, X, Xt = data_cleaning(housing)

# print(stats,'\n',X,'\n',Xt)
# print('done')
