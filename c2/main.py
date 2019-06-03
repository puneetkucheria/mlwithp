#import pandas as pd
import numpy as np
from c2_def import create_train_set, load_housing_data, data_cleaning




housing = load_housing_data()
housing["income_cat"] =np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5,5.0,inplace=True)
strain_train_set, strain_test_set = create_train_set(housing)

housing=strain_train_set.copy()
housing["rooms_per_household"]= housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]= housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]= housing["population"]/housing["households"]

stats, X, Xt = data_cleaning(housing)

print(stats,'\n',X,'\n',Xt)
print('done')
