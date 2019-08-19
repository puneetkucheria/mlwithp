import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer as Imputer

def load_housing_data():
    '''Load data form housing.csv in datasets folder
        no variable required csv_path Hardcoded'''
    csv_path="../datasets/housing/housing.csv"
    return pd.read_csv(csv_path)

def create_train_set(df):
    '''get test and train set for the data Frame
    need to pass dataframe'''
    split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["income_cat"]):
        strain_train_set = df.loc[train_index]
        strain_test_set = df.loc[test_index]
        for set_ in (strain_train_set, strain_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
        return strain_train_set, strain_test_set

def data_cleaning(df):

    #Data Cleaning for numbers
    imputer = Imputer(strategy="median")
    dfn = df.drop("ocean_proximity", axis=1)
    imputer.fit(dfn)
    Xn=pd.DataFrame(imputer.transform(dfn),columns=dfn.columns)

    #Data Cleaning for Text and Categorical Attributes
    encoder = LabelBinarizer(sparse_output=True)
    Xt = encoder.fit_transform(df["ocean_proximity"])

    return imputer.statistics_,Xn,Xt

