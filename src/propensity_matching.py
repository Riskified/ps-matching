from typing import List
from numpy import log
from pandas import DataFrame, Series, concat, read_csv
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

CLF_PARAMS = {
    "C": 1e4,
    "class_weight": "balanced",
    "penalty": "l2",
    "max_iter": 1000
    }


class PrepData:
    label = Series(dtype=bool)
    minority_class_in_group: str

    def __init__(self, file_path: str, group):
        data: DataFrame = read_csv(file_path)
        self.input_data = data.drop(group, axis=1)
        self.group_data = data[group]
        print(f'loaded data with {len(data)} observations')
        self.create_label()

    def get_minority_class(self):
        return self.group_data.value_counts().tail(1).index[0]

    def create_label(self):
        self.label: Series = self.group_data == self.get_minority_class()
        self.label.name = f'{self.group_data.name}_logical'
        print(self.label.value_counts(normalize=True))
        print(
            f'The minority class contains {self.label.value_counts()[True]} observations'
            )  # [self.get_minority_class()]


class PScorer(BaseEstimator):
    """
        PScorer Class -- creates balanced dataset using propensity score matching
        Parameters
        ----------
        data : DataFrame
            Data with the group variable and the set of covariatesto be balanced
        group : str
            The variable that indicates the intervention
        minority_class_in_group : str
            The minority class of the group (the intervention)
        categorical_columns : list
            List of categorical variables
    """

    def __init__(self, categorical_columns):
        ct = ColumnTransformer(
            [
                ("onehot", OneHotEncoder(sparse_output=False, drop="first"), categorical_columns)
                ], remainder='passthrough'
            )

        self.pipe_ = Pipeline(
            steps=[
                ('catfeatures', ct),
                ('clf', LogisticRegression(**CLF_PARAMS))
                ]
            )

    @staticmethod
    def logit(ps_score):
        return log(ps_score / (1 - ps_score))

    def fit(self, X, y):
        self.pipe_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        ps_score = self.pipe_.predict_proba(X)[:, 1]
        return self.logit(ps_score)
