from typing import List
from numpy import log
from pandas import DataFrame, Series, concat, read_csv
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

CLF_PARAMS = {
    "C": 10,  # 1e4,
    "class_weight": "balanced",
    "penalty": "l2",
    "max_iter": 1000
    }


class PrepData:
    group_label = Series(dtype=bool)
    target_label = Series(dtype=bool)

    def __init__(self, file_path: str, group: str, target: str, index_col: str):
        data: DataFrame = read_csv(file_path, index_col=index_col)
        self.input: DataFrame = data.drop([group, target], axis=1)
        print(f'loaded data with {len(data)} observations')
        self.group_label = self.create_group_label(data[group])
        self.target_label = self.create_group_label(data[target])

    @staticmethod
    def get_minority_class(label_col) -> str:
        return label_col.value_counts().tail(1).index[0]

    @staticmethod
    def create_group_label(label: Series):
        logical_label: Series = label == PrepData.get_minority_class(label)
        print(logical_label.value_counts(normalize=True))
        print(f'The minority class contains {(logical_label == True).sum()} observations')
        return logical_label


class PScorer(BaseEstimator):
    """
        PScorer Class -- creates balanced dataset using propensity score matching
        Parameters
        ----------
        data : DataFrame
            Data with the group variable and the label_col of covariatesto be balanced
        group : str
            The variable that indicates the intervention
        minority_class_in_group : str
            The minority class of the group (the intervention)
        categorical_columns : list
            List of categorical variables
    """

    ct = make_column_transformer(
        (OneHotEncoder(), make_column_selector(dtype_include=object)),
        remainder='passthrough'
        )

    pipe_ = make_pipeline(
        ct,
        LogisticRegression(**CLF_PARAMS)
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
