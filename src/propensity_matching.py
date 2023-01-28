from numpy import log
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

CLF_PARAMS = {
    "C": 10,  # 1e4,
    "class_weight": "balanced",
    "penalty": "l2",
    "max_iter": 1000
    }


class PScorer(BaseEstimator):
    ct = make_column_transformer(
        (OneHotEncoder(), make_column_selector(dtype_include="category")),
        remainder='passthrough'
        )

    pipe_ = make_pipeline(ct, LogisticRegression(**CLF_PARAMS))

    @staticmethod
    def logit(ps_score):
        return log(ps_score / (1 - ps_score))

    def fit(self, X: DataFrame, y: Series):
        self.pipe_.fit(X, y)
        return self

    def predict(self, X: DataFrame) -> Series:
        # check_is_fitted(self)
        ps_score = self.pipe_.predict_proba(X)[:, 1]
        return Series(self.logit(ps_score), X.index)
