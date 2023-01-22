# get relevant packages
from typing import List
import math

from numpy import array
from pandas import DataFrame, Series, concat
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# pd.options.mode.chained_assignment = None  # default='warn'


class PsMatch:
    """
        PsMatch Class -- creates balanced dataset using propensity score matching
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
    matched_data = DataFrame()
    label = Series()
    minority_class_in_group: str

    def __init__(self, data: DataFrame, group, categorical_columns: List[str]):
        self.input_data = data.drop(group, axis=1)
        self.group_data = data[group]
        self.categorical_columns = categorical_columns
        print(f'loaded data with {len(data)} observations')
        self.create_label()

    def get_minority_class(self):
        return self.group_data.value_counts().tail(1).index[0]

    def create_label(self):
        self.label: Series = self.group_data == self.get_minority_class()
        self.label.name = f'{self.group_data.name}_logical'
        print(self.label.value_counts(normalize=True))
        # print(f'The minority class contains {self.label.value_counts()[self.group]} observations')

    def transform_categorical_columns(self):
        ct = ColumnTransformer(
            [
                ("onehot", OneHotEncoder(sparse=False, drop="first"), self.categorical_columns)],
            remainder='drop'
            )
        # todo: change remainder='passthrough'
        x_data = self.data.drop([self.group, self.logical_column_name], axis=1)
        x_categorical = x_data[self.categorical_columns]
        x_other = x_data.drop(self.categorical_columns, axis=1)
        x_categorical_transformed = ct.fit_transform(x_categorical)
        return concat(
            [x_other, DataFrame(
                x_categorical_transformed,
                columns=ct.get_feature_names_out()
                )], axis=1
            )

    @staticmethod
    def logit(p):
        logit_value = math.log(p / (1 - p))
        return logit_value

    @staticmethod
    def fit_ps_model(x_data, y_col):
        return LogisticRegression(C=1e6).fit(x_data, y_col)

    @staticmethod
    def predict_ps(data, model):
        data_with_ps = data.assign(propensity_score=model.predict_proba(data)[:, 1])
        data_with_ps['log_propensity_score'] = array([PsMatch.logit(x) for x in data_with_ps['propensity_score']])

    # todo: add function that predict on new data

    def get_propensity_score(self):
        # x_transformed = self.transform_categorical_columns()
        # model = self.fit_ps_model(x_transformed, self.data[self.logical_column_name])

        ct = ColumnTransformer(
            [
                ("onehot", OneHotEncoder(sparse=False, drop="first"), self.categorical_columns)
                ], remainder='passthrough'
            )

        pipe = Pipeline(
            steps=[
                ('catfeatures', ct),
                ('clf', LogisticRegression(C=1e6))
                ]
            )

        pipe.fit(self.input_data, self.label)
        ps_score = pipe.predict_proba(self.input_data)[:, 1]
        pipe.predict_log_proba(self.input_data)[:, 1]

        # data_with_ps = self.data.assign(propensity_score=model.predict_proba(x_transformed)[:, 1])
        # data_with_ps['log_propensity_score'] = np.array([self.logit(x) for x in data_with_ps['propensity_score']])
        # return data_with_ps
