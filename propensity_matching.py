# get relevant packages
from typing import List

import pandas as pd
import numpy as np
import math
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'


class PsMatch:
    """
        PsMatch Class -- creates balanced dataset using propensity score matching
        Parameters
        ----------
        data : pd.DataFrame
            Data with the group variable and the set of covariatesto be balanced
        group : str
            The variable that indicates the intervention
        minority_class_in_group : str
            The minority class of the group (the intervention)
        categorical_columns : list
            List of categorical variables
    """
    def __init__(self, data: pd.DataFrame, group: str, minority_class_in_group: str, categorical_columns: List[str]):
        self.data = data
        self.group = group
        self.minority_class_in_group = minority_class_in_group
        self.categorical_columns = categorical_columns
        self.n_observations: int = len(data)
        print('loaded data with {} observations'.format(self.n_observations))

        self.data[f'{self.group}_logical'] = self.data[self.group] == self.minority_class_in_group
        self.logical_column_name = self.data[['{}_logical'.format(self.group)]].columns[0]
        self.matched_data = None
        print('The minority class contains {} observations'.format(
            self.data.loc[self.data[self.group] == self.minority_class_in_group].shape[0])
        )

    def transform_categorical_columns(self):
        ct = ColumnTransformer([("onehot", OneHotEncoder(sparse=False, drop="first"), self.categorical_columns)],
                               remainder='drop')
        # todo: change remainder='passthrough'
        x_data = self.data.drop([self.group, self.logical_column_name], axis=1)
        x_categorical = x_data[self.categorical_columns]
        x_other = x_data.drop(self.categorical_columns, axis=1)
        x_categorical_transformed = ct.fit_transform(x_categorical)
        return pd.concat([x_other, pd.DataFrame(x_categorical_transformed,
                                                columns=ct.get_feature_names_out())], axis=1)

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
        data_with_ps['log_propensity_score'] = np.array([PsMatch.logit(x) for x in data_with_ps['propensity_score']])

    # todo: add function that predict on new data

    def get_propensity_score(self):
        x_transformed = self.transform_categorical_columns()
        model = self.fit_ps_model(x_transformed, self.data[self.logical_column_name])
        data_with_ps = self.data.assign(propensity_score=model.predict_proba(x_transformed)[:, 1])
        data_with_ps['log_propensity_score'] = np.array([self.logit(x) for x in data_with_ps['propensity_score']])
        return data_with_ps

    def create_match_df(self, nmatches=1, caliper=0.001):
        """
            create_match_df method -- finds similar observations and match between intervention and control
            Parameters
            ----------
            nmatches : int
                set the number of n controls for each minor class (intervention group)
            caliper : float
                set the minimal distance for matching between intervention and control
            :return
                matched Dataframe
        """
        self.nmatches = nmatches
        self.caliper = caliper

        data_scores = self.get_propensity_score()
        intervention_group = data_scores[data_scores[self.logical_column_name] == True][['propensity_score']]
        control_group = data_scores[data_scores[self.logical_column_name] == False][['propensity_score']]
        result, match_ids = [], []
        print('starting matching process, this might take a time')

        for i in range(len(intervention_group)):
            match_id = i
            score = intervention_group.iloc[i]

            # todo: add potential matches according to caliper
            matches = abs(control_group - score).sort_values('propensity_score')
            matches['in_caliper'] = matches['propensity_score'] <= self.caliper
            matches_in_caliper = matches[matches['in_caliper'] == True]
            matches_in_caliper_n = matches_in_caliper.head(self.nmatches)

            if len(matches_in_caliper_n) == 0:
                continue

            if len(matches_in_caliper_n) < self.nmatches:
                select = matches_in_caliper_n.shape[0]
            else:
                select = self.nmatches

            chosen = np.random.choice(matches_in_caliper_n.index, min(select, self.nmatches), replace=False)
            result.extend([intervention_group.index[i]] + list(chosen))
            match_ids.extend([i] * (len(chosen) + 1))
            control_group = control_group.drop(index=chosen)

        matched_data = data_scores.loc[result]
        matched_data['match_id'] = match_ids
        matched_data['record_id'] = matched_data.index
        self.matched_data = matched_data
        print('please note: the matched dataset contains {} observations from the minority class'.format(
            self.matched_data.loc[self.matched_data[self.group] == self.minority_class_in_group].shape[0]))

        return self.matched_data

    def plot_roc_curve(self):

        data_with_preds = self.get_propensity_score()
        # calculate the fpr and tpr for all thresholds of the classification
        preds = data_with_preds['propensity_score']
        y_true = data_with_preds[self.group]
        fpr, tpr, threshold = metrics.roc_curve(y_true, preds, pos_label=self.minority_class_in_group)
        roc_auc = metrics.auc(fpr, tpr)

        # method I: plt
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        return plt.show()

    @staticmethod
    def calculate_smd(data, col_name, treatment_name):

        cat_column = data[[col_name, treatment_name]].select_dtypes(['object']).columns
        data[cat_column] = data[cat_column].apply(lambda x: x.astype('category'))
        data[cat_column] = data[cat_column].apply(lambda x: x.cat.codes)

        treated_metric = data[data[treatment_name] == True][col_name]
        untreated_metric = data[data[treatment_name] == False][col_name]

        mean_treated = treated_metric.mean()
        mean_untreated = untreated_metric.mean()
        count_treated = treated_metric.count()
        count_untreated = untreated_metric.count()
        std_treated = treated_metric.std()
        std_untreated = untreated_metric.std()

        d = (mean_treated - mean_untreated) \
            / math.sqrt(((count_treated - 1) * std_treated ** 2 + (count_untreated - 1) * std_untreated ** 2) /
                        (count_treated + count_untreated - 2))
        return round(d, 3)

    def plot_smd_comparison(self, cols):
        smd_data = []
        for cl in cols:
            smd_data.append([cl, 'before_match', self.calculate_smd(data=self.data,
                                                                    col_name=cl,
                                                                    treatment_name=self.logical_column_name)])
            smd_data.append([cl, 'after_match', self.calculate_smd(data=self.matched_data,
                                                                   col_name=cl,
                                                                   treatment_name=self.logical_column_name)])

        res = pd.DataFrame(smd_data, columns=['variable', 'matching', 'mean_difference'])

        sn_plot = sns.barplot(data=res, y='variable', x='mean_difference', hue='matching', orient='h')
        sn_plot.set(title='Standardised Mean differences')
        return plt.show()
