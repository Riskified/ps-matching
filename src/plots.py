import math

from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sns
from seaborn import barplot
from sklearn import metrics


class SimilarObs:
    def __init__(self, matched_data: DataFrame):
        self.matched_data = matched_data
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
        data[cat_column] = data[cat_column].apply(lambda x: x.astype('category')).apply(lambda x: x.cat.codes)
        treated_metric = data[data[treatment_name] == True][col_name]
        untreated_metric = data[data[treatment_name] == False][col_name]

        mean_treated = treated_metric.mean()
        mean_untreated = untreated_metric.mean()
        count_treated = treated_metric.count()
        count_untreated = untreated_metric.count()
        std_treated = treated_metric.std()
        std_untreated = untreated_metric.std()

        d = (mean_treated - mean_untreated) \
            / math.sqrt \
                (((count_treated
                    - 1) * std_treated ** 2 + (count_untreated - 1) * std_untreated ** 2) /
                 (count_treated +
                    count_untreated - 2
                     )
                    )
        return round(d, 3)

    def plot_smd_comparison(self, cols, sns=None):
        smd_data = []
        for cl in cols:
            smd_data.append \
                (
                    [cl, 'before_match', self.calculate_smd
                    (
                        data=self.data,
                        col_name=cl,
                        treatment_name=self.logical_column_name
                    )
            ])
            smd_data.append \
                (
                    [cl, 'after_match', self.calculate_smd
                    (
                        data=self.matched_data,
                        col_name=cl,
                        treatment_name=self.logical_column_name
                    )
            ])

            res = DataFrame(smd_data, columns=['variable', 'matching', 'mean_difference'])

            sn_plot = barplot(data=res, y='variable', x='mean_difference', hue='matching', orient='h')
            sn_plot.set(title='Standardised Mean differences')
            return plt.show()
