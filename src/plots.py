from math import sqrt
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from seaborn import barplot
from sklearn import metrics
from pandas.api.types import is_categorical_dtype


class ScorePlotter:

    @staticmethod
    def plot_roc_curve(predictions, y_true):
        fpr, tpr, threshold = metrics.roc_curve(y_true, predictions)
        roc_auc = metrics.auc(fpr, tpr)
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
    def calc_standardization(treated_metric: Series, untreated_metric: Series) -> float:
        return sqrt(
            (
                (treated_metric.count() - 1) * treated_metric.std() ** 2 +
                (untreated_metric.count() - 1) * untreated_metric.std() ** 2) / (
                treated_metric.count() + untreated_metric.count() - 2)
            )

        # return round(
        #     (mean_treated - mean_untreated) / sqrt(
        #         ((count_treated - 1) * pow(std_treated, 2) + (count_untreated - 1) * pow(std_untreated, 2)) / (
        #                 count_treated + count_untreated - 2)
        #         ),
        #     3)

    @staticmethod
    def calculate_smd(array: Series, treatment: Series) -> float:
        if is_categorical_dtype(array):
            array: Series = array.cat.codes
        treated_metric: Series = array[array.index.isin(treatment[treatment].index)]
        untreated_metric: Series = array[~array.index.isin(treatment[treatment].index)]
        mean_difference: float = treated_metric.mean() - untreated_metric.mean()
        denominator: float = ScorePlotter.calc_standardization(treated_metric, untreated_metric)
        return round(mean_difference / denominator, 3)

    @staticmethod
    def plot_smd_comparison(data, matched_index, treatment):
        smd_data = []
        matched_data = data[data.index.isin(matched_index)]
        for col in data.columns:
            feature_before_match: float = ScorePlotter.calculate_smd(
                array=data[col],
                treatment=treatment
                )
            feature_after_match: float = ScorePlotter.calculate_smd(
                array=matched_data[col],
                treatment=treatment
                )
            smd_data.append([col, 'before_match', feature_before_match])
            smd_data.append([col, 'after_match', feature_after_match])

        res = DataFrame(smd_data, columns=['variable', 'matching', 'mean_difference'])
        sn_plot = barplot(data=res, y='variable', x='mean_difference', hue='matching', orient='h')
        sn_plot.set(title='Standardised Mean differences')
        return plt.show()
