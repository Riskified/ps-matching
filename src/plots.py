from math import sqrt, pow
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from seaborn import barplot
from sklearn import metrics


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
    def calc_d(mean_treated, mean_untreated, count_treated, count_untreated, std_treated, std_untreated):
        return round(
            (mean_treated - mean_untreated) / sqrt(
                ((count_treated - 1) * pow(std_treated, 2) + (count_untreated - 1) * pow(std_untreated, 2)) / (
                        count_treated + count_untreated - 2)
                ), 3
            )

    @staticmethod
    def calculate_smd(data: DataFrame, col_name: str, treatment_name: Series) -> float:
        cat_column = data[[col_name]].select_dtypes(['object']).columns
        data[cat_column] = data[cat_column].apply(lambda x: x.astype('category')).apply(lambda x: x.cat.codes)
        treated_metric = data[treatment_name][col_name]
        untreated_metric = data[treatment_name][col_name]

        mean_treated: float = treated_metric.mean()
        mean_untreated: float = untreated_metric.mean()
        count_treated: float = treated_metric.count()
        count_untreated: float = untreated_metric.count()
        std_treated: float = treated_metric.std()
        std_untreated: float = untreated_metric.std()

        return ScorePlotter.calc_d(
            mean_treated, mean_untreated, count_treated, count_untreated, std_treated, std_untreated
            )

    @staticmethod
    def plot_smd_comparison(cols, data, matched_data, treatment):
        smd_data = []
        for cl in cols:
            feature_before_match: float = ScorePlotter.calculate_smd(
                data=data,
                col_name=cl,
                treatment_name=treatment
                )
            feature_after_match: float = ScorePlotter.calculate_smd(
                data=matched_data,
                col_name=cl,
                treatment_name=treatment
                )
            smd_data.append([cl, 'before_match', feature_before_match])
            smd_data.append([cl, 'after_match', feature_after_match])
            res = DataFrame(smd_data, columns=['variable', 'matching', 'mean_difference'])
            sn_plot = barplot(data=res, y='variable', x='mean_difference', hue='matching', orient='h')
            sn_plot.set(title='Standardised Mean differences')
            return plt.show()
