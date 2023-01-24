import pandas as pd

from src.plots import ScorePlotter
from src.propensity_matching import PrepData, PScorer
from src.find_similarities import ObsMatcher

PS_GROUP = 'acquirer'
TARGET = 'target'
FILE_PATH = 'data/df.csv'

if __name__ == "__main__":
    # load data
    data = PrepData(FILE_PATH, group=PS_GROUP, target="target",  index_col="id")

    # calculate ps scores
    scorer = PScorer()
    scorer.fit(data.input, data.group_label)
    propensity_scores = scorer.predict(data.input)
    ps_scores = pd.Series(propensity_scores, data.input.index)

    # check ps scores in roc curve
    plotter = ScorePlotter(data)
    plotter.plot_roc_curve(ps_scores, data.group_label)

    # match scores between groups
    matcher = ObsMatcher(n_matches=1, caliper=0.001)
    matcher.match_scores(ps_scores, data.group_label)
    plotter.plot_smd_comparison(cols=data.input.columns.to_list()) # data.input, data.input.columns.to_list(), PS_GROUP
