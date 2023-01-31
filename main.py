from typing import List

import pandas as pd
from pandas import Series
from src.plots import ScorePlotter
from src.prepare_data import PrepData
from src.propensity_matching import PScorer
from src.find_similarities import ObsMatcher

PS_GROUP = 'acquirer'
TARGET = 'target'
FILE_PATH = 'data/df.csv'


if __name__ == "__main__":
    data = PrepData(FILE_PATH, group=PS_GROUP, target=TARGET,  index_col="id")
    scorer = PScorer()
    scorer.fit(data.input, data.group_label)
    ps_scores: Series = scorer.predict(data.input)
    ScorePlotter.plot_roc_curve(ps_scores, data.group_label)

    matcher = ObsMatcher(n_matches=1, caliper=0.001)
    matched_index: List[int] = matcher.match_scores(ps_scores, data.group_label)

    ScorePlotter.plot_smd_comparison(
        data=data.input,
        matched_index=matched_index,
        treatment=data.group_label
        )

    matched_data = data.input[data.input.index.isin(matched_index)].join(data.target_label).join(data.group_label)





