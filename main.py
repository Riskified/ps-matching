import pandas as pd

from src.propensity_matching import PrepData, PScorer
from src.find_similarities import ObsMatcher

CAT_FEATURES = ['target', 'order_total_spent', 'credit_card_type', 'credit_card_company', 'model_score_category']
PS_GROUP = 'acquirer'
TARGET = 'target'

if __name__ == "__main__":
    data = PrepData('data/df.csv', group=PS_GROUP, target="target",  index_col="id")
    scorer = PScorer() # categorical_columns=CAT_FEATURES
    scorer.fit(data.input, data.group_label)
    propensity_scores = scorer.predict(data.input)



    temp = pd.Series(propensity_scores, data.input.index)
    matcher = ObsMatcher(n_matches=1, caliper=0.001)
    matcher.match_scores(temp, data.group_label)
