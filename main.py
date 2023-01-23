from src.propensity_matching import PrepData, PScorer
from src.find_similarities import ObsMatcher

CAT_FEATURES = ['target', 'order_total_spent', 'credit_card_type', 'credit_card_company', 'model_score_category']
PS_GROUP = 'acquirer'

if __name__ == "__main__":
    data = PrepData('data/df.csv', group=PS_GROUP)
    scorer = PScorer(categorical_columns=CAT_FEATURES)
    scorer.fit(data.input_data, data.label)
    scorer.predict(data.input_data)
    matcher = ObsMatcher(scorer, n_matches=1, caliper=0.001)
    matcher.create_match_df(data.input_data, data.label)
