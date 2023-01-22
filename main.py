from pandas import read_csv
from src.propensity_matching import PsMatch


if __name__ == "__main__":
    df = read_csv("data/df.csv")
    match_init = PsMatch(
        data=df,
        group='acquirer',
        categorical_columns=['target', 'order_total_spent', 'credit_card_type', 'credit_card_company', 'model_score_category']
        )
    match_init.get_propensity_score()
