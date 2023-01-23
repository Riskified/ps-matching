from tqdm import tqdm
from numpy import random
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class ObsMatcher:
    def __init__(self, p_scorer: BaseEstimator, n_matches: int, caliper: float):
        """
            create_match_df method -- finds similar observations and match between intervention and control
            Parameters
            ----------
            n_matches : int
                set the number of n controls for each minor class (intervention group)
            caliper : float
                set the minimal distance for matching between intervention and control
            :return
                matched Dataframe
        """
        if check_is_fitted(p_scorer) is False:
            raise ModuleNotFoundError
        self.p_scorer = p_scorer
        self.n_matches = n_matches
        self.caliper = caliper

    def create_match_df(self, data: DataFrame, label):

        data['propensity_score'] = self.p_scorer.predict(data)
        intervention_group: Series = data[label]['propensity_score']
        control_group: Series = data[~label]['propensity_score']

        result, match_ids = [], []
        print('starting matching process, this might take a time')

        for i in tqdm(range(len(intervention_group))):
            score = intervention_group.iloc[i]

            # todo: add potential matches according to caliper
            matches: Series = abs(control_group - score).sort_values()
            in_caliper: Series = matches <= self.caliper
            matches_in_caliper: Series = matches[in_caliper]
            matches_in_caliper_n = matches_in_caliper.head(self.n_matches)

            if len(matches_in_caliper_n) == 0:
                continue

            if len(matches_in_caliper_n) < self.n_matches:
                select: int = len(matches_in_caliper_n)
            else:
                select: int = self.n_matches

            chosen = random.choice(matches_in_caliper_n.index, min(select, self.n_matches), replace=False)
            result.extend([intervention_group.index[i]] + list(chosen))
            match_ids.extend([i] * (len(chosen) + 1))

        matched_data = data.loc[result]
        matched_data['match_id'] = match_ids
        matched_data['record_id'] = matched_data.index

        print(f'please note:'
              f'the matched dataset contains {matched_data.index.isin(label).sum()} observations from minority class')
        return matched_data
