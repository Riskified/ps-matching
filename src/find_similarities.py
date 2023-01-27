from numpy import random
from pandas import DataFrame, Series
from tqdm import tqdm


class ObsMatcher:
    def __init__(self, n_matches: int, caliper: float):
        """
            create_match_df method -- finds similar observations and match between intervention and control
            Parameters
            ----------
            n_matches : int
                label_col the number of n controls for each minor class (intervention group)
            caliper : float
                label_col the minimal distance for matching between intervention and control
            :return
                matched Dataframe
        """
        self.n_matches = n_matches
        self.caliper = caliper

    def match_scores(self, p_scores: DataFrame, label):

        intervention_group: Series = p_scores[label]
        control_group: Series = p_scores[~label]
        len(control_group) # 38708
        match_ids = {}
        print('starting matching process, this might take a time')

        for index, score in tqdm(intervention_group.items(), total=len(intervention_group)):
            matches: Series = abs(control_group - score)
            matches_in_caliper: Series = matches[matches <= self.caliper]
            select: int = min(len(matches_in_caliper), self.n_matches)
            if select > 0:
                chosen = random.choice(matches_in_caliper.index, select, replace=False).tolist()
                match_ids.update({index: chosen})
                control_group = control_group.drop(index=chosen)

        len(control_group)
        len(p_scores[~label])
        random.choice(matches_in_caliper.index, select, replace=False)

        matched_ids = DataFrame(match_ids)
        print(
            f'please note:'
            f'the matched dataset contains {matched_ids.index.isin(label).sum()} observations from minority class'
            )
        return matched_ids
