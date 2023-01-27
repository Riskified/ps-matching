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

        match_idss = {}
        print('starting matching process, this might take a time')
        popi = intervention_group.head(4)  # .items()
        for index, score in tqdm(popi.items()):  # intervention_group.items()
            matches: Series = abs(control_group - score).sort_values()

            in_caliper: Series = matches <= self.caliper
            matches_in_caliper: Series = matches[in_caliper]
            matches_in_caliper_n = matches_in_caliper.head(self.n_matches)
            if len(matches_in_caliper_n) == 0:
                break
            select: int = min(len(matches_in_caliper_n), self.n_matches)
            chosen = random.choice(matches_in_caliper.index, select, replace=False)
            match_idss.append({index: list(chosen)})
            control_group = control_group.drop(index=chosen)

        matched_ids = DataFrame(match_idss)
        print(
            f'please note:'
            f'the matched dataset contains {matched_ids.index.isin(label).sum()} observations from minority class'
            )
        return matched_ids
