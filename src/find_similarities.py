from typing import Any, Dict, List

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

    def create_matches_table(self, match_ids: Dict[str, List[Any]]) -> DataFrame:
        col_names: List[str] = [f"matched_{i}" for i in range(1, self.n_matches + 1)]
        print(
            f'please note:'
            f'the matched dataset contains {len(match_ids)} observations from minority class'
            )
        return DataFrame(match_ids.values(), index=match_ids.keys(), columns=col_names)

    def match_scores(self, p_scores: Series, label: Series) -> List[int]:
        intervention_group: Series = p_scores[label]
        control_group: Series = p_scores[~label]
        match_ids = {}
        print('starting matching process, this might take a time')
        for index, score in tqdm(intervention_group.items(), total=len(intervention_group)):
            matches: Series = abs(control_group - score)
            matches_in_caliper: Series = matches[matches <= self.caliper]
            select: int = min(len(matches_in_caliper), self.n_matches)
            if select > 0:
                chosen: List[int] = random.choice(matches_in_caliper.index, select, replace=False).tolist()
                match_ids.update({index: chosen})
                control_group: Series = control_group.drop(index=chosen)

        matched_table: DataFrame = self.create_matches_table(match_ids)
        return matched_table.reset_index().melt()["value"].to_list()
