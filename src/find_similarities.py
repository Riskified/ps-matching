from numpy import random


class SimilarObs:
    def create_match_df(self, nmatches=1, caliper=0.001):
        """
            create_match_df method -- finds similar observations and match between intervention and control
            Parameters
            ----------
            nmatches : int
                set the number of n controls for each minor class (intervention group)
            caliper : float
                set the minimal distance for matching between intervention and control
            :return
                matched Dataframe
        """
        self.nmatches = nmatches
        self.caliper = caliper

        data_scores = self.get_propensity_score()
        intervention_group = data_scores[data_scores[self.logical_column_name] == True][['propensity_score']]
        control_group = data_scores[data_scores[self.logical_column_name] == False][['propensity_score']]
        result, match_ids = [], []
        print('starting matching process, this might take a time')

        for i in range(len(intervention_group)):
            match_id = i
            score = intervention_group.iloc[i]

            # todo: add potential matches according to caliper
            matches = abs(control_group - score).sort_values('propensity_score')
            matches['in_caliper'] = matches['propensity_score'] <= self.caliper
            matches_in_caliper = matches[matches['in_caliper'] == True]
            matches_in_caliper_n = matches_in_caliper.head(self.nmatches)

            if len(matches_in_caliper_n) == 0:
                continue

            if len(matches_in_caliper_n) < self.nmatches:
                select = matches_in_caliper_n.shape[0]
            else:
                select = self.nmatches

            chosen = random.choice(matches_in_caliper_n.index, min(select, self.nmatches), replace=False)
            result.extend([intervention_group.index[i]] + list(chosen))
            match_ids.extend([i] * (len(chosen) + 1))
            control_group = control_group.drop(index=chosen)

        matched_data = data_scores.loc[result]
        matched_data['match_id'] = match_ids
        matched_data['record_id'] = matched_data.index
        self.matched_data = matched_data
        print \
            ('please note: the matched dataset contains {} observations from the minority class'.format(
            self.matched_data.loc[self.matched_data[self.group] == self.minority_class_in_group].shape[0
                ]
            ))

        return self.matched_data
