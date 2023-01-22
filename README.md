# ps-matching
python class to perform propensity score matching

# How to Use the class:
## Class initiation:
match_init = PsMatch(data=df, group='the matching group', minority_class_in_group='the group minority', categorical_columns=[list of categorical columns])

## Create matched dataset:
match_data = match_init.create_match_df(nmatches=1, caliper=0.001)

## Get ROC curve to asses model fit:
match_init.plot_roc_curve()

## Plot SMD before and after matching:
init.plot_smd_comparison(cols=[list of columns to compere])

