# ps-matching
python class to perform propensity score matching

# How to Use the ps-matching code:
## Set global variables:
```
PS_GROUP = 'treatment'    # set the group variable (treatment/control)
TARGET = 'target'         # set the target variable, the outcome of interest
FILE_PATH = 'data/df.csv' # dataframe contains all dependant and independent variables
```

## PrepData Class initiation:
```
data = PrepData(FILE_PATH, group=PS_GROUP, target=TARGET,  index_col="id")
```

## initiate the PScorer and estimate the Propensity Score:
```
scorer = PScorer()
scorer.fit(data.input, data.group_label)
ps_scores: Series = scorer.predict(data.input)
```

## Get ROC curve to assess model fit:
```
ScorePlotter.plot_roc_curve(ps_scores, data.group_label) 
```

## Initiate the ObsMatcher class, set the matching ratio and the caliper:
```
matcher = ObsMatcher(n_matches=1, caliper=0.001)
matched_index: List[int] = matcher.match_scores(ps_scores, data.group_label) 
```

## Plot SMD before and after matching:
```
 ScorePlotter.plot_smd_comparison(
        data=data.input,
        matched_index=matched_index,
        treatment=data.group_label
        )
```

